"""
XGBoost Walk-Forward Trainer
==============================
Trains one XGBClassifier per walk-forward fold using the labeled ensemble
dataset. Each fold trains on all data up to month T and predicts month T+1.
Uses early stopping on the last 6 months of training data as eval set.

Output
------
  models/ensemble_models.pkl
    List of dicts: [{date, model, feature_names, train_size, eval_auc}]

CLI
---
  python -m src.strategies.ensemble.model_trainer
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from xgboost import XGBClassifier

load_dotenv()
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
with open(ROOT / "config" / "settings.yaml") as f:
    CFG = yaml.safe_load(f)

PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Feature columns — everything except metadata and target
META_COLS = {"date", "ticker", "fwd_return", "xlk_return", "label"}

# Fixed XGBoost hyperparameters
XGB_PARAMS = dict(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=1.0,
    eval_metric="auc",
    early_stopping_rounds=20,
    random_state=42,
    use_label_encoder=False,
    verbosity=0,
    n_jobs=-1,
)

# Eval set: use last N months of training data
EVAL_MONTHS = 6


# ── Feature extraction ────────────────────────────────────────────────────────

def _get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return feature column names (everything except meta/target)."""
    return [c for c in df.columns if c not in META_COLS]


def _prepare_xy(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    """Extract X, y from a DataFrame. Fill NaNs with 0.0 (neutral signal)."""
    X = df[feature_cols].fillna(0.0).copy()
    y = df["label"].astype(int)
    return X, y


# ── Public API ────────────────────────────────────────────────────────────────

def train_walk_forward(
    labeled_df: Optional[pd.DataFrame] = None,
) -> list[dict]:
    """
    Train XGBClassifier models via walk-forward cross-validation.

    For each fold:
      - Train on all data up to month T
      - Use last 6 months of training data as eval_set for early stopping
      - Predict month T+1
      - Store the trained model + metadata

    Parameters
    ----------
    labeled_df : pre-loaded labeled dataset; if None, loads from parquet

    Returns
    -------
    list of dicts:
        [{date, model, feature_names, train_size, eval_auc, best_iteration}]
    """
    from src.strategies.ensemble.target_builder import walk_forward_folds

    # Load data
    if labeled_df is None:
        path = PROCESSED / "ensemble_labeled.parquet"
        if not path.exists():
            logger.error(f"Labeled dataset not found: {path}")
            logger.error("Run: python -m src.strategies.ensemble.target_builder")
            return []
        labeled_df = pd.read_parquet(path)
        labeled_df["date"] = pd.to_datetime(labeled_df["date"])

    feature_cols = _get_feature_cols(labeled_df)
    logger.info(f"Features ({len(feature_cols)}): {feature_cols}")

    # Check label balance for scale_pos_weight adjustment
    label_counts = labeled_df["label"].value_counts()
    n_neg = label_counts.get(0, 0)
    n_pos = label_counts.get(1, 0)
    imbalance = n_pos / max(n_neg, 1)
    params = XGB_PARAMS.copy()
    if imbalance < 0.67 or imbalance > 1.5:
        params["scale_pos_weight"] = n_neg / max(n_pos, 1)
        logger.info(f"Label imbalance detected — scale_pos_weight={params['scale_pos_weight']:.2f}")
    else:
        logger.info(f"Label balance OK: {n_pos}/{n_neg} ({imbalance:.2f} ratio)")

    # Walk-forward training
    results = []
    fold_num = 0

    for train_df, test_df in walk_forward_folds(labeled_df):
        fold_num += 1
        test_date = test_df["date"].iloc[0]

        # Split eval set from end of training data
        train_months = sorted(train_df["date"].unique())
        if len(train_months) > EVAL_MONTHS:
            eval_cutoff = train_months[-EVAL_MONTHS]
            eval_mask = train_df["date"] >= eval_cutoff
            fit_df = train_df[~eval_mask]
            eval_df = train_df[eval_mask]
        else:
            # Not enough months for a separate eval set — use last 20% of rows
            n = len(train_df)
            fit_df = train_df.iloc[: int(n * 0.8)]
            eval_df = train_df.iloc[int(n * 0.8):]

        X_fit, y_fit = _prepare_xy(fit_df, feature_cols)
        X_eval, y_eval = _prepare_xy(eval_df, feature_cols)
        X_test, y_test = _prepare_xy(test_df, feature_cols)

        # Train
        model = XGBClassifier(**params)
        model.fit(
            X_fit, y_fit,
            eval_set=[(X_eval, y_eval)],
            verbose=False,
        )

        # Eval AUC (from training history)
        eval_results = model.evals_result()
        eval_key = list(eval_results.keys())[0]  # 'validation_0'
        auc_history = eval_results[eval_key]["auc"]
        best_auc = max(auc_history)
        best_iter = model.best_iteration if hasattr(model, "best_iteration") else len(auc_history) - 1

        results.append({
            "date": test_date,
            "model": model,
            "feature_names": feature_cols,
            "train_size": len(fit_df),
            "eval_size": len(eval_df),
            "test_size": len(test_df),
            "eval_auc": best_auc,
            "best_iteration": best_iter,
        })

        if fold_num % 10 == 0 or fold_num <= 3:
            logger.info(
                f"  Fold {fold_num:3d} │ test={test_date.date()} │ "
                f"train={len(fit_df):5d} │ eval={len(eval_df):4d} │ "
                f"AUC={best_auc:.4f} │ iters={best_iter}"
            )

    logger.info(f"Training complete: {fold_num} folds")

    if results:
        aucs = [r["eval_auc"] for r in results]
        logger.info(
            f"  AUC: mean={np.mean(aucs):.4f}, "
            f"std={np.std(aucs):.4f}, "
            f"min={np.min(aucs):.4f}, "
            f"max={np.max(aucs):.4f}"
        )

    return results


# ── Persistence ───────────────────────────────────────────────────────────────

def save_models(results: list[dict], path: Path = None) -> None:
    """Save trained models to pickle."""
    if path is None:
        path = MODELS_DIR / "ensemble_models.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(f"Saved {len(results)} models → {path}  ({path.stat().st_size / 1024:.0f} KB)")


def load_models(path: Path = None) -> list[dict]:
    """Load trained models from pickle."""
    if path is None:
        path = MODELS_DIR / "ensemble_models.pkl"
    if not path.exists():
        logger.error(f"Model file not found: {path}")
        return []
    with open(path, "rb") as f:
        return pickle.load(f)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s │ %(levelname)-8s │ %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger.info("Starting XGBoost walk-forward training…")
    results = train_walk_forward()

    if not results:
        logger.warning("No models trained")
        return

    # Save
    save_models(results)

    # Summary table
    print(f"\n{'═' * 76}")
    print(f"  XGBoost Walk-Forward Training Summary")
    print(f"{'═' * 76}")
    print(f"  Total folds:  {len(results)}")
    print(f"  Features:     {len(results[0]['feature_names'])}")

    aucs = [r["eval_auc"] for r in results]
    print(f"  AUC mean:     {np.mean(aucs):.4f}")
    print(f"  AUC std:      {np.std(aucs):.4f}")
    print(f"  AUC [min,max]: [{np.min(aucs):.4f}, {np.max(aucs):.4f}]")
    print(f"{'═' * 76}")

    # Per-fold table
    print(f"\n  {'Fold':>4}  {'Test Date':>12}  {'Train':>6}  {'Eval':>5}  "
          f"{'Test':>5}  {'AUC':>7}  {'Iters':>5}")
    print(f"  {'─' * 4}  {'─' * 12}  {'─' * 6}  {'─' * 5}  "
          f"{'─' * 5}  {'─' * 7}  {'─' * 5}")

    for i, r in enumerate(results):
        auc_str = f"{r['eval_auc']:.4f}"
        marker = " ★" if r["eval_auc"] >= 0.55 else ""
        print(
            f"  {i+1:4d}  {r['date'].date()!s:>12}  {r['train_size']:6d}  "
            f"{r['eval_size']:5d}  {r['test_size']:5d}  {auc_str:>7}{marker}"
            f"  {r['best_iteration']:5d}"
        )

    # Feature importance (average across all folds)
    print(f"\n  Average Feature Importance (gain):")
    feat_names = results[0]["feature_names"]
    importance_sum = np.zeros(len(feat_names))
    for r in results:
        imp = r["model"].feature_importances_
        importance_sum += imp
    importance_avg = importance_sum / len(results)

    sorted_idx = np.argsort(importance_avg)[::-1]
    for rank, idx in enumerate(sorted_idx[:15]):
        bar = "█" * int(importance_avg[idx] / importance_avg[sorted_idx[0]] * 20)
        print(f"    {rank+1:2d}. {feat_names[idx]:<28s}  {bar}  {importance_avg[idx]:.4f}")

    print()


if __name__ == "__main__":
    main()
