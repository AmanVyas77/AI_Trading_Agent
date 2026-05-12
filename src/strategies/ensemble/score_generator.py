"""
Prediction Score Generator
============================
Loads walk-forward trained XGBoost models and the labeled feature matrix,
generates outperformance probability scores for each ticker-month, and
saves the scored dataset.

For each model:
  - model["date"] is the month the model was trained to predict
  - We score all rows in the labeled dataset matching that month
  - predict_proba[:, 1] gives P(outperform XLK next month)

Output
------
  data/processed/ensemble_scores.parquet
  Columns: [date, ticker, ensemble_score, label]

CLI
---
  python -m src.strategies.ensemble.score_generator
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"

META_COLS = {"date", "ticker", "fwd_return", "xlk_return", "label"}


# ── Load helpers ──────────────────────────────────────────────────────────────

def _load_models(path: Path = None) -> list[dict]:
    if path is None:
        path = MODELS_DIR / "ensemble_models.pkl"
    if not path.exists():
        logger.error(f"Model file not found: {path}")
        return []
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_labeled(path: Path = None) -> pd.DataFrame:
    if path is None:
        path = PROCESSED / "ensemble_labeled.parquet"
    if not path.exists():
        logger.error(f"Labeled dataset not found: {path}")
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


# ── Public API ────────────────────────────────────────────────────────────────

def generate_scores(
    models: Optional[list[dict]] = None,
    labeled_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Generate ensemble outperformance scores for each ticker-month.

    For each trained model, finds the matching month in the labeled dataset
    and computes P(outperform XLK) via predict_proba.

    Parameters
    ----------
    models     : list of model dicts from model_trainer; loads from pkl if None
    labeled_df : labeled dataset; loads from parquet if None

    Returns
    -------
    pd.DataFrame with columns: [date, ticker, ensemble_score, label]
    """
    if models is None:
        models = _load_models()
    if not models:
        logger.error("No models available")
        return pd.DataFrame(columns=["date", "ticker", "ensemble_score", "label"])

    if labeled_df is None:
        labeled_df = _load_labeled()
    if labeled_df.empty:
        logger.error("Labeled dataset is empty")
        return pd.DataFrame(columns=["date", "ticker", "ensemble_score", "label"])

    logger.info(f"Scoring {len(models)} models against {labeled_df.shape[0]} labeled rows")

    all_scores = []

    for i, entry in enumerate(models):
        model = entry["model"]
        test_date = entry["date"]
        feature_names = entry["feature_names"]

        # Find rows for this prediction month
        month_mask = labeled_df["date"] == test_date
        month_df = labeled_df[month_mask]

        if month_df.empty:
            logger.warning(f"  No data for {test_date.date()} — skipping")
            continue

        # Prepare features — use the model's feature list, fill NaN with 0.0
        # Handle columns that may have been dropped (e.g., fcf_yield)
        X = pd.DataFrame(0.0, index=month_df.index, columns=feature_names)
        for col in feature_names:
            if col in month_df.columns:
                X[col] = month_df[col].fillna(0.0).values

        # Score
        proba = model.predict_proba(X)[:, 1]

        month_scores = pd.DataFrame({
            "date": month_df["date"].values,
            "ticker": month_df["ticker"].values,
            "ensemble_score": proba,
            "label": month_df["label"].values,
        })
        all_scores.append(month_scores)

        if (i + 1) % 20 == 0:
            logger.info(
                f"  Scored {i+1}/{len(models)} months "
                f"(last: {test_date.date()}, {len(month_df)} tickers)"
            )

    if not all_scores:
        logger.warning("No scores generated")
        return pd.DataFrame(columns=["date", "ticker", "ensemble_score", "label"])

    result = pd.concat(all_scores, ignore_index=True)
    result = result.sort_values(["date", "ticker"]).reset_index(drop=True)

    logger.info(f"Generated {len(result)} scores across {result['date'].nunique()} months")

    return result


# ── Analysis helpers ──────────────────────────────────────────────────────────

def _compute_top_n_accuracy(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """
    For each month, select the top-N tickers by ensemble_score.
    Compute what fraction of them actually outperformed (label=1).
    """
    records = []
    for date, grp in df.groupby("date"):
        top = grp.nlargest(min(top_n, len(grp)), "ensemble_score")
        hit_rate = top["label"].mean()
        records.append({
            "date": date,
            "n_selected": len(top),
            "n_outperform": int(top["label"].sum()),
            "hit_rate": hit_rate,
            "mean_score": top["ensemble_score"].mean(),
        })
    return pd.DataFrame(records)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s │ %(levelname)-8s │ %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    scores = generate_scores()

    if scores.empty:
        logger.warning("No scores generated — parquet NOT saved")
        return

    # Save
    out_path = PROCESSED / "ensemble_scores.parquet"
    scores.to_parquet(out_path, index=False, engine="pyarrow")
    logger.info(f"Saved → {out_path}  ({out_path.stat().st_size / 1024:.0f} KB)")

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'═' * 72}")
    print(f"  Ensemble Score Generation Summary")
    print(f"{'═' * 72}")
    print(f"  Rows:        {len(scores)}")
    print(f"  Tickers:     {scores['ticker'].nunique()}")
    print(f"  Months:      {scores['date'].nunique()}")
    print(f"  Date range:  {scores['date'].min().date()} → {scores['date'].max().date()}")
    print(f"  Mean score:  {scores['ensemble_score'].mean():.4f}")
    print(f"  Score std:   {scores['ensemble_score'].std():.4f}")
    print(f"  Score range: [{scores['ensemble_score'].min():.4f}, "
          f"{scores['ensemble_score'].max():.4f}]")
    print(f"{'═' * 72}")

    # ── Top-20 directional accuracy ───────────────────────────────────
    accuracy = _compute_top_n_accuracy(scores, top_n=20)
    majority_correct = (accuracy["hit_rate"] > 0.5).sum()
    total_months = len(accuracy)
    pct_correct = majority_correct / total_months * 100

    print(f"\n  Top-20 Selection Accuracy:")
    print(f"    Months where majority (>50%) outperformed XLK: "
          f"{majority_correct}/{total_months} ({pct_correct:.1f}%)")
    print(f"    Average hit rate: {accuracy['hit_rate'].mean():.1%}")
    print(f"    Best month:  {accuracy.loc[accuracy['hit_rate'].idxmax(), 'date'].date()} "
          f"({accuracy['hit_rate'].max():.0%})")
    print(f"    Worst month: {accuracy.loc[accuracy['hit_rate'].idxmin(), 'date'].date()} "
          f"({accuracy['hit_rate'].min():.0%})")

    # ── Per-month table ───────────────────────────────────────────────
    print(f"\n  {'Month':>12}  {'Hit Rate':>9}  {'#Out':>5}  {'Avg Score':>10}")
    print(f"  {'─' * 12}  {'─' * 9}  {'─' * 5}  {'─' * 10}")

    for _, row in accuracy.iterrows():
        marker = " ★" if row["hit_rate"] > 0.6 else ""
        print(
            f"  {row['date'].date()!s:>12}  "
            f"{row['hit_rate']:8.0%}  "
            f"{row['n_outperform']:5d}  "
            f"{row['mean_score']:10.4f}{marker}"
        )

    # ── Score distribution by label ───────────────────────────────────
    print(f"\n  Score distribution by label:")
    for lab in [0, 1]:
        subset = scores[scores["label"] == lab]["ensemble_score"]
        label_name = "Outperform (1)" if lab == 1 else "Underperform (0)"
        print(f"    {label_name}:  mean={subset.mean():.4f}, "
              f"median={subset.median():.4f}, std={subset.std():.4f}")

    # AUC
    try:
        from sklearn.metrics import roc_auc_score
        overall_auc = roc_auc_score(scores["label"], scores["ensemble_score"])
        print(f"\n  Overall out-of-sample AUC: {overall_auc:.4f}")
    except ImportError:
        pass

    print()


if __name__ == "__main__":
    main()
