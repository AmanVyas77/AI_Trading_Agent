"""
Live / holdout scorer for the frozen ensemble model.

Sprint 7 Prompt 3 Part B: the production model is the LAST fold in
models/ensemble_models.pkl (trained through 2024-07-31). This module
scores forward months using that single fold — a genuine walk-forward
run needs a training pass, which is intentionally NOT done in Sprint 7
(the model is frozen for the true-holdout evaluation).

Feature-matrix vs frozen model: the rebuilt feature matrix has 24
factor columns; the frozen model was fit on 23 (fcf_yield was all-NaN
at Sprint 5 training and got dropped). We reindex X on the model's
own feature_names, filling missing columns with 0.0 to match
model_trainer._prepare_xy.
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"

FROZEN_MODEL_PATH = MODELS_DIR / "ensemble_models.pkl"
FEATURE_MATRIX_PATH = PROCESSED / "ensemble_feature_matrix.parquet"
OUTPUT_PATH = PROCESSED / "holdout_scores.parquet"


def _load_last_fold(model_path: Path) -> dict:
    with open(model_path, "rb") as f:
        folds = pickle.load(f)
    if not isinstance(folds, list) or not folds:
        raise ValueError(f"{model_path}: expected non-empty list of fold dicts")
    return folds[-1]


def score_months(
    start: str,
    end: str,
    model_path: Optional[Path] = None,
    feature_matrix_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Score every (date, ticker) row of the feature matrix falling in
    [start, end] using the frozen model's last fold.

    Returns a DataFrame with columns [date, ticker, ensemble_score] and
    also writes it to output_path (default: holdout_scores.parquet).
    """
    model_path = Path(model_path) if model_path else FROZEN_MODEL_PATH
    fm_path = Path(feature_matrix_path) if feature_matrix_path else FEATURE_MATRIX_PATH
    out_path = Path(output_path) if output_path else OUTPUT_PATH

    fold = _load_last_fold(model_path)
    model = fold["model"]
    feat_names = list(fold["feature_names"])
    effective_train_end = fold.get("effective_train_end")

    logger.info(
        "AUDIT: frozen model — path=%s  fold_date=%s  "
        "effective_train_end=%s  n_features=%d",
        model_path.name, fold.get("date"), effective_train_end, len(feat_names),
    )

    if not fm_path.exists():
        raise FileNotFoundError(f"feature matrix not found: {fm_path}")
    fm = pd.read_parquet(fm_path)
    fm["date"] = pd.to_datetime(fm["date"])

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    df = fm.loc[(fm["date"] >= start_ts) & (fm["date"] <= end_ts)].copy()
    if df.empty:
        raise ValueError(
            f"feature matrix has no rows in [{start}, {end}] — did you rebuild it?"
        )

    X = df.reindex(columns=feat_names).fillna(0.0)

    assert X.shape[1] == 23, f"expected 23 features, got {X.shape[1]}"
    assert list(X.columns) == feat_names, "column order does not match frozen model"

    proba = model.predict_proba(X)[:, 1]

    out = pd.DataFrame({
        "date": df["date"].values,
        "ticker": df["ticker"].values,
        "ensemble_score": proba,
    })
    out = out.sort_values(["date", "ticker"]).reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)

    logger.info(
        "Scored %d rows across %d months for %d tickers → %s",
        len(out), out["date"].nunique(), out["ticker"].nunique(), out_path,
    )
    return out


if __name__ == "__main__":
    import argparse
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s │ %(levelname)-8s │ %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    args = ap.parse_args()
    score_months(args.start, args.end)
