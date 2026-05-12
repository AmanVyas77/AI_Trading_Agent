"""
Regime Attribution Analysis
=============================
Computes feature importance by macro regime and evaluates how the ensemble
model's accuracy varies across different market environments.

Regime Definitions (using raw macro data from SQLite)
-----------------------------------------------------
  risk_on:        VIX < 20 AND yield_spread > 0
  risk_off:       VIX > 25 OR yield_spread < 0
  high_inflation: CPI YoY > 4.0%
  rate_rising:    Fed Funds 3-month change > 0.25

Output
------
  data/processed/regime_attribution.json

CLI
---
  python -m src.strategies.ensemble.regime_analysis
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from sqlalchemy import create_engine, text, inspect

load_dotenv()
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
with open(ROOT / "config" / "settings.yaml") as f:
    CFG = yaml.safe_load(f)

DB_PATH = ROOT / CFG["data"]["paths"]["db"]
DB_URL = f"sqlite:///{DB_PATH}"
PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"


# ── Loaders ───────────────────────────────────────────────────────────────────

def _load_models(path: Path = None) -> list[dict]:
    if path is None:
        path = MODELS_DIR / "ensemble_models.pkl"
    if not path.exists():
        logger.error(f"Models not found: {path}")
        return []
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_labeled(path: Path = None) -> pd.DataFrame:
    if path is None:
        path = PROCESSED / "ensemble_labeled.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def _load_scores(path: Path = None) -> pd.DataFrame:
    if path is None:
        path = PROCESSED / "ensemble_scores.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def _load_raw_macro() -> pd.DataFrame:
    """Load raw (un-z-scored) macro series from SQLite, resample to monthly."""
    engine = create_engine(DB_URL, echo=False)
    if "macro_series" not in inspect(engine).get_table_names():
        return pd.DataFrame()

    targets = ["vix", "yield_spread_10y2y", "fed_funds_rate", "cpi"]
    ph = ",".join(f":s{i}" for i in range(len(targets)))
    params = {f"s{i}": s for i, s in enumerate(targets)}

    sql = f"""
        SELECT series_name, date, value
        FROM macro_series
        WHERE series_name IN ({ph})
        ORDER BY date
    """
    with engine.connect() as conn:
        raw = pd.read_sql_query(text(sql), conn, params=params)

    if raw.empty:
        return pd.DataFrame()

    raw["date"] = pd.to_datetime(raw["date"])
    wide = raw.pivot_table(index="date", columns="series_name", values="value", aggfunc="last")
    monthly = wide.resample("ME").last().ffill()
    return monthly


# ── 1. Global Feature Importance ──────────────────────────────────────────────

def _compute_global_importance(models: list[dict]) -> dict[str, float]:
    """Average feature_importances_ across all walk-forward models."""
    if not models:
        return {}

    feat_names = models[0]["feature_names"]
    importance_sum = np.zeros(len(feat_names))

    for entry in models:
        importance_sum += entry["model"].feature_importances_

    importance_avg = importance_sum / len(models)

    # Return as sorted dict (highest first)
    ranked = sorted(
        zip(feat_names, importance_avg),
        key=lambda x: x[1],
        reverse=True,
    )
    return {name: round(float(val), 6) for name, val in ranked}


# ── 2. Regime Classification ─────────────────────────────────────────────────

def _classify_regimes(macro: pd.DataFrame) -> pd.DataFrame:
    """
    Classify each month into regime buckets using raw macro data.

    Returns a DataFrame indexed by month-end date with boolean columns:
      risk_on, risk_off, high_inflation, rate_rising
    """
    regimes = pd.DataFrame(index=macro.index)

    # VIX and yield spread
    vix = macro.get("vix", pd.Series(dtype=float))
    spread = macro.get("yield_spread_10y2y", pd.Series(dtype=float))
    ff = macro.get("fed_funds_rate", pd.Series(dtype=float))
    cpi = macro.get("cpi", pd.Series(dtype=float))

    # risk_on: VIX < 20 AND yield spread > 0
    regimes["risk_on"] = (vix < 20) & (spread > 0)

    # risk_off: VIX > 25 OR yield spread < 0
    regimes["risk_off"] = (vix > 25) | (spread < 0)

    # high_inflation: CPI YoY > 4%
    if not cpi.empty:
        cpi_yoy = cpi.pct_change(12) * 100
        regimes["high_inflation"] = cpi_yoy > 4.0
    else:
        regimes["high_inflation"] = False

    # rate_rising: fed funds 3-month change > 0.25
    if not ff.empty:
        ff_delta_3m = ff.diff(3)
        regimes["rate_rising"] = ff_delta_3m > 0.25
    else:
        regimes["rate_rising"] = False

    return regimes.fillna(False)


# ── 3. Per-Regime Analysis ────────────────────────────────────────────────────

def _analyze_regime(
    regime_name: str,
    regime_months: pd.DatetimeIndex,
    models: list[dict],
    scores_df: pd.DataFrame,
    top_n: int = 20,
) -> dict:
    """Compute importance, accuracy, and score stats for one regime."""
    # Find models whose test date falls in this regime
    regime_models = [m for m in models if m["date"] in regime_months]
    n_months = len(regime_models)

    if n_months == 0:
        return {
            "importance": {},
            "accuracy": None,
            "n_months": 0,
            "mean_score_selected": None,
            "mean_score_rejected": None,
        }

    # Feature importance for this regime
    feat_names = regime_models[0]["feature_names"]
    imp_sum = np.zeros(len(feat_names))
    for m in regime_models:
        imp_sum += m["model"].feature_importances_
    imp_avg = imp_sum / n_months
    importance = {name: round(float(val), 6) for name, val in zip(feat_names, imp_avg)}

    # Accuracy: % of months where top-20 by score had majority label=1
    months_correct = 0
    score_selected_all = []
    score_rejected_all = []

    for m in regime_models:
        month_scores = scores_df[scores_df["date"] == m["date"]]
        if month_scores.empty:
            continue

        top = month_scores.nlargest(min(top_n, len(month_scores)), "ensemble_score")
        bottom = month_scores[~month_scores.index.isin(top.index)]

        hit_rate = top["label"].mean()
        if hit_rate > 0.5:
            months_correct += 1

        score_selected_all.append(top["ensemble_score"].mean())
        if not bottom.empty:
            score_rejected_all.append(bottom["ensemble_score"].mean())

    accuracy = months_correct / n_months if n_months > 0 else None
    mean_sel = float(np.mean(score_selected_all)) if score_selected_all else None
    mean_rej = float(np.mean(score_rejected_all)) if score_rejected_all else None

    return {
        "importance": importance,
        "accuracy": round(accuracy, 4) if accuracy is not None else None,
        "n_months": n_months,
        "mean_score_selected": round(mean_sel, 4) if mean_sel is not None else None,
        "mean_score_rejected": round(mean_rej, 4) if mean_rej is not None else None,
    }


# ── Public API ────────────────────────────────────────────────────────────────

def run_regime_analysis(
    models: Optional[list[dict]] = None,
    labeled_df: Optional[pd.DataFrame] = None,
    scores_df: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Run regime attribution analysis.

    Returns dict with:
      global_importance: {feature: avg_importance}
      by_regime: {regime_name: {importance, accuracy, n_months, ...}}
    """
    if models is None:
        models = _load_models()
    if not models:
        logger.error("No models available")
        return {}

    if labeled_df is None:
        labeled_df = _load_labeled()
    if scores_df is None:
        scores_df = _load_scores()

    # 1. Global feature importance
    logger.info("Computing global feature importance…")
    global_imp = _compute_global_importance(models)

    # 2. Load raw macro and classify regimes
    logger.info("Loading raw macro data for regime classification…")
    raw_macro = _load_raw_macro()
    if raw_macro.empty:
        logger.warning("No raw macro data — skipping regime analysis")
        return {"global_importance": global_imp, "by_regime": {}}

    regimes = _classify_regimes(raw_macro)

    # 3. Map model test dates to regime months
    model_dates = pd.DatetimeIndex([m["date"] for m in models])

    # Align regime index to model dates
    aligned_regimes = regimes.reindex(model_dates, method="ffill")

    # 4. Per-regime analysis
    logger.info("Analyzing per-regime performance…")
    regime_results = {}
    for regime_name in ["risk_on", "risk_off", "high_inflation", "rate_rising"]:
        if regime_name not in aligned_regimes.columns:
            continue
        mask = aligned_regimes[regime_name].fillna(False)
        regime_months = model_dates[mask]
        logger.info(f"  {regime_name}: {len(regime_months)} months")

        result = _analyze_regime(regime_name, regime_months, models, scores_df)
        regime_results[regime_name] = result

    output = {
        "global_importance": global_imp,
        "by_regime": regime_results,
    }

    return output


# ── Save ──────────────────────────────────────────────────────────────────────

def _safe_json(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if (np.isnan(obj) or np.isinf(obj)) else float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    raise TypeError(f"Not JSON serializable: {type(obj)}")


def save_attribution(results: dict, path: Path = None) -> None:
    if path is None:
        path = PROCESSED / "regime_attribution.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=_safe_json)
    logger.info(f"Saved → {path}  ({path.stat().st_size / 1024:.0f} KB)")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s │ %(levelname)-8s │ %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    results = run_regime_analysis()

    if not results:
        logger.error("Regime analysis failed")
        return

    save_attribution(results)

    # Print summary
    print(f"\n{'═' * 72}")
    print(f"  Regime Attribution Analysis")
    print(f"{'═' * 72}")

    # Global importance
    print(f"\n  Global Feature Importance (avg across {len(_load_models())} folds):")
    for i, (feat, imp) in enumerate(results["global_importance"].items()):
        bar = "█" * int(imp / max(results["global_importance"].values()) * 25)
        print(f"    {i+1:2d}. {feat:<28s}  {bar}  {imp:.4f}")

    # Per-regime
    print(f"\n  Per-Regime Performance:")
    print(f"  {'Regime':<18s}  {'Months':>6s}  {'Accuracy':>8s}  {'Avg Score (sel)':>15s}  {'Top Feature'}")
    print(f"  {'─' * 18}  {'─' * 6}  {'─' * 8}  {'─' * 15}  {'─' * 25}")

    for regime_name, data in results.get("by_regime", {}).items():
        n = data["n_months"]
        acc = f"{data['accuracy']:.0%}" if data["accuracy"] is not None else "—"
        score = f"{data['mean_score_selected']:.4f}" if data["mean_score_selected"] is not None else "—"
        top_feat = max(data["importance"], key=data["importance"].get) if data["importance"] else "—"
        print(f"  {regime_name:<18s}  {n:6d}  {acc:>8s}  {score:>15s}  {top_feat}")

    # Regime-specific top-3 features
    for regime_name, data in results.get("by_regime", {}).items():
        if not data["importance"]:
            continue
        sorted_imp = sorted(data["importance"].items(), key=lambda x: x[1], reverse=True)
        print(f"\n  {regime_name} — Top 5 features:")
        for feat, imp in sorted_imp[:5]:
            print(f"    {feat:<28s}  {imp:.4f}")

    print(f"\n{'═' * 72}")
    print()


if __name__ == "__main__":
    main()
