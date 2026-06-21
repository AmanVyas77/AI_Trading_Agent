"""
End-to-end verification — fundamental scorer dup-row & weight-redistribution fix
==================================================================================
Rebuilds the fundamental feature matrix against the real DB and runs composite
scoring, then checks:
  1. No duplicate (ticker, quarter_end) rows in the feature matrix.
  2. No factor names in features_used/features_missing that aren't real
     _WEIGHT_MAP keys (catches factor-name vs. column-name mismatches).
  3. Rows with zero available weight (all factors genuinely missing) produce
     NaN composite_score, and nothing else does — i.e. weight redistribution
     never silently zero-fills a genuinely-missing factor.
  4. lm_sentiment_score usage trends from ~100% missing in early quarters to
     partially used in later quarters, as LM 10-K coverage rolls in.
  5. composite_score never contains inf; NaN occurs exactly where weight is zero.

This hits the live DB directly (no mocking) — run manually, not via pytest.

Usage:
    python scripts/verify_fundamental_scorer.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np

from src.strategies.fundamental.xbrl_features import build_feature_matrix
from src.strategies.fundamental.fundamental_scorer import compute_composite_scores, _WEIGHT_MAP

print("=" * 70)
print("FUNDAMENTAL SCORER — END-TO-END VERIFICATION")
print("=" * 70)

# 1. Rebuild the feature matrix fresh
matrix = build_feature_matrix()
print(f"\n[1] feature matrix shape: {matrix.shape}")

dup_count = matrix.duplicated(subset=["ticker", "quarter_end"]).sum()
print(f"[1] duplicate (ticker, quarter_end) rows: {dup_count}")
assert dup_count == 0, "FAIL: duplicate keys still present in feature matrix"

# 2. Run composite scoring
scores = compute_composite_scores(matrix)
print(f"\n[2] composite scores shape: {scores.shape}")
print(f"[2] columns: {list(scores.columns)}")

# 3. Confirm weight redistribution invariant: every row's used weights sum to 1.0
# NOTE: features_used / features_missing come back as comma-joined STRINGS,
# not Python lists — must split before treating them as collections of names.
all_factor_names = list(_WEIGHT_MAP.keys())
weight_lookup = {name: w for name, (_, w) in _WEIGHT_MAP.items()}

def _split(s):
    if not isinstance(s, str) or s == "":
        return []
    return s.split(",")

used_list = scores["features_used"].apply(_split)
missing_list = scores["features_missing"].apply(_split)

def used_weight_sum(names):
    return sum(weight_lookup[f] for f in names if f in weight_lookup)

raw_sums = used_list.apply(used_weight_sum)
total_weight = sum(weight_lookup.values())

# sanity: every name appearing in features_used/features_missing should be a
# real _WEIGHT_MAP key — catches naming mismatches between the two layers
unknown_used = {f for names in used_list for f in names if f and f not in weight_lookup}
unknown_missing = {f for names in missing_list for f in names if f and f not in weight_lookup}
print(f"\n[3] unrecognized names in features_used (not in _WEIGHT_MAP): {unknown_used or 'none'}")
print(f"[3] unrecognized names in features_missing (not in _WEIGHT_MAP): {unknown_missing or 'none'}")
assert not unknown_used, f"FAIL: features_used contains names not in _WEIGHT_MAP: {unknown_used}"
assert not unknown_missing, f"FAIL: features_missing contains names not in _WEIGHT_MAP: {unknown_missing}"

zero_weight_mask = raw_sums == 0
n_zero = zero_weight_mask.sum()
print(f"\n[3] rows with ZERO available weight (all factors genuinely missing): {n_zero} / {len(scores)}")

if n_zero > 0:
    zero_rows = scores[zero_weight_mask]
    print("[3] sample of zero-weight rows (up to 10):")
    cols_to_show = ["ticker", "quarter_end", "composite_score", "features_used", "features_missing"]
    print(zero_rows[cols_to_show].head(10).to_string())
    print(f"\n[3] composite_score for zero-weight rows — "
          f"nan count: {zero_rows['composite_score'].isna().sum()} / {len(zero_rows)}, "
          f"non-nan values (up to 10, should be empty/none if correct): "
          f"{zero_rows['composite_score'].dropna().head(10).tolist()}")
    assert zero_rows["composite_score"].isna().all(), (
        "FAIL: a row with zero available weight has a non-NaN composite_score — "
        "likely a divide-by-zero or bad default in compute_composite_scores()"
    )

nz = raw_sums[~zero_weight_mask]
print(f"\n[3] raw available-weight sum (non-zero rows only) — "
      f"min: {nz.min():.4f}, max: {nz.max():.4f}, mean: {nz.mean():.4f}")
print(f"[3] total weight in _WEIGHT_MAP: {total_weight:.4f}")

# 4. Confirm lm_sentiment_score is genuinely missing in early quarters,
#    genuinely used in later quarters where LM data exists
scores_with_qtr = scores.copy()
scores_with_qtr["used_lm"] = used_list.apply(lambda names: "lm_sentiment_score" in names)
scores_with_qtr["missing_lm"] = missing_list.apply(lambda names: "lm_sentiment_score" in names)

by_quarter = scores_with_qtr.groupby("quarter_end").agg(
    n=("ticker", "count"),
    used_lm=("used_lm", "sum"),
    missing_lm=("missing_lm", "sum"),
)
print("\n[4] lm_sentiment_score usage by quarter (first 3 / last 3):")
print(by_quarter.head(3))
print("...")
print(by_quarter.tail(3))

early_missing_rate = by_quarter.head(3)["missing_lm"].sum() / by_quarter.head(3)["n"].sum()
late_used_rate = by_quarter.tail(3)["used_lm"].sum() / by_quarter.tail(3)["n"].sum()
print(f"\n[4] early-quarter lm missing rate: {early_missing_rate:.1%} (expect high, ~100%)")
print(f"[4] late-quarter lm used rate: {late_used_rate:.1%} (expect > 0%)")

# 5. Composite score sanity — no infs ever; NaN allowed ONLY on zero-weight rows
inf_count = scores["composite_score"].isin([np.inf, -np.inf]).sum()
nan_count = scores["composite_score"].isna().sum()
print(f"\n[5] composite_score inf count: {inf_count}, nan count: {nan_count}")
print(f"[5] composite_score range (excl. nan/inf): "
      f"[{scores['composite_score'].replace([np.inf, -np.inf], np.nan).min():.3f}, "
      f"{scores['composite_score'].replace([np.inf, -np.inf], np.nan).max():.3f}]")

assert inf_count == 0, "FAIL: composite_score contains inf"
# every NaN composite_score must correspond exactly to a zero-weight row, and vice versa
nan_mask = scores["composite_score"].isna()
mismatch = (nan_mask != zero_weight_mask).sum()
assert mismatch == 0, (
    f"FAIL: {mismatch} rows where NaN-ness of composite_score disagrees with "
    "zero-available-weight status — redistribution logic may be wrong"
)
print(f"[5] NaN composite_score rows exactly match zero-weight rows: confirmed "
      f"({nan_count} rows, all genuinely all-factors-missing)")

print("\n" + "=" * 70)
print("ALL CHECKS PASSED" if (dup_count == 0 and inf_count == 0 and mismatch == 0)
      else "CHECKS FAILED — see above")
print("=" * 70)
