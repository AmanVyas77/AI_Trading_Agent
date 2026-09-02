#!/usr/bin/env python
"""
Diagnostic backtest of the Markov exit layer over the 2025-26 window.

This is a PROTOTYPE QUALITY CHECK, not a production verdict. It compares three
monthly-return paths on the frozen-ensemble TOP_N book:

    baseline      buy-and-hold the TOP_N book until the next monthly rebalance
                  (the current production exit rule)
    experimental  baseline, but zero out (to cash) any held ticker whose
                  monthly exit review says SELL — do NOT re-buy on the
                  ensemble signal alone
    hard-stop     baseline, plus a daily Zhang Case II mandatory-sell: a held
                  ticker that breaches x*_0 on any trading day is sold at that
                  day's close and sits in cash for the remainder of the month

Assumptions (explicit)
----------------------
  - Zhang (f1,f2,λ1,λ2) recalibrated MONTHLY per ticker via
    exit_manager.calibrate_for_month (dt=1/252). Cached per (ticker, month);
    NOT recomputed for the daily hard-stop.
  - Andrade DHMMs retrained monthly per ticker with fixed random_state=42.
    Reproducibility of the Sharpe numbers is preferred over avoiding one
    seed's local optimum — no seed sweep in the monthly loop.
  - K = K_fraction × entry_price (REV 4 fix A): a FIXED dollar transaction cost
    per position, not the flat K=0.01 of REV 3. K_fraction = 0.001 (0.1%).
  - entry_price (backtest simplification): the first close of the ~250d
    calibration lookback window ending at month_end — a stable proxy for the
    position's cost basis at the start of the calibration period. A future LIVE
    wiring MUST replace this with the actual broker cost basis of the held
    position; the first-close proxy is only defensible for a diagnostic.
  - rho = 0.03 risk-free assumption (≈ 2025-26 avg 3M T-bill).
  - Ignoring slippage, dividends, position-sizing. This diagnoses the exit
    SIGNAL alone.
  - Equal-weight book (1/n). When a ticker is exited its weight goes to cash
    (return 0); it is NOT redistributed to survivors.

Scale note (REV 4 — why the hard-stop can now fire on real names)
-----------------------------------------------------------------
  REV 3 passed a flat K=0.01 (paper-scale) against ~$100 prices, so Case II
  x*_0 = rho·K/(rho−f1) collapsed to ~cents and a Case II name hard-stopped on
  the first trading day (or the condition degenerated to a price-independent
  constant — Opus's Prompt-2 flag). REV 4 references K to entry_price, so
  x*_0 = rho·(K_fraction·entry_price)/(rho−f1) is a genuine dollar LEVEL the
  live price crosses. Case I names (f1 > rho) still never hard-stop; Case II
  names now trigger only when the live price actually breaches x*_0. Report the
  hard-stop trigger count to see how often that happens on this slice.

Regime gate (REV 4 fix B)
-------------------------
  The Andrade STRONG_SELL override is suppressed when the cached regime
  multiplier > NEUTRAL_MULT (a RISK_ON tape).

  LOOKAHEAD FIX (2026-08-03). REV 4 called get_live_regime_signal() inside
  calibrate_for_month, i.e. inside this backtest loop — against that function's
  own docstring ("Only for live/forward use — never called inside the backtest
  loop"). It has no as-of parameter, so every (ticker, month) calibration cached
  the SAME run-time regime: today's macro gating 2025 decisions, all-or-nothing
  across the window, and silently different on every re-run as the macro tables
  refreshed. It now uses get_regime_signal_asof(month_end), which reads only
  macro observations dated on or before that month-end. Suppression is now a
  genuine per-month property; the summary reports the per-month reading.

Usage
-----
    python scripts/backtest_exit_layer.py --start 2025-01-01 --end 2026-06-30
"""
from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))  # allow `python scripts/…` without install

from src.exit.exit_manager import (  # noqa: E402
    DEFAULT_K_FRACTION,
    _ANDRADE_TO_STATE,
    calibrate_for_month,
    daily_hard_stop,
    monthly_exit_review,
)
from src.exit.zhang_optimal import (  # noqa: E402
    case1_threshold,
    case2_thresholds,
    phi,
)
from src.live.scorer import score_months  # noqa: E402
from src.strategies.ensemble.portfolio_builder import MIN_SCORE, TOP_N  # noqa: E402
from src.strategies.ensemble.regime_gate import (  # noqa: E402
    NEUTRAL_MULT,
    get_regime_signal_asof,
)
from src.utils.data_vintage import format_vintage, price_vintage  # noqa: E402
with open(ROOT / "config" / "settings.yaml") as f:
    _CFG = yaml.safe_load(f)
DB_PATH = ROOT / _CFG["data"]["paths"]["db"]
HOLDOUT_SCORES = ROOT / "data" / "processed" / "holdout_scores.parquet"
LOG_DIR = ROOT / "logs"

FEATURE_MATRIX = ROOT / "data" / "processed" / "ensemble_feature_matrix.parquet"
# Set by --vintage; echoed into every report so no run is ambiguous about what
# it read. None = live repo data (subject to nightly AV backfill revision).
VINTAGE_DIR: Path | None = None
VINTAGE_MANIFEST: dict | None = None

RHO = 0.03
# REV 4: transaction cost is now K_fraction × entry_price (a fixed dollar amount
# per position), computed inside calibrate_for_month — not a flat K=0.01. We pass
# K_fraction explicitly for auditability; it mirrors exit_manager.DEFAULT_K_FRACTION.
K_FRACTION = DEFAULT_K_FRACTION  # 0.001 = 0.1% of entry price
CALIB_LOOKBACK = 250

# REV 3 Sharpe numbers (commit 4a2a699) — the diagnostic FAIL we are re-testing.
REV3_SHARPE = {"baseline": 1.5871, "experimental": 1.0783, "hard_stop": 1.5871}

logger = logging.getLogger("backtest_exit_layer")

# Prompt 1C guard: Prompt 1 was initially run under /opt/anaconda3/bin/python3
# (numpy 1.26.3 / sklearn 1.2.2) by mistake. Fail loudly rather than silently
# producing numbers under the wrong stack.
EXPECTED_VERSIONS = {"numpy": "2.4.4", "sklearn": "1.8.0", "pandas": "2.3.3"}


def _assert_interpreter() -> dict:
    import numpy
    import pandas
    import sklearn
    block = {
        "sys_executable": sys.executable,
        "numpy": numpy.__version__,
        "sklearn": sklearn.__version__,
        "pandas": pandas.__version__,
    }
    if ".venv" not in block["sys_executable"] or "anaconda" in block["sys_executable"].lower():
        raise SystemExit(f"ABORT: not the repo .venv → {block['sys_executable']}")
    bad = {k: block[k] for k, v in EXPECTED_VERSIONS.items() if block[k] != v}
    if bad:
        raise SystemExit(
            f"ABORT: interpreter version mismatch {bad}, expected {EXPECTED_VERSIONS}")
    logger.info("AUDIT: interpreter %s  numpy=%s sklearn=%s pandas=%s",
                block["sys_executable"], block["numpy"], block["sklearn"],
                block["pandas"])
    return block


# ── Frozen vintage ───────────────────────────────────────────────────────────

def use_vintage(vintage_dir: str | Path) -> dict:
    """Repoint every data input at a frozen snapshot from freeze_vintage.py.

    Verifies each frozen file's sha256 against MANIFEST.json before use, so a
    tampered or partially-copied snapshot fails loudly rather than silently
    producing numbers attributed to the wrong vintage.

    Rebinds this module's DB_PATH *and* regime_gate.DB_PATH — the as-of regime
    signal reads macro_series from its own module-level path, so repointing only
    this module would leave the regime leg reading live data.
    """
    import hashlib
    import json

    global VINTAGE_DIR, VINTAGE_MANIFEST, DB_PATH, FEATURE_MATRIX

    vdir = Path(vintage_dir)
    if not vdir.is_absolute():
        vdir = ROOT / vdir
    manifest_path = vdir / "MANIFEST.json"
    if not manifest_path.exists():
        raise SystemExit(f"ABORT: no MANIFEST.json in {vdir}")
    manifest = json.loads(manifest_path.read_text())

    for name, meta in manifest["frozen"].items():
        f = vdir / meta["snapshot_relpath"]
        if not f.exists():
            raise SystemExit(f"ABORT: frozen file missing: {f}")
        h = hashlib.sha256()
        with open(f, "rb") as fh:
            while block := fh.read(1 << 20):
                h.update(block)
        if h.hexdigest() != meta["sha256"]:
            raise SystemExit(
                f"ABORT: {name} sha256 mismatch — snapshot corrupt or modified\n"
                f"  manifest={meta['sha256']}\n  actual  ={h.hexdigest()}")

    # Assert the frozen model is still the one the snapshot was taken against.
    model_meta = manifest["referenced"].get("ensemble_models.pkl", {})
    if model_meta:
        h = hashlib.md5()
        with open(ROOT / model_meta["source_relpath"], "rb") as fh:
            while block := fh.read(1 << 20):
                h.update(block)
        if h.hexdigest() != model_meta.get("md5"):
            raise SystemExit(
                f"ABORT: models/ensemble_models.pkl md5 {h.hexdigest()} != "
                f"manifest {model_meta.get('md5')}")

    DB_PATH = vdir / "quant_research.db"
    FEATURE_MATRIX = vdir / "ensemble_feature_matrix.parquet"
    VINTAGE_DIR = vdir
    VINTAGE_MANIFEST = manifest

    # The as-of regime signal reads macro_series via its OWN module global.
    from src.strategies.ensemble import regime_gate as _rg
    _rg.DB_PATH = DB_PATH

    logger.info("VINTAGE: frozen snapshot %s (all %d files sha256-verified)",
                vdir, len(manifest["frozen"]))
    logger.info("VINTAGE: price_vintage sha256=%s  git_head=%s",
                manifest["price_vintage"]["sha256"], manifest["git_head"])
    return manifest


# ── Data loading ─────────────────────────────────────────────────────────────

def _load_prices_wide() -> pd.DataFrame:
    """adj_close pivoted to (date × ticker), date-sorted DatetimeIndex."""
    with sqlite3.connect(str(DB_PATH)) as conn:
        df = pd.read_sql_query(
            "SELECT date, ticker, adj_close FROM prices ORDER BY date", conn
        )
    df["date"] = pd.to_datetime(df["date"])
    wide = df.pivot(index="date", columns="ticker", values="adj_close").sort_index()
    return wide


def _ticker_closes(wide: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Per-ticker close frame (index=date, column 'close'), NaNs dropped."""
    s = wide[ticker].dropna()
    return pd.DataFrame({"close": s})


def _price_on_or_before(wide: pd.DataFrame, ticker: str, when: pd.Timestamp) -> float:
    col = wide[ticker].dropna()
    sub = col.loc[col.index <= when]
    if sub.empty:
        return float("nan")
    return float(sub.iloc[-1])


# ── Return helpers ───────────────────────────────────────────────────────────

def _annualized_sharpe(monthly: pd.Series) -> float:
    m = monthly.dropna()
    if len(m) < 2:
        return float("nan")
    sd = m.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return float("nan")
    return float(m.mean() / sd * np.sqrt(12))


# ── Core backtest ────────────────────────────────────────────────────────────

def run_backtest(start: str, end: str, allow_andrade: bool = True,
                 emit_decisions: bool = False) -> dict:
    # Pin the price vintage FIRST: the prices table is retroactively rewritten by
    # the AV backfill, so a result is only comparable against another run on the
    # same fingerprint. See src/utils/data_vintage.py.
    vintage = price_vintage(DB_PATH, start, end)
    logger.info("AUDIT: %s", format_vintage(vintage))
    logger.info("AUDIT: data source = %s",
                f"FROZEN VINTAGE {VINTAGE_DIR}" if VINTAGE_DIR
                else "LIVE repo data (subject to AV backfill revision)")
    logger.info("AUDIT: allow_andrade = %s", allow_andrade)

    # Scores are regenerated rather than read from the snapshot, then checked
    # against the frozen copy — that verifies the frozen feature matrix + frozen
    # model still reproduce the frozen scores bit-for-bit.
    scores_out = (LOG_DIR / "holdout_scores_run.parquet") if VINTAGE_DIR else None
    logger.info("Regenerating scores for [%s, %s] …", start, end)
    score_months(start=start, end=end,
                 feature_matrix_path=FEATURE_MATRIX,
                 output_path=scores_out)

    scores_path = scores_out or HOLDOUT_SCORES
    if VINTAGE_DIR is not None:
        frozen_scores = pd.read_parquet(VINTAGE_DIR / "holdout_scores.parquet")
        regen = pd.read_parquet(scores_path)
        a = frozen_scores.sort_values(["date", "ticker"]).reset_index(drop=True)
        b = regen.sort_values(["date", "ticker"]).reset_index(drop=True)
        same = (len(a) == len(b)
                and bool((a["ensemble_score"].values == b["ensemble_score"].values).all()))
        logger.info("VINTAGE: regenerated scores match frozen copy bit-for-bit: %s",
                    same)
        if not same:
            raise SystemExit(
                "ABORT: regenerated scores differ from the frozen snapshot — the "
                "snapshot is not capturing every input the scorer reads")

    scores = pd.read_parquet(scores_path)
    scores["date"] = pd.to_datetime(scores["date"])
    scores = scores.sort_values(["date", "ticker"])
    month_ends = sorted(scores["date"].unique())
    logger.info("Loaded %d score rows across %d month-ends",
                len(scores), len(month_ends))

    wide = _load_prices_wide()

    rows: list[dict] = []          # per-month path returns
    trigger_rows: list[dict] = []  # per-month trigger_source attribution
    big_moves: list[dict] = []     # |monthly return| > 5% highlights
    suppression_rows: list[dict] = []  # per-month Andrade-suppressed-by-regime count
    k_abs_values: list[float] = []     # every K_absolute used (for distribution)
    hard_stop_rows: list[dict] = []    # per-month hard-stop trigger count
    regime_asof_rows: list[dict] = []  # per-month point-in-time regime reading
    zhang_rows: list[dict] = []        # per-calibration Zhang case / threshold
    decision_rows: list[dict] = []     # Prompt 2 per-decision emit (all evaluated
                                       # (month, ticker) pairs, not just big-moves)

    # Prompt 2: pre-compute the top-N book at each month_end so we can populate
    # in_book_next_month for each SELL flag without re-running the selection
    # inside the ticker loop.
    _book_by_month: dict[str, set[str]] = {}
    for _me in month_ends:
        _me_ts = pd.Timestamp(_me)
        _ms = scores[scores["date"] == _me_ts]
        _above = _ms[_ms["ensemble_score"] > MIN_SCORE]
        _sel = _above.nlargest(min(TOP_N, len(_above)), "ensemble_score")
        _book_by_month[_me_ts.date().isoformat()] = set(_sel["ticker"].tolist())

    for i, me in enumerate(month_ends[:-1]):
        me = pd.Timestamp(me)
        me_next = pd.Timestamp(month_ends[i + 1])
        me_next2 = pd.Timestamp(month_ends[i + 2]) if (i + 2) < len(month_ends) else None
        next_book = _book_by_month.get(me_next.date().isoformat(), set())

        # POINT-IN-TIME regime for this month (2026-08-03 lookahead fix).
        # Previously calibrate_for_month called get_live_regime_signal() per
        # (ticker, month), which reads the LATEST macro snapshot — today's macro
        # gating 2025 decisions, and a result that changed whenever the macro
        # tables were refreshed. Fetched once per month and reused across the
        # book: it does not vary by ticker.
        regime_asof = get_regime_signal_asof(me)
        regime_asof_rows.append({
            "month": me.date().isoformat(),
            "multiplier": regime_asof["multiplier"],
            "rule_signal": regime_asof["rule_signal"],
            "stale": regime_asof["stale"],
        })

        month_scores = scores[scores["date"] == me]
        above = month_scores[month_scores["ensemble_score"] > MIN_SCORE]
        book = above.nlargest(min(TOP_N, len(above)), "ensemble_score")
        tickers = [t for t in book["ticker"].tolist() if t in wide.columns]
        n = len(tickers)
        if n == 0:
            logger.warning("%s: empty book, skipping", me.date())
            continue

        base_rets, exp_rets, hs_rets = [], [], []
        cnt = {"zhang": 0, "andrade": 0, "both": 0, "none": 0}
        suppressed = 0   # Andrade STRONG_SELLs suppressed by RISK_ON regime
        hs_fired = 0     # hard-stop SELL triggers this month

        for t in tickers:
            p0 = _price_on_or_before(wide, t, me)
            p1 = _price_on_or_before(wide, t, me_next)
            if np.isnan(p0) or np.isnan(p1) or p0 <= 0:
                continue
            ret_full = p1 / p0 - 1.0

            # ── calibrate ONCE, reuse for both monthly + daily decisions ──
            closes_df = _ticker_closes(wide, t)
            # entry_price = first close of the ~250d calibration window ending at
            # me (cost-basis proxy for the diagnostic — see header). Mirrors the
            # window calibrate_for_month slices internally.
            window = closes_df.loc[closes_df.index <= me].iloc[-CALIB_LOOKBACK:]
            entry_price = float(window["close"].iloc[0]) if not window.empty else float("nan")
            try:
                calib = calibrate_for_month(
                    t, me, closes_df,
                    entry_price=entry_price,
                    K_fraction=K_FRACTION,
                    calibration_lookback=CALIB_LOOKBACK,
                    regime_signal=regime_asof,
                )
            except ValueError as exc:
                logger.warning("%s %s: calibration skipped (%s)", me.date(), t, exc)
                base_rets.append(ret_full)
                exp_rets.append(ret_full)
                hs_rets.append(ret_full)
                continue

            k_abs_values.append(calib.K_absolute)

            # ── Zhang regime / threshold diagnostics (Prompt 1C Task B) ──
            _zp = calib.zhang_params
            _args = (RHO, _zp.f1, _zp.f2, _zp.lam1, _zp.lam2)
            _phi = phi(*_args)
            if _phi <= 0.0:
                _case, _xs = "never_sell", float("nan")
            elif RHO <= _zp.f1:
                _case = "case1"
                _xs = case1_threshold(*_args, calib.K_absolute)
            else:
                _case = "case2"
                _xs, _ = case2_thresholds(*_args, calib.K_absolute)
            zhang_rows.append({
                "month": me.date().isoformat(), "ticker": t, "case": _case,
                "state": 2 if calib.andrade_signal.action in ("SELL", "STRONG_SELL") else 1,
                "phi": _phi, "f1": _zp.f1, "x_star": _xs, "p0": p0,
                "x_star_over_p0": (_xs / p0) if (p0 > 0 and np.isfinite(_xs)) else float("nan"),
            })
            # A STRONG_SELL that the regime gate suppressed (RISK_ON tape).
            if (calib.andrade_signal.action == "STRONG_SELL"
                    and calib.regime_signal["multiplier"] > NEUTRAL_MULT):
                suppressed += 1

            # ── monthly experimental review ──
            review = monthly_exit_review(calib, p0, rho=RHO,
                                         allow_andrade=allow_andrade)
            cnt[review.trigger_source] += 1
            exp_ret = 0.0 if review.action == "SELL" else ret_full

            # ── daily hard-stop over (me, me_next] using the SAME calib ──
            day_prices = wide[t].dropna()
            day_prices = day_prices.loc[
                (day_prices.index > me) & (day_prices.index <= me_next)
            ]
            hs_ret = ret_full
            for day, day_close in day_prices.items():
                hs = daily_hard_stop(calib, float(day_close), rho=RHO)
                if hs.action == "SELL":
                    hs_ret = float(day_close) / p0 - 1.0
                    hs_fired += 1
                    break

            base_rets.append(ret_full)
            exp_rets.append(exp_ret)
            hs_rets.append(hs_ret)

            if abs(ret_full) > 0.05 and review.action == "SELL":
                big_moves.append({
                    "month": me.date().isoformat(),
                    "ticker": t,
                    "ret_full": ret_full,
                    "trigger_source": review.trigger_source,
                })

            # ── Prompt 2: per-decision emit ──────────────────────────────
            if emit_decisions:
                # Compute pivotality on the observed row.  Semantics from
                # Prompt 1D §B.1: force each condition True and False with all
                # others held fixed, pivotal iff the two decisions differ.  On
                # this window Case II is 0/225, so pivotality is nonzero only
                # on Case I rows (the never-sell / Case II branches cannot
                # produce a Zhang SELL under any state or price).
                _andr_action_str = calib.andrade_signal.action
                _state = _ANDRADE_TO_STATE[_andr_action_str]
                _mult = calib.regime_signal["multiplier"]
                _andr_override = (
                    allow_andrade
                    and _andr_action_str == "STRONG_SELL"
                    and _mult <= NEUTRAL_MULT
                )
                _price_pivotal = False
                _state_pivotal = False
                _case_pivotal = False
                _xstar0 = float("nan")
                if _case == "case1":
                    _price_pivotal = (_state == 2) and (not _andr_override)
                    _state_pivotal = (p0 >= _xs) and (not _andr_override)
                    _case_pivotal = (_state == 2) and (p0 >= _xs) and (not _andr_override)
                    # x*_0 is only defined in Case II; NaN for Case I.
                elif _case == "case2":
                    try:
                        _, _xstar0 = case2_thresholds(*_args, calib.K_absolute)
                    except ValueError:
                        _xstar0 = float("nan")

                # ret_next_month = ticker's return over the *following* month
                # (me_next → me_next2). NaN when we're at the last month.
                _ret_next_month = float("nan")
                if me_next2 is not None:
                    _p_next2 = _price_on_or_before(wide, t, me_next2)
                    _p_next = p1
                    if (not np.isnan(_p_next)) and (not np.isnan(_p_next2)) and _p_next > 0:
                        _ret_next_month = _p_next2 / _p_next - 1.0

                decision_rows.append({
                    "month": me.date().isoformat(),
                    "ticker": t,
                    "action": review.action,
                    "trigger_source": review.trigger_source,
                    "andrade_action": _andr_action_str,
                    "andrade_confidence": float("nan"),  # DHMM Signal exposes no scalar confidence
                    "zhang_case": _case,
                    "zhang_threshold_xstar": float(_xs) if np.isfinite(_xs) else float("nan"),
                    "zhang_xstar0": _xstar0,
                    "p0": float(p0),
                    "ret_full": float(ret_full),
                    "regime_multiplier": float(_mult),
                    "in_book_next_month": bool(t in next_book),
                    "ret_next_month": _ret_next_month,
                    "price_test_pivotal": bool(_price_pivotal),
                    "state_test_pivotal": bool(_state_pivotal),
                    "case_gate_pivotal": bool(_case_pivotal),
                    "phi": float(_phi),
                    "f1": float(_zp.f1),
                    "f2": float(_zp.f2),
                    "lam1": float(_zp.lam1),
                    "lam2": float(_zp.lam2),
                    "state": int(_state),
                    "andrade_override_active": bool(_andr_override),
                    "K_absolute": float(calib.K_absolute),
                    "entry_price": float(calib.entry_price),
                })

        if not base_rets:
            continue
        base_m = float(np.mean(base_rets))
        exp_m = float(np.sum(exp_rets) / n)   # exits → cash (weight kept at 1/n)
        hs_m = float(np.mean(hs_rets))
        rows.append({
            "month": me.date().isoformat(),
            "baseline": base_m,
            "experimental": exp_m,
            "hard_stop": hs_m,
            "n_held": n,
        })
        trigger_rows.append({"month": me.date().isoformat(), **cnt})
        suppression_rows.append({"month": me.date().isoformat(), "suppressed": suppressed})
        hard_stop_rows.append({"month": me.date().isoformat(), "hs_fired": hs_fired})
        logger.info(
            "%s  n=%2d  base=%+.4f  exp=%+.4f  hs=%+.4f  "
            "[zhang=%d andrade=%d both=%d]  suppressed=%d  hs_fired=%d",
            me.date(), n, base_m, exp_m, hs_m,
            cnt["zhang"], cnt["andrade"], cnt["both"], suppressed, hs_fired,
        )

    ret_df = pd.DataFrame(rows).set_index("month") if rows else pd.DataFrame()
    trig_df = pd.DataFrame(trigger_rows).set_index("month") if trigger_rows else pd.DataFrame()
    supp_df = pd.DataFrame(suppression_rows).set_index("month") if suppression_rows else pd.DataFrame()
    hs_df = pd.DataFrame(hard_stop_rows).set_index("month") if hard_stop_rows else pd.DataFrame()

    sharpe = {
        "baseline": _annualized_sharpe(ret_df["baseline"]) if not ret_df.empty else float("nan"),
        "experimental": _annualized_sharpe(ret_df["experimental"]) if not ret_df.empty else float("nan"),
        "hard_stop": _annualized_sharpe(ret_df["hard_stop"]) if not ret_df.empty else float("nan"),
    }
    k_series = pd.Series(k_abs_values, dtype=float)
    k_abs_summary = {
        "n": int(k_series.size),
        "min": float(k_series.min()) if k_series.size else float("nan"),
        "median": float(k_series.median()) if k_series.size else float("nan"),
        "max": float(k_series.max()) if k_series.size else float("nan"),
    }
    return {
        "returns": ret_df,
        "triggers": trig_df,
        "suppression": supp_df,
        "hard_stop_triggers": hs_df,
        "k_abs_summary": k_abs_summary,
        "sharpe": sharpe,
        "big_moves": big_moves,
        "price_vintage": vintage,
        "regime_asof": regime_asof_rows,
        "zhang_diag": zhang_rows,
        "allow_andrade": allow_andrade,
        "decision_rows": decision_rows,
    }


# ── Reporting ────────────────────────────────────────────────────────────────

def write_report(result: dict, start: str, end: str) -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = LOG_DIR / f"exit_backtest_{ts}.log"

    ret_df: pd.DataFrame = result["returns"]
    trig_df: pd.DataFrame = result["triggers"]
    supp_df: pd.DataFrame = result["suppression"]
    hs_df: pd.DataFrame = result["hard_stop_triggers"]
    k_abs = result["k_abs_summary"]
    sharpe = result["sharpe"]
    big = result["big_moves"]

    d_exp = sharpe["experimental"] - sharpe["baseline"]
    d_hs = sharpe["hard_stop"] - sharpe["baseline"]

    lines: list[str] = []
    lines.append("═" * 72)
    lines.append(f"  Markov Exit Layer — DIAGNOSTIC backtest  [{start} → {end}]")
    lines.append(f"  generated {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"  rho={RHO}  K_fraction={K_FRACTION}  "
                 f"calibration_lookback={CALIB_LOOKBACK}d  "
                 f"MIN_SCORE={MIN_SCORE}  TOP_N={TOP_N}")
    v = result["price_vintage"]
    lines.append(f"  {format_vintage(v)}")
    lines.append(f"  price_vintage sha256 (full): {v['sha256']}")
    if VINTAGE_DIR is not None:
        lines.append(f"  DATA SOURCE: FROZEN VINTAGE {VINTAGE_DIR}")
        lines.append(f"    manifest git_head={VINTAGE_MANIFEST['git_head']}  "
                     f"created={VINTAGE_MANIFEST['created_utc']}")
        lines.append(f"    frozen files sha256-verified: "
                     f"{', '.join(sorted(VINTAGE_MANIFEST['frozen']))}")
    else:
        lines.append("  DATA SOURCE: LIVE repo data (NOT frozen — the nightly AV "
                     "backfill revises history)")
    lines.append(f"  allow_andrade = {result['allow_andrade']}"
                 f"{'   ← ANDRADE DISABLED (Zhang-only)' if not result['allow_andrade'] else ''}")
    lines.append("═" * 72)

    lines.append("\n── Annualized Sharpe (×√12) ──")
    lines.append(f"  baseline      : {sharpe['baseline']:+.4f}")
    lines.append(f"  experimental  : {sharpe['experimental']:+.4f}   "
                 f"(Δ vs baseline: {d_exp:+.4f})")
    lines.append(f"  hard_stop     : {sharpe['hard_stop']:+.4f}   "
                 f"(Δ vs baseline: {d_hs:+.4f})")

    lines.append("\n── Monthly returns (baseline / experimental / hard_stop) ──")
    if not ret_df.empty:
        lines.append(f"  {'month':<12}{'base':>10}{'exp':>10}{'hs':>10}"
                     f"{'exp-base':>11}{'hs-base':>10}{'n':>4}")
        for month, r in ret_df.iterrows():
            lines.append(
                f"  {month:<12}{r['baseline']:>+10.4f}{r['experimental']:>+10.4f}"
                f"{r['hard_stop']:>+10.4f}"
                f"{r['experimental']-r['baseline']:>+11.4f}"
                f"{r['hard_stop']-r['baseline']:>+10.4f}{int(r['n_held']):>4d}"
            )
        lines.append("  " + "-" * 60)
        lines.append(
            f"  {'MEAN':<12}{ret_df['baseline'].mean():>+10.4f}"
            f"{ret_df['experimental'].mean():>+10.4f}"
            f"{ret_df['hard_stop'].mean():>+10.4f}"
        )
    else:
        lines.append("  (no monthly returns computed)")

    lines.append("\n── Override attribution per month (monthly baseline layer) ──")
    lines.append("  zhang-only  = Zhang SELL, Andrade not STRONG_SELL")
    lines.append("  andrade-only= Andrade STRONG_SELL, Zhang HOLD")
    lines.append("  both        = both fired the same month")
    if not trig_df.empty:
        lines.append(f"  {'month':<12}{'zhang':>8}{'andrade':>9}{'both':>7}{'none':>7}")
        for month, r in trig_df.iterrows():
            lines.append(f"  {month:<12}{int(r['zhang']):>8d}{int(r['andrade']):>9d}"
                         f"{int(r['both']):>7d}{int(r['none']):>7d}")
        lines.append("  " + "-" * 42)
        lines.append(f"  {'TOTAL':<12}{int(trig_df['zhang'].sum()):>8d}"
                     f"{int(trig_df['andrade'].sum()):>9d}"
                     f"{int(trig_df['both'].sum()):>7d}{int(trig_df['none'].sum()):>7d}")
    else:
        lines.append("  (no overrides)")

    lines.append("\n── Large moves (|monthly return| > 5%) the exit layer would have exited ──")
    if big:
        for m in big:
            lines.append(f"  {m['month']}  {m['ticker']:<6}  "
                         f"ret={m['ret_full']:+.4f}  caught_by={m['trigger_source']}")
    else:
        lines.append("  (none — no exited ticker had a >5% monthly move)")

    # ── Regime suppression (REV 4 fix B) ──────────────────────────────────
    lines.append("\n── Regime suppression of Andrade STRONG_SELL (REV 4 fix B) ──")
    lines.append("  A STRONG_SELL suppressed because the live-regime multiplier > "
                 f"NEUTRAL_MULT ({NEUTRAL_MULT}).")
    if not supp_df.empty:
        total_supp = int(supp_df["suppressed"].sum())
        months_with_supp = int((supp_df["suppressed"] > 0).sum())
        lines.append(f"  {'month':<12}{'suppressed':>12}")
        for month, r in supp_df.iterrows():
            lines.append(f"  {month:<12}{int(r['suppressed']):>12d}")
        lines.append("  " + "-" * 24)
        lines.append(f"  {'TOTAL':<12}{total_supp:>12d}")
        lines.append(f"  months with ≥1 suppression: {months_with_supp} / {len(supp_df)}")
    else:
        lines.append("  (no calibrations)")

    # ── Hard-stop trigger count ───────────────────────────────────────────
    lines.append("\n── Daily hard-stop triggers (REV 3 = 0) ──")
    if not hs_df.empty:
        total_hs = int(hs_df["hs_fired"].sum())
        months_with_hs = int((hs_df["hs_fired"] > 0).sum())
        lines.append(f"  {'month':<12}{'hs_fired':>10}")
        for month, r in hs_df.iterrows():
            lines.append(f"  {month:<12}{int(r['hs_fired']):>10d}")
        lines.append("  " + "-" * 22)
        lines.append(f"  {'TOTAL':<12}{total_hs:>10d}")
        lines.append(f"  months with ≥1 hard-stop: {months_with_hs} / {len(hs_df)}")
    else:
        lines.append("  (no calibrations)")

    # ── Point-in-time regime per month (2026-08-03 lookahead fix) ─────────
    lines.append("\n── Point-in-time regime per month (as-of, NOT run-time) ──")
    lines.append("  Was: get_live_regime_signal() — latest snapshot applied to every month.")
    lines.append("  Now: get_regime_signal_asof(month_end) — macro dated ≤ month_end only.")
    ra = result["regime_asof"]
    if ra:
        lines.append(f"  {'month':<12}{'mult':>7}{'rule':>10}{'stale':>8}")
        for r in ra:
            lines.append(f"  {r['month']:<12}{r['multiplier']:>7.2f}"
                         f"{r['rule_signal']:>10}{str(r['stale']):>8}")
        n_supp_possible = sum(1 for r in ra if r["multiplier"] > NEUTRAL_MULT)
        lines.append(f"  months where regime could suppress Andrade "
                     f"(mult > {NEUTRAL_MULT}): {n_supp_possible} / {len(ra)}")
    else:
        lines.append("  (none)")

    # ── Zhang Case I / Case II split (Prompt 1C Task B) ───────────────────
    lines.append("\n── Zhang regime split across ALL calibrations ──")
    zd = pd.DataFrame(result["zhang_diag"])
    if not zd.empty:
        counts = zd["case"].value_counts().to_dict()
        lines.append(f"  n_calibrations = {len(zd)}")
        for k in ("case1", "case2", "never_sell"):
            n = int(counts.get(k, 0))
            lines.append(f"    {k:<12} {n:4d}  ({n/len(zd)*100:5.1f}%)")
        lines.append("  Case I  = rho ≤ f1 → sell only in state 2 above x*; never hard-stops")
        lines.append("  Case II = rho > f1 → x* and mandatory x*_0 both live")
        st = zd["state"].value_counts().to_dict()
        lines.append(f"  Andrade-derived state: state1(uptick)={int(st.get(1,0))}  "
                     f"state2(downtick)={int(st.get(2,0))}")
        lines.append("  (Zhang can only SELL in state 2 — state 1 never sells in either case)")

        r = zd["x_star_over_p0"].replace([np.inf, -np.inf], np.nan).dropna()
        lines.append("\n── x* relative to p0 (x*/p0; SELL needs price ≥ x*, i.e. ratio ≤ 1) ──")
        if not r.empty:
            lines.append(f"  n={len(r)}  min={r.min():.4g}  p05={r.quantile(.05):.4g}  "
                         f"median={r.median():.4g}  p95={r.quantile(.95):.4g}  max={r.max():.4g}")
            n_reach = int((r <= 1.0).sum())
            lines.append(f"  calibrations where p0 ≥ x* at month-end (Zhang would fire "
                         f"if state 2): {n_reach} / {len(r)}")
            n_state2_reach = int(((zd["x_star_over_p0"] <= 1.0) & (zd["state"] == 2)).sum())
            lines.append(f"  … AND in state 2 (actual Zhang SELL): {n_state2_reach}")
        else:
            lines.append("  (no finite x* — all never-sell)")
    else:
        lines.append("  (no calibrations)")

    # ── K_absolute distribution ───────────────────────────────────────────
    lines.append("\n── K_absolute distribution across all (ticker, month) calibrations ──")
    lines.append(f"  K_fraction = {K_FRACTION}  (K_absolute = K_fraction × entry_price)")
    lines.append(f"  n={k_abs['n']}  min={k_abs['min']:.4f}  "
                 f"median={k_abs['median']:.4f}  max={k_abs['max']:.4f}")
    lines.append(f"  (median should be ≈ {K_FRACTION} × median entry price in the universe)")

    # ── REV 3 vs REV 4 comparison ─────────────────────────────────────────
    lines.append("\n── REV 3 vs REV 4 comparison ──")
    lines.append(f"  {'Path':<14}{'REV 3 Sharpe':>14}{'REV 4 Sharpe':>14}"
                 f"{'Δ (REV4−REV3)':>16}")
    for path_label, key in (("baseline", "baseline"),
                            ("experimental", "experimental"),
                            ("hard-stop", "hard_stop")):
        r3 = REV3_SHARPE[key]
        r4 = sharpe[key]
        lines.append(f"  {path_label:<14}{r3:>+14.4f}{r4:>+14.4f}{r4 - r3:>+16.4f}")

    # ── Standing decision rule (diagnostic batch 2026-07-30) ──────────────
    # Prompt 1D Task C: the old rule-2/rule-3 "PASS/FAIL" verdict was retired.
    # It compared this run's experimental Sharpe against the REV 3 numbers and
    # emitted FULL PASS / MINIMUM PASS / FAIL — a verdict on the layer that the
    # batch's actual KEEP/SHELVE rule does not authorise this script to make.
    # None of (a)/(b)/(c) below is computable from what run_backtest returns:
    # (a) needs per-SELL ret_full, (b) needs the paired monthly series under a
    # sign test and a paired t-test, (c) needs the redistribute-to-survivors
    # variant and a max-drawdown series. This block therefore only RESTATES the
    # rule and reports the integrity check it can actually run.
    exp_r4 = sharpe["experimental"]
    base_drift = abs(sharpe["baseline"] - REV3_SHARPE["baseline"])

    lines.append("\n" + "═" * 72)
    lines.append("  STANDING DECISION RULE — diagnostic batch 2026-07-30")
    lines.append("  KEEP the layer only if ALL THREE hold:")
    lines.append("    (a) SELL-flagged positions have a NEGATIVE mean ret_full")
    lines.append("    (b) the paired monthly difference is significant at 0.05 by")
    lines.append("        BOTH sign test and paired t-test, in the layer's favour")
    lines.append("    (c) the redistribute-to-survivors variant still beats baseline")
    lines.append("        on Sharpe OR cuts max drawdown by ≥ 3 percentage points")
    lines.append("  SHELVE if (a) fails.  INCONCLUSIVE (→ shelve, noted) if (a) holds")
    lines.append("  but (b) fails.")
    lines.append("")
    lines.append("  NOT EVALUATED HERE — this script measures none of (a), (b), (c).")
    lines.append("  Sharpe alone does not decide the layer. Prompt 2 tests (a).")
    lines.append("")
    lines.append(f"  integrity check — baseline drift vs REV 3: {base_drift:.6f} "
                 f"({'OK' if base_drift <= 1e-4 else 'DRIFT > 1e-4 — INVESTIGATE'})")
    lines.append(f"  observed: experimental Sharpe {exp_r4:+.4f} "
                 f"(Δ vs baseline {d_exp:+.4f}) — reported, not adjudicated")
    lines.append("═" * 72)

    report = "\n".join(lines)
    path.write_text(report + "\n")
    print(report)
    return path


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s │ %(levelname)-7s │ %(message)s",
        datefmt="%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description="Diagnostic exit-layer backtest")
    ap.add_argument("--start", default="2025-01-01")
    ap.add_argument("--end", default="2026-06-30")
    ap.add_argument(
        "--vintage", default=None, metavar="DIR",
        help="Read all data inputs from a frozen snapshot created by "
             "scripts/freeze_vintage.py (sha256-verified against its "
             "MANIFEST.json). Omit to read live repo data, which the nightly AV "
             "backfill retroactively revises.")
    ap.add_argument(
        "--no-andrade", action="store_true",
        help="Disable the Andrade override entirely, leaving Zhang as the only "
             "exit trigger (Prompt 1C Task B).")
    ap.add_argument(
        "--emit-decisions", default=None, metavar="PATH",
        help="Prompt 2: write a per-decision parquet for every (month, ticker) "
             "evaluated on the Andrade-ON path, and re-run the Zhang-only path "
             "to assert baseline/experimental/hard_stop/Zhang-only Sharpes are "
             "bit-identical to the frozen Prompt-1D reference values (tolerance "
             "1e-6). Fails loudly on drift.")
    args = ap.parse_args()

    _assert_interpreter()
    if args.vintage:
        use_vintage(args.vintage)
    else:
        logger.warning("No --vintage given — reading LIVE repo data, which the "
                       "nightly AV backfill revises. Results may not be "
                       "comparable across runs.")

    if args.emit_decisions:
        # Reference Sharpes from Prompt 1D §"Reference Sharpe figures"
        # (backtests/exit_layer_units_and_attribution_2026-08-04.md), which
        # were themselves reproduced from Prompt 1C at 0.0e+00 drift on the
        # frozen vintage. Tolerance 1e-6 per Prompt 2.
        EXPECTED = {
            "baseline":       1.587144507439707,
            "experimental":   1.282216655915358,
            "hard_stop":      1.587144507439707,
            "zhang_only":     1.591542820613977,
        }
        # Run Andrade-ON (this is the run whose decisions we emit) then
        # Andrade-OFF (to check the Zhang-only Sharpe).
        result_on = run_backtest(args.start, args.end,
                                 allow_andrade=True, emit_decisions=True)
        result_off = run_backtest(args.start, args.end,
                                  allow_andrade=False, emit_decisions=False)

        got = {
            "baseline":     result_on["sharpe"]["baseline"],
            "experimental": result_on["sharpe"]["experimental"],
            "hard_stop":    result_on["sharpe"]["hard_stop"],
            "zhang_only":   result_off["sharpe"]["experimental"],
        }
        drift = {k: abs(got[k] - EXPECTED[k]) for k in EXPECTED}
        for k, d in drift.items():
            logger.info("SHARPE ASSERT  %-12s got=%.15f expected=%.15f drift=%.3e",
                        k, got[k], EXPECTED[k], d)
        bad = {k: d for k, d in drift.items() if d > 1e-6}
        if bad:
            raise SystemExit(
                f"ABORT: Sharpe drift beyond 1e-6 — instrumentation is not "
                f"return-neutral: {bad}")

        emit_path = Path(args.emit_decisions)
        if not emit_path.is_absolute():
            emit_path = ROOT / emit_path
        emit_path.parent.mkdir(parents=True, exist_ok=True)
        dec_df = pd.DataFrame(result_on["decision_rows"])
        dec_df.to_parquet(emit_path, index=False)
        logger.info("Wrote %d per-decision rows → %s", len(dec_df), emit_path)

        # Still write the standard report for the Andrade-ON run.
        report_path = write_report(result_on, args.start, args.end)
        logger.info("Report written → %s", report_path)
        return

    result = run_backtest(args.start, args.end,
                          allow_andrade=not args.no_andrade)
    path = write_report(result, args.start, args.end)
    logger.info("Report written → %s", path)


if __name__ == "__main__":
    main()
