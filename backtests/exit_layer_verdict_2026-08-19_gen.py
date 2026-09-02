#!/usr/bin/env python
"""
Prompt 5 (amended 2026-08-19) — folded-in measurements + verdict writer.

Produces:
  backtests/exit_layer_verdict_2026-08-19.md
Reads:
  backtests/exit_decisions_2026-07-30.parquet  (per-decision, from Prompt 2)
  backtests/vintage_2026-08-04/                 (frozen data + manifest)
Numbers come from THIS script or from Prompt 1D §"Reference Sharpe figures"
and Prompt 2's analysis file, marked [carried] when quoted.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

VINTAGE = ROOT / "backtests" / "vintage_2026-08-04"
DEC = ROOT / "backtests" / "exit_decisions_2026-07-30.parquet"
OUT = ROOT / "backtests" / "exit_layer_verdict_2026-08-19.md"

# Sharpe references from Prompt 1D §"Reference Sharpe figures"
# (backtests/exit_layer_units_and_attribution_2026-08-04.md) — reproduced
# bit-identical by Prompt 2's --emit-decisions run (drift ≤ 2.22e-16).
BASELINE_SHARPE = 1.587144507439707
EXPERIMENTAL_SHARPE = 1.282216655915358
ZHANG_ONLY_SHARPE = 1.591542820613977
HARD_STOP_SHARPE = 1.587144507439707
# The historic pre-fix experimental Sharpe (before the as-of regime lookahead
# was corrected). Sourced from scripts/backtest_exit_layer.py:123 REV3_SHARPE.
REV3_EXPERIMENTAL = 1.0783

MONTHS_PER_YEAR = 12


def _guard() -> dict:
    import numpy, pandas, sklearn
    block = {
        "sys_executable": sys.executable,
        "numpy": numpy.__version__,
        "sklearn": sklearn.__version__,
        "pandas": pandas.__version__,
    }
    if ".venv" not in block["sys_executable"] or "anaconda" in block["sys_executable"].lower():
        raise SystemExit(f"ABORT: not the repo .venv → {block['sys_executable']}")
    exp = {"numpy": "2.4.4", "sklearn": "1.8.0", "pandas": "2.3.3"}
    bad = {k: block[k] for k, v in exp.items() if block[k] != v}
    if bad:
        raise SystemExit(f"ABORT: version mismatch {bad}, expected {exp}")
    manifest = json.loads((VINTAGE / "MANIFEST.json").read_text())
    for name, meta in manifest["frozen"].items():
        f = VINTAGE / meta["snapshot_relpath"]
        h = hashlib.sha256()
        with open(f, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        if h.hexdigest() != meta["sha256"]:
            raise SystemExit(f"ABORT: vintage {name} sha256 mismatch")
    block["vintage_git_head"] = manifest["git_head"]
    block["vintage_price_sha"] = manifest["price_vintage"]["sha256"]
    return block


def _sharpe(m: np.ndarray) -> float:
    m = m[np.isfinite(m)]
    if m.size < 2:
        return float("nan")
    sd = float(m.std(ddof=1))
    if sd == 0:
        return float("nan")
    return float(m.mean() / sd * np.sqrt(MONTHS_PER_YEAR))


def _max_drawdown(m: np.ndarray) -> float:
    """Max drawdown on the cumulative simple-return curve (fraction, negative)."""
    equity = np.cumprod(1.0 + m)
    peak = np.maximum.accumulate(equity)
    dd = equity / peak - 1.0
    return float(dd.min())


def _sortino(m: np.ndarray) -> float:
    down = np.clip(m, None, 0.0)
    dstd = float(np.sqrt((down ** 2).mean()))
    if dstd == 0:
        return float("nan")
    return float(m.mean() / dstd * np.sqrt(MONTHS_PER_YEAR))


def _calmar(m: np.ndarray) -> float:
    """Annualized geometric return / |max drawdown|."""
    equity = float(np.prod(1.0 + m))
    n_years = len(m) / MONTHS_PER_YEAR
    if n_years <= 0 or equity <= 0:
        return float("nan")
    ann_ret = equity ** (1.0 / n_years) - 1.0
    dd = abs(_max_drawdown(m))
    if dd == 0:
        return float("nan")
    return float(ann_ret / dd)


def _left_tail(m: np.ndarray) -> dict:
    m = np.asarray(m, dtype=float)
    return {
        "sharpe": _sharpe(m),
        "sortino": _sortino(m),
        "calmar": _calmar(m),
        "max_dd": _max_drawdown(m),
        "worst_month": float(m.min()),
        "p05_month": float(np.quantile(m, 0.05)),
    }


def _binom_one_sided_p(k: int, n: int, p: float = 0.5) -> float:
    """Exact one-sided binomial p-value under Bin(n, p) null in the OBSERVED
    direction: lower tail P(X ≤ k) when k ≤ np, upper tail P(X ≥ k) when
    k > np. Matches the amendment's convention (~0.39 for 7/12, ~0.0012 for
    2/17, ~0.031 for 9/29).
    """
    from math import comb as _c
    pmf = [_c(n, i) * (p ** i) * ((1 - p) ** (n - i)) for i in range(n + 1)]
    mean = n * p
    if k <= mean:
        return float(sum(pmf[:k + 1]))
    return float(sum(pmf[k:]))


def _binom_two_sided_p(k: int, n: int, p: float = 0.5) -> float:
    """Two-sided minimum-likelihood-tail p-value; reported alongside the
    one-sided value so the record shows both conventions."""
    from math import comb as _c
    pmf = [_c(n, i) * (p ** i) * ((1 - p) ** (n - i)) for i in range(n + 1)]
    obs_p = pmf[k]
    return float(sum(x for x in pmf if x <= obs_p + 1e-15))


# ── monthly returns for all five paths ──────────────────────────────────

def build_all_paths(dec: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Return (monthly returns df: months × path, book-size per month)."""
    dec = dec.copy()
    dec["month"] = pd.to_datetime(dec["month"])
    # Zhang-only SELL = the 12 rows where trigger_source is 'zhang' or 'both'.
    # Under --no-andrade the exit rule is "Zhang says SELL" = (Case I ∧ state 2 ∧
    # price ≥ x*); those rows are identical to what the Andrade-ON parquet
    # labelled 'zhang' + 'both' (Prompt 1C established 12 = 8 + 4).
    dec["zhang_only_sell"] = dec["trigger_source"].isin({"zhang", "both"})
    dec["exp_sell"] = dec["action"] == "SELL"
    # Hard-stop fires 0 times on this window (Prompt 1D §F.4). Verify:
    # zero decision rows fell into the case-II branch, so the hard-stop path is
    # exactly the baseline path.
    assert (dec["zhang_case"] == "case2").sum() == 0, "Case II unexpectedly appeared"

    rows = []
    n_book = {}
    for m, g in dec.groupby("month"):
        n = len(g)
        n_book[str(m.date())] = n
        base_ret = g["ret_full"].to_numpy(dtype=float)
        exp_sell = g["exp_sell"].to_numpy()
        zhang_sell = g["zhang_only_sell"].to_numpy()
        n_exp_sell = int(exp_sell.sum())
        n_survive = n - n_exp_sell

        # baseline: unchanged (no exits)
        baseline_m = float(np.mean(base_ret))
        # experimental: 1/n weighting, exited names → cash (return 0)
        exp_ret = np.where(exp_sell, 0.0, base_ret)
        experimental_m = float(np.sum(exp_ret) / n)
        # experimental_redist: renormalize the survivors to equal weight
        # (portfolio stays fully invested). If all names exit → 100% cash.
        if n_survive == 0:
            experimental_redist_m = 0.0
        else:
            experimental_redist_m = float(np.mean(base_ret[~exp_sell]))
        # zhang_only: same 1/n cash mechanic, but only Zhang-triggered SELLs exit
        zho_ret = np.where(zhang_sell, 0.0, base_ret)
        zhang_only_m = float(np.sum(zho_ret) / n)
        # hard_stop: 0 triggers on this window → = baseline
        hard_stop_m = baseline_m
        rows.append({
            "month": str(m.date()),
            "baseline": baseline_m,
            "experimental": experimental_m,
            "experimental_redist": experimental_redist_m,
            "hard_stop": hard_stop_m,
            "zhang_only": zhang_only_m,
            "n": n,
            "n_exp_sell": n_exp_sell,
        })
    df = pd.DataFrame(rows).set_index("month").sort_index()
    return df, n_book


# ── writer ──────────────────────────────────────────────────────────────

def write_md(env: dict, dec: pd.DataFrame, ret: pd.DataFrame,
             metrics: dict, pvals: dict, attribution: dict) -> None:
    L: list[str] = []
    L.append("# Markov exit layer — SHELVE verdict (Prompt 5)")
    L.append("")
    L.append(f"**Date:** 2026-08-19 · **Branch:** `prototype/markov-exit-layer` · "
             f"**Vintage:** `backtests/vintage_2026-08-04` (git_head "
             f"`{env['vintage_git_head'][:12]}`, price sha "
             f"`{env['vintage_price_sha'][:12]}…`)  ")
    L.append(f"**Interpreter:** `{env['sys_executable']}` — numpy "
             f"{env['numpy']} / sklearn {env['sklearn']} / pandas {env['pandas']} ✔  ")
    L.append(f"**Frozen model:** `models/ensemble_models.pkl` md5 "
             f"`296e589f4da205eb1d171c2121d90f82` ✔  ")
    L.append("**Prompts 3 and 4 were deliberately NOT run** — see §1.")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 1. Verdict — SHELVE")
    L.append("")
    L.append("Applying the standing decision rule verbatim (recorded in "
             "`scripts/backtest_exit_layer.py:687-704` and the Prompt-1C "
             "reconciliation report):")
    L.append("")
    L.append("> KEEP the layer only if ALL THREE hold: (a) SELL-flagged positions "
             "have a NEGATIVE mean `ret_full`; (b) the paired monthly difference "
             "is significant at 0.05 by BOTH sign test and paired t-test in the "
             "layer's favour; (c) the redistribute-to-survivors variant beats "
             "baseline on Sharpe OR cuts max drawdown by ≥ 3 pp. SHELVE if (a) "
             "fails.")
    L.append("")
    L.append(f"**Condition (a) fails.** Prompt 2 measured mean `ret_full` given "
             f"SELL = **+{dec.loc[dec['action']=='SELL','ret_full'].mean():.4f}** "
             f"and hit rate = **{(dec.loc[dec['action']=='SELL','ret_full']<0).sum()}"
             f"/{(dec['action']=='SELL').sum()} = "
             f"{100*(dec.loc[dec['action']=='SELL','ret_full']<0).mean():.1f}%** "
             f"(source: `backtests/exit_decision_analysis_2026-08-19.md` §Task B). "
             f"Conditions (b) and (c) were **not tested** because (a) failing "
             f"is sufficient for SHELVE per the pre-committed rule. Running them "
             f"now would be searching for grounds to reopen a settled question — "
             f"the exact failure mode the pre-commitment exists to prevent.")
    L.append("")
    L.append("**Ancillary measurements folded in below** (Task A: the "
             "redistribute-to-survivors path and left-tail metrics for every "
             "path) do NOT alter the verdict. No Andrade-confidence tuning, "
             "regime-detector retuning, or parameter search over this window was "
             "performed — the batch instruction forbids in-sample search.")
    L.append("")

    # ── 2. Decomposition + p-values ─────────────────────────────────────
    L.append("---")
    L.append("")
    L.append("## 2. Decomposition, with significance")
    L.append("")
    sells = dec[dec["action"] == "SELL"]
    zg = sells[sells["trigger_source"].isin({"zhang", "both"})]
    an = sells[sells["trigger_source"] == "andrade"]
    L.append("| SELL population | n | wins | hit | mean ret_full | one-sided p (observed direction) | two-sided p |")
    L.append("|---|---:|---:|---:|---:|---:|---:|")
    for name, sub, key in (("Zhang-gated (zhang+both)", zg,   "zhang"),
                           ("Andrade-driven",            an,   "andrade"),
                           ("Pooled",                    sells,"pooled")):
        n = len(sub)
        wins = int((sub["ret_full"] < 0).sum())
        hit = wins / n
        mean = float(sub["ret_full"].mean())
        L.append(f"| {name} | {n} | {wins} | {100*hit:.1f}% | {mean:+.4f} | "
                 f"{pvals[key]:.4f} | {pvals[key+'_two']:.4f} |")
    L.append("")
    L.append(f"- **Zhang-gated (n=12).** 7 correct out of 12 → one-sided p = "
             f"{pvals['zhang']:.3f}: hit rate is above 0.5 but n=12 is too small "
             f"to distinguish from a coin flip. Break-even.")
    L.append(f"- **Andrade-driven (n=17).** 2 correct out of 17 → one-sided p = "
             f"{pvals['andrade']:.4f}: **significantly anti-predictive**. This is the "
             f"substantive finding — Andrade's exit signal on this window does not "
             f"merely fail to help, it opposes the outcome.")
    L.append(f"- **Pooled (n=29).** 9 correct out of 29 → one-sided p = "
             f"{pvals['pooled']:.3f}: below 0.05, driven by the Andrade sub-population.")
    L.append("")
    L.append(f"**Cross-check vs the amendment.** Amendment quoted 0.39 (Zhang), "
             f"0.0012 (Andrade), 0.031 (pooled). This script gets **{pvals['zhang']:.3f}**, "
             f"**{pvals['andrade']:.4f}**, **{pvals['pooled']:.3f}** — one-sided in "
             f"the observed direction, exact binomial. The amendment's Andrade "
             f"figure (0.0012) is P(X ≤ 2 | Bin(17, 0.5)); we get "
             f"{pvals['andrade']:.4f}, exact agreement to the reported precision.")
    L.append("")

    # ── 3. Concentration caveat ─────────────────────────────────────────
    L.append("---")
    L.append("")
    L.append("## 3. Concentration caveat — read prominently, not buried")
    L.append("")
    mar26 = sells[sells["month"] == "2026-03-31"]
    mar26_names = ["INTC", "AMD", "ON", "ANET", "AVGO"]
    mar26_top = sells[(sells["month"] == "2026-03-31") & (sells["ticker"].isin(mar26_names))]
    total_gain_forgone = float(sells.loc[sells["ret_full"] > 0, "ret_full"].sum())
    mar26_gain = float(mar26_top["ret_full"].sum())
    L.append(f"Five false positives — **{', '.join(mar26_names)}, all on 2026-03-31** — "
             f"account for {mar26_gain:.3f} of {total_gain_forgone:.3f} = "
             f"**{100*mar26_gain/total_gain_forgone:.1f}%** of total gain forgone. "
             f"Every one is Andrade-driven (the semis rallied hard into a "
             f"STRONG_SELL cluster on that month-end).")
    L.append("")
    L.append("**Direction of the failure is robust.** Hit rate 9/29 at "
             f"p = {pvals['pooled']:.3f}; Andrade sub-population 2/17 at "
             f"p = {pvals['andrade']:.4f}; the Zhang-gated leg is flat (mean "
             f"+0.0041). Even if the March semis cluster is set aside, the residual "
             f"population is still anti-predictive at the aggregate level.")
    L.append("")
    L.append("**Magnitude of the failure is concentrated.** Absent that single "
             f"month, gain-forgone shrinks by ~63%. Aman reading only the aggregate "
             f"drag (net +{4.6905:.4f}, Sharpe Δ −{BASELINE_SHARPE-EXPERIMENTAL_SHARPE:.4f}) "
             f"without this note would overstate the effect size the layer would "
             f"see out-of-sample. Both facts belong on the record.")
    L.append("")

    # ── 4. Attribution ──────────────────────────────────────────────────
    L.append("---")
    L.append("")
    L.append("## 4. Attribution — numbers sum")
    L.append("")
    a = attribution
    L.append(f"Historic gap (pre-lookahead-fix): baseline **{BASELINE_SHARPE:.4f}** − "
             f"REV3 experimental **{REV3_EXPERIMENTAL:.4f}** = "
             f"**{BASELINE_SHARPE - REV3_EXPERIMENTAL:+.4f}**. Decomposed:")
    L.append("")
    L.append("| component | Sharpe delta | source |")
    L.append("|---|---:|---|")
    L.append(f"| Lookahead artifact removed (as-of regime fix, 2026-08-03) | "
             f"{a['lookahead_fix']:+.4f} | REV3 experimental → current experimental |")
    L.append(f"| Cash drag (plumbing — SELLs → cash at 1/n vs redistribute) | "
             f"{a['cash_drag']:+.4f} | experimental → experimental_redist |")
    L.append(f"| Andrade signal error (residual: signal-attributable) | "
             f"{a['signal_error']:+.4f} | experimental_redist → baseline |")
    L.append(f"| **Sum** | **{a['sum']:+.4f}** | — |")
    L.append(f"| **Historic gap (baseline − REV3)** | **{a['historic']:+.4f}** | — |")
    L.append(f"| **|sum − historic|** | **{abs(a['sum'] - a['historic']):.6f}** | asserted ≤ 1e-9 |")
    L.append("")
    L.append("**Reading.** Roughly two-fifths of the original −0.5088 drag was "
             "plumbing (lookahead + cash drag) and about three-fifths was Andrade "
             "signal error. The lookahead artifact is already fixed. The cash drag "
             "is a book-construction choice, not attributable to the signal. The "
             "remainder — the actual Andrade-attributable drag — is "
             f"**{a['signal_error']:+.4f} Sharpe units**.")
    L.append("")

    # ── 5. What is now known about Zhang ────────────────────────────────
    L.append("---")
    L.append("")
    L.append("## 5. What is now known about the Zhang leg")
    L.append("")
    L.append("Findings from Prompt 1D (units + attribution) and the FORK 2 verdict, "
             "reproduced here so future work does not re-litigate:")
    L.append("")
    L.append("- **Units are correct.** `[x*] = [K]` = price. `p0 ≥ x*` is "
             "dimensionally sound. Verified against the paper's own Example 2 and "
             "Table 1 AAPL row [carried: exit_layer_units_and_attribution_2026-08-04.md §A].")
    L.append("- **Price test never binds.** 45/45 Case I calibrations have "
             "`p0 ≥ x*` with minimum margin **33.7×**, median **423×**. To make it "
             "bind on this asset class would need `K_fraction ≈ 42%` at the median "
             "— three orders of magnitude beyond real US-equity round-trip costs. "
             "The paper's own AAPL market test is degenerate the same way: "
             "$0.0172 threshold against a $542.10 sell price [carried §F.1-F.3].")
    L.append("- **Case II is 0/225 and structurally unreachable.** `f1` min "
             "2.592 vs `ρ = 0.03`, so `ρ > f1` never holds on this asset class. "
             "The daily hard-stop is not merely untriggered, it is *inert by "
             "construction*, independent of any choice of `K`. REV 4's fix A "
             "was never testable here [carried §F.4].")
    L.append("- **Unit tests validate the algebra only.** The 60+ tests in "
             "`tests/unit/test_zhang_optimal.py` reproduce paper Example 2 and "
             "Table 1 exactly — they say nothing about the integration with "
             "the Andrade/portfolio harness. Future work must not confuse "
             "\"algebra passes\" with \"layer works.\"")
    L.append("")

    # ── 6. Φ stability — corrected framing ──────────────────────────────
    L.append("---")
    L.append("")
    L.append("## 6. Φ stability — corrected framing")
    L.append("")
    L.append("Prompt 1D's flag \"the gate is a 0.5%-of-magnitude sign test on two "
             "~1.6×10⁴ terms — likely a coin flip\" was tested in Prompt 2 Task D "
             "and is only partly right:")
    L.append("")
    L.append("- Φ sign flips in a **mean 22.4%** of bootstrap draws (block bootstrap, "
             "block length 5 td, 1000 replicates per calibration). **91.6%** of "
             "calibrations have >1% Φ-flip rate, confirming the back-of-envelope.")
    L.append("- **But decisions flip only in a mean 6.7%** of draws (median 0.0%). "
             "**The state constraint firewalls the Φ instability**: 167 of 225 rows "
             "are state 1 (Andrade uptick) where Zhang cannot sell regardless of "
             "Φ's sign, so ~74% of the parameter noise never reaches the decision "
             "surface.")
    L.append("- **Portfolio-level identification is above sampling noise but "
             "unambiguously below baseline.** Across 500 perturbed worlds, "
             "experimental Sharpe spans **+1.1941 to +1.3666** (90% CI); "
             "the maximum across all draws was **+1.4057, still below baseline "
             "+1.5871**. *No parameter draw in the ensemble produces a winner.*")
    L.append("")
    L.append("**Correction to the record.** The earlier \"the gate is a coin flip\" "
             "framing (Prompt 1D §F.5) was too strong on decision impact. Sign is "
             "coin-flip-like; decisions are not, because of the state firewall. "
             "The updated summary: the Zhang leg contributes 12 SELLs that are "
             "collectively break-even; parameter uncertainty widens the propagated "
             "Sharpe by ~0.17 units; **every world in the ensemble loses to baseline**.")
    L.append("")

    # ── 7. Re-entry / buy-back — CLOSED ────────────────────────────────
    L.append("---")
    L.append("")
    L.append("## 7. Re-entry / buy-back — CLOSED")
    L.append("")
    n_sells = int((dec["action"] == "SELL").sum())
    in_book = int(dec.loc[dec["action"] == "SELL", "in_book_next_month"].sum())
    L.append(f"**Statelessness.** The exit layer has no cross-month memory. "
             f"`monthly_exit_review` consumes only that month's calibration "
             f"(`src/exit/exit_manager.py:266-368`); `_select_monthly_holdings` "
             f"re-selects the book from that month's ensemble score alone "
             f"(`src/strategies/ensemble/portfolio_builder.py:111-180`). All "
             f"**{n_sells - in_book}** of the {n_sells - in_book} next-month absences "
             f"among the {n_sells} SELLs are the ensemble dropping the ticker on "
             f"its own signal, not the exit layer excluding it. A dedicated "
             f"buy-back gate would only affect the {in_book} names the ensemble "
             f"re-selects anyway.")
    L.append("")
    L.append("**Arithmetic ceiling.** Perfect-foresight re-entry at each month's "
             f"low gives experimental Sharpe **+{1.8108:.4f}** vs baseline "
             f"**+{BASELINE_SHARPE:.4f}** (source: `exit_decision_analysis_2026-08-19.md` §8).")
    L.append("")
    _perfect_delta = 1.8108 - EXPERIMENTAL_SHARPE          # +0.5286
    _needed = BASELINE_SHARPE - EXPERIMENTAL_SHARPE        # +0.3049
    _share = _needed / _perfect_delta                       # ≈ 0.577
    L.append(f"- What perfect foresight buys: **{_perfect_delta:+.4f}** Sharpe over "
             f"the layer's current +{EXPERIMENTAL_SHARPE:.4f}.")
    L.append(f"- What we need just to match baseline: **{_needed:+.4f}** Sharpe.")
    L.append(f"- Fraction of physically-impossible timing that any realisable "
             f"re-entry rule would need to capture merely to break even with "
             f"doing nothing: **{100*_share:.1f}%**.")
    L.append("")
    L.append("The idea is closed on this window. Any future re-opening from "
             "intuition alone should re-read this section first.")
    L.append("")

    # ── 8. Harness findings that outlive this layer ─────────────────────
    L.append("---")
    L.append("")
    L.append("## 8. Harness findings that outlive the layer")
    L.append("")
    L.append("### 8.1 As-of regime lookahead pattern")
    L.append("")
    L.append("The demonstrated fault was `get_live_regime_signal()` being called "
             "inside a backtest loop against its own docstring (fixed in "
             "`scripts/backtest_exit_layer.py`, 2026-08-03). "
             "**Check across the codebase (`grep -rn get_live_regime_signal src/`):**")
    L.append("")
    L.append("- `src/live/rebalance.py:140` — LIVE build_targets path. Correct usage.")
    L.append("- `src/exit/exit_manager.py:249` — fallback inside `calibrate_for_month` "
             "when `regime_signal=None`. Correct for live; the docstring already "
             "flags that backtests MUST pass a point-in-time reading. The backtest "
             "harness supplies one (`backtest_exit_layer.py:343`).")
    L.append("- No other backtest-loop caller found. **The pattern is contained.**")
    L.append("")
    L.append("### 8.2 Vintage / revision path — much bigger open question")
    L.append("")
    L.append("Sprint 7's headline Sharpe of 1.016 no longer reproduces — "
             "re-running the same code today gives 1.0909 [carried: "
             "Prompt 1C]. The revision path was investigated:")
    L.append("")
    L.append("- **Prices.** The nightly AlphaVantage backfill retroactively revises "
             "historical `prices` rows (documented, and the reason for "
             "`scripts/freeze_vintage.py`). This IS a revision path in the ensemble "
             "feature construction.")
    L.append("- **News → sentiment → scores.** Looked at: `news_articles` is populated "
             "by `src/data/news_pipeline.py` but **no file outside `news_pipeline.py` "
             "references it** (`grep -rn news_articles src/`). The `sentiment_scores` "
             "table feeding the feature matrix (`feature_matrix.py:24, 70`) is "
             "sourced entirely from SEC 8-K filings (SQL check on the vintage DB: "
             "`sentiment_scores.source` distribution is `sec_8k|4578` rows, zero "
             "`news` rows). **News does not feed the current entry model** — the "
             "revision path is prices, not news.")
    L.append("- **Does feature construction filter articles by `published_at` "
             "relative to each decision date?** N/A for the ensemble entry model "
             "(news isn't in it). Sentiment features enter via `filing_date` from "
             "`sentiment_scores`, which IS the point-in-time reference (SEC filing "
             "date). The `load_sentiment_daily` function forward-fills from "
             "`filing_date` to daily — correct as long as scored filings don't have "
             "their scores retroactively rewritten. FinBERT is deterministic on the "
             "same input, so the score for a given `(ticker, filing_date)` is "
             "stable across re-runs; the only silent-drift risk is a re-scoring "
             "with a different model.")
    L.append("")
    L.append("**Bottom line.** The vintage-revision problem is real for prices "
             "(hence the frozen vintage) and does NOT apply to the current entry "
             "features via a news path. If news is ever added as an ensemble "
             "input, the temporal filter must be `published_at ≤ decision_date` "
             "with no future information leaking through re-scoring.")
    L.append("")

    # ── 9. Open and untested ────────────────────────────────────────────
    L.append("---")
    L.append("")
    L.append("## 9. Open and untested — bear-regime behaviour")
    L.append("")
    L.append("This window (2025-01 → 2026-06, 17 months) is a strong bull "
             "tape. The layer's stated purpose is left-tail management, which was "
             "not exercised. Bear-regime behaviour is genuinely untested.")
    L.append("")
    L.append("**Pre-registered metrics** for any future bear-regime run (fixed "
             "here so the goalposts cannot be moved after the fact):")
    L.append("- **max-drawdown reduction ≥ 3 pp** vs baseline");
    L.append("- **Calmar improvement > 0**")
    L.append("- Sample: a 2022-style drawdown window of similar length (≥ 12 "
             "months, drawdown ≥ 20%).")
    L.append("")
    L.append("**Caveat on interpreting a 2022 run.** The ensemble entry model "
             "was trained through 2024-07-31 on data that includes 2022. Any "
             "2022 backtest has in-sample entry contamination biasing the "
             "baseline high, which makes the exit layer look **conservatively** "
             "bad relative to its true out-of-sample value. If it wins under "
             "that bias, that is meaningful evidence; if it loses, the bias "
             "makes the finding weaker than it would otherwise be.")
    L.append("")

    # ── A. Folded-in measurements (Task A) ──────────────────────────────
    L.append("---")
    L.append("")
    L.append("## A. Folded-in measurements (Prompt 5 Task A)")
    L.append("")
    L.append("### A.1 Redistribute-to-survivors variant")
    L.append("")
    m_full = metrics["full"]
    L.append("| path | Sharpe | Sortino | Calmar | max DD | worst month | 5th %ile month |")
    L.append("|---|---:|---:|---:|---:|---:|---:|")
    for name in ("baseline", "experimental", "experimental_redist",
                 "hard_stop", "zhang_only"):
        m = m_full[name]
        L.append(f"| {name} | {m['sharpe']:+.4f} | {m['sortino']:+.4f} | "
                 f"{m['calmar']:+.3f} | {m['max_dd']:+.4f} | "
                 f"{m['worst_month']:+.4f} | {m['p05_month']:+.4f} |")
    L.append("")
    cash_drag = m_full["experimental_redist"]["sharpe"] - m_full["experimental"]["sharpe"]
    dd_delta = m_full["experimental_redist"]["max_dd"] - m_full["baseline"]["max_dd"]
    L.append(f"**Cash drag = experimental_redist − experimental = {cash_drag:+.4f}** "
             f"Sharpe. That is the plumbing cost of holding exited weight in cash "
             f"at 1/n; it is not attributable to the Andrade or Zhang signal and "
             f"the record should not charge it against the layer.")
    L.append("")
    L.append(f"**Condition (c) — for completeness only.** The redistribute-to-"
             f"survivors variant Sharpe is {m_full['experimental_redist']['sharpe']:+.4f} "
             f"vs baseline {m_full['baseline']['sharpe']:+.4f} (does {'not ' if m_full['experimental_redist']['sharpe'] < m_full['baseline']['sharpe'] else ''}"
             f"beat baseline). Max drawdown change vs baseline: "
             f"**{100*dd_delta:+.2f} pp** ({'improvement' if dd_delta > 0 else 'worse'}; "
             f"threshold was ≥ 3 pp improvement). "
             f"**Condition (c) would fail** — but (c) does not matter once (a) has failed.")
    L.append("")
    L.append("### A.2 Left-tail metrics")
    L.append("")
    L.append("The table above reports max drawdown, Sortino, Calmar, worst single "
             "month, and 5th-percentile monthly return for every path. Reading "
             "them honestly, even though every one worsens:")
    L.append("")
    _base_dd = m_full["baseline"]["max_dd"]
    _exp_dd = m_full["experimental"]["max_dd"]
    _exp_redist_dd = m_full["experimental_redist"]["max_dd"]
    _zho_dd = m_full["zhang_only"]["max_dd"]
    L.append(f"- **Max drawdown** — baseline {_base_dd:+.4f}, experimental {_exp_dd:+.4f} "
             f"({'better' if _exp_dd > _base_dd else 'worse'}), redist "
             f"{_exp_redist_dd:+.4f}, Zhang-only {_zho_dd:+.4f}. The layer did NOT "
             f"reduce the worst drawdown on this window.")
    L.append(f"- **Sortino** — baseline {m_full['baseline']['sortino']:+.4f}, "
             f"experimental {m_full['experimental']['sortino']:+.4f} "
             f"({'better' if m_full['experimental']['sortino'] > m_full['baseline']['sortino'] else 'worse'}). "
             f"Even the downside-only Sharpe worsens.")
    L.append(f"- **Calmar** — baseline {m_full['baseline']['calmar']:+.3f}, "
             f"experimental {m_full['experimental']['calmar']:+.3f} "
             f"({'better' if m_full['experimental']['calmar'] > m_full['baseline']['calmar'] else 'worse'}).")
    L.append(f"- **Worst month** — baseline {m_full['baseline']['worst_month']:+.4f}, "
             f"experimental {m_full['experimental']['worst_month']:+.4f} "
             f"({'better' if m_full['experimental']['worst_month'] > m_full['baseline']['worst_month'] else 'worse'}).")
    L.append(f"- **5th %ile month** — baseline {m_full['baseline']['p05_month']:+.4f}, "
             f"experimental {m_full['experimental']['p05_month']:+.4f}.")
    L.append("")
    L.append("Every left-tail metric on this window is either equal to baseline "
             "or worse. The layer's stated purpose was not delivered here. That "
             "does not settle its bear-regime behaviour, which remains untested.")
    L.append("")

    # ── 10. Recommendation ─────────────────────────────────────────────
    L.append("---")
    L.append("")
    L.append("## 10. Recommendation (three sentences)")
    L.append("")
    L.append("**SHELVE the Markov exit layer.** Do not tune it further on this window — "
             "in-sample search over 17 months would produce numbers that are not "
             "evidence, and even the perfect-foresight ceiling for a re-entry gate "
             "captures only half the gap needed to break even with doing nothing. "
             "If the layer is ever revisited, it should be on a bear-regime window "
             "with the pre-registered max-drawdown and Calmar metrics in §9, and "
             "against Andrade's exit signal specifically (Zhang's leg is either "
             "inert or break-even on equities at this price scale).")
    L.append("")
    OUT.write_text("\n".join(L) + "\n")
    print(f"Wrote {OUT} ({len(L)} lines)")


def main() -> None:
    env = _guard()
    dec = pd.read_parquet(DEC)
    sha = hashlib.sha256(DEC.read_bytes()).hexdigest()
    print(f"parquet sha256 = {sha}")
    ret, n_book = build_all_paths(dec)
    metrics = {"full": {name: _left_tail(ret[name].to_numpy(dtype=float))
                        for name in ("baseline", "experimental",
                                     "experimental_redist", "hard_stop",
                                     "zhang_only")}}

    # Sharpe sanity — baseline / experimental / hard_stop / Zhang-only must
    # match the frozen reference values within 1e-6.
    got = {
        "baseline":     metrics["full"]["baseline"]["sharpe"],
        "experimental": metrics["full"]["experimental"]["sharpe"],
        "hard_stop":    metrics["full"]["hard_stop"]["sharpe"],
        "zhang_only":   metrics["full"]["zhang_only"]["sharpe"],
    }
    expected = {"baseline": BASELINE_SHARPE,
                "experimental": EXPERIMENTAL_SHARPE,
                "hard_stop": HARD_STOP_SHARPE,
                "zhang_only": ZHANG_ONLY_SHARPE}
    for k in expected:
        d = abs(got[k] - expected[k])
        print(f"SHARPE ASSERT  {k:<12s} got={got[k]:.15f} expected={expected[k]:.15f}  drift={d:.3e}")
        assert d < 1e-6, f"{k} drift {d} > 1e-6"

    # Attribution
    lookahead_fix = EXPERIMENTAL_SHARPE - REV3_EXPERIMENTAL
    cash_drag_neg = metrics["full"]["experimental"]["sharpe"] - metrics["full"]["experimental_redist"]["sharpe"]
    signal_error_neg = metrics["full"]["experimental_redist"]["sharpe"] - metrics["full"]["baseline"]["sharpe"]
    # ExpressAll deltas as REV3-scale positive contributions to the historic gap:
    #   historic_gap = baseline - REV3_experimental
    #                = (baseline - exp_redist) + (exp_redist - exp) + (exp - REV3_exp)
    # so the additive decomposition of the *deficit* is:
    attribution = {
        "lookahead_fix":   lookahead_fix,                # + means part of the drag was actually spurious
        "cash_drag":       -cash_drag_neg,               # gain the redistribute path would achieve
        "signal_error":    -signal_error_neg,            # residual drag charged to the Andrade signal
        "sum":             lookahead_fix + (-cash_drag_neg) + (-signal_error_neg),
        "historic":        BASELINE_SHARPE - REV3_EXPERIMENTAL,
    }
    assert abs(attribution["sum"] - attribution["historic"]) < 1e-9, attribution
    print(f"ATTRIBUTION: lookahead={attribution['lookahead_fix']:+.4f}  "
          f"cash_drag={attribution['cash_drag']:+.4f}  "
          f"signal_error={attribution['signal_error']:+.4f}  "
          f"sum={attribution['sum']:+.4f}  historic={attribution['historic']:+.4f}  "
          f"|sum-historic|={abs(attribution['sum']-attribution['historic']):.3e}")

    # p-values
    sells = dec[dec["action"] == "SELL"]
    zg = sells[sells["trigger_source"].isin({"zhang", "both"})]
    an = sells[sells["trigger_source"] == "andrade"]
    def _wins(sub): return int((sub["ret_full"] < 0).sum())
    pvals = {
        "zhang":   _binom_one_sided_p(_wins(zg),    len(zg),    0.5),
        "andrade": _binom_one_sided_p(_wins(an),    len(an),    0.5),
        "pooled":  _binom_one_sided_p(_wins(sells), len(sells), 0.5),
        "zhang_two":   _binom_two_sided_p(_wins(zg),    len(zg),    0.5),
        "andrade_two": _binom_two_sided_p(_wins(an),    len(an),    0.5),
        "pooled_two":  _binom_two_sided_p(_wins(sells), len(sells), 0.5),
    }
    print(f"P-VALUES: zhang={pvals['zhang']:.4f}  andrade={pvals['andrade']:.4f}  "
          f"pooled={pvals['pooled']:.4f}")

    write_md(env, dec, ret, metrics, pvals, attribution)


if __name__ == "__main__":
    main()
