"""
TimesFM Momentum Factor
=======================
Runs Google's TimesFM foundation model over per-ticker daily adj_close
histories and produces a monthly "expected 1-month return" signal that
augments the existing quant factor stack.

Design notes
------------
* Installed package: timesfm 2.0.2 (ships TimesFM 2.5 architecture).
  The 2.x API is `TimesFM_2p5_200M_torch.from_pretrained(...)` +
  `.compile(ForecastConfig(...))` + `.forecast(horizon, inputs)`.
  We do NOT use the older 1.x `TimesFm(hparams=..., checkpoint=...)`
  constructor because it does not exist in this package.
* Model weights (~800 MB) are downloaded once from HuggingFace on first
  instantiation and cached under ~/.cache/huggingface.
* Backend forced to CPU + torch threads=2 to stay inside the ~2 GB RAM
  budget on this machine.
* Batching: for each month-end we forecast ALL eligible tickers in one
  `.forecast()` call. This turns ~6,407 (ticker × month) pairs into
  ~120 batched calls (one per month).
* Manual normalisation: `series / series.iloc[0]` puts each ticker on a
  common scale near 1.0. We therefore set `normalize_inputs=False` on
  the ForecastConfig to avoid double-normalising.

Public entry point
------------------
    compute_timesfm_predictions(price_df, month_ends) -> DataFrame
        columns = ["ticker", "date", "timesfm_pred_return_1m"]
"""

from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Module-level constants ────────────────────────────────────────────────────
# NOTE: intentionally NOT in settings.yaml — these are algorithm parameters
# specific to this factor and should be edited here.

CONTEXT_LEN = 252   # trading days of adj_close history per inference
HORIZON_LEN = 21    # forecast 21 days ahead (~1 calendar month)
MIN_CONTEXT = 63    # minimum non-NaN days required; below this → NaN

# TimesFM 2.5 repo. The 1.0 checkpoint referenced in earlier design docs
# will NOT load into the installed TimesFM_2p5_200M_torch class.
HF_REPO = "google/timesfm-2.5-200m-pytorch"

# TimesFM 2.5's default quantile list; median is at index 4.
_TIMESFM_QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
_MEDIAN_IDX = _TIMESFM_QUANTILES.index(0.5)

# Small forecast batch size keeps peak RAM predictable on 8 GB machines.
# TimesFM pads the input list up to a multiple of this value internally.
_PER_CORE_BATCH_SIZE = 8

_OUTPUT_COL = "timesfm_pred_return_1m"


# ── Model loader (cached) ─────────────────────────────────────────────────────

_MODEL = None  # process-level cache — avoid reloading 800 MB per call


def _load_model():
    """Load and compile TimesFM once per process.

    Raises with a clear message on missing package, HuggingFace auth/download
    failures, or ForecastConfig mismatches. Returns the compiled model.
    """
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    try:
        import timesfm
    except ImportError as e:
        raise RuntimeError(
            "timesfm is not installed. Run "
            "`.venv/bin/pip install timesfm` from the project root."
        ) from e

    try:
        import torch
        torch.set_num_threads(2)  # RAM-conservative on 8 GB Mac
    except ImportError as e:
        raise RuntimeError(
            "torch not available — timesfm 2.5 requires PyTorch."
        ) from e

    logger.info("Loading TimesFM checkpoint from %s …", HF_REPO)
    try:
        model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(HF_REPO)
    except Exception as e:
        msg = str(e).lower()
        if "401" in msg or "403" in msg or "auth" in msg or "gated" in msg:
            raise RuntimeError(
                f"HuggingFace authentication error loading {HF_REPO}. "
                "Run `huggingface-cli login` if the repo is gated."
            ) from e
        if "not found" in msg or "404" in msg:
            raise RuntimeError(
                f"HuggingFace repo {HF_REPO} not found. "
                "Check network connectivity and repo id."
            ) from e
        raise RuntimeError(
            f"Failed to load TimesFM checkpoint from {HF_REPO}: {e}"
        ) from e

    logger.info(
        "Compiling TimesFM (context=%d, horizon=%d, batch=%d, backend=cpu) …",
        CONTEXT_LEN, HORIZON_LEN, _PER_CORE_BATCH_SIZE,
    )
    try:
        model.compile(
            timesfm.ForecastConfig(
                max_context=CONTEXT_LEN,
                max_horizon=HORIZON_LEN,
                normalize_inputs=False,      # we normalise manually
                per_core_batch_size=_PER_CORE_BATCH_SIZE,
                use_continuous_quantile_head=False,
                force_flip_invariance=True,
                infer_is_positive=True,
                fix_quantile_crossing=True,
            )
        )
    except TypeError as e:
        raise RuntimeError(
            f"TimesFM ForecastConfig API mismatch (installed timesfm may be "
            f"a different version than expected): {e}"
        ) from e

    _MODEL = model
    return _MODEL


# ── Public API ────────────────────────────────────────────────────────────────

def compute_timesfm_predictions(
    price_df: pd.DataFrame,
    month_ends: Iterable[pd.Timestamp],
) -> pd.DataFrame:
    """Compute one-month-ahead TimesFM return forecasts.

    Parameters
    ----------
    price_df : pd.DataFrame
        Wide format — DatetimeIndex × ticker columns, values = daily
        adj_close prices (the same format `_load_prices_wide` returns
        inside factor_export_quant.py).
    month_ends : list or DatetimeIndex
        Month-end dates at which to forecast.

    Returns
    -------
    pd.DataFrame
        Long format with columns ["ticker", "date", "timesfm_pred_return_1m"],
        including every (ticker, month_end) pair — NaN where the ticker had
        insufficient history or the batch call failed.
    """
    if price_df is None or price_df.empty:
        raise ValueError("compute_timesfm_predictions: price_df is empty")

    if not isinstance(price_df.index, pd.DatetimeIndex):
        price_df = price_df.copy()
        price_df.index = pd.to_datetime(price_df.index)

    month_ends = pd.DatetimeIndex(pd.to_datetime(list(month_ends)))
    tickers = list(price_df.columns)

    model = _load_model()  # raises on failure — fail loud, not silent NaN

    logger.info(
        "TimesFM: %d tickers × %d month-ends (%d potential inferences)",
        len(tickers), len(month_ends), len(tickers) * len(month_ends),
    )

    records: list[dict] = []

    for m_end in month_ends:
        history_slice = price_df.loc[:m_end]
        if history_slice.empty:
            for tkr in tickers:
                records.append({"ticker": tkr, "date": m_end,
                                _OUTPUT_COL: np.nan})
            continue

        # (a) + (b) + (c): collect eligible normalised series
        # Store (ticker, normalised_array, last_normalised_value) so we can
        # denormalise the forecast directly with a single ratio.
        eligible: list[tuple[str, np.ndarray, float]] = []
        skipped: list[str] = []
        for tkr in tickers:
            series = history_slice[tkr].dropna()
            if len(series) < MIN_CONTEXT:
                skipped.append(tkr)
                continue
            series = series.iloc[-CONTEXT_LEN:]
            first = float(series.iloc[0])
            if not np.isfinite(first) or first <= 0.0:
                skipped.append(tkr)
                continue
            normalised = (series.values / first).astype(np.float32)
            last_normalised = float(normalised[-1])
            if not np.isfinite(last_normalised) or last_normalised <= 0.0:
                skipped.append(tkr)
                continue
            eligible.append((tkr, normalised, last_normalised))

        # Skipped tickers get NaN
        for tkr in skipped:
            records.append({"ticker": tkr, "date": m_end,
                            _OUTPUT_COL: np.nan})

        if not eligible:
            logger.warning(
                "TimesFM %s: no eligible tickers (all under MIN_CONTEXT=%d)",
                m_end.date(), MIN_CONTEXT,
            )
            continue

        # (d): single batched forecast for this month-end
        inputs = [arr for _, arr, _ in eligible]
        try:
            _point, quantile_preds = model.forecast(
                horizon=HORIZON_LEN, inputs=inputs,
            )
        except Exception as e:
            logger.warning(
                "TimesFM forecast failed at %s (%d tickers): %s — "
                "returning NaN for this month.",
                m_end.date(), len(eligible), e,
            )
            for tkr, _, _ in eligible:
                records.append({"ticker": tkr, "date": m_end,
                                _OUTPUT_COL: np.nan})
            continue

        # quantile_preds shape: (N, HORIZON_LEN, len(_TIMESFM_QUANTILES))
        # (e) + (f): median at last horizon step → 1-month return.
        # Since we normalised by dividing by the first price, both the
        # model output and last_normalised live in the same scale, so:
        #   pred_return = pred_normalised / last_normalised - 1
        for i, (tkr, _norm, last_normalised) in enumerate(eligible):
            try:
                pred_normalised = float(
                    quantile_preds[i, HORIZON_LEN - 1, _MEDIAN_IDX]
                )
            except (IndexError, TypeError, ValueError) as e:
                logger.warning(
                    "TimesFM %s %s: could not read median forecast (%s) — NaN",
                    m_end.date(), tkr, e,
                )
                records.append({"ticker": tkr, "date": m_end,
                                _OUTPUT_COL: np.nan})
                continue

            pred_return = pred_normalised / last_normalised - 1.0
            if not np.isfinite(pred_return):
                pred_return = np.nan
            records.append({"ticker": tkr, "date": m_end,
                            _OUTPUT_COL: float(pred_return)})

        logger.info(
            "TimesFM %s: forecast OK (%d eligible, %d skipped)",
            m_end.date(), len(eligible), len(skipped),
        )

    out = pd.DataFrame.from_records(records)
    if out.empty:
        return pd.DataFrame(columns=["ticker", "date", _OUTPUT_COL])

    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values(["date", "ticker"]).reset_index(drop=True)
    return out
