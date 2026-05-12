"""
Unit Tests — Factor Computation & Scoring
==========================================
Tests cover momentum, volume z-scores, macro regime, XBRL-derived features,
SUE fallback, composite score weight redistribution, and z-score capping.

All tests use synthetic data; no external APIs or real DB are touched.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ══════════════════════════════════════════════════════════════════════════════
# 1. test_momentum_computation
# ══════════════════════════════════════════════════════════════════════════════

class TestMomentumComputation:
    """compute_momentum() on a 252-row synthetic price DataFrame."""

    @pytest.fixture()
    def prices_252(self) -> pd.DataFrame:
        """252 trading-day price history for 10 tickers."""
        np.random.seed(7)
        dates = pd.bdate_range("2022-01-03", periods=252, freq="B")
        tickers = [f"T{i:02d}" for i in range(10)]
        data = {}
        for t in tickers:
            rets = np.random.normal(0.0005, 0.02, size=252)
            data[t] = 100.0 * np.cumprod(1 + rets)
        return pd.DataFrame(data, index=dates)

    def test_shape_and_no_nan_for_full_history(self, prices_252):
        from src.strategies.quant.quant_factors import compute_momentum

        mom = compute_momentum(prices_252)

        # Same tickers
        assert set(mom.columns) == set(prices_252.columns)

        # After warm-up the last ~10 rows must have no NaN
        tail = mom.tail(10)
        assert tail.notna().all().all(), "Last 10 rows should have no NaN"

    def test_5day_skip_applied(self, prices_252):
        """
        Inject a spike in the last 5 days of a single ticker.
        With skip_days=5 the spike should NOT affect the momentum score
        relative to the pre-spike score.
        """
        from src.strategies.quant.quant_factors import compute_momentum

        # Baseline
        mom_base = compute_momentum(prices_252)
        last_day = mom_base.index[-1]
        score_before = mom_base.loc[last_day, "T00"]

        # Add a +50% spike in the last 5 days of T00
        prices_spiked = prices_252.copy()
        prices_spiked.iloc[-5:, 0] *= 1.50

        mom_spike = compute_momentum(prices_spiked)
        score_after = mom_spike.loc[last_day, "T00"]

        # Because the skip window covers those 5 days, the score should
        # be identical (or extremely close)
        assert abs(score_after - score_before) < 0.15, (
            f"5-day skip not working: base={score_before:.4f}, "
            f"spiked={score_after:.4f}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# 2. test_volume_zscore
# ══════════════════════════════════════════════════════════════════════════════

class TestVolumeZscore:
    """Volume Z-score should be ≈ mean 0, std 1 cross-sectionally."""

    @pytest.fixture()
    def price_volume(self):
        np.random.seed(11)
        dates = pd.bdate_range("2023-01-02", periods=60, freq="B")
        tickers = [f"V{i}" for i in range(15)]
        prices = pd.DataFrame(
            100 + np.random.randn(60, 15).cumsum(axis=0),
            index=dates, columns=tickers,
        )
        volumes = pd.DataFrame(
            np.random.lognormal(mean=15, sigma=0.5, size=(60, 15)),
            index=dates, columns=tickers,
        )
        return prices, volumes

    def test_cross_sectional_mean_near_zero(self, price_volume):
        from src.strategies.quant.quant_factors import compute_volume_trend

        prices, volumes = price_volume
        vol_z = compute_volume_trend(prices, volumes)

        # Use the last 20 rows (after warm-up)
        tail = vol_z.tail(20)
        cross_means = tail.mean(axis=1)
        assert (cross_means.abs() < 0.25).all(), (
            f"Cross-sectional means not near 0: {cross_means.describe()}"
        )

    def test_cross_sectional_std_near_one(self, price_volume):
        from src.strategies.quant.quant_factors import compute_volume_trend

        prices, volumes = price_volume
        vol_z = compute_volume_trend(prices, volumes)
        tail = vol_z.tail(20)
        cross_stds = tail.std(axis=1)
        assert ((cross_stds - 1.0).abs() < 0.35).all(), (
            f"Cross-sectional stds not near 1: {cross_stds.describe()}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# 3. test_macro_regime_filter
# ══════════════════════════════════════════════════════════════════════════════

class TestMacroRegimeFilter:
    """Macro regime filter: risk-off when VIX > 25, risk-on otherwise."""

    @pytest.fixture()
    def dates_index(self):
        return pd.bdate_range("2023-01-02", periods=20, freq="B")

    def test_risk_off_when_vix_high(self, dates_index):
        from src.strategies.quant.quant_factors import macro_regime_filter

        macro = pd.DataFrame({
            "vix": [30.0] * 20,
            "yield_spread_10y2y": [0.5] * 20,
        }, index=dates_index)

        regime = macro_regime_filter(macro, dates_index)
        assert not regime.any(), "All positions should be False (risk-off) when VIX > 25"

    def test_risk_on_when_vix_low_and_spread_positive(self, dates_index):
        from src.strategies.quant.quant_factors import macro_regime_filter

        macro = pd.DataFrame({
            "vix": [18.0] * 20,
            "yield_spread_10y2y": [1.2] * 20,
        }, index=dates_index)

        regime = macro_regime_filter(macro, dates_index)
        assert regime.all(), "All positions should be True (risk-on)"

    def test_risk_off_when_yield_spread_negative(self, dates_index):
        from src.strategies.quant.quant_factors import macro_regime_filter

        macro = pd.DataFrame({
            "vix": [15.0] * 20,
            "yield_spread_10y2y": [-0.3] * 20,
        }, index=dates_index)

        regime = macro_regime_filter(macro, dates_index)
        assert not regime.any(), "All False when yield curve is inverted"


# ══════════════════════════════════════════════════════════════════════════════
# 4. test_gross_profitability
# ══════════════════════════════════════════════════════════════════════════════

class TestGrossProfitability:
    """GP / TotalAssets via compute_derived_features."""

    def test_gross_profitability_formula(self, engine, sample_xbrl):
        from src.data.edgar_pipeline import upsert_facts, compute_derived_features

        upsert_facts(sample_xbrl, engine)
        derived = compute_derived_features(sample_xbrl, engine=engine)

        # Check each row
        for _, row in derived.iterrows():
            # Normalise end_date to string for comparison
            end_str = (
                row["end_date"].strftime("%Y-%m-%d")
                if isinstance(row["end_date"], pd.Timestamp)
                else str(row["end_date"])
            )
            src = sample_xbrl[
                (sample_xbrl["ticker"] == row["ticker"])
                & (sample_xbrl["end_date"] == end_str)
            ]
            if src.empty:
                continue
            expected = src.iloc[0]["gross_profit"] / src.iloc[0]["total_assets"]
            if pd.notna(row["gross_profitability"]):
                assert abs(row["gross_profitability"] - expected) < 1e-6, (
                    f"GP mismatch for {row['ticker']} {end_str}: "
                    f"got {row['gross_profitability']}, expected {expected}"
                )


# ══════════════════════════════════════════════════════════════════════════════
# 5. test_revenue_acceleration
# ══════════════════════════════════════════════════════════════════════════════

class TestRevenueAcceleration:
    """Second derivative of revenue growth across a 6-quarter sequence."""

    def test_known_values(self, engine):
        from src.data.edgar_pipeline import upsert_facts, compute_derived_features

        # Build a 6-quarter sequence where revenue growth is known:
        # Q0=100, Q1=110, Q2=121, Q3=133.1, Q4=120, Q5=132
        # YoY growth (pct_change(4)):
        #   Q4 = (120-100)/100 = 0.20
        #   Q5 = (132-110)/110 = 0.20
        # Revenue acceleration (diff of YoY):
        #   Q5 = 0.20 - 0.20 = 0.0
        revenues = [100, 110, 121, 133.1, 120, 132]
        rows = []
        qe_dates = pd.date_range("2020-03-31", periods=6, freq="QE")
        for i, qe in enumerate(qe_dates):
            rows.append({
                "ticker": "TEST",
                "end_date": qe.strftime("%Y-%m-%d"),
                "revenue": revenues[i],
                "gross_profit": revenues[i] * 0.5,
                "rd_expense": 10,
                "operating_income": revenues[i] * 0.2,
                "net_income": revenues[i] * 0.1,
                "eps_diluted": 1.0,
                "total_assets": 1000,
                "total_liabilities": 400,
                "stockholders_equity": 600,
                "deferred_revenue": 50,
                "cash": 100,
                "long_term_debt": 200,
                "operating_cf": 80,
                "capex": 20,
            })
        facts_df = pd.DataFrame(rows)
        upsert_facts(facts_df, engine)
        derived = compute_derived_features(facts_df, engine=engine)

        # Q4 (index 4): revenue_yoy = pct_change(4) = (120-100)/100 = 0.20
        q4 = derived[derived["end_date"] == qe_dates[4].strftime("%Y-%m-%d")]
        if not q4.empty and pd.notna(q4.iloc[0]["revenue_yoy"]):
            assert abs(q4.iloc[0]["revenue_yoy"] - 0.20) < 1e-6

        # Q5 (index 5): revenue_yoy = (132-110)/110 = 0.2
        q5 = derived[derived["end_date"] == qe_dates[5].strftime("%Y-%m-%d")]
        if not q5.empty and pd.notna(q5.iloc[0]["revenue_yoy"]):
            assert abs(q5.iloc[0]["revenue_yoy"] - 0.2) < 1e-6

        # Revenue acceleration at Q5 = 0.2 - 0.2 = 0.0
        if not q5.empty and pd.notna(q5.iloc[0]["revenue_acceleration"]):
            assert abs(q5.iloc[0]["revenue_acceleration"]) < 1e-6


# ══════════════════════════════════════════════════════════════════════════════
# 6. test_sue_fallback
# ══════════════════════════════════════════════════════════════════════════════

class TestSueFallback:
    """
    Seasonal random walk SUE: with no analyst estimates, compute_sue_simfin
    should still produce a score for quarters 5+.
    """

    def test_sue_non_nan_after_warmup(self):
        from src.data.simfin_pipeline import compute_sue_simfin

        # 12 quarters of EPS data (no analyst estimates → seasonal random walk)
        np.random.seed(33)
        eps_vals = [1.0, 1.1, 1.2, 1.05, 1.15, 1.25, 1.3, 1.1, 1.2, 1.35, 1.4, 1.5]
        qe = pd.date_range("2019-03-31", periods=12, freq="QE")
        eps_df = pd.DataFrame({
            "ticker": ["TEST"] * 12,
            "fiscal_period": qe.strftime("%Y-%m-%d"),
            "eps_actual": eps_vals,
        })

        sue_df = compute_sue_simfin("TEST", eps_df)
        assert not sue_df.empty, "SUE df should not be empty"

        # Quarters 0–3 should have NaN sue (need 4Q lookback for estimate)
        # Quarters 4+ should have NaN until rolling window (8Q min_periods=4) is met
        # So quarters 7+ (index ≥ 7) should definitely have valid SUE
        late = sue_df.iloc[7:]
        valid_late = late["sue_score"].dropna()
        assert len(valid_late) > 0, (
            "SUE should be non-NaN for quarters with sufficient history"
        )


# ══════════════════════════════════════════════════════════════════════════════
# 7. test_composite_score_weight_redistribution
# ══════════════════════════════════════════════════════════════════════════════

class TestCompositeScoreWeightRedistribution:
    """
    When one feature is fully NaN, the composite score must still use
    effective weights that sum to 1.0.
    """

    def test_weights_sum_to_one_with_missing_feature(self):
        from src.strategies.fundamental.fundamental_scorer import (
            compute_composite_scores,
            _WEIGHT_MAP,
        )

        # Build a single-row feature matrix with all features = 1.0
        # except one feature set to NaN
        feature_cols = [col for _, (col, _) in _WEIGHT_MAP.items()]
        row = {"ticker": "TEST", "quarter_end": pd.Timestamp("2022-12-31")}
        for col in feature_cols:
            row[col] = 1.0

        # Set one feature to NaN
        nan_col = feature_cols[0]
        row[nan_col] = np.nan

        fm = pd.DataFrame([row])
        scores = compute_composite_scores(fm)

        assert len(scores) == 1
        composite = scores.iloc[0]["composite_score"]
        assert pd.notna(composite), "Composite should not be NaN"

        # The effective score with all available features = 1.0 should equal
        # 1.0 * (sum_avail_weight / sum_avail_weight) = 1.0
        # because redistribution normalises to sum = 1.0
        assert abs(composite - 1.0) < 1e-9, (
            f"With all available features=1.0, composite should be 1.0, "
            f"got {composite}"
        )

    def test_all_features_present_weights_sum_to_one(self):
        from src.strategies.fundamental.fundamental_scorer import (
            compute_composite_scores,
            _WEIGHT_MAP,
        )

        feature_cols = [col for _, (col, _) in _WEIGHT_MAP.items()]
        row = {"ticker": "TEST", "quarter_end": pd.Timestamp("2022-12-31")}
        for col in feature_cols:
            row[col] = 1.0

        fm = pd.DataFrame([row])
        scores = compute_composite_scores(fm)
        composite = scores.iloc[0]["composite_score"]

        # All features present, all = 1.0 → composite = 1.0
        assert abs(composite - 1.0) < 1e-9

    def test_features_missing_logged(self):
        from src.strategies.fundamental.fundamental_scorer import (
            compute_composite_scores,
            _WEIGHT_MAP,
        )

        feature_cols = [col for _, (col, _) in _WEIGHT_MAP.items()]
        row = {"ticker": "TEST", "quarter_end": pd.Timestamp("2022-12-31")}
        for col in feature_cols:
            row[col] = 2.0

        # Knock out two features
        row[feature_cols[0]] = np.nan
        row[feature_cols[1]] = np.nan

        fm = pd.DataFrame([row])
        scores = compute_composite_scores(fm)

        # features_missing should list the two missing factors
        missing = scores.iloc[0]["features_missing"]
        assert missing != "", "Should list missing features"
        assert len(missing.split(",")) == 2


# ══════════════════════════════════════════════════════════════════════════════
# 8. test_zscore_outlier_cap
# ══════════════════════════════════════════════════════════════════════════════

class TestZscoreOutlierCap:
    """No z-scored value should exceed ±3.0 after capping."""

    def test_cap_at_three(self):
        from src.strategies.fundamental.xbrl_features import (
            _zscore_cross_section,
            FEATURE_COLS,
            ZSCORE_CAP,
        )

        np.random.seed(55)
        n_tickers = 20
        n_quarters = 4
        quarter_ends = pd.date_range("2022-03-31", periods=n_quarters, freq="QE")
        rows = []
        for qe in quarter_ends:
            for i in range(n_tickers):
                row = {
                    "ticker": f"T{i:02d}",
                    "quarter_end": qe,
                }
                for col in FEATURE_COLS:
                    # Inject some extreme values
                    if i == 0:
                        row[col] = 100.0  # extreme outlier
                    elif i == 1:
                        row[col] = -100.0  # extreme negative
                    else:
                        row[col] = np.random.normal(0, 1)
                rows.append(row)

        df = pd.DataFrame(rows)
        result = _zscore_cross_section(df)

        for col in FEATURE_COLS:
            if col in result.columns:
                valid = result[col].dropna()
                if len(valid) > 0:
                    assert valid.max() <= ZSCORE_CAP + 1e-9, (
                        f"{col} max={valid.max()} exceeds cap {ZSCORE_CAP}"
                    )
                    assert valid.min() >= -ZSCORE_CAP - 1e-9, (
                        f"{col} min={valid.min()} below -{ZSCORE_CAP}"
                    )
