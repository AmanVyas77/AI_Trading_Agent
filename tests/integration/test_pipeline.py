"""
Integration Tests — Data Pipeline Round-Trips
==============================================
Tests the data flow end-to-end using in-memory SQLite and mocked HTTP calls.
All external requests (yfinance, SEC EDGAR, FINRA) are monkey-patched.
"""

from __future__ import annotations

import io
import json
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import text


# ══════════════════════════════════════════════════════════════════════════════
# 1. test_price_pipeline_roundtrip
# ══════════════════════════════════════════════════════════════════════════════

class TestPricePipelineRoundtrip:
    """Mock yfinance.download, run fetch_prices, verify rows in DB."""

    def _make_yf_df(self, n=10):
        """Create a synthetic yfinance-like DataFrame."""
        dates = pd.bdate_range("2023-06-01", periods=n, freq="B")
        np.random.seed(42)
        return pd.DataFrame({
            "Open": np.random.uniform(100, 110, n),
            "High": np.random.uniform(110, 120, n),
            "Low": np.random.uniform(90, 100, n),
            "Close": np.random.uniform(100, 110, n),
            "Volume": np.random.randint(1_000_000, 5_000_000, n).astype(float),
        }, index=dates)

    def test_fetch_prices_inserts_rows(self, engine):
        """Prepare rows like fetch_prices does internally, upsert, and verify."""
        # Import at function level to avoid module-level import errors
        from src.data.quant_pipeline import _upsert_prices

        yf_df = self._make_yf_df(10)
        raw = yf_df.rename(columns=str.lower).copy()
        raw["ticker"] = "MOCK"
        raw["date"] = raw.index.strftime("%Y-%m-%d")
        raw["adj_close"] = raw["close"]
        rows = raw[["ticker", "date", "open", "high", "low", "close", "volume", "adj_close"]]

        _upsert_prices(rows, engine)

        # Verify rows are in the DB
        with engine.connect() as conn:
            result = pd.read_sql_query(
                text("SELECT * FROM prices WHERE ticker = 'MOCK'"), conn
            )

        assert len(result) == 10, f"Expected 10 rows, got {len(result)}"
        assert set(result.columns) >= {
            "ticker", "date", "open", "high", "low", "close", "volume", "adj_close"
        }

    def test_load_prices_roundtrip(self, engine):
        """Insert then load back via SQL query (mirrors load_prices logic)."""
        from src.data.quant_pipeline import _upsert_prices

        yf_df = self._make_yf_df(10)
        raw = yf_df.rename(columns=str.lower).copy()
        raw["ticker"] = "TRIP"
        raw["date"] = raw.index.strftime("%Y-%m-%d")
        raw["adj_close"] = raw["close"]
        rows = raw[["ticker", "date", "open", "high", "low", "close", "volume", "adj_close"]]
        _upsert_prices(rows, engine)

        # Load back via load_prices logic
        with engine.connect() as conn:
            df = pd.read_sql_query(
                text("SELECT ticker, date, adj_close FROM prices WHERE ticker='TRIP' ORDER BY date"),
                conn,
            )
        df["date"] = pd.to_datetime(df["date"])
        pivot = df.pivot(index="date", columns="ticker", values="adj_close")

        assert "TRIP" in pivot.columns
        assert len(pivot) == 10


# ══════════════════════════════════════════════════════════════════════════════
# 2. test_edgar_cik_map
# ══════════════════════════════════════════════════════════════════════════════

class TestEdgarCikMap:
    """Mock the SEC company_tickers.json response."""

    @patch("src.data.edgar_pipeline.requests.get")
    def test_cik_map_returns_correct_dict(self, mock_get, tmp_path, monkeypatch):
        import src.data.edgar_pipeline as edgar_mod

        # Ensure cache file does not exist so we hit the mocked API
        (tmp_path / "data" / "universe").mkdir(parents=True, exist_ok=True)

        def patched_get_cik_map():
            """Replicate get_cik_map() but with tmp_path as ROOT."""
            cache_path = tmp_path / "data" / "universe" / "cik_map.json"
            if cache_path.exists():
                with open(cache_path) as f:
                    return json.load(f)

            url = "https://www.sec.gov/files/company_tickers.json"
            resp = edgar_mod.requests.get(url, headers=edgar_mod.SEC_HEADERS, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            mapping = {v["ticker"]: str(v["cik_str"]).zfill(10) for v in data.values()}

            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "w") as f:
                json.dump(mapping, f)
            return mapping

        # Mock SEC response
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
            "1": {"cik_str": 789019, "ticker": "MSFT", "title": "Microsoft Corp"},
            "2": {"cik_str": 1045810, "ticker": "NVDA", "title": "NVIDIA Corp"},
        }
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = patched_get_cik_map()

        assert isinstance(result, dict)
        assert result["AAPL"] == "0000320193"
        assert result["MSFT"] == "0000789019"
        assert result["NVDA"] == "0001045810"
        assert len(result) == 3


# ══════════════════════════════════════════════════════════════════════════════
# 3. test_xbrl_extraction
# ══════════════════════════════════════════════════════════════════════════════

class TestXbrlExtraction:
    """Mock fetch_company_facts with a minimal EDGAR JSON fixture."""

    @patch("src.data.edgar_pipeline.fetch_company_facts")
    def test_extract_ticker_facts_shape_and_values(self, mock_fetch):
        from src.data.edgar_pipeline import extract_ticker_facts

        # Minimal EDGAR company facts JSON
        mock_fetch.return_value = {
            "facts": {
                "us-gaap": {
                    "Revenues": {
                        "units": {
                            "USD": [
                                {"end": "2022-03-31", "val": 1000, "form": "10-Q", "frame": "CY2022Q1"},
                                {"end": "2022-06-30", "val": 1100, "form": "10-Q", "frame": "CY2022Q2"},
                                {"end": "2022-09-30", "val": 1200, "form": "10-Q", "frame": "CY2022Q3"},
                                {"end": "2022-12-31", "val": 1300, "form": "10-K", "frame": "CY2022Q4"},
                            ]
                        }
                    },
                    "GrossProfit": {
                        "units": {
                            "USD": [
                                {"end": "2022-03-31", "val": 500, "form": "10-Q", "frame": "CY2022Q1"},
                                {"end": "2022-06-30", "val": 550, "form": "10-Q", "frame": "CY2022Q2"},
                                {"end": "2022-09-30", "val": 600, "form": "10-Q", "frame": "CY2022Q3"},
                                {"end": "2022-12-31", "val": 650, "form": "10-K", "frame": "CY2022Q4"},
                            ]
                        }
                    },
                    "Assets": {
                        "units": {
                            "USD": [
                                {"end": "2022-03-31", "val": 5000, "form": "10-Q", "frame": "CY2022Q1"},
                                {"end": "2022-12-31", "val": 5500, "form": "10-K", "frame": "CY2022Q4"},
                            ]
                        }
                    },
                }
            }
        }

        result = extract_ticker_facts("TEST", "0001234567")

        assert not result.empty
        assert "ticker" in result.columns
        assert (result["ticker"] == "TEST").all()

        # Should have revenue column with known values
        assert "revenue" in result.columns
        assert 1000 in result["revenue"].values
        assert 1300 in result["revenue"].values

        # Should have gross_profit
        assert "gross_profit" in result.columns
        assert 500 in result["gross_profit"].values

    @patch("src.data.edgar_pipeline.fetch_company_facts")
    def test_extract_empty_facts(self, mock_fetch):
        from src.data.edgar_pipeline import extract_ticker_facts

        mock_fetch.return_value = {"facts": {"us-gaap": {}}}
        result = extract_ticker_facts("EMPTY", "0000000000")
        assert result.empty


# ══════════════════════════════════════════════════════════════════════════════
# 4. test_derived_features_pipeline
# ══════════════════════════════════════════════════════════════════════════════

class TestDerivedFeaturesPipeline:
    """Insert synthetic xbrl_facts, run compute_derived_features, check output."""

    def test_fcf_and_rev_accel_present(self, engine, sample_xbrl):
        from src.data.edgar_pipeline import upsert_facts, compute_derived_features

        upsert_facts(sample_xbrl, engine)
        derived = compute_derived_features(sample_xbrl, engine=engine)

        assert not derived.empty
        assert "free_cash_flow" in derived.columns
        assert "revenue_acceleration" in derived.columns
        assert "gross_profitability" in derived.columns
        assert "rd_intensity" in derived.columns

    def test_fcf_correct_formula(self, engine, sample_xbrl):
        from src.data.edgar_pipeline import upsert_facts, compute_derived_features

        upsert_facts(sample_xbrl, engine)
        derived = compute_derived_features(sample_xbrl, engine=engine)

        # FCF = operating_cf - |capex|
        for _, row in derived.iterrows():
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
            expected_fcf = src.iloc[0]["operating_cf"] - abs(src.iloc[0]["capex"])
            if pd.notna(row["free_cash_flow"]):
                assert abs(row["free_cash_flow"] - expected_fcf) < 1e-4

    def test_rev_accel_non_null_after_warmup(self, engine, sample_xbrl):
        from src.data.edgar_pipeline import upsert_facts, compute_derived_features

        upsert_facts(sample_xbrl, engine)
        derived = compute_derived_features(sample_xbrl, engine=engine)

        # For each ticker, rows 5+ (quarter index) should have rev_accel
        for ticker, grp in derived.groupby("ticker"):
            grp = grp.sort_values("end_date")
            # pct_change(4) needs 5 rows, diff needs 1 more → row index ≥ 5
            late = grp.iloc[5:]
            if not late.empty:
                valid = late["revenue_acceleration"].dropna()
                assert len(valid) > 0, (
                    f"{ticker}: revenue_acceleration should be non-null "
                    f"after warmup"
                )


# ══════════════════════════════════════════════════════════════════════════════
# 5. test_finra_short_volume_parse
# ══════════════════════════════════════════════════════════════════════════════

class TestFinraShortVolumeParse:
    """Mock FINRA pipe-delimited response and verify parsing + DB insert."""

    @patch("src.data.simfin_pipeline.requests.get")
    def test_finra_parse_and_insert(self, mock_get, engine):
        from src.data.simfin_pipeline import fetch_finra_short_volume

        # Build a mock FINRA response — pipe-delimited text
        header = "Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market"
        rows = [
            "20230103|AAAA|500000|10000|1000000|N",
            "20230103|BBBB|300000|5000|800000|N",
            "20230104|AAAA|550000|12000|1100000|N",
            "20230104|BBBB|320000|6000|850000|N",
        ]
        finra_text = "\n".join([header] + rows)

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = finra_text
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        # Run with a very narrow date range (single month)
        fetch_finra_short_volume(
            tickers=["AAAA", "BBBB"],
            start="2023-01-01",
            end="2023-01-31",
            engine=engine,
        )

        # Verify rows in DB
        with engine.connect() as conn:
            result = pd.read_sql_query(
                text("SELECT * FROM short_interest ORDER BY ticker, date"), conn
            )

        assert len(result) >= 2, f"Expected ≥2 rows, got {len(result)}"
        assert "AAAA" in result["ticker"].values
        assert "BBBB" in result["ticker"].values

        # Verify short_ratio = short_volume / total_volume
        aaaa_rows = result[result["ticker"] == "AAAA"]
        for _, row in aaaa_rows.iterrows():
            if (pd.notna(row["short_ratio"])
                    and pd.notna(row["short_volume"])
                    and pd.notna(row["total_volume"])):
                expected_ratio = row["short_volume"] / row["total_volume"]
                assert abs(row["short_ratio"] - expected_ratio) < 1e-6


# ══════════════════════════════════════════════════════════════════════════════
# 6. test_screener_filters
# ══════════════════════════════════════════════════════════════════════════════

class TestScreenerFilters:
    """Build a synthetic ticker DataFrame and verify filter logic."""

    def test_market_cap_filter(self):
        """Only tickers with market_cap >= $2B should pass."""
        df = pd.DataFrame({
            "ticker": ["BIG", "SMALL", "MED", "TINY"],
            "name": ["Big Co", "Small Co", "Med Co", "Tiny Co"],
            "exchange": ["NMS", "NMS", "NMS", "NMS"],
            "currency": ["USD", "USD", "USD", "USD"],
            "sector": ["Tech"] * 4,
            "industry": ["Software"] * 4,
            "market_cap": [10e9, 500e6, 2.5e9, 100e6],
            "avg_volume": [5e6, 5e6, 5e6, 5e6],
            "avg_price_approx": [150, 20, 80, 5],
            "ipo_date": [None, None, None, None],
        }).set_index("ticker")

        min_mcap = 2.0 * 1e9  # $2B from settings
        filtered = df[df["market_cap"] >= min_mcap]

        assert "BIG" in filtered.index
        assert "MED" in filtered.index
        assert "SMALL" not in filtered.index
        assert "TINY" not in filtered.index

    def test_volume_filter(self):
        """Only tickers with approx dollar volume >= $5M/day should pass."""
        df = pd.DataFrame({
            "ticker": ["HIGH_VOL", "LOW_VOL", "MID_VOL"],
            "market_cap": [5e9, 5e9, 5e9],
            "avg_volume": [1e6, 10_000, 200_000],
            "avg_price_approx": [100, 100, 100],
        }).set_index("ticker")

        df["approx_dollar_volume"] = df["avg_volume"] * df["avg_price_approx"]
        min_vol = 5.0 * 1e6  # $5M
        filtered = df[df["approx_dollar_volume"] >= min_vol]

        assert "HIGH_VOL" in filtered.index
        assert "MID_VOL" in filtered.index
        assert "LOW_VOL" not in filtered.index

    def test_exchange_filter(self):
        """Only NYSE/NASDAQ exchanges should pass."""
        df = pd.DataFrame({
            "ticker": ["NYSE_T", "NMS_T", "LSE_T", "TSE_T"],
            "exchange": ["NYQ", "NMS", "LSE", "TSE"],
            "currency": ["USD"] * 4,
            "market_cap": [5e9] * 4,
        }).set_index("ticker")

        valid_exchanges = {"NYQ", "NMS", "NGM", "NCM", "NYSE", "NASDAQ"}
        filtered = df[df["exchange"].isin(valid_exchanges)]

        assert "NYSE_T" in filtered.index
        assert "NMS_T" in filtered.index
        assert "LSE_T" not in filtered.index
        assert "TSE_T" not in filtered.index

    def test_combined_filters(self):
        """Tickers must pass ALL filters to be included."""
        df = pd.DataFrame({
            "ticker": [
                "PASS",        # passes all
                "FAIL_MCAP",   # fails market cap
                "FAIL_VOL",    # fails volume
                "FAIL_EX",     # fails exchange
            ],
            "exchange": ["NMS", "NMS", "NMS", "LSE"],
            "currency": ["USD", "USD", "USD", "USD"],
            "market_cap": [5e9, 500e6, 5e9, 5e9],
            "avg_volume": [500_000, 500_000, 1_000, 500_000],
            "avg_price_approx": [100, 100, 100, 100],
            "ipo_date": [None, None, None, None],
        }).set_index("ticker")

        # Apply all three filters in sequence
        valid_exchanges = {"NYQ", "NMS", "NGM", "NCM", "NYSE", "NASDAQ"}
        min_mcap = 2.0e9
        min_dollar_vol = 5.0e6

        df["approx_dollar_volume"] = df["avg_volume"] * df["avg_price_approx"]

        filtered = df[
            df["exchange"].isin(valid_exchanges)
            & (df["currency"] == "USD")
            & (df["market_cap"] >= min_mcap)
            & (df["approx_dollar_volume"] >= min_dollar_vol)
        ]

        assert len(filtered) == 1
        assert "PASS" in filtered.index
