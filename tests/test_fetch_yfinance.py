import sys
from pathlib import Path
from datetime import datetime
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.fetch_yfinance import (
    UNIVERSE_COLUMNS,
    _retry,
    _universe_cache_path,
    build_universe_row,
    count_consecutive_dividend_growth_years,
    fetch_fundamentals,
    fetch_ohlcv,
    fetch_universe,
)


def _dummy_yf_history(n=120, start=1000.0, end=1200.0):
    dates = pd.date_range("2026-01-01", periods=n, freq="B")
    price = np.linspace(start, end, n)
    return pd.DataFrame(
        {
            "Open": price,
            "High": price * 1.01,
            "Low": price * 0.99,
            "Close": price,
            "Volume": np.full(n, 1_000_000.0),
        },
        index=dates,
    )


def _to_ohlcv(df):
    return df.rename(
        columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}
    )[["open", "high", "low", "close", "volume"]]


class TestRetry:
    def test_retries_then_succeeds(self):
        calls = {"n": 0}

        def flaky():
            calls["n"] += 1
            if calls["n"] < 2:
                raise RuntimeError("temporary failure")
            return "ok"

        assert _retry(flaky, retries=3, backoff=0) == "ok"
        assert calls["n"] == 2

    def test_raises_after_max_retries(self):
        def always_fail():
            raise RuntimeError("permanent failure")

        with pytest.raises(RuntimeError):
            _retry(always_fail, retries=2, backoff=0)


class TestFetchOhlcv:
    def test_returns_expected_columns(self):
        with patch("src.data.fetch_yfinance.yf.Ticker") as mock_ticker:
            mock_ticker.return_value.history.return_value = _dummy_yf_history()
            df = fetch_ohlcv("7203.T")
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
        assert len(df) == 120

    def test_raises_when_empty(self):
        with patch("src.data.fetch_yfinance.yf.Ticker") as mock_ticker:
            mock_ticker.return_value.history.return_value = pd.DataFrame()
            with pytest.raises(ValueError):
                fetch_ohlcv("0000.T")


class TestDividendGrowthYears:
    def test_consecutive_increase(self):
        idx = pd.to_datetime([f"{y}-03-01" for y in range(2019, 2026)])
        divs = pd.Series([10, 12, 14, 16, 18, 20, 22], index=idx)
        assert count_consecutive_dividend_growth_years(divs) == 6

    def test_broken_streak(self):
        idx = pd.to_datetime([f"{y}-03-01" for y in range(2019, 2026)])
        # 2021→2022年で減配。直近2023→2024→2025の3年分のみ連続増配としてカウントされる
        divs = pd.Series([10, 12, 14, 10, 18, 20, 22], index=idx)
        assert count_consecutive_dividend_growth_years(divs) == 3

    def test_empty_series_returns_zero(self):
        assert count_consecutive_dividend_growth_years(pd.Series(dtype=float)) == 0


class TestFetchFundamentals:
    def test_maps_info_fields(self):
        mock_info = {
            "revenueGrowth": 0.10,
            "operatingMargins": 0.12,
            "returnOnEquity": 0.15,
            "trailingPE": 18.5,
            "priceToBook": 1.6,
            "dividendYield": 2.5,  # yfinance 1.5系以降は%表記そのもの(2.5 = 2.5%)
            "recommendationMean": 2.0,
        }
        with patch("src.data.fetch_yfinance.yf.Ticker") as mock_ticker:
            mock_ticker.return_value.info = mock_info
            mock_ticker.return_value.dividends = pd.Series(dtype=float)
            result = fetch_fundamentals("7203.T")

        assert result["growth_pct"] == 10.0
        assert result["op_margin_pct"] == 12.0
        assert result["roe_pct"] == 15.0
        assert result["per"] == 18.5
        assert result["pbr"] == 1.6
        assert result["div_yield_pct"] == 2.5
        assert result["analyst_rating"] == 4.0  # 6 - recommendationMean
        assert np.isnan(result["margin_ratio"])

    def test_missing_fields_become_nan(self):
        with patch("src.data.fetch_yfinance.yf.Ticker") as mock_ticker:
            mock_ticker.return_value.info = {}
            mock_ticker.return_value.dividends = pd.Series(dtype=float)
            result = fetch_fundamentals("9999.T")

        assert np.isnan(result["per"])
        assert np.isnan(result["analyst_rating"])
        assert result["div_growth_years"] == 0


class TestBuildUniverseRow:
    def test_returns_universe_columns(self):
        ohlcv = _to_ohlcv(_dummy_yf_history())
        fundamentals = {
            "growth_pct": 10.0, "op_margin_pct": 12.0, "roe_pct": 15.0,
            "per": 18.5, "pbr": 1.6, "div_yield_pct": 2.5,
            "div_growth_years": 5, "margin_ratio": np.nan, "analyst_rating": 4.0,
        }
        with patch("src.data.fetch_yfinance.fetch_ohlcv", return_value=ohlcv), \
             patch("src.data.fetch_yfinance.fetch_fundamentals", return_value=fundamentals):
            row = build_universe_row("7203.T", "大型株", "トヨタ自動車")

        assert set(row.keys()) == set(UNIVERSE_COLUMNS)
        assert row["code"] == "7203"
        assert row["name"] == "トヨタ自動車"
        assert row["segment"] == "大型株"
        assert row["per"] == 18.5


class TestFetchUniverse:
    def test_builds_dataframe_and_caches(self, tmp_path, monkeypatch):
        import src.data.fetch_yfinance as mod

        monkeypatch.setattr(mod, "CACHE_DIR", tmp_path)
        dummy_row = {c: 0 for c in UNIVERSE_COLUMNS}
        with patch.object(mod, "build_universe_row", return_value=dummy_row):
            df = fetch_universe({"7203.T": ("大型株", "トヨタ")}, cache=True, request_interval=0)

        assert len(df) == 1
        cached_files = list(tmp_path.glob("universe_*.csv"))
        assert len(cached_files) == 1

    def test_uses_cache_if_exists(self, tmp_path, monkeypatch):
        import src.data.fetch_yfinance as mod

        monkeypatch.setattr(mod, "CACHE_DIR", tmp_path)
        segment_map = {"7203.T": ("大型株", "トヨタ")}
        cache_file = _universe_cache_path(segment_map)
        pd.DataFrame([{c: 0 for c in UNIVERSE_COLUMNS}]).to_csv(cache_file, index=False)

        with patch.object(mod, "build_universe_row") as mock_build:
            df = fetch_universe(segment_map, cache=True)

        mock_build.assert_not_called()
        assert len(df) == 1

    def test_skips_failed_ticker(self, tmp_path, monkeypatch):
        import src.data.fetch_yfinance as mod

        monkeypatch.setattr(mod, "CACHE_DIR", tmp_path)
        with patch.object(mod, "build_universe_row", side_effect=ValueError("fetch failed")):
            df = fetch_universe({"0000.T": ("大型株", "ダミー")}, cache=False, request_interval=0)

        assert len(df) == 0
        assert list(df.columns) == UNIVERSE_COLUMNS

    def test_different_segment_maps_use_different_cache_files(self, tmp_path, monkeypatch):
        """異なる銘柄セットが同じ日付のキャッシュファイルを誤って共有しないこと"""
        import src.data.fetch_yfinance as mod

        monkeypatch.setattr(mod, "CACHE_DIR", tmp_path)
        dummy_row = {c: 0 for c in UNIVERSE_COLUMNS}

        with patch.object(mod, "build_universe_row", return_value=dummy_row):
            df1 = fetch_universe({"7203.T": ("大型株", "トヨタ")}, cache=True, request_interval=0)
            df2 = fetch_universe(
                {"6758.T": ("大型株", "ソニー"), "8306.T": ("銀行", "三菱UFJ")},
                cache=True, request_interval=0,
            )

        assert len(df1) == 1
        assert len(df2) == 2
        cached_files = list(tmp_path.glob("universe_*.csv"))
        assert len(cached_files) == 2
