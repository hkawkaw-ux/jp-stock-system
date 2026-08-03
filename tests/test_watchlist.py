import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.watchlist import (
    WATCHLIST_COLUMNS,
    compute_signal_row,
    compute_watchlist_signals,
    load_watchlist,
    save_watchlist,
)


def _dummy_ohlcv(n=120):
    dates = pd.date_range("2026-01-01", periods=n, freq="B")
    price = np.linspace(1000, 1200, n)
    return pd.DataFrame(
        {
            "open": price, "high": price * 1.01, "low": price * 0.99,
            "close": price, "volume": np.full(n, 1_000_000.0),
        },
        index=dates,
    )


class TestLoadSaveWatchlist:
    def test_save_and_load_roundtrip(self, tmp_path):
        path = tmp_path / "watchlist.yaml"
        tickers = [{"code": "7203", "name": "トヨタ自動車"}, {"code": "8306", "name": "三菱UFJ"}]

        save_watchlist(tickers, path)
        loaded = load_watchlist(path)

        assert loaded == tickers

    def test_load_missing_file_returns_empty_list(self, tmp_path):
        assert load_watchlist(tmp_path / "not_exist.yaml") == []


class TestComputeSignalRow:
    def test_returns_expected_fields(self):
        with patch("src.watchlist.fetch_ohlcv", return_value=_dummy_ohlcv()):
            row = compute_signal_row("7203", "トヨタ自動車")

        assert row["code"] == "7203"
        assert row["name"] == "トヨタ自動車"
        assert row["action"] in ("BUY", "SELL", "HOLD")
        assert set(row.keys()) == set(WATCHLIST_COLUMNS)


class TestComputeWatchlistSignals:
    def test_builds_dataframe_for_multiple_tickers(self):
        tickers = [{"code": "7203", "name": "トヨタ自動車"}, {"code": "8306", "name": "三菱UFJ"}]
        with patch("src.watchlist.fetch_ohlcv", return_value=_dummy_ohlcv()):
            df = compute_watchlist_signals(tickers, request_interval=0)

        assert len(df) == 2
        assert list(df.columns) == WATCHLIST_COLUMNS

    def test_failed_ticker_marked_as_error(self):
        tickers = [{"code": "0000", "name": "ダミー"}]
        with patch("src.watchlist.fetch_ohlcv", side_effect=ValueError("fetch failed")):
            df = compute_watchlist_signals(tickers, request_interval=0)

        assert df.iloc[0]["action"] == "ERROR"
        assert df.iloc[0]["code"] == "0000"
