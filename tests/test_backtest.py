import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtest import (
    compute_equity_curve,
    compute_max_drawdown,
    compute_metrics,
    compute_metrics_by_pattern,
    compute_metrics_by_sector,
    run_backtest,
    run_backtest_multi,
)


def _empty_signal_df(n=5, open_price=100):
    dates = pd.date_range("2026-01-01", periods=n, freq="B")
    df = pd.DataFrame({"open": [open_price] * n, "close": [open_price] * n}, index=dates)
    df["BUY"] = False
    df["SELL"] = False
    for p in ["A", "B", "C"]:
        df[f"pattern_{p}"] = False
    return df


def _make_signal_df():
    """
    day0: BUY(pattern_A)   -> day1始値100でエントリー
    day3: SELL             -> day4始値110でイグジット => +10%
    day5: BUY(pattern_B)   -> day6始値90でエントリー
    day8: SELL             -> day9始値81でイグジット => -10%
    """
    n = 12
    dates = pd.date_range("2026-01-01", periods=n, freq="B")
    open_ = [100, 100, 100, 100, 110, 100, 90, 100, 100, 81, 100, 100]
    df = pd.DataFrame({"open": open_, "close": open_}, index=dates)
    df["BUY"] = False
    df["SELL"] = False
    for p in ["A", "B", "C"]:
        df[f"pattern_{p}"] = False

    df.iloc[0, df.columns.get_loc("BUY")] = True
    df.iloc[0, df.columns.get_loc("pattern_A")] = True
    df.iloc[3, df.columns.get_loc("SELL")] = True
    df.iloc[5, df.columns.get_loc("BUY")] = True
    df.iloc[5, df.columns.get_loc("pattern_B")] = True
    df.iloc[8, df.columns.get_loc("SELL")] = True
    return df


class TestRunBacktest:
    def test_generates_expected_trades(self):
        df = _make_signal_df()
        trades = run_backtest(df)

        assert len(trades) == 2
        assert trades.iloc[0]["pattern"] == "A"
        assert trades.iloc[0]["entry_price"] == 100
        assert trades.iloc[0]["exit_price"] == 110
        assert trades.iloc[0]["return_pct"] == 10.0
        assert trades.iloc[1]["pattern"] == "B"
        assert trades.iloc[1]["entry_price"] == 90
        assert trades.iloc[1]["exit_price"] == 81
        assert trades.iloc[1]["return_pct"] == -10.0

    def test_no_signal_returns_empty(self):
        df = _empty_signal_df()
        trades = run_backtest(df)
        assert trades.empty
        assert list(trades.columns) == [
            "entry_date", "entry_price", "exit_date", "exit_price", "pattern", "return_pct",
        ]

    def test_unclosed_position_is_excluded(self):
        df = _empty_signal_df()
        df.iloc[0, df.columns.get_loc("BUY")] = True
        trades = run_backtest(df)
        assert trades.empty  # SELLシグナルが出ないまま終了 -> 未決済トレードは含めない


class TestComputeMetrics:
    def test_win_rate_and_expected_value(self):
        trades = pd.DataFrame({
            "return_pct": [10.0, -10.0, 5.0, -2.0],
            "pattern": ["A", "B", "A", "C"],
        })
        m = compute_metrics(trades)
        assert m["trade_count"] == 4
        assert m["win_rate"] == 50.0
        assert m["avg_return_pct"] == round((10 - 10 + 5 - 2) / 4, 2)

    def test_empty_trades(self):
        trades = pd.DataFrame(columns=["return_pct", "pattern"])
        m = compute_metrics(trades)
        assert m["trade_count"] == 0
        assert np.isnan(m["win_rate"])


class TestEquityCurveAndDrawdown:
    def test_equity_curve_compounding(self):
        trades = pd.DataFrame({"return_pct": [10.0, -10.0]})
        curve = compute_equity_curve(trades)
        assert curve.iloc[0] == 100.0
        assert curve.iloc[1] == pytest.approx(110.0)
        assert curve.iloc[2] == pytest.approx(99.0)

    def test_max_drawdown(self):
        curve = pd.Series([100, 120, 90, 110])
        dd = compute_max_drawdown(curve)
        assert dd == pytest.approx(-25.0)  # ピーク120→90: (90-120)/120*100


class TestMetricsByPattern:
    def test_groups_by_pattern(self):
        trades = pd.DataFrame({
            "return_pct": [10.0, -10.0, 5.0],
            "pattern": ["A", "B", "A"],
        })
        by_pattern = compute_metrics_by_pattern(trades)
        assert by_pattern.loc["A", "trade_count"] == 2
        assert by_pattern.loc["B", "trade_count"] == 1
        assert by_pattern.loc["C", "trade_count"] == 0


class TestRunBacktestMulti:
    def test_combines_trades_with_ticker_sector_name(self):
        df_a = _make_signal_df()      # 2トレード発生
        df_b = _empty_signal_df()     # 0トレード

        def fake_fetch_ohlcv(ticker, period="2y"):
            return {"AAAA.T": df_a, "BBBB.T": df_b}[ticker]

        with patch("src.backtest.fetch_ohlcv", side_effect=fake_fetch_ohlcv), \
             patch("src.backtest.add_indicators", side_effect=lambda df: df), \
             patch("src.backtest.evaluate", side_effect=lambda df: df):
            trades = run_backtest_multi({
                "AAAA.T": ("銀行", "銀行A"),
                "BBBB.T": ("証券", "証券B"),
            })

        assert len(trades) == 2
        assert set(trades["ticker"]) == {"AAAA.T"}
        assert set(trades["sector"]) == {"銀行"}
        assert set(trades["name"]) == {"銀行A"}

    def test_skips_failed_ticker(self):
        def fake_fetch_ohlcv(ticker, period="2y"):
            raise ValueError("fetch failed")

        with patch("src.backtest.fetch_ohlcv", side_effect=fake_fetch_ohlcv):
            trades = run_backtest_multi({"ZZZZ.T": ("銀行", "ダミー")})

        assert trades.empty

    def test_trades_sorted_by_entry_date_across_tickers(self):
        # AAAA: 2026年2月にエントリー成立（辞書順は先頭だが日付は後）
        dates_a = pd.date_range("2026-02-01", periods=12, freq="B")
        open_a = [100] * 12
        df_a = pd.DataFrame({"open": open_a, "close": open_a}, index=dates_a)
        df_a["BUY"] = False
        df_a["SELL"] = False
        for p in ["A", "B", "C"]:
            df_a[f"pattern_{p}"] = False
        df_a.iloc[5, df_a.columns.get_loc("BUY")] = True
        df_a.iloc[9, df_a.columns.get_loc("SELL")] = True

        # BBBB: 2026年1月にエントリー成立（辞書順は後だが日付は先）
        dates_b = pd.date_range("2026-01-01", periods=12, freq="B")
        open_b = [100] * 12
        df_b = pd.DataFrame({"open": open_b, "close": open_b}, index=dates_b)
        df_b["BUY"] = False
        df_b["SELL"] = False
        for p in ["A", "B", "C"]:
            df_b[f"pattern_{p}"] = False
        df_b.iloc[0, df_b.columns.get_loc("BUY")] = True
        df_b.iloc[2, df_b.columns.get_loc("SELL")] = True

        def fake_fetch_ohlcv(ticker, period="2y"):
            return {"AAAA.T": df_a, "BBBB.T": df_b}[ticker]

        with patch("src.backtest.fetch_ohlcv", side_effect=fake_fetch_ohlcv), \
             patch("src.backtest.add_indicators", side_effect=lambda df: df), \
             patch("src.backtest.evaluate", side_effect=lambda df: df):
            trades = run_backtest_multi({
                "AAAA.T": ("セクタA", "A社"),
                "BBBB.T": ("セクタB", "B社"),
            })

        assert trades["entry_date"].is_monotonic_increasing
        assert trades.iloc[0]["ticker"] == "BBBB.T"
        assert trades.iloc[-1]["ticker"] == "AAAA.T"


class TestMetricsBySector:
    def test_groups_by_sector(self):
        trades = pd.DataFrame({
            "return_pct": [10.0, -10.0, 5.0],
            "sector": ["銀行", "証券", "銀行"],
        })
        by_sector = compute_metrics_by_sector(trades)
        assert by_sector.loc["銀行", "trade_count"] == 2
        assert by_sector.loc["証券", "trade_count"] == 1

    def test_empty_trades_returns_empty_df(self):
        trades = pd.DataFrame(columns=["return_pct", "sector"])
        result = compute_metrics_by_sector(trades)
        assert result.empty
