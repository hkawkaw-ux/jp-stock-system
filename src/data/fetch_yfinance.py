"""
yfinance を用いた実データ取得モジュール
------------------------------------------------------------
segment_scoring.py の make_demo_universe() と同じ列構成の DataFrame を返す。
J-Quants ではなく yfinance を採用（無料・APIキー不要・約15分遅延）。

注意:
  yfinance は非公式ライブラリであり、Yahoo!側の仕様変更で
  突然取得できなくなるリスクがある。商用利用不可。
  個人の投資判断補助（非商用）用途として利用すること。
"""

import hashlib
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parents[2] / "data"

UNIVERSE_COLUMNS = [
    "code", "name", "segment",
    "growth_pct", "op_margin_pct", "roe_pct", "per", "pbr",
    "div_yield_pct", "div_growth_years", "tech_score", "vol_ratio",
    "margin_ratio", "analyst_rating", "ret_1m_pct", "ret_3m_pct",
]


# ============================================================
# 共通: リトライ（Yahoo側の断続的な取得失敗に対応）
# ============================================================
def _retry(func, retries=3, backoff=1.0):
    last_exc = None
    for attempt in range(retries):
        try:
            return func()
        except Exception as e:
            last_exc = e
            wait = backoff * (2 ** attempt)
            logger.warning(f"取得失敗(試行{attempt + 1}/{retries}): {e}")
            if wait > 0:
                time.sleep(wait)
    raise last_exc


# ============================================================
# 1. 日足株価・出来高取得（signal_engine.py 互換）
# ============================================================
def fetch_ohlcv(ticker: str, period: str = "6mo") -> pd.DataFrame:
    hist = _retry(lambda: yf.Ticker(ticker).history(period=period))
    if hist is None or hist.empty:
        raise ValueError(f"{ticker}: 日足データを取得できませんでした")

    hist = hist.rename(columns={
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Volume": "volume",
    })[["open", "high", "low", "close", "volume"]]
    if hist.index.tz is not None:
        hist.index = hist.index.tz_localize(None)
    return hist


# ============================================================
# 2. 配当履歴 → 連続増配年数
# ============================================================
def count_consecutive_dividend_growth_years(dividends: pd.Series) -> int:
    if dividends is None or dividends.empty:
        return 0
    yearly = dividends.groupby(dividends.index.year).sum().sort_index()
    years = 0
    for i in range(len(yearly) - 1, 0, -1):
        if yearly.iloc[i] > yearly.iloc[i - 1]:
            years += 1
        else:
            break
    return years


def _to_pct(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    return round(value * 100, 2)


# ============================================================
# 3. 財務指標取得
# ============================================================
def fetch_fundamentals(ticker: str) -> dict:
    t = yf.Ticker(ticker)
    info = _retry(lambda: t.info) or {}
    dividends = _retry(lambda: t.dividends)

    recommendation = info.get("recommendationMean")
    # yfinance は 1=Strong Buy 〜 5=Sell。本システムは 5=最高評価の設計のため反転する。
    analyst_rating = (6 - recommendation) if recommendation is not None else np.nan

    # yfinance 1.5系以降、dividendYield は既に%表記(例: 3.26 = 3.26%)で返る。
    # revenueGrowth 等の小数比率(例: 0.019 = 1.9%)とは仕様が異なるため個別に変換する。
    div_yield_raw = info.get("dividendYield")
    div_yield_pct = round(div_yield_raw, 2) if div_yield_raw is not None else np.nan

    return {
        "growth_pct": _to_pct(info.get("revenueGrowth")),
        "op_margin_pct": _to_pct(info.get("operatingMargins")),
        "roe_pct": _to_pct(info.get("returnOnEquity")),
        "per": info.get("trailingPE", np.nan),
        "pbr": info.get("priceToBook", np.nan),
        "div_yield_pct": div_yield_pct,
        "div_growth_years": count_consecutive_dividend_growth_years(dividends),
        "margin_ratio": np.nan,  # yfinanceでは信用倍率は取得不可（任意項目）
        "analyst_rating": analyst_rating,
    }


def _return_pct(close: pd.Series, lookback: int):
    if len(close) <= lookback:
        return np.nan
    return round((close.iloc[-1] / close.iloc[-lookback - 1] - 1) * 100, 2)


# ============================================================
# 4. 1銘柄分の統合行（テクニカル + ファンダメンタルズ）
# ============================================================
def build_universe_row(ticker: str, segment: str, name: str = None) -> dict:
    from src.signal_engine import add_indicators, evaluate  # 循環import回避のため遅延import

    ohlcv = fetch_ohlcv(ticker)
    evaluated = evaluate(add_indicators(ohlcv))
    latest = evaluated.iloc[-1]
    close = evaluated["close"]

    row = {
        "code": ticker.replace(".T", ""),
        "name": name or ticker,
        "segment": segment,
        "tech_score": float(latest["score"]) if not pd.isna(latest["score"]) else np.nan,
        "vol_ratio": float(latest["vol_ratio"]) if not pd.isna(latest["vol_ratio"]) else np.nan,
        "ret_1m_pct": _return_pct(close, 21),
        "ret_3m_pct": _return_pct(close, 63),
    }
    row.update(fetch_fundamentals(ticker))
    return row


# ============================================================
# 5. 複数銘柄をまとめて取得（segment_scoring.py 互換 DataFrame）
# ============================================================
def _universe_cache_path(segment_map: dict) -> Path:
    """
    銘柄セットごとに異なるキャッシュファイルを使う。
    日付のみをキーにすると、異なる銘柄セット（例: 検証用28銘柄とデモ2銘柄）が
    同じキャッシュファイルを誤って共有してしまうため、銘柄構成のハッシュを含める。
    """
    key_source = ",".join(sorted(segment_map.keys()))
    cache_key = hashlib.md5(key_source.encode()).hexdigest()[:8]
    return CACHE_DIR / f"universe_{datetime.now():%Y%m%d}_{cache_key}.csv"


def fetch_universe(segment_map: dict, cache: bool = True, request_interval: float = 1.0) -> pd.DataFrame:
    """
    segment_map: {ticker: (segment, name)} 形式
    戻り値: segment_scoring.py の make_demo_universe() と同じ列構成の DataFrame
    """
    cache_path = _universe_cache_path(segment_map)
    if cache and cache_path.exists():
        logger.info(f"キャッシュを使用: {cache_path}")
        return pd.read_csv(cache_path)

    rows = []
    tickers = list(segment_map.items())
    for i, (ticker, (segment, name)) in enumerate(tickers):
        try:
            rows.append(build_universe_row(ticker, segment, name))
        except Exception as e:
            logger.warning(f"{ticker}: 取得失敗のためスキップ ({e})")
        if request_interval and i < len(tickers) - 1:
            time.sleep(request_interval)

    df = pd.DataFrame(rows, columns=UNIVERSE_COLUMNS)
    if cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path, index=False, encoding="utf-8-sig")
        logger.info(f"キャッシュに保存: {cache_path}")
    return df
