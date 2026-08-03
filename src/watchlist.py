"""
ウォッチリスト（複数銘柄のBUY/SELL/HOLDシグナル一覧）管理
------------------------------------------------------------
銘柄コードの読み書きと、各銘柄のシグナル計算をまとめる。
"""

import logging
import time
from pathlib import Path

import pandas as pd
import yaml

from src.data.fetch_yfinance import fetch_ohlcv
from src.signal_engine import add_indicators, evaluate, latest_signal

logger = logging.getLogger(__name__)

WATCHLIST_COLUMNS = ["code", "name", "close", "action", "patterns", "rsi", "score"]


# ============================================================
# 1. ウォッチリスト銘柄コードの読み書き
# ============================================================
def load_watchlist(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data.get("tickers", [])


def save_watchlist(tickers: list[dict], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.dump({"tickers": tickers}, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


# ============================================================
# 2. 個別銘柄のシグナル計算
# ============================================================
def compute_signal_row(code: str, name: str, period: str = "6mo") -> dict:
    ticker = f"{code}.T"
    ohlcv = fetch_ohlcv(ticker, period=period)
    df = evaluate(add_indicators(ohlcv))
    signal = latest_signal(df)
    return {
        "code": code,
        "name": name,
        "close": signal["close"],
        "action": signal["action"],
        "patterns": ",".join(signal["patterns"]) if signal["patterns"] else "",
        "rsi": signal["rsi"],
        "score": signal["score"],
    }


# ============================================================
# 3. 複数銘柄のシグナル一覧
# ============================================================
def compute_watchlist_signals(tickers: list[dict], request_interval: float = 2.5) -> pd.DataFrame:
    rows = []
    for i, t in enumerate(tickers):
        code, name = t["code"], t.get("name", t["code"])
        try:
            rows.append(compute_signal_row(code, name))
        except Exception as e:
            logger.warning(f"{code}: シグナル計算に失敗 ({e})")
            rows.append({
                "code": code, "name": name, "close": None,
                "action": "ERROR", "patterns": "", "rsi": None, "score": None,
            })
        if request_interval and i < len(tickers) - 1:
            time.sleep(request_interval)
    return pd.DataFrame(rows, columns=WATCHLIST_COLUMNS)
