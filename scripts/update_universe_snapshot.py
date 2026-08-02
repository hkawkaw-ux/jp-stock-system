"""
セグメント評価用ユニバースのスナップショットを更新するスクリプト。
GitHub Actions（.github/workflows/update_universe.yml）またはローカルで実行する。

実行例:
    python scripts/update_universe_snapshot.py
"""

import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtest import SAMPLE_TICKERS
from src.data.fetch_yfinance import fetch_universe
from src.snapshot import save_universe_snapshot

JST = timezone(timedelta(hours=9))
CSV_PATH = Path(__file__).resolve().parents[1] / "app" / "universe_snapshot.csv"
META_PATH = Path(__file__).resolve().parents[1] / "app" / "universe_snapshot_meta.json"


def main():
    df = fetch_universe(SAMPLE_TICKERS, cache=False)
    fetched_at = datetime.now(JST).isoformat()
    save_universe_snapshot(df, fetched_at, CSV_PATH, META_PATH)
    print(f"[更新] {CSV_PATH} ({len(df)}件, 取得日時: {fetched_at})")


if __name__ == "__main__":
    main()
