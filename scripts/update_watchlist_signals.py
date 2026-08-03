"""
ウォッチリスト銘柄のシグナルスナップショットを更新するスクリプト。
GitHub Actions（.github/workflows/update_universe.yml）またはローカルで実行する。

実行例:
    python scripts/update_watchlist_signals.py
"""

import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.snapshot import save_universe_snapshot
from src.watchlist import compute_watchlist_signals, load_watchlist

JST = timezone(timedelta(hours=9))
WATCHLIST_PATH = Path(__file__).resolve().parents[1] / "config" / "watchlist.yaml"
CSV_PATH = Path(__file__).resolve().parents[1] / "app" / "watchlist_signals.csv"
META_PATH = Path(__file__).resolve().parents[1] / "app" / "watchlist_signals_meta.json"


def main():
    tickers = load_watchlist(WATCHLIST_PATH)
    if not tickers:
        print(f"[警告] {WATCHLIST_PATH} にウォッチリスト銘柄が登録されていません")
        return

    df = compute_watchlist_signals(tickers)
    fetched_at = datetime.now(JST).isoformat()
    save_universe_snapshot(df, fetched_at, CSV_PATH, META_PATH)
    print(f"[更新] {CSV_PATH} ({len(df)}件, 取得日時: {fetched_at})")


if __name__ == "__main__":
    main()
