"""
セグメント評価用ユニバースの静的スナップショット読み書き
------------------------------------------------------------
Streamlit Cloud上でのリアルタイムyfinance取得はレート制限を受けやすいため、
GitHub Actions等で定期取得したCSVスナップショットを読み込む方式にする。
"""

import json
from pathlib import Path

import pandas as pd


def save_universe_snapshot(df: pd.DataFrame, fetched_at: str, csv_path: Path, meta_path: Path) -> None:
    csv_path = Path(csv_path)
    meta_path = Path(meta_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    meta_path.write_text(
        json.dumps({"fetched_at": fetched_at, "row_count": len(df)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_universe_snapshot(csv_path: Path, meta_path: Path) -> tuple[pd.DataFrame, str | None]:
    csv_path = Path(csv_path)
    meta_path = Path(meta_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"スナップショットCSVが見つかりません: {csv_path}")

    df = pd.read_csv(csv_path)
    fetched_at = None
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        fetched_at = meta.get("fetched_at")
    return df, fetched_at
