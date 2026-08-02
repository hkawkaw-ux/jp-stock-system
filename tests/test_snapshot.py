import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.snapshot import load_universe_snapshot, save_universe_snapshot


class TestSaveUniverseSnapshot:
    def test_writes_csv_and_meta_json(self, tmp_path):
        df = pd.DataFrame([{"code": "7203", "segment": "自動車", "score": 50.0}])
        csv_path = tmp_path / "universe_snapshot.csv"
        meta_path = tmp_path / "universe_snapshot_meta.json"

        save_universe_snapshot(df, "2026-08-02T10:00:00+09:00", csv_path, meta_path)

        assert csv_path.exists()
        saved = pd.read_csv(csv_path)
        assert saved.iloc[0]["code"] == 7203

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        assert meta["fetched_at"] == "2026-08-02T10:00:00+09:00"
        assert meta["row_count"] == 1


class TestLoadUniverseSnapshot:
    def test_loads_dataframe_and_timestamp(self, tmp_path):
        df = pd.DataFrame([{"code": "7203", "segment": "自動車", "score": 50.0}])
        csv_path = tmp_path / "universe_snapshot.csv"
        meta_path = tmp_path / "universe_snapshot_meta.json"
        save_universe_snapshot(df, "2026-08-02T10:00:00+09:00", csv_path, meta_path)

        loaded_df, fetched_at = load_universe_snapshot(csv_path, meta_path)

        assert len(loaded_df) == 1
        assert fetched_at == "2026-08-02T10:00:00+09:00"

    def test_missing_meta_returns_none_timestamp(self, tmp_path):
        df = pd.DataFrame([{"code": "7203", "segment": "自動車", "score": 50.0}])
        csv_path = tmp_path / "universe_snapshot.csv"
        df.to_csv(csv_path, index=False)
        meta_path = tmp_path / "does_not_exist.json"

        loaded_df, fetched_at = load_universe_snapshot(csv_path, meta_path)

        assert len(loaded_df) == 1
        assert fetched_at is None

    def test_missing_csv_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_universe_snapshot(tmp_path / "no_such.csv", tmp_path / "no_such_meta.json")
