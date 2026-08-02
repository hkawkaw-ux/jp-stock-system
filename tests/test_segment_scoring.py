import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.segment_scoring import top_n_per_segment


def _make_scored_df(segment_scores: dict) -> pd.DataFrame:
    rows = []
    for segment, scores in segment_scores.items():
        for i, score in enumerate(scores):
            rows.append({"code": f"{segment}{i}", "segment": segment, "score": score})
    return pd.DataFrame(rows)


class TestTopNPerSegment:
    def test_works_with_default_demo_segments(self):
        df = _make_scored_df({
            "大型株": [80, 60, 90],
            "中型株": [50, 70],
        })
        result = top_n_per_segment(df, n=10)
        assert set(result.keys()) == {"大型株", "中型株"}
        assert list(result["大型株"]["score"]) == [90, 80, 60]

    def test_works_with_arbitrary_sector_names(self):
        """SEGMENTS定数にない任意のセグメント名（業種名等）でも動作すること"""
        df = _make_scored_df({
            "銀行": [30, 50],
            "証券": [90],
            "不動産": [10, 20, 40],
        })
        result = top_n_per_segment(df, n=10)
        assert set(result.keys()) == {"銀行", "証券", "不動産"}
        assert list(result["不動産"]["score"]) == [40, 20, 10]

    def test_respects_top_n_limit(self):
        df = _make_scored_df({"銀行": [10, 20, 30, 40, 50]})
        result = top_n_per_segment(df, n=2)
        assert len(result["銀行"]) == 2
        assert list(result["銀行"]["score"]) == [50, 40]
