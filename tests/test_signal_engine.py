import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.signal_engine import DEFAULT_PARAMS, add_indicators, evaluate, load_params, make_demo_data


class TestLoadParams:
    def test_returns_default_when_file_missing(self, tmp_path):
        params = load_params(tmp_path / "not_exist.yaml")
        assert params == DEFAULT_PARAMS

    def test_loads_and_overrides_from_yaml(self, tmp_path):
        custom = {
            "signal": {
                "patterns": {"A_rsi": [40, 60]},
                "rsi": {"period": 21},
            }
        }
        path = tmp_path / "params.yaml"
        path.write_text(yaml.dump(custom), encoding="utf-8")

        params = load_params(path)
        assert params["patterns"]["A_rsi"] == [40, 60]
        assert params["rsi"]["period"] == 21
        # 明示していない項目はデフォルトのまま残る
        assert params["rsi"]["overbought"] == DEFAULT_PARAMS["rsi"]["overbought"]
        assert params["moving_average"] == DEFAULT_PARAMS["moving_average"]

    def test_reads_project_config_params_yaml(self):
        """実際の config/params.yaml がデフォルト値と一致していることを確認する"""
        project_config = Path(__file__).resolve().parents[1] / "config" / "params.yaml"
        params = load_params(project_config)
        assert params["patterns"]["A_rsi"] == [50, 65]
        assert params["rsi"]["overbought"] == 75


class TestEvaluateWithParams:
    def _demo(self):
        return add_indicators(make_demo_data(seed=7, n=180))

    def test_default_params_match_original_hardcoded_behavior(self):
        df_default = evaluate(self._demo())
        df_explicit = evaluate(self._demo(), params=DEFAULT_PARAMS)
        pd.testing.assert_series_equal(df_default["BUY"], df_explicit["BUY"])
        pd.testing.assert_series_equal(df_default["SELL"], df_explicit["SELL"])

    def test_changing_pattern_a_rsi_range_changes_signal(self):
        df = self._demo()
        narrow_params = {**DEFAULT_PARAMS, "patterns": {**DEFAULT_PARAMS["patterns"], "A_rsi": [61, 62]}}

        wide_result = evaluate(df, params=DEFAULT_PARAMS)
        narrow_result = evaluate(df, params=narrow_params)

        # レンジを極端に狭めるとパターンA成立数は減るか同数になる（増えることはない）
        assert narrow_result["pattern_A"].sum() <= wide_result["pattern_A"].sum()

    def test_changing_overbought_threshold_changes_sell_signal(self):
        df = self._demo()
        strict_params = {**DEFAULT_PARAMS, "rsi": {**DEFAULT_PARAMS["rsi"], "overbought": 50}}

        default_result = evaluate(df, params=DEFAULT_PARAMS)
        strict_result = evaluate(df, params=strict_params)

        # 買われ過ぎ閾値を下げるとSELLが出やすくなる（同数以上になる）
        assert strict_result["SELL"].sum() >= default_result["SELL"].sum()
