"""
バックテストモジュール
------------------------------------------------------------
signal_engine.py の BUY/SELL シグナルに従って売買したと仮定し、
勝率・平均損益・期待値・最大ドローダウンを算出する。

売買ルール:
  ・BUYシグナル成立日の翌営業日「始値」でエントリー
  ・保有中にSELLシグナルが成立した日の翌営業日「始値」でイグジット
  ・同時保有は1ポジションのみ
  ・手数料・スリッページは未考慮（引継ぎ資料の制約事項どおり）

翌営業日の始値で約定させるのは、シグナルは当日終値確定後にしか
判定できないため、当日終値で約定すると未来のデータを先取りしてしまうことを防ぐため。
"""

import logging
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.fetch_yfinance import fetch_ohlcv
from src.signal_engine import add_indicators, evaluate

logger = logging.getLogger(__name__)

PATTERNS = ["A", "B", "C"]
TRADE_COLUMNS = ["entry_date", "entry_price", "exit_date", "exit_price", "pattern", "return_pct"]


# ============================================================
# 1. シグナルに従った売買シミュレーション
# ============================================================
def run_backtest(df: pd.DataFrame) -> pd.DataFrame:
    trades = []
    position = None
    n = len(df)

    for i in range(n):
        row = df.iloc[i]

        if position is None:
            if row["BUY"] and i + 1 < n:
                entry_idx = i + 1
                pattern = next((p for p in PATTERNS if row.get(f"pattern_{p}", False)), None)
                position = {
                    "entry_idx": entry_idx,
                    "entry_date": df.index[entry_idx],
                    "entry_price": df["open"].iloc[entry_idx],
                    "pattern": pattern,
                }
        else:
            if i > position["entry_idx"] and row["SELL"] and i + 1 < n:
                exit_idx = i + 1
                exit_price = df["open"].iloc[exit_idx]
                return_pct = round((exit_price / position["entry_price"] - 1) * 100, 2)
                trades.append({
                    "entry_date": position["entry_date"],
                    "entry_price": position["entry_price"],
                    "exit_date": df.index[exit_idx],
                    "exit_price": exit_price,
                    "pattern": position["pattern"],
                    "return_pct": return_pct,
                })
                position = None

    return pd.DataFrame(trades, columns=TRADE_COLUMNS)


# ============================================================
# 2. 累積損益カーブ・最大ドローダウン
# ============================================================
def compute_equity_curve(trades: pd.DataFrame) -> pd.Series:
    """トレード順に複利で資産推移を算出（初期値100）"""
    equity = [100.0]
    for r in trades["return_pct"]:
        equity.append(equity[-1] * (1 + r / 100))
    return pd.Series(equity)


def compute_max_drawdown(equity_curve: pd.Series) -> float:
    """最大下落率(%)を負の値で返す"""
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max * 100
    return float(drawdown.min())


# ============================================================
# 3. 成績指標
# ============================================================
def compute_metrics(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {
            "trade_count": 0, "win_rate": np.nan, "avg_return_pct": np.nan,
            "avg_win_pct": np.nan, "avg_loss_pct": np.nan,
            "expected_value_pct": np.nan, "max_drawdown_pct": np.nan,
        }

    wins = trades[trades["return_pct"] > 0]
    losses = trades[trades["return_pct"] <= 0]
    win_rate = len(wins) / len(trades)
    avg_win = wins["return_pct"].mean() if not wins.empty else 0.0
    avg_loss = losses["return_pct"].mean() if not losses.empty else 0.0
    expected_value = win_rate * avg_win + (1 - win_rate) * avg_loss

    equity_curve = compute_equity_curve(trades)
    max_dd = compute_max_drawdown(equity_curve)

    return {
        "trade_count": len(trades),
        "win_rate": round(win_rate * 100, 1),
        "avg_return_pct": round(trades["return_pct"].mean(), 2),
        "avg_win_pct": round(avg_win, 2),
        "avg_loss_pct": round(avg_loss, 2),
        "expected_value_pct": round(expected_value, 2),
        "max_drawdown_pct": round(max_dd, 2),
    }


def compute_metrics_by_pattern(trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for p in PATTERNS:
        sub = trades[trades["pattern"] == p] if not trades.empty else trades
        m = compute_metrics(sub)
        m["pattern"] = p
        rows.append(m)
    return pd.DataFrame(rows).set_index("pattern")


def compute_metrics_by_sector(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty or "sector" not in trades.columns:
        return pd.DataFrame()
    rows = []
    for sector in sorted(trades["sector"].dropna().unique()):
        sub = trades[trades["sector"] == sector]
        m = compute_metrics(sub)
        m["sector"] = sector
        rows.append(m)
    return pd.DataFrame(rows).set_index("sector")


# ============================================================
# 3b. 複数銘柄への一括適用
# ============================================================
def run_backtest_multi(ticker_sectors: dict, period: str = "2y") -> pd.DataFrame:
    """
    ticker_sectors: {ticker: (sector, name)} 形式
    各銘柄のバックテスト結果を結合し、ticker/sector/name列を付与して返す
    """
    all_trades = []
    for ticker, (sector, name) in ticker_sectors.items():
        try:
            ohlcv = fetch_ohlcv(ticker, period=period)
            evaluated = evaluate(add_indicators(ohlcv))
            trades = run_backtest(evaluated)
        except Exception as e:
            logger.warning(f"{ticker}: バックテスト失敗のためスキップ ({e})")
            continue
        trades = trades.copy()
        trades["ticker"] = ticker
        trades["sector"] = sector
        trades["name"] = name
        all_trades.append(trades)

    columns = TRADE_COLUMNS + ["ticker", "sector", "name"]
    if not all_trades:
        return pd.DataFrame(columns=columns)
    combined = pd.concat(all_trades, ignore_index=True)[columns]
    # 銘柄ごとの結果を単純連結すると時系列が崩れ、累積損益カーブが無意味になるため
    # エントリー日でソートし、時系列順の複利計算ができるようにする。
    return combined.sort_values("entry_date").reset_index(drop=True)


# ============================================================
# 4. 可視化
# ============================================================
def _setup_japanese_font():
    for name in ["Yu Gothic", "Meiryo", "MS Gothic", "IPAexGothic"]:
        try:
            matplotlib.rcParams["font.family"] = name
            return
        except Exception:
            continue


def plot_results(trades: pd.DataFrame, save_path: str = None):
    _setup_japanese_font()
    equity_curve = compute_equity_curve(trades)
    by_pattern = compute_metrics_by_pattern(trades)
    has_sector = "sector" in trades.columns and not trades.empty

    ncols = 3 if has_sector else 2
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5))

    axes[0].plot(equity_curve.values, marker="o", markersize=3)
    axes[0].set_title("累積損益カーブ（初期値100）")
    axes[0].set_xlabel("トレード回数")
    axes[0].set_ylabel("資産評価額")
    axes[0].grid(True)

    win_rates = by_pattern["win_rate"].fillna(0)
    axes[1].bar(by_pattern.index, win_rates)
    axes[1].set_title("パターン別勝率(%)")
    axes[1].set_ylabel("勝率(%)")
    axes[1].set_ylim(0, 100)

    if has_sector:
        by_sector = compute_metrics_by_sector(trades)
        sector_win_rates = by_sector["win_rate"].fillna(0)
        axes[2].bar(by_sector.index, sector_win_rates)
        axes[2].set_title("セクター別勝率(%)")
        axes[2].set_ylabel("勝率(%)")
        axes[2].set_ylim(0, 100)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
    plt.close(fig)
    return fig


# ============================================================
# 5. 実行
# ============================================================
# 検証用サンプル銘柄（技術検証目的。特定銘柄の売買を推奨するものではない）
# 6セクター×5銘柄=30銘柄。統計的目安（トレード数30件以上）に近づけるための拡張。
SAMPLE_TICKERS = {
    # 銀行
    "8306.T": ("銀行", "三菱UFJフィナンシャル・グループ"),
    "8316.T": ("銀行", "三井住友フィナンシャルグループ"),
    "8411.T": ("銀行", "みずほフィナンシャルグループ"),
    "8309.T": ("銀行", "三井住友トラスト・ホールディングス"),
    "8355.T": ("銀行", "静岡銀行"),
    # 証券
    "8604.T": ("証券", "野村ホールディングス"),
    "8601.T": ("証券", "大和証券グループ本社"),
    "8628.T": ("証券", "松井証券"),
    "8616.T": ("証券", "東海東京フィナンシャル・ホールディングス"),
    "8703.T": ("証券", "楽天証券ホールディングス"),
    # 不動産
    "8801.T": ("不動産", "三井不動産"),
    "8802.T": ("不動産", "三菱地所"),
    "3289.T": ("不動産", "東急不動産ホールディングス"),
    "8804.T": ("不動産", "東京建物"),
    "8830.T": ("不動産", "住友不動産"),
    # 自動車
    "7203.T": ("自動車", "トヨタ自動車"),
    "7267.T": ("自動車", "ホンダ"),
    "7201.T": ("自動車", "日産自動車"),
    "7269.T": ("自動車", "スズキ"),
    "7211.T": ("自動車", "三菱自動車工業"),
    # 電機
    "6758.T": ("電機", "ソニーグループ"),
    "6501.T": ("電機", "日立製作所"),
    "6752.T": ("電機", "パナソニックホールディングス"),
    "6702.T": ("電機", "富士通"),
    "6503.T": ("電機", "三菱電機"),
    # 商社
    "8058.T": ("商社", "三菱商事"),
    "8031.T": ("商社", "三井物産"),
    "8001.T": ("商社", "伊藤忠商事"),
    "8053.T": ("商社", "住友商事"),
    "2768.T": ("商社", "双日"),
}

if __name__ == "__main__":
    trades = run_backtest_multi(SAMPLE_TICKERS, period="2y")

    print("=" * 60)
    print(f" 複数銘柄バックテスト結果（過去2年・{len(SAMPLE_TICKERS)}銘柄）")
    print("=" * 60)
    metrics = compute_metrics(trades)
    for k, v in metrics.items():
        print(f"  {k:20}: {v}")

    print("\n■ セクター別成績")
    print(compute_metrics_by_sector(trades).to_string())

    print("\n■ パターン別成績")
    print(compute_metrics_by_pattern(trades).to_string())

    print("\n■ 銘柄別トレード件数")
    if not trades.empty:
        print(trades.groupby(["sector", "ticker", "name"]).size().rename("trade_count").to_string())

    data_dir = Path(__file__).resolve().parents[1] / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    trades.to_csv(data_dir / "backtest_trades_multi.csv", index=False, encoding="utf-8-sig")
    plot_results(trades, save_path=str(data_dir / "backtest_multi.png"))
    print(f"\n[出力] {data_dir / 'backtest_trades_multi.csv'}")
    print(f"[出力] {data_dir / 'backtest_multi.png'}")
