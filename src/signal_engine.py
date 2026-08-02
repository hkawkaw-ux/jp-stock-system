"""
日本株 短期売買シグナル判定エンジン (プロトタイプ)
------------------------------------------------------------
前回提示した手法をルール化し、買い/売りシグナルを自動判定する。
入力: 日足データ (date, open, high, low, close, volume) の DataFrame
出力: 各種指標 + シグナル + パターン判定 (A/B/C)

実データを使う場合は fetch_data() を
  - J-Quants API (公式・無料枠あり)
  - yfinance (証券コード + ".T" 例: "7203.T")
  - 証券会社(SBI/楽天)のCSVエクスポート
のいずれかに差し替えるだけで動く。
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# ============================================================
# 0. パラメータ管理（config/params.yaml から閾値・配点を読み込む）
# ============================================================
DEFAULT_PARAMS = {
    "moving_average": {"short": 5, "mid": 25, "long": 75},
    "rsi": {"period": 14, "overbought": 75, "oversold": 30},
    "volume": {"surge_prev": 1.5, "surge_avg20": 2.0},
    "adjustment": {"min_days": 10},
    "breakout": {"lookback": 20},
    "stoploss": {"ma25_deviation": -0.05},
    "patterns": {
        "A_rsi": [50, 65],
        "B_rsi": [35, 45],
        "B_ma25_dev": 0.03,
        "C_rsi": [55, 70],
    },
}

DEFAULT_PARAMS_PATH = Path(__file__).resolve().parents[1] / "config" / "params.yaml"


def load_params(path=None) -> dict:
    """
    config/params.yaml の signal セクションを読み込む。
    ファイルが無い、または一部項目が未定義の場合は DEFAULT_PARAMS で補う。
    """
    path = Path(path) if path else DEFAULT_PARAMS_PATH
    if not path.exists():
        return {k: (dict(v) if isinstance(v, dict) else v) for k, v in DEFAULT_PARAMS.items()}

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    signal_params = data.get("signal", {})

    merged = {k: (dict(v) if isinstance(v, dict) else v) for k, v in DEFAULT_PARAMS.items()}
    for key, value in signal_params.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    return merged


# ============================================================
# 1. テクニカル指標の計算
# ============================================================
def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def macd(close: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def bollinger(close: pd.Series, period=20, k=2):
    ma = close.rolling(period).mean()
    sd = close.rolling(period).std()
    return ma + k * sd, ma, ma - k * sd


def add_indicators(df: pd.DataFrame, params: dict = None) -> pd.DataFrame:
    p = params or load_params()
    ma = p["moving_average"]

    df = df.copy()
    # 移動平均
    df["ma5"] = df["close"].rolling(ma["short"]).mean()
    df["ma25"] = df["close"].rolling(ma["mid"]).mean()
    df["ma75"] = df["close"].rolling(ma["long"]).mean()
    # 出来高
    df["vol_ma20"] = df["volume"].rolling(20).mean()
    df["vol_ratio"] = df["volume"] / df["vol_ma20"]          # 出来高 / 20日平均
    df["vol_vs_prev"] = df["volume"] / df["volume"].shift(1)  # 前日比
    # RSI
    df["rsi"] = rsi(df["close"], p["rsi"]["period"])
    # MACD
    df["macd"], df["macd_sig"], df["macd_hist"] = macd(df["close"])
    # ボリンジャー
    df["bb_up"], df["bb_mid"], df["bb_low"] = bollinger(df["close"])
    df["bb_width"] = (df["bb_up"] - df["bb_low"]) / df["bb_mid"]  # バンド幅(収縮判定)
    # 直近高値(ブレイク判定用: 当日を除く)
    df["hh20"] = df["high"].rolling(p["breakout"]["lookback"]).max().shift(1)
    # 調整日数: 直近高値からの経過日数
    df["days_since_high"] = _days_since_recent_high(df["close"], lookback=25)
    return df


def _days_since_recent_high(close: pd.Series, lookback=25):
    out = np.full(len(close), np.nan)
    vals = close.values
    for i in range(len(vals)):
        start = max(0, i - lookback + 1)
        window = vals[start:i + 1]
        peak_idx = np.argmax(window)
        out[i] = (len(window) - 1) - peak_idx  # 高値からの経過日数
    return out


# ============================================================
# 2. シグナル / パターン判定
# ============================================================
def evaluate(df: pd.DataFrame, params: dict = None) -> pd.DataFrame:
    p = params or load_params()
    min_days = p["adjustment"]["min_days"]
    a_lo, a_hi = p["patterns"]["A_rsi"]
    b_lo, b_hi = p["patterns"]["B_rsi"]
    b_dev = p["patterns"]["B_ma25_dev"]
    c_lo, c_hi = p["patterns"]["C_rsi"]

    df = df.copy()
    c = df["close"]

    up_trend = (c > df["ma25"]) & (df["ma25"] > df["ma75"])
    golden5_25 = (df["ma5"] > df["ma25"]) & (df["ma5"].shift(1) <= df["ma25"].shift(1))
    breakout = c > df["hh20"]
    vol_surge = df["vol_vs_prev"] >= p["volume"]["surge_prev"]
    vol_big = df["vol_ratio"] >= p["volume"]["surge_avg20"]
    macd_up = (df["macd"] > df["macd_sig"]) & (df["macd"].shift(1) <= df["macd_sig"].shift(1))

    # --- パターンA: 成長株順張り(ブレイクアウト) ---
    df["pattern_A"] = (
        (c > df["ma25"])
        & df["rsi"].between(a_lo, a_hi)
        & vol_big
        & (df["days_since_high"] >= min_days)
        & breakout
    )

    # --- パターンB: 押し目買い ---
    df["pattern_B"] = (
        up_trend
        & df["rsi"].between(b_lo, b_hi)
        & (abs(c - df["ma25"]) / df["ma25"] <= b_dev)  # 25日線付近
        & (df["vol_ratio"] < 1.0)                      # 出来高減少
    )

    # --- パターンC: 機関投資家追随型(最強候補) ---
    df["pattern_C"] = (
        (c > df["ma75"])
        & (c > df["ma25"])
        & (df["days_since_high"] >= min_days)
        & vol_surge
        & df["rsi"].between(c_lo, c_hi)
        & breakout
    )

    # --- 総合 買いシグナル ---
    df["BUY"] = df["pattern_A"] | df["pattern_B"] | df["pattern_C"] | (golden5_25 & up_trend & vol_surge)

    # --- 売り / 手仕舞いシグナル ---
    df["SELL"] = (
        (df["rsi"] >= p["rsi"]["overbought"])                                       # 買われ過ぎ過熱
        | ((df["ma5"] < df["ma25"]) & (df["ma5"].shift(1) >= df["ma25"].shift(1)))   # デッドクロス
        | (c < df["ma25"] * (1 + p["stoploss"]["ma25_deviation"]))                   # 25日線割れ(損切)
        | ((df["macd"] < df["macd_sig"]) & (df["macd"].shift(1) >= df["macd_sig"].shift(1)) & (df["rsi"] > 60))
    )

    # --- シグナル強度スコア (0-100) ---
    score = (
        up_trend.astype(int) * 20
        + (df["vol_ratio"].clip(0, 3) / 3 * 25)
        + (df["rsi"].between(50, 70).astype(int) * 15)
        + (df["days_since_high"] >= min_days).astype(int) * 15
        + breakout.astype(int) * 15
        + macd_up.astype(int) * 10
    )
    df["score"] = score.round(0)
    return df


def latest_signal(df: pd.DataFrame) -> dict:
    row = df.iloc[-1]
    patterns = [p for p in ["A", "B", "C"] if row.get(f"pattern_{p}", False)]
    if row["BUY"]:
        action = "BUY"
    elif row["SELL"]:
        action = "SELL"
    else:
        action = "HOLD"
    return {
        "date": str(row.name.date()) if hasattr(row.name, "date") else str(row.name),
        "close": round(float(row["close"]), 1),
        "action": action,
        "patterns": patterns,
        "score": None if pd.isna(row["score"]) else int(row["score"]),
        "rsi": None if pd.isna(row["rsi"]) else round(float(row["rsi"]), 1),
        "vol_ratio": None if pd.isna(row["vol_ratio"]) else round(float(row["vol_ratio"]), 2),
    }


# ============================================================
# 3. デモ用: 擬似株価データ生成 (実データが無い環境用)
# ============================================================
def make_demo_data(seed=7, n=180):
    rng = np.random.default_rng(seed)
    price = [1000.0]
    vol = []
    trend = 0.0008
    for i in range(n - 1):
        # 調整→急騰のサイクルを人工的に作る
        if 60 <= i < 75:      # 調整局面
            drift = -0.004
        elif 75 <= i < 95:    # 急騰局面
            drift = 0.012
        else:
            drift = trend
        ret = drift + rng.normal(0, 0.012)
        price.append(price[-1] * (1 + ret))
    price = np.array(price)
    base_vol = 1_000_000
    for i in range(n):
        v = base_vol * (1 + rng.normal(0, 0.3))
        if 75 <= i < 82:  # 急騰時の出来高急増
            v *= 2.5
        vol.append(max(v, 100_000))
    dates = pd.date_range("2026-01-06", periods=n, freq="B")
    high = price * (1 + rng.uniform(0, 0.01, n))
    low = price * (1 - rng.uniform(0, 0.01, n))
    openp = price * (1 + rng.uniform(-0.005, 0.005, n))
    return pd.DataFrame(
        {"open": openp, "high": high, "low": low, "close": price, "volume": vol},
        index=dates,
    )


if __name__ == "__main__":
    df = make_demo_data()
    df = add_indicators(df)
    df = evaluate(df)

    print("=" * 60)
    print(" 最新シグナル")
    print("=" * 60)
    for k, v in latest_signal(df).items():
        print(f"  {k:12}: {v}")

    print("\n" + "=" * 60)
    print(" 直近の買い/売りシグナル発生日")
    print("=" * 60)
    sig = df[(df["BUY"]) | (df["SELL"])].tail(12)
    for idx, r in sig.iterrows():
        act = "🟢BUY " if r["BUY"] else "🔴SELL"
        pats = "".join(p for p in ["A", "B", "C"] if r[f"pattern_{p}"])
        print(f"  {idx.date()}  {act}  終値{r['close']:.0f}  RSI{r['rsi']:.0f}  "
              f"出来高比{r['vol_ratio']:.1f}  score{r['score']:.0f}  {pats}")
