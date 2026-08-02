"""
日本株 セグメント別スコアリング試作エンジン
============================================================
・大型株150 / 中型株150 / 小型成長株100 = 約400銘柄を評価
・11項目・100点満点で採点（配点はユーザー指定どおり）
・任意項目（信用倍率・アナリスト評価）が欠損しても比例配分で100点換算
・セグメント自体の強弱を偏差値化して相対表示
・各セグメント代表10社をスコア順にランキング

※本試作はデモ用擬似データ（make_demo_universe）で動作確認する。
  実運用時は load_universe() を J-Quants / 財務API 等に差し替える。
"""

import numpy as np
import pandas as pd

# ============================================================
# 配点定義（ユーザー指定）
# ============================================================
WEIGHTS = {
    "growth":      20,  # 業績成長率
    "op_margin":   10,  # 営業利益率
    "roe":         10,  # ROE
    "per":         10,  # PER（低いほど高評価）
    "pbr":          5,  # PBR（低いほど高評価）
    "div_yield":   10,  # 配当利回り
    "div_growth":   5,  # 増配実績
    "technical":   15,  # テクニカル
    "volume":       5,  # 出来高
    "margin_ratio": 5,  # 信用倍率（任意）
    "analyst":      5,  # アナリスト評価（任意）
}
OPTIONAL = ["margin_ratio", "analyst"]      # 欠損許容項目
TOTAL_MAX = sum(WEIGHTS.values())            # 100

SEGMENTS = {"大型株": 150, "中型株": 150, "小型成長株": 100}


# ============================================================
# 1. 各指標を 0～1 の評価スコアへ正規化
#    （項目ごとに「高いほど良い / 低いほど良い / 最適レンジ」を定義）
# ============================================================
def _clip01(x):
    return np.clip(x, 0.0, 1.0)


def score_metrics(df: pd.DataFrame) -> pd.DataFrame:
    s = pd.DataFrame(index=df.index)

    # --- 高いほど良い（下限0・上限で頭打ち） ---
    s["growth"]     = _clip01(df["growth_pct"] / 30)          # 成長率30%で満点
    s["op_margin"]  = _clip01(df["op_margin_pct"] / 25)       # 営業利益率25%で満点
    s["roe"]        = _clip01(df["roe_pct"] / 20)             # ROE20%で満点
    s["div_yield"]  = _clip01(df["div_yield_pct"] / 4.0)      # 配当利回り4%で満点
    s["div_growth"] = _clip01(df["div_growth_years"] / 10)    # 連続増配10年で満点
    s["technical"]  = _clip01(df["tech_score"] / 100)         # 前回エンジンのscore(0-100)
    s["volume"]     = _clip01((df["vol_ratio"] - 0.5) / 2.0)  # 出来高比0.5→2.5で0→1
    s["analyst"]    = _clip01((df["analyst_rating"] - 1) / 4) # 1-5段階を0-1へ

    # --- 低いほど良い（バリュエーション） ---
    #   PER: 8倍以下=満点, 40倍以上=0
    s["per"] = _clip01((40 - df["per"]) / (40 - 8))
    #   PBR: 0.8倍以下=満点, 5倍以上=0
    s["pbr"] = _clip01((5.0 - df["pbr"]) / (5.0 - 0.8))

    # --- 最適レンジ型（信用倍率）---
    #   信用倍率は低い（売り長=逆日歩妙味 / 需給良好）ほど加点、高倍率は減点
    #   1倍以下=満点, 6倍以上=0
    s["margin_ratio"] = _clip01((6.0 - df["margin_ratio"]) / (6.0 - 1.0))

    return s


# ============================================================
# 2. 重み付け合計 → 100点満点スコア（任意項目欠損は比例配分）
# ============================================================
def total_score(df: pd.DataFrame, use_optional=True) -> pd.DataFrame:
    m = score_metrics(df)
    out = df.copy()

    active = list(WEIGHTS.keys())
    if not use_optional:
        active = [k for k in active if k not in OPTIONAL]

    # 欠損している任意項目は自動的に母数から除外
    available = [k for k in active if not m[k].isna().all()]
    denom = sum(WEIGHTS[k] for k in available)

    raw = sum(m[k].fillna(0) * WEIGHTS[k] for k in available)
    # denom 点満点 → 100 点換算
    out["score"] = (raw / denom * 100).round(1)

    # カテゴリ別内訳も保持
    out["s_growth_block"] = ((m["growth"]*20 + m["op_margin"]*10 + m["roe"]*10)).round(1)
    out["s_value_block"]  = ((m["per"]*10 + m["pbr"]*5 + m["div_yield"]*10 + m["div_growth"]*5)).round(1)
    out["s_tech_block"]   = ((m["technical"]*15 + m["volume"]*5 + m["margin_ratio"].fillna(0)*5)).round(1)
    out["denom"] = denom
    return out


# ============================================================
# 3. セグメント強弱（代表銘柄を集約→偏差値化）
# ============================================================
def segment_strength(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("segment")
    agg = pd.DataFrame({
        "avg_score":   g["score"].mean(),
        "momentum_1m": g["ret_1m_pct"].mean(),
        "momentum_3m": g["ret_3m_pct"].mean(),
        "inflow":      g["vol_ratio"].mean(),
        "avg_roe":     g["roe_pct"].mean(),
        "avg_growth":  g["growth_pct"].mean(),
    })

    # 3軸を偏差値化して合成（モメンタム・資金流入・ファンダ健全性）
    def z(col):
        v = agg[col]
        return (v - v.mean()) / (v.std(ddof=0) + 1e-9)

    strength_z = (
        0.40 * (z("momentum_1m") * 0.4 + z("momentum_3m") * 0.6)  # モメンタム
        + 0.30 * z("inflow")                                       # 資金流入
        + 0.30 * (z("avg_roe") * 0.5 + z("avg_growth") * 0.5)      # ファンダ健全性
    )
    # 偏差値(50±10)へ変換
    agg["strength"] = (50 + strength_z * 10).round(1)
    agg = agg.sort_values("strength", ascending=False)
    return agg.round(2)


def top_n_per_segment(df: pd.DataFrame, n=10, segments=None) -> dict:
    """
    segments省略時は df 内に実際に存在する segment 値でループする。
    （SEGMENTS定数は既定の3区分専用のため、業種名など任意の区分にも対応できるようにしている）
    """
    seg_list = segments if segments is not None else df["segment"].unique()
    result = {}
    for seg in seg_list:
        sub = df[df["segment"] == seg].sort_values("score", ascending=False).head(n)
        result[seg] = sub
    return result


# ============================================================
# 4. デモ用 擬似ユニバース生成（実データが無い環境用）
# ============================================================
def make_demo_universe(seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    code = 1301
    # セグメントごとに“性格”を変えて生成
    profiles = {
        # segment:       growth, opm,  roe,  per,  pbr, dy,  divg, tech, vol, mgn, analyst, ret1, ret3
        "大型株":      dict(gm=6,  op=12, roe=9,  per=16, pbr=1.3, dy=2.6, dg=6, tc=55, vr=1.0, mg=2.5, an=3.6, r1=1.5, r3=4),
        "中型株":      dict(gm=10, op=11, roe=11, per=15, pbr=1.5, dy=2.2, dg=4, tc=58, vr=1.2, mg=3.0, an=3.5, r1=3,   r3=8),
        "小型成長株":  dict(gm=22, op=14, roe=13, per=28, pbr=2.8, dy=0.8, dg=2, tc=63, vr=1.5, mg=3.6, an=3.4, r1=5,   r3=14),
    }
    names = ["ホールディングス","製作所","商事","電機","化学","運輸","システムズ","フーズ",
             "メディカル","エナジー","マテリアル","ネットワーク","バイオ","ロジ","テック"]
    for seg, cnt in SEGMENTS.items():
        p = profiles[seg]
        for i in range(cnt):
            rows.append({
                "code": code,
                "name": f"{seg[:2]}{names[i % len(names)]}{i+1:03d}",
                "segment": seg,
                "growth_pct":      max(-10, rng.normal(p["gm"], 8)),
                "op_margin_pct":   max(0,   rng.normal(p["op"], 5)),
                "roe_pct":         max(-5,  rng.normal(p["roe"], 5)),
                "per":             max(4,   rng.normal(p["per"], 6)),
                "pbr":             max(0.4, rng.normal(p["pbr"], 0.7)),
                "div_yield_pct":   max(0,   rng.normal(p["dy"], 1.0)),
                "div_growth_years":max(0, int(rng.normal(p["dg"], 3))),
                "tech_score":      float(np.clip(rng.normal(p["tc"], 18), 0, 100)),
                "vol_ratio":       max(0.2, rng.normal(p["vr"], 0.5)),
                "margin_ratio":    max(0.3, rng.normal(p["mg"], 1.8)),
                "analyst_rating":  float(np.clip(rng.normal(p["an"], 0.8), 1, 5)),
                "ret_1m_pct":      rng.normal(p["r1"], 6),
                "ret_3m_pct":      rng.normal(p["r3"], 12),
            })
            code += 1
    return pd.DataFrame(rows)


# ============================================================
# 5. 実行
# ============================================================
if __name__ == "__main__":
    uni = make_demo_universe()
    scored = total_score(uni, use_optional=True)

    print("=" * 64)
    print(f" 評価ユニバース: {len(scored)}銘柄  "
          f"(大型{SEGMENTS['大型株']}/中型{SEGMENTS['中型株']}/小型{SEGMENTS['小型成長株']})")
    print(f" 採点: 11項目 100点満点 / 実効母数 denom={scored['denom'].iloc[0]}")
    print("=" * 64)

    strength = segment_strength(scored)
    print("\n■ セグメント強弱（偏差値・降順）")
    print(strength[["strength", "avg_score", "momentum_1m", "momentum_3m",
                    "inflow", "avg_roe", "avg_growth"]].to_string())

    tops = top_n_per_segment(scored, n=10)
    for seg, sub in tops.items():
        st = strength.loc[seg, "strength"]
        print("\n" + "=" * 64)
        print(f"■ {seg}  代表10社ランキング（セグメント強弱 偏差値 {st}）")
        print("=" * 64)
        view = sub[["code", "name", "score", "growth_pct", "roe_pct",
                    "per", "div_yield_pct", "tech_score", "vol_ratio"]].copy()
        view.columns = ["コード", "銘柄", "総合", "成長%", "ROE%",
                        "PER", "配当%", "テク", "出来高比"]
        print(view.to_string(index=False,
              formatters={"成長%": "{:.1f}".format, "ROE%": "{:.1f}".format,
                          "PER": "{:.1f}".format, "配当%": "{:.1f}".format,
                          "テク": "{:.0f}".format, "出来高比": "{:.2f}".format}))

    # CSV出力（全銘柄スコア）
    out_cols = ["code","name","segment","score","s_growth_block","s_value_block",
                "s_tech_block","growth_pct","op_margin_pct","roe_pct","per","pbr",
                "div_yield_pct","div_growth_years","tech_score","vol_ratio",
                "margin_ratio","analyst_rating","ret_1m_pct","ret_3m_pct"]
    scored.sort_values(["segment","score"], ascending=[True,False])[out_cols]\
        .to_csv("/mnt/user-data/outputs/segment_scores.csv", index=False, encoding="utf-8-sig")
    print("\n[CSV] /mnt/user-data/outputs/segment_scores.csv を出力しました")
