"""
日本株スコアリング・売買シグナルシステム — Streamlitダッシュボード
------------------------------------------------------------
タブ1: 銘柄コード入力 → signal_engineで判定 → チャート＋シグナル表示
タブ2: セグメント（業種）強弱
タブ3: 代表銘柄ランキング

※タブ2・3は本来の400銘柄評価ではなく、バックテスト検証用の28銘柄
  （銀行・証券・不動産・自動車・電機・商社の6業種）による簡易版。
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtest import SAMPLE_TICKERS
from src.data.fetch_yfinance import fetch_ohlcv, fetch_universe
from src.segment_scoring import segment_strength, top_n_per_segment, total_score
from src.signal_engine import add_indicators, evaluate, latest_signal

PLOTLY_FONT = dict(family="Yu Gothic, Meiryo, MS Gothic, sans-serif")

st.set_page_config(page_title="日本株スコアリング・売買シグナルシステム", layout="wide")


@st.cache_data(ttl=3600, show_spinner=False)
def load_signal(ticker: str, period: str):
    ohlcv = fetch_ohlcv(ticker, period=period)
    df = evaluate(add_indicators(ohlcv))
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def load_universe():
    df = fetch_universe(SAMPLE_TICKERS, cache=True)
    return total_score(df)


def render_signal_tab():
    st.subheader("銘柄シグナル判定")
    col_input, col_period, col_button = st.columns([2, 1, 1])
    with col_input:
        code = st.text_input("証券コード（4桁）", value="7203")
    with col_period:
        period = st.selectbox("取得期間", ["3mo", "6mo", "1y", "2y"], index=2)
    with col_button:
        st.write("")
        st.write("")
        run = st.button("判定する", use_container_width=True)

    if not run:
        return

    ticker = f"{code.strip()}.T"
    try:
        with st.spinner(f"{ticker} のデータを取得中..."):
            df = load_signal(ticker, period)
    except Exception as e:
        st.error(f"データ取得に失敗しました（{ticker}）: {e}")
        return

    signal = latest_signal(df)
    action_label = {"BUY": "🟢 BUY", "SELL": "🔴 SELL", "HOLD": "⚪ HOLD"}[signal["action"]]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("アクション", action_label)
    c2.metric("終値", f"{signal['close']:.1f}")
    c3.metric("RSI", signal["rsi"] if signal["rsi"] is not None else "-")
    c4.metric("スコア", signal["score"] if signal["score"] is not None else "-")

    if signal["patterns"]:
        st.info(f"成立パターン: {', '.join(signal['patterns'])}")
    else:
        st.caption("成立パターンなし")

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        name="株価",
    ))
    for col, name in [("ma5", "MA5"), ("ma25", "MA25"), ("ma75", "MA75")]:
        fig.add_trace(go.Scatter(x=df.index, y=df[col], name=name, line=dict(width=1)))

    buy_points = df[df["BUY"]]
    sell_points = df[df["SELL"]]
    if not buy_points.empty:
        fig.add_trace(go.Scatter(
            x=buy_points.index, y=buy_points["low"] * 0.98, mode="markers",
            marker=dict(symbol="triangle-up", color="green", size=11), name="BUY",
        ))
    if not sell_points.empty:
        fig.add_trace(go.Scatter(
            x=sell_points.index, y=sell_points["high"] * 1.02, mode="markers",
            marker=dict(symbol="triangle-down", color="red", size=11), name="SELL",
        ))

    fig.update_layout(
        title=f"{ticker} 日足チャート",
        xaxis_title="日付", yaxis_title="価格",
        font=PLOTLY_FONT,
        xaxis_rangeslider_visible=False,
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_segment_tab():
    st.subheader("業種別 強弱スコア")
    st.caption(
        "※本来の400銘柄評価（大型株/中型株/小型成長株）ではなく、"
        "バックテスト検証用の28銘柄（銀行・証券・不動産・自動車・電機・商社の6業種）による簡易版です。"
    )
    with st.spinner("データを取得中...（初回は数分かかる場合があります）"):
        try:
            scored = load_universe()
        except Exception as e:
            st.error(f"データ取得に失敗しました: {e}")
            return

    strength = segment_strength(scored)
    fig = go.Figure(go.Bar(
        x=strength.index, y=strength["strength"],
        marker_color=["#d62728" if v < 45 else "#2ca02c" if v > 55 else "#7f7f7f" for v in strength["strength"]],
    ))
    fig.update_layout(
        title="業種別 強弱偏差値（50が中立）",
        yaxis_title="偏差値", font=PLOTLY_FONT, height=400,
    )
    fig.add_hline(y=50, line_dash="dash", line_color="gray")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(strength, use_container_width=True)


def render_ranking_tab():
    st.subheader("業種別 代表銘柄ランキング")
    st.caption("※検証用28銘柄の中でのスコア順ランキングです。")
    with st.spinner("データを取得中...（初回は数分かかる場合があります）"):
        try:
            scored = load_universe()
        except Exception as e:
            st.error(f"データ取得に失敗しました: {e}")
            return

    tops = top_n_per_segment(scored, n=10)
    for sector, sub in tops.items():
        st.markdown(f"**{sector}**")
        view = sub[["code", "name", "score", "growth_pct", "roe_pct", "per", "div_yield_pct", "tech_score"]].copy()
        view.columns = ["コード", "銘柄", "総合スコア", "成長率%", "ROE%", "PER", "配当利回り%", "テクニカル"]
        st.dataframe(view, use_container_width=True, hide_index=True)


st.title("日本株スコアリング・売買シグナルシステム")
st.caption("本システムは投資判断の補助を目的としています。シグナルの的中・利益を保証しません。")

tab1, tab2, tab3 = st.tabs(["銘柄シグナル判定", "セグメント強弱", "代表銘柄ランキング"])
with tab1:
    render_signal_tab()
with tab2:
    render_segment_tab()
with tab3:
    render_ranking_tab()
