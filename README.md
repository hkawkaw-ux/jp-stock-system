# 日本株スコアリング・売買シグナルシステム

日本株の売買判断を支援する2階建てツール群。
「**どのセグメント・銘柄が強いか（選別）**」→「**いつ売買するか（タイミング）**」の
2段階で絞り込む。

> ⚠️ 本システムは投資判断の**補助**を目的とする。シグナルの的中・利益を保証しない。
> 実データ適用前に**バックテスト検証**を必須とし、損切りルールの徹底を前提とすること。

---

## 構成する2エンジン

| エンジン | ファイル | 時間軸 | 役割 |
|---|---|---|---|
| セグメント評価 | `src/segment_scoring.py` | 数週間〜数ヶ月 | 強いセグメント・上位銘柄の選別（11項目100点） |
| 短期シグナル | `src/signal_engine.py` | 数日〜数週間 | 選んだ銘柄の売買タイミング判定（BUY/SELL/HOLD） |

## 2段階連携イメージ

```
① セグメント評価  segment_scoring.py
   約400銘柄を11項目で採点 → セグメント強弱 + 代表銘柄ランキング
        ▼ 上位銘柄を渡す
② 短期シグナル  signal_engine.py
   選定銘柄の日足を判定 → BUY / SELL / HOLD + 強度スコア
        ▼
   売買候補リスト（銘柄 + タイミング + 根拠）
```

---

## セットアップ

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# APIキーを設定
cp .env.example .env             # 中身を編集
```

## 動作確認（デモデータ）

```bash
python src/signal_engine.py      # 短期シグナルのデモ
python src/segment_scoring.py    # セグメント評価のデモ（CSV出力）
```

---

## 開発ロードマップ（優先度順）

| No | タスク | 内容 | 優先度 |
|---|---|---|---|
| 1 | 実データ接続 | `src/data/fetch_yfinance.py` で `make_demo_*` を実データに置換（yfinance採用、APIキー不要・約15分遅延） | ★★★ |
| 2 | パラメータ外出し | 閾値・配点を `config/params.yaml` へ分離 | ★★★ |
| 3 | バックテスト | `src/backtest.py` で勝率・期待値・最大DDを算出 | ★★★ |
| 4 | 2段階連携 | `src/pipeline.py` で評価→シグナルを直列化 | ★★☆ |
| 5 | UI化 | `app/dashboard.py`（Streamlit）で可視化 | ★★☆ |
| 6 | テスト整備 | `tests/` で指標計算の回帰テスト | ★★☆ |
| 7 | 通知連携 | 日次バッチでTeams/メール通知 | ★☆☆ |
| 8 | パラメータ最適化 | 実データ分布に合わせ正規化閾値を調整 | ★☆☆ |

詳細は `docs/開発引継ぎ資料_Claude_Code向け.docx` を参照。

## ディレクトリ構成（推奨）

```
jp-stock-system/
├─ README.md
├─ requirements.txt
├─ .env.example
├─ config/params.yaml
├─ src/
│   ├─ data/fetch_yfinance.py    # 実データ接続（yfinance）
│   ├─ signal_engine.py          # 既存
│   ├─ segment_scoring.py        # 既存
│   ├─ pipeline.py               # 実装対象
│   └─ backtest.py               # 実装対象
├─ app/dashboard.py              # 実装対象
├─ tests/
└─ docs/                         # 設計書・引継ぎ資料
```

## ライセンス・留意

- `yfinance` は非公式・商用利用不可。社内利用可否を確認すること。
- APIキー・トークンは `.env` で管理し **Git にコミットしない**。
- Claude Code の社内利用可否・データ取扱いは情報システム部門の承認を得ること。
