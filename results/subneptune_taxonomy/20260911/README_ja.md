# 局所化学の検証記録（2026-09-11）

この記録は実装PR #185–#188に対応します。既存の `completed_dry`、
`source_full`、`gas_exchange` の結果は置き換えていません。

| 記録 | 検証した内容 |
| --- | --- |
| `hydrogen.json` | 共通H2標準・有限H/He・native Fe–Si–O–Hの条件付き局所解 |
| `melts_present.json` | 実MELTSを使った有限量のmelt–metal–gas局所解 |
| `melts_absent.json` | 同じ有限予算でmetalを厳密に除いた局所解 |
| `sulfur_carbon_source.json` | 固定GCEのS/N版37成分とCarbon版26成分の原式再現、各2温度 |
| `sulfide.json` | 人工的なSCSSを使った有限Fe/S収支と硫化物出現の数値対照 |

`receipt.json` は出力のSHA256、実行時のコードcommit、検証条件を保持します。
各JSONにも入力、成分基底、実際のimport先、物理モデルの制限を記録しています。
成分量から元素収支を再計算する検証は、リポジトリのルートで次を実行します。

```sh
python results/subneptune_taxonomy/20260911/verify.py
```

再計算用の環境は `PYTHONPATH=src:/path/to/exoeos/src`、
`JAX_ENABLE_X64=1`、`JAX_PLATFORMS=cpu` です。H、S/C原式、SCSS対照はそれぞれ
`examples/metal_silicate/hydrogen.py`、`sulfur_source.py`、`sulfide.py` を実行します。
実MELTSのコマンドとruntime/interpreterのパスは各結果の
`provenance.command` に残しています。別環境ではそのパスを実在する固定runtimeへ
変更してください。外部backendのダウンロードは実行処理に含みません。

既存全体テストは `1223 passed, 1 skipped`、今回の追加テストは
`70 passed` でした。既存suiteは新規testの作成前に収集を開始したため、
新規70件を別途まとめて検証しています。skipは隔離checkoutにsetuptools-scmの
生成versionファイルがないためです。テストの全出力もここに保存しています。
HTMLは `./update_doc.sh -D sphinx_gallery_conf.plot_gallery=0` で再構築しました。
日本語文書は `doc_ExoGibbs` のcommit `7471341` に対応します。

## まだ成立していない物理判定

これらの残差は、実装した式の局所的整合性を検証します。2350 K / 1 barの
MELTS連成は旧H2則などの適用域外を含む条件付きモデルであり、実験校正済みの
惑星予測ではありません。metalあり・なしの両方に局所根があっても、安定枝の
選択やヒステリシスの証拠にはなりません。

残った条件は、2025年H2のホスト別数表と比較、異なる相の標準状態の独立な校正、
最終ホスト組成での競合固相・不在nonideal alloyの安定性、実測SCSSと元のS/C
分配校正、S+Cの共通自由エネルギー、graphite/carbide・metal N/nitrideの制約、
Nを省略する定量的上限、ガス種の収束です。未取得の係数を一律倍率で補っていません。
ExoInventoryの惑星圧力・総量閉包、ExoJAXの不透明度/RCEは今回の変更に含みません。
