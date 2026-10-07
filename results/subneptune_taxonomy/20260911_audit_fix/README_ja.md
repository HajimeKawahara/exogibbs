# provider監査への修正と再検証（2026-09-11）

`subneptune_taxonomy_provider_pr_audit_20260911_en.md` の修正依頼に対応した記録です。
実装は既存の未merge PR #185〜#189へ反映しています。

- #185: 全PR baseをCI対象とし、固定ExoEOS・実import先・native Hモデルを確認する独立jobを追加。skipを失敗扱いにします。
- #188: SCSS定義域・数値失敗を枝と初期値ごとに記録し、他の試行とcontinuationを継続。硫化物なしの反復ではSCSSを使わず、最終組成で判定します。
- #189: 保存成分量からH、S/N、C、SCSSの化学残差を再計算。metadataと不受理枝も確認し、実MELTSの再評価を明示的なoptionとして追加しました。

隣の `20260911/` にある元の5つの数値JSONは変更していません。
このdirectoryの `revalidated_archive.json` は、元の量を再評価した結果であり、
新しい平衡根を解き直した記録ではありません。標準実行は外部providerを必要とせず、
MELTS化学を `not_run` と明示します。

```sh
PYTHONPATH=src JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu \
  python results/subneptune_taxonomy/20260911/verify.py
```

実MELTSを含む再評価の完全な実行コマンド、実装・ExoEOS・日本語文書のcommit、
codeと出力のhashは `receipt.json` にあります。別環境ではcheckout、固定runtime、
worker Pythonのパスを実在するものへ変更してください。

再評価した反応残差の最大値は、Hが `1.85e-14`、S/N・Cが `7.11e-15`、
SCSSが `1.56e-15`、実MELTSの金属ありが `1.33e-14`、金属なしが `1.78e-14` です。
SCSSの不受理枝2件は再判定でも不受理です。元素を保存する0.001 molの反応方向摂動は、
hashを更新して残差の保存値を据え置いても検証に失敗することを回帰テストで確認しました。

金属なしの根へ金属ありの合金組成を挿入した値は、試験合金1 molあたり
`D/(RT)=-2.892303537`、同一予算の `G_present-G_absent` をRTで割った値は
`-0.225360740` です。金属なしの根はこのformalなモデルでも不安定であり、
金属相を抑制した診断用の根です。金属ありの根の全競合相に対する安定性や、
実験校正の成立を意味しません。

ローカルの全体テストは `1330 passed, 1 skipped, 24 warnings in 1265.77s (0:21:05)`、provider接続は `88 passed, 0 skipped` でした。
1件のskipは隔離checkoutにsetuptools-scm生成versionモジュールがないためです。
GitHubでも5つのPRの全checksが成功しています。

全体テストとprovider接続テストの全出力は `pytest_full.log` と
`pytest_provider.log`、集計は `receipt.json` に保存します。全体テストは今回追加した
全testファイルが揃ってから収集しています。HTMLビルドは成功し、既存警告86件、
新規ページの警告0件でした。日本語文書は `doc_ExoGibbs` の `33bc107` に更新し、
commit・push済みです。日本語PDF全体の生成は、XeLaTeXと必要packageがないため未検証です。

実験に基づくH/S/SCSS校正、同じMELTS母相への有限S接続、S+C共通モデル、
競合相・省略種の制約などは引き続き未完了の物理条件として保持しています。
