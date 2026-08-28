# 手書き数字ブラウザー推論アプリ 要求定義

## 1. 文書の位置づけ

この文書は、`mnist/plot_digits_classification.ipynb` で学習したモデルを使うブラウザー推論アプリの要求定義である。

要求の優先順位は次のとおりとする。

1. ユーザーが承認した要求とデフォルト選択
2. この要求定義
3. リポジトリ内の既存実装・文書
4. 参照する公式資料

承認されていない機能は追加しない。

## 2. 確認済みの前提

- 対象 Notebook は `mnist/plot_digits_classification.ipynb` である。
- Notebook が使用するデータは、MNIST 28×28 ではなく `sklearn.datasets.load_digits()` の8×8画像である。
- 入力は64特徴で、各特徴の値域は0から16である。
- 分類器は `sklearn.svm.SVC(gamma=0.001)` である。
- 学習データとテストデータは `test_size=0.5, shuffle=False` で分割される。
- `executed-notebooks/mnist/plot_digits_classification.executed.ipynb` に保存された分類レポートと混同行列では、テスト899件中871件正解、精度約96.9%である。
- 現在の分類器は確率推定を有効化していない。決定値や投票数を確率として表示してはならない。
- `setup.ps1 -RunNotebooks` と `setup.sh --run-notebooks` は既存の Notebook 自動実行手段である。

## 3. スコープ

### 3.1 機能要件

| ID | 要求 |
|---|---|
| FR-01 | `mnist/plot_digits_predition.html` をPCまたはMacのブラウザーで直接開けること。ファイル名は指定どおり `predition` とする。 |
| FR-02 | 画面左側に手書き数字の描画領域を表示し、マウス、タッチ、ペンの主ポインターで描画できること。 |
| FR-03 | 1ストロークの終了時に自動推論し、明示的な「推論」ボタンでも再実行できること。 |
| FR-04 | 描画を切り出して中央配置し、8×8、値域0–16、64特徴へ変換すること。描画ピクセルを変更した完了ストロークがない状態は推論しないこと。 |
| FR-05 | Notebook と同じRBFカーネル、学習済み支持ベクトル、双対係数、切片、クラス別支持ベクトル数を使い、one-vs-one投票で0–9を分類すること。 |
| FR-06 | 画面右側に予測数字、クラス別投票数、推論時間、8×8入力プレビュー、テスト精度、予測クラスのprecision・recall・F1・support、支持ベクトル総数を表示すること。 |
| FR-07 | 投票数および決定値は確率ではないことを画面上で明示すること。 |
| FR-08 | 「クリア」操作で描画と現在の推論結果を初期化できること。 |
| FR-09 | 起動時に埋め込みモデルを検査し、欠落・JSON不正・形状不正の場合は推論を無効化すること。 |
| FR-10 | モデルが利用できない場合、対象 Notebook の全コードセルを上から実行する手順、既存のWindows/macOS自動実行手段、実行後の再読み込み操作を表示すること。 |
| FR-11 | 静的HTMLからローカルプロセスを直接起動するボタンは設けないこと。代わりに安全な手順表示と再読み込みボタンを設けること。 |
| FR-12 | Notebook の最後の既存空コードセルが、学習済みモデルと評価指標をHTML内の専用区間へ埋め込み、再実行で更新できること。 |

### 3.2 非機能要件

| ID | 要求 |
|---|---|
| NFR-01 | 実行時にCDN、外部ネットワーク、Webサーバー、Pythonランタイムを必要としない単一HTMLとすること。 |
| NFR-02 | React、ONNX、TensorFlow.js、Pyodideなどの新規実行時依存を追加しないこと。 |
| NFR-03 | デスクトップでは左右2カラム、狭い画面では縦1カラムにすること。 |
| NFR-04 | ライト・ダークテーマに対応し、Clawpilotテーマ変数だけを色指定に使用すること。 |
| NFR-05 | 描画領域に説明を付け、操作ボタンをキーボード操作可能にし、推論結果を支援技術へ通知すること。 |
| NFR-06 | Notebook の既存セル、学習条件、既存出力処理を変更せず、最後の空セルだけへエクスポート処理を追加すること。 |
| NFR-07 | 同じモデルから生成するHTML内容は決定的であること。生成日時など、実行ごとに不要な差分が生じる値を埋め込まないこと。 |
| NFR-08 | Windowsで実測していないmacOS固有動作を「検証済み」と表現しないこと。 |

## 4. モデル埋め込み契約

HTMLには、開始マーカーと終了マーカーに囲まれた `application/json` のモデルデータ区間を1つ設ける。

最小データは次を含む。

- スキーマバージョン
- モデル種別とカーネル
- クラス配列
- 特徴数
- gamma
- 支持ベクトル
- 双対係数
- 切片
- クラス別支持ベクトル数
- テスト件数と正解数
- テスト精度
- クラス別precision・recall・F1・support
- scikit-learnのバージョン

起動時には少なくとも次を検証する。

- スキーマバージョンが対応値であること
- クラスが0–9の10件であること
- 特徴数が64であること
- gammaが正の有限数であること
- 支持ベクトルの各行が64件であること
- 双対係数、切片、クラス別支持ベクトル数の形状がSVC多クラスモデルとして整合すること
- 支持ベクトル、双対係数、切片、gammaなど、推論に使う全数値が `NaN`、正負の無限大、`null` ではないこと

## 5. 入力前処理

1. 高DPI表示でも座標がずれない内部解像度で描画する。
2. 白背景に黒い線で利用者へ表示する。
3. 背景との差分からインク領域を検出する。
4. 描画ピクセルを変更した完了ストロークがなければ推論を停止し、数字の描画を促す。
5. インク領域を正方形へ余白付きで収め、中央配置する。
6. 面積平均相当で8×8へ縮小する。
7. 明度を反転して0–16へ写像し、行優先の64特徴へ平坦化する。

## 6. 推論と統計

RBFカーネルは次式とする。

$$
K(x,s_i)=\exp\left(-\gamma\lVert x-s_i\rVert^2\right)
$$

10クラスのため45個の1対1判定を行う。予測ラベルはscikit-learn/libsvmの既定投票規則に合わせる。同票時もクラス順を含めて参照実装と一致させる。

推論時間はブラウザーの `performance.now()` の差分をミリ秒で表示する。画面に表示する投票数や決定値には「確率ではない」と明記する。

## 7. モデル不足時の動作

モデル不足時は通常の推論UIに進ませず、次を表示する。

次のいずれか一方を実施するよう案内する。

1. 対象 Notebook だけを実行する場合は、`mnist/plot_digits_classification.ipynb` を開き、全コードセルを上から実行する。
2. リポジトリのセットアップとNotebook実行をまとめて自動化する場合は、Windowsでは既存の `setup.ps1 -RunNotebooks`、macOSでは既存の `setup.sh --run-notebooks` を実行する。

どちらの場合も、完了後にHTMLを再読み込みする。

ブラウザーからPowerShell、Bash、Jupyterを直接起動しない。

## 8. 対象外

- 真のMNIST 28×28モデルへの置き換え
- CNNモデルの利用
- 確率推定を有効にするための再学習
- 描画履歴、画像保存、複数モデル選択、線幅設定
- バックエンドAPI、クラウド配置、ユーザー認証
- 新規ビルドパイプラインやテストフレームワーク
- 正しい綴りの別名HTML追加

## 9. 受入条件

| ID | 条件 |
|---|---|
| AC-01 | HTMLを `file://` で開き、外部通信なしで起動する。 |
| AC-02 | マウス描画とタッチ相当Pointer Eventの双方でストロークを記録する。 |
| AC-03 | ストローク終了時に右側へ予測と統計を表示する。 |
| AC-04 | JavaScript推論が、Notebookと同じテスト899件すべてで `clf.predict()` と一致する。 |
| AC-05 | Notebook全コードセルの再実行時に算出した `y_test` と `predicted` の評価値が、899件中871件正解という保存済み基準と一致する。HTML起動時にテストデータ全件を再評価する要件ではない。 |
| AC-06 | 空白を推論せず、クリア後は結果が初期状態へ戻る。 |
| AC-07 | モデル欠落・不正時に推論を停止し、実行手順と再読み込みを表示する。 |
| AC-08 | 投票数・決定値を確率と表示しない。 |
| AC-09 | デスクトップと狭幅表示、ライトとダーク表示で利用できる。 |
| AC-10 | Notebookの変更は最後の既存空コードセルに限定される。 |
| AC-11 | `README.md` と `SETUP.md` に利用方法と復旧方法が記載される。 |

## 10. 実装タスク対応

| タスク | 主な要求 | 対象ファイル |
|---|---|---|
| 要求固定 | 全要求 | `docs/requirement-definition.md` |
| モデル出力 | FR-12、NFR-06、NFR-07 | `mnist/plot_digits_classification.ipynb` |
| HTML基盤と起動検査 | FR-01、FR-09–FR-11、NFR-01–NFR-04 | `mnist/plot_digits_predition.html` |
| 描画と前処理 | FR-02–FR-04、FR-08、NFR-05 | 同上 |
| SVC推論と統計 | FR-05–FR-07 | 同上 |
| 参照一致・UI検証 | AC-01–AC-10 | 一時検証物のみ。恒久的なテストファイルは追加しない |
| 利用者向け文書 | AC-11 | `README.md`、`SETUP.md` |
| 変更履歴 | リリース要件 | `CHANGELOG.md` |

## 11. 変更履歴とパッケージ版

全実装・検証後、Keep a Changelog形式の `CHANGELOG.md` に `Unreleased` セクションを作り、変更概要を追加する。

このリポジトリには、現時点で `pyproject.toml`、`package.json`、`setup.py` などのパッケージマニフェストが存在しない。最終確認時に再検索し、対象パッケージが存在せず今回変更していない場合は、該当するPATCHバージョンはないため架空のパッケージや版番号を作成しない。

## 12. 情報源

### リポジトリ内一次資料

- `mnist/plot_digits_classification.ipynb`
- `executed-notebooks/mnist/plot_digits_classification.executed.ipynb`
- `README.md`
- `SETUP.md`
- `setup.ps1`
- `setup.sh`
- `environment.yml`

### 公式資料

- scikit-learn, Recognizing hand-written digits: <https://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html>
- scikit-learn, `load_digits`: <https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html>
- scikit-learn, `SVC`: <https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html>
- scikit-learn, Support Vector Machines: <https://scikit-learn.org/stable/modules/svm.html>
- MDN, Pointer events: <https://developer.mozilla.org/en-US/docs/Web/API/Pointer_events>
- MDN, Canvas API: <https://developer.mozilla.org/en-US/docs/Web/API/Canvas_API>
- MDN, Same-origin policy — file origins: <https://developer.mozilla.org/en-US/docs/Web/Security/Defenses/Same-origin_policy#file_origins>
- MDN, `performance.now()`: <https://developer.mozilla.org/en-US/docs/Web/API/Performance/now>
