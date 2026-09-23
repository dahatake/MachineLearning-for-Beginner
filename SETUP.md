# セットアップおよび全コード実行手順

最終確認日: **2026-08-28**

この文書は、`setup/setup-windows.ps1`（Windows）または `setup/setup-mac.sh`（macOS）を使い、現在リポジトリに存在する全コードを実行する手順です。

## 対象範囲

現在の実行対象は、`mnist/` にある次の **2 個の Jupyter Notebook** です。リポジトリ内に、それ以外の Python ソースや実行対象 Notebook はありません。[R1][R2][R3]

| 実行対象 | 主な直接依存 | データ |
|---|---|---|
| `mnist/plot_digits_classification.ipynb` | Matplotlib、scikit-learn | `sklearn.datasets.load_digits()` [R2][S2][S3] |
| `mnist/mnist_pytorch.ipynb` | PyTorch、torchvision | `torchvision.datasets.MNIST`。Notebook 自体も `download=True` を指定しています。[R3][T1][T2] |

`mnist/plot_digits_predition.html` は、1つ目のNotebookで学習したSVCをブラウザー内で実行する推論アプリです。Notebookの最後のセルが、学習済みモデルと評価指標をHTML内へ保存します。[R2]

`README.md` に記載される Lobe、Azure Machine Learning、Azure for Students は、外部 GUI／クラウド演習であり、このリポジトリ内の実行コードではありません。そのため本スクリプトの自動実行対象外です。[R1]

## 対応環境

| OS | 対応 CPU | 前提 |
|---|---|---|
| Windows 10 以降 | x86_64 | PowerShell 7 以降、インターネット接続 [M1][P1] |
| macOS 11 以降 | Intel x86_64 / Apple Silicon arm64 | 標準の Bash、`curl`、`shasum`、インターネット接続 [M1] |

Windows ARM64 は、採用した Miniforge リリースにネイティブ Windows ARM64 インストーラーがないため、本手順では対応を主張しません。[M1][M2]

Miniforge 公式資料では Apple Silicon ビルドを experimental と注記しています。本スクリプトは公式 arm64 インストーラーと arm64 Conda パッケージを使用しますが、この留保も適用されます。[M1]

GPU は必須ではありません。PyTorch Notebook は CUDA、MPS、CPU の順で利用可否を判定し、アクセラレーターが無ければ CPU を使用します。[R3] `environment.yml` は元の `mnist.yml` と同じ PyTorch 2.3.1／torchvision 0.18.1 を指定しています。[R4]

## スクリプトが行うこと

1. 既存の Conda を検索します。
2. Conda が無い場合だけ、固定した **Miniforge 26.5.3-0** をユーザー領域へ非対話インストールします。[M1][M2]
3. ダウンロードした Miniforge インストーラーの SHA-256 を、公式 GitHub Release API 公開値と照合します。一致しなければ停止します。[M2]
4. 選択した Conda が対象 OS のネイティブ版か確認します。
5. クロスプラットフォーム用 `environment.yml` から、このリポジトリ専用の `mlfb-mnist` 環境を作成します。既に存在する場合は、`--prune` を指定して同じ定義へ更新します。[C1] Windows と macOS では PyTorch 2.3.1 を取得するチャンネルが異なるため、環境の依存解決中だけ `CONDA_CHANNEL_PRIORITY=flexible` を指定します。ユーザーの `.condarc` は変更しません。strict mode は同名パッケージの下位チャンネルへのフォールバックを禁止するため、このクロスプラットフォーム定義には使用しません。[C3]
6. 全直接依存を import し、scikit-learn の Digits データ形状 `(1797, 64)` を検査します。[S2]
7. torchvision の MNIST 学習・テストデータを `data/MNIST/` に準備します。`download=True` は、未取得時のみインターネットから取得する torchvision の公式仕様です。[T1]
8. 全コード実行オプションを指定した場合、リポジトリ内の `.ipynb` を自動検出し、nbconvert で順番に実行して `executed-notebooks/` に保存します。生成済み Notebook と `.ipynb_checkpoints` は再実行対象から除外します。セルで例外が発生すると処理は失敗します（`--allow-errors` は使いません）。完走後はカーネルを即時終了し、長い終了処理を残しません。[J1][J2]
9. `plot_digits_classification.ipynb` の最後のセルは、学習したSVCの支持ベクトル、双対係数、切片、評価指標を `mnist/plot_digits_predition.html` 内へ保存します。HTMLの他の部分は変更しません。[R2]

既存の `mnist.yml` は Windows 固有ビルドを含む当時の完全スナップショットなので変更していません。新しい `environment.yml` は、Notebook が直接使うパッケージと元ファイルのバージョンだけを記載しています。Conda 公式資料でも、完全な明示仕様は通常プラットフォーム固有で、クロスプラットフォーム共有には直接指定したパッケージのみを使う方法が案内されています。[R4][C1]

## Windows の手順

### 1. PowerShell 7 を確認

PowerShell 7 を起動し、次を実行します。

```powershell
$PSVersionTable.PSVersion
```

Major が 7 以上でない場合、Microsoft 公式手順では Windows クライアントへの WinGet インストールが推奨されています。[P1]

```powershell
winget install --id Microsoft.PowerShell --source winget
```

Windows 10 などで WinGet を利用できない場合は、同じ Microsoft 公式ページに掲載される安定版 MSI インストーラーを使用してください。[P1]

インストール後、新しい PowerShell 7（`pwsh.exe`）を開きます。

### 2. セットアップのみ実行

リポジトリのルートへ移動し、スクリプトと `environment.yml` の内容を確認してから実行します。

```powershell
pwsh.exe -NoLogo -NoProfile -File .\setup\setup-windows.ps1
```

これにより環境構築、依存確認、MNIST データ準備まで行います。管理者権限は要求しません。Conda が無い場合の既定インストール先は `$HOME\Miniforge3` です。

Miniforge 公式資料には Windows のインストール先で空白や特殊文字を避けるよう注意があります。[M1] 既定パスにそれらが含まれる場合、スクリプトは推測で別の場所へ導入せず停止します。書き込み可能な ASCII パスを明示して再実行してください。

```powershell
pwsh.exe -NoLogo -NoProfile -File .\setup\setup-windows.ps1 -MiniforgePrefix 'D:\Miniforge3'
```

### 3. 全コードを自動実行

```powershell
pwsh.exe -NoLogo -NoProfile -File .\setup\setup-windows.ps1 -RunNotebooks
```

セットアップとデータ取得も同時に行うため、初回からこのコマンドだけを実行しても構いません。2 回目以降は既存環境と既存データを再利用します。[C1][T1]

### 4. 必要な場合だけ使うオプション

既存 Conda の場所を明示する場合:

```powershell
pwsh.exe -NoLogo -NoProfile -File .\setup\setup-windows.ps1 -CondaExecutable 'C:\path\to\conda.exe' -RunNotebooks
```

データの事前取得を省略する場合:

```powershell
pwsh.exe -NoLogo -NoProfile -File .\setup\setup-windows.ps1 -SkipDataDownload
```

ダウンロードした ZIP から展開した場合など、実行ポリシーによりスクリプトだけがブロックされたときは、内容を確認した後に対象ファイルを `Unblock-File` で解除できます。これはポリシー全体を変更しません。[P2]

```powershell
Unblock-File -LiteralPath .\setup\setup-windows.ps1
```

## macOS の手順

### 1. セットアップのみ実行

Terminal でリポジトリのルートへ移動し、スクリプトと `environment.yml` の内容を確認してから実行します。実行権限の付与は不要です。

```bash
bash ./setup/setup-mac.sh
```

Conda が無い場合の既定インストール先は `${HOME}/miniforge3` です。Miniforge 公式の非対話インストール方式 `bash <installer> -b -p <prefix>` を使用し、シェル初期化ファイルは変更しません。[M1]

### 2. 全コードを自動実行

```bash
bash ./setup/setup-mac.sh --run-notebooks
```

初回からこのコマンドだけを実行しても構いません。

### 3. 必要な場合だけ使うオプション

既存 Conda の場所を明示する場合:

```bash
bash ./setup/setup-mac.sh --conda /path/to/conda --run-notebooks
```

Miniforge の導入先を変える場合:

```bash
bash ./setup/setup-mac.sh --miniforge-prefix "${HOME}/custom-miniforge"
```

データの事前取得を省略する場合:

```bash
bash ./setup/setup-mac.sh --skip-data-download
```

## 実行結果

全コード実行に成功すると、元 Notebook は変更せず次のファイルが生成されます。

```text
executed-notebooks/
└── mnist/
	├── mnist_pytorch.executed.ipynb
	└── plot_digits_classification.executed.ipynb
```

取得データは `data/MNIST/` に置かれます。`data/` と `executed-notebooks/` は `.gitignore` に追加済みです。

`mnist/plot_digits_predition.html` 内のモデルも、`plot_digits_classification.ipynb` の実行結果に更新されます。HTMLは生成済みNotebookではなく、リポジトリに残すブラウザー推論アプリです。

対話的に Notebook を開く場合は、セットアップ完了時に表示される `conda run ... jupyter notebook ...` コマンドをそのまま実行してください。`conda run` は環境を shell で activate せずに、その環境内の実行ファイルを起動する Conda の公式機能です。[C2]

## ブラウザー推論アプリの利用

1. 全コード実行の完了後、`mnist/plot_digits_predition.html` をPCまたはMacのブラウザーで開きます。
2. 左側の描画エリアへ、マウス、タッチ、またはペンで数字を描きます。Pointer Eventsはこれらの入力を共通のイベントモデルで扱います。[W1]
3. ストロークを離すと自動推論されます。必要な場合は **推論** を選んで再実行できます。
4. 右側で予測数字、クラス別投票、SVC推論時間、8×8入力、Notebookのテスト評価を確認します。

アプリはCanvas APIで描画と画像縮小を行い、8×8・画素値0–16の64特徴を、Notebookと同じRBF SVCへ入力します。[W2][S2] モデルデータとJavaScriptはHTML内に含まれるため、実行時のCDN、Webサーバー、Pythonは不要です。ブラウザーからローカルの別ファイルを読む方式は、`file://` originの制限を受ける場合があるため使用していません。[W3]

表示されるクラス別投票数と決定値は確率ではありません。対象の `SVC(gamma=0.001)` は確率推定を有効にしていません。[S4] 推論時間は `performance.now()` の差分です。[W4]

モデルが欠落または不正な場合、アプリは推論を開始せず、次の2つの復旧方法を画面に表示します。

- `mnist/plot_digits_classification.ipynb` の全コードセルを上から実行する。
- Windowsでは `setup/setup-windows.ps1 -RunNotebooks`、macOSでは `setup/setup-mac.sh --run-notebooks` を実行する。

完了後、HTMLを再読み込みしてください。静的HTMLからローカルのPowerShell、Bash、Jupyterを直接起動するボタンは設けていません。

## 失敗時の確認

- **SHA-256 不一致**: スクリプトはインストールを実行せず停止します。ネットワークキャッシュやプロキシを確認し、再取得してください。期待値を手動で変更せず、公式 Release API と照合してください。[M2]
- **ネットワークエラー**: 初回は Miniforge、Conda パッケージ、torchvision MNIST の取得に外部接続が必要です。[M1][T1]
- **Notebook 実行が長い**: `mnist_pytorch.ipynb` は 5 epoch の CNN 学習を行います。[R3] スクリプトは長時間セルに対応するため nbconvert のセルタイムアウトを無制限（`-1`）にしています。[J1] また、Windows の実測で学習完了後の graceful shutdown が長時間残ったため、公式設定の `shutdown_kernel=immediate` を指定しています。[J2]
- **推論アプリにモデル未準備と表示される**: `mnist/plot_digits_classification.ipynb` の全コードセルを上から実行するか、OS別の全コード自動実行手順を実施し、HTMLを再読み込みしてください。最後のセルでHTMLが見つからない場合は、Notebookと `plot_digits_predition.html` が同じ `mnist/` ディレクトリにあることを確認してください。[R2]
- **既存 `mlfb-mnist` 環境の不整合**: Conda 公式手順に従い、不要なら専用環境を削除してからセットアップを再実行できます。[C1]

```text
conda env remove --name mlfb-mnist
```

Conda が PATH に無い場合は、セットアップ完了時に表示された Conda の絶対パスを `conda` の代わりに使ってください。

## 検証記録

2026-08-28 に次を確認しました。

- Windows ホストで `setup.ps1` の PowerShell 構文解析に成功。
- `setup.sh` は Bash 構文検査と ShellCheck に成功。
- Conda 26.5.3 solver と `CONDA_CHANNEL_PRIORITY=flexible` を使う dry-run で、`win-64`、`osx-64`、`osx-arm64` の3対象について `environment.yml` の依存解決に成功。異OS向け dry-run では、Conda 公式資料に従い検証用の `CONDA_OVERRIDE_OSX=11.0` を指定しました。[C1][C3]
- strict mode では、チャンネル順により Windows または macOS 用の PyTorch 同名パッケージが除外されることを実測しました。このため、スクリプトは環境作成・更新プロセスだけを flexible mode に固定しています。[C3]
- Windows では固定版 Miniforge の SHA-256 検証、環境作成、依存 import、Digits データ検査、MNIST データ取得を実行して成功。
- Windows では最新版の `setup.ps1 -RunNotebooks` により、自動検出した現在の2 Notebook をエラー許容なしで最後まで実行して成功。生成物を解析し、`plot_digits_classification` は非空コードセル8/8、`mnist_pytorch` は5/5が出力または実行番号を持ち、例外出力は両方0件でした。CNNは第5 epochとテスト評価の出力まで確認しました。
- Windowsで `plot_digits_classification.ipynb` をエラー許容なしで全セル実行し、支持ベクトル519件のSVCとテスト899件中871件正解（96.885%）の評価指標がHTMLへ保存されることを確認しました。
- 埋め込みモデルの全配列を同条件で再学習したscikit-learnモデルと照合し、JavaScriptのone-vs-one推論がテスト899件すべてで `clf.predict()` と一致することを確認しました。
- WindowsのChromium系統合ブラウザーで、`file://` 起動、合成したマウス／タッチPointer Eventによる描画・自動推論、クリア、モデル欠落時の案内、デスクトップ／狭幅表示、ライト／ダークテーマ、外部リソース要求0件を確認しました。物理タッチデバイスでの実機操作は未実施です。

macOS 実機でのランタイム実行は、この Windows 検証環境からは実施していません。macOS について確認済みなのは、公式対応条件、両アーキテクチャ向け依存解決、Bash 構文、ShellCheck です。この制約を超える実機検証済みという主張はしていません。

## 出典

すべて 2026-08-28 に確認しました。

### リポジトリ内一次資料

- [R1] [`README.md`](README.md) — 演習範囲、既存セットアップ、外部 GUI／Azure 演習の記載。
- [R2] [`mnist/plot_digits_classification.ipynb`](mnist/plot_digits_classification.ipynb) — import、Digits 読み込み、SVC 学習・評価、ブラウザーモデル保存コード。
- [R3] [`mnist/mnist_pytorch.ipynb`](mnist/mnist_pytorch.ipynb) — import、デバイス選択、MNIST 取得、5 epoch の CNN 学習コード。
- [R4] [`mnist.yml`](mnist.yml) — 元の Python／パッケージバージョンと Windows 固有ビルドの完全スナップショット。

### 公式外部資料

- [C1] Conda, **Managing environments** — YAML 環境作成・更新、クロスプラットフォーム共有、異OS dry-run の制約: <https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-environments.html>
- [C2] Conda, **conda run** — 環境内コマンド実行: <https://docs.conda.io/projects/conda/en/stable/commands/run.html>
- [C3] Conda, **Managing channels** — チャンネル順序と strict channel priority の動作: <https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-channels.html>
- [M1] conda-forge, **Miniforge README** — Windows/macOS 対応条件、インストーラー、非対話インストール: <https://github.com/conda-forge/miniforge>
- [M2] conda-forge, **Miniforge 26.5.3-0 Release API** — 固定リリースの資産名、URL、SHA-256 digest: <https://api.github.com/repos/conda-forge/miniforge/releases/tags/26.5.3-0>
- [J1] Jupyter nbconvert, **Executing notebooks** — `--execute`、kernel、セル timeout、例外時の動作: <https://nbconvert.readthedocs.io/en/latest/execute_api.html>
- [J2] Jupyter nbconvert, **Configuration options** — `shutdown_kernel` の `graceful`／`immediate` 設定: <https://nbconvert.readthedocs.io/en/latest/config_options.html>
- [T1] torchvision, **MNIST dataset API** — `root`、`train`、`download` の仕様: <https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html>
- [T2] PyTorch Examples, **Basic MNIST Example** — 公式サンプルと直接依存 `torch`／`torchvision`: <https://github.com/pytorch/examples/tree/main/mnist>
- [S1] scikit-learn, **Installing scikit-learn** — 分離環境と Matplotlib 要件: <https://scikit-learn.org/stable/install.html>
- [S2] scikit-learn, **load_digits API** — 1797 samples、64 features、画像形状: <https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html>
- [S3] scikit-learn, **Recognizing hand-written digits** — 対象 Notebook の公式元サンプル: <https://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html>
- [S4] scikit-learn, **Support Vector Machines — Scores and probabilities** — SVCの決定値と確率推定: <https://scikit-learn.org/stable/modules/svm.html#scores-probabilities>
- [P1] Microsoft Learn, **Install PowerShell 7 on Windows** — WinGet 推奨手順と `Microsoft.PowerShell` package ID: <https://learn.microsoft.com/powershell/scripting/install/install-powershell-on-windows?view=powershell-7.6>
- [P2] Microsoft Learn, **about_Execution_Policies** — 実行ポリシーの scope と `Unblock-File` の位置づけ: <https://learn.microsoft.com/powershell/module/microsoft.powershell.core/about/about_execution_policies?view=powershell-7.6>
- [W1] MDN, **Pointer events** — マウス、ペン、タッチの共通入力モデル: <https://developer.mozilla.org/en-US/docs/Web/API/Pointer_events>
- [W2] MDN, **Canvas API** — JavaScriptによる2D描画と画像処理: <https://developer.mozilla.org/en-US/docs/Web/API/Canvas_API>
- [W3] MDN, **Same-origin policy — File origins** — `file://` originから別ファイルを読む際の制限: <https://developer.mozilla.org/en-US/docs/Web/Security/Defenses/Same-origin_policy#file_origins>
- [W4] MDN, **Performance: now()** — 単調増加する高精度タイムスタンプ: <https://developer.mozilla.org/en-US/docs/Web/API/Performance/now>
