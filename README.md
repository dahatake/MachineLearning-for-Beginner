# MachineLearning-for-Beginner
機械学習の原理や基礎をこれから学ぶ方向けのサンプルコードです。

## セットアップ

Windows、macOS、Linux の手順と、Anaconda（既定）、Miniconda、venv の選び方は [`SETUP.md`](SETUP.md) を参照してください。

ここでは以下の2つを体験を通して。機械学習について学び実際にモデルの作成を行います。

1. MNIST

機械学習での Hello World ともいえる極めて有名なデータセットです。手書き画像の0-9の**分類**を行います。
画像処理ですからGPUがあった方が処理は早いです。ですが、小さな画像ですので、GPUの無いコンピューターでも、それ程時間はかかりません。
アルゴリズムは、`SVC` と、`Neural Network` の双方を試します。

2. Computer Vision 

ツールを使って `Image Classification` を行います。主に、アノテーション(ラベリング)と、学習の実行を行います。推論は行いません。

> [!NOTE]
> AutoMLについて触れていますが、それにはSoftware Engineeringの方向けである、Microsoft Azure など環境が別途必要です。そのため、**オプション**として記載しています。

# 対象者
- 一般的なPythonのプログラミングの基本コースを終了した方。あるいは何らかのプログラミング経験者
- 理工学部だが、情報工学は専攻ではない方
- Jupyter Notebook あるいは Jupyter Lab の一般的な使い方を知っている方
    - https://jupyter.org/try

# 利用ツール

- Anaconda
- Jupyter Notebook
- Lobe.ai

# 1. 環境構築

機械学習のモデル開発に必要な環境を整備します。

以下のアプリケーションをインストールします。
- Anaconda
    - Pythonの複数の環境の分離を行うために使います
    - インストール時にPythonがインストールされていないと、Pythonもインストールされます

ダウンロード元:

https://www.anaconda.com/download

Anacondaを使うのは、関連モジュールのバージョンの整合性を保つためです。このテキストのコードはPythonを使用しますが、Pythonのモジュールは、個々に開発が行われているため、特定のモジュールのバージョンを上げる事で不整合が起こる事があります。
Anacondaは、そのような問題を解決するために、モジュールのバージョンを管理するためのツールです。Anacondaでは、「環境」(Environment)という概念があって、作業する環境毎に異なるバージョンのモジュールを使う事が出来ます。そして、Anaconda経由でモジュールをインストールする事で、その環境に合わせたバージョンのモジュールをインストールする事が出来ます。

他にも似た事をするツールはvenvなど幾つかあります。

## 1.1 作業フォルダの作成

このワークショップのための`作業フォルダ`を作成します。例えば、以下の様な場所に作成します。

```shell
C:\Work
```

> [!NOTE]
> このテキストでは、`作業フォルダ`を **C:\Work** とします。他の場所に作成した場合は、適宜読み替えてください。


## 1.2. このワークショップで使うファイルのダウンロード

複数のファイルを1つのZipファイルに圧縮してダウンロードします。

この演習では最もお勧めです。

![download-zip](/images/github-download-as-zip.jpg)

- zip圧縮のファイルは、展開してください。

> [!WARNING]
> **zipのダウンロードができなかった場合**
>
> ブラウザーで「ダウンロードできませんでした - ウイルスが検出されました」と表示されることがあります。これは Microsoft Defender が GitHub の生成した zip ファイルを誤って検知したもので、ファイルの内容にウイルスは含まれていません。
> この場合は、zip を使わずに、winget で Git をインストールしてクローンしてください。Windows の PowerShell で、以下を上から順に 1 つずつ実行します。
>
> ```powershell
> winget install --id Git.Git -e --source winget --accept-package-agreements --accept-source-agreements
> ```
>
> ```powershell
> $env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [Environment]::GetEnvironmentVariable("Path", "User")
> ```
>
> ```powershell
> New-Item -ItemType Directory -Force -Path C:\Work
> ```
>
> ```powershell
> Set-Location C:\Work
> ```
>
> ```powershell
> git clone https://github.com/dahatake/MachineLearning-for-Beginner.git
> ```
>
> 各コマンドの意味、Git を使わない方法（tar.gz 形式での取得）、macOS の手順、うまくいかない場合の対処は、[zip を使わずにファイルを取得する手順](docs/download-without-zip.md) を参照してください。

> [!Tips]
> **クローン**は、ダウンロードとほぼ同じ意味です。厳密には勿論、異なるのですが、今の段階ではあまり気にしなくてよいです。
> クローンをする場合は、Gitというツールを使って、GitHubなどにあるファイルの複製を、自分のPCあるいはMacにダウンロードして構成します。クローンをした場合は、その後のPCあるいはMacでの作業での変更点を、GitHubなどのクローン元に反映がしやすくなります。このテキストでは、大本のファイルを変更することはありません。
> クローンをする場合は、以下の手順を実行します。
> - (自分のPCもしくはMacにGitがインストール済みの場合) git clone コマンドでクローンします。Git のインストール方法は [zip を使わずにファイルを取得する手順](docs/download-without-zip.md) を参照してください。
> ```shell
> git clone https://github.com/dahatake/MachineLearning-for-Beginner.git
> ```

ファイルをダウンロードした後は、ファイルの内容を展開します。

- 展開あるいはクローンしたファイルを、`作業フォルダ`にコピーします。
- 作業フォルダ中は、以下の様な構成になっています。**例示しているだけ**なので、まったく同じではなくてよいです!


```text
C:\Work   --- ここは別でもいいです
   |---MachineLearning-for-Beginner --- ディレクトリ
        |--- mnist --- ディレクトリ
        |       |--- plot_digits_classification.ipynb --- SVCでのモデル作成
        |       |--- plot_digits_predition.html       --- SVCでのブラウザー推論
        |       |--- mnist_pytorch.ipynb --- CNNでのモデル作成
        | --- images  --- 画像ファイル
        |--- README.md  --- このファイル
```

## 1.3. Anaconda のインストール

自分のPCあるいはMacにインストールがされていない場合はインストールを行ってください。
以下の公式サイトから**無料版 | Free** をダウンロードして、インストールします。

> [!NOTE]
> ダウンロード画面にメールアドレスでの登録を促して `Registration`の項目がありす。ここでは、**スキップ**しても構いません。

![anaconda-download](/images/anaconda-skip-registration.jpg)

インストール時間は、環境にも寄りますが**5分程度**かかるかと思います。

https://www.anaconda.com/

### 1.3.1. Anaconda の環境 (Environment) の作成

既存の Anaconda の`環境`ファイルを取り込んで、自分のPCあるいはMacに同じ環境を作成します。環境は、特定のプログラムを動かす際に必要となるモジュール(ライブラリとも言います)を一度にダウンロードするなりして、動作実行を行える状態を指します。まさに動作**環境**ですね。

環境ファイルは、このリポジトリの中にあります。**mnist.yml** です。

このテキストでは Anaconda の Environment 名を **mnist** としています。

- OSのスタートメニューにある**Anaconda Prompt** を起動します。


> [!CAUTION]
> OSのターミナルやコマンドプロンプトではありません。

- 作業フォルダまで移動します。以下は例です。自分の環境に合わせて変更してください。

```cmd
cd C:\Work\MachineLearning-for-Beginner\
```

- 以下のコマンドを入力して、`Channel`を追加します。Anacondaでの多くのモジュールが、用意されている`defaults`のチャネル以外の`conda-forge`や `pytorch`にあります。

```cmd
conda config --add channels conda-forge
```

```cmd
conda config --add channels pytorch
```

- 以下のコマンドを入力して、`環境 (Environment)`を作成します。**5分程度**かかります。

```shell
conda env create -f mnist.yml
```

mnist.ymlで指定しているパッケージが、Windowsで動作確認をしているため、Macで動作しない可能性があります。

scikit-learn のインストールでエラーが出る場合は、以下のサイトを参考にして、手動でインストールを行ってください。

https://scikit-learn.org/stable/install.html

また、PyTorch のインストールでエラーが出る場合は、以下のサイトを参考にして、手動でインストールを行ってください。

https://github.com/pytorch/examples/tree/main/mnist


### 1.3.2. (オプション) Jupyter Notebook のインストール

> [!NOTE]
> 自分のPCあるいはMacにJuputer Notebookがインストールされていない場合は、必ずこの作業を行ってください。

Anaconda Navigator の [Home] で、Jupyter Notebook が表示されていない場合は、インストールを行います。表示されている場合は、このセクションはスキップしてください。

- [Environments] に移動します。
- 先ほど作成した Environment を選択します。ここでは、**mnist** です。
- 画面上部で、インストール済みかどうかなどを選択できるようになっています。ここでは **[Not installed]** を選択します。 

![install-notebook](/images/anaconda-install-notebook.jpg)


- 画面右上の検索ボックスに **notebook** と入力します
- 検索結果から **notebook** を選択します。
- 画面下の **Apply** を押します

![search-notebook](/images/anaconda-install-notebook-check-notebook.jpg)

- Anacondaは、依存関係などを自動的に調査した上で、関連するモジュールをインストールします。**Apply** を押します

![apply-dependency](/images/anaconda-install-notebook-apply-related-packages.jpg)

何らかの理由でインストールがうまくいかない場合は、公式サイトからダウンロードしてください。

https://anaconda.org/anaconda/jupyter


### 1.3.3. Jupyter Notebook の作業フォルダの変更

> [!NOTE]
> この作業は必須ではありません。ですが、演習を実施する上で、設定を行う事を強く推奨します。


初期設定ですと、Jupyter Notebook は、ユーザーの**ホームディレクトリ**に作成されます。作業フォルダを変更するには、以下の手順を実行します:

- **Anaconda Prompt** を起動します。


`環境`を`mnist`に変更します。

```shell
conda activate mnist
```

- 以下のコマンドを入力して、設定ファイルを作成します。

```shell
jupyter notebook --generate-config
```

設定ファイルは、**jupyter_notebook_config.py**です。

Windowsの場合は、以下の様な場所に出来ています。

```cmd
C:\Users\<<ユーザーアカウント名>>\.jupyter
```

- 作成されたファイルを開き、**c.ServerApp.root_dir**を検索し、コメントを外します。コメントは行頭の「#」の文字です。

- **c.ServerApp.root_dir**の後に、作業フォルダのパスを指定します。

例:
```shell
c.ServerApp.root_dir = r'C:/Work'
```
- 設定ファイルを保存します。


### 1.3.4. Jupyter Notebook の起動

Jupyter Notebook が起動できるかを確認します。

- [Anaconda Navigator]を起動します。

- [Environments] に移動します。

- [Environment] - [mnist] を選択します
- 三角のアイコンをクリックして、**Open with Jupyter Notebook** を選択します

![Jupyter Notebook run](/images/anaconda-run-notebook.jpg)


以下の画面の様に、作業フォルダーの内容が表示されていれば設定完了です。この画面では`C:\Work`が作業フォルダーになります。

![Jupyter Notebook open](/images/jupyternotebook-open-workfolder.jpg)


## 1.4. Lobe.ai のインストール

以下のサイトからインストーラーをダウンロードして、インストールを行います。

~~https://github.com/lobe/lobe~~

> [!IMPORTANT]

自分のPCあるいはMacのCPUがAVXという命令セットに対応していないと、Lobeのインストールが出来ません。

> [!IMPORTANT]
> 公式サイトからアプリケーションのインストーラーへのリンクが無くなっています。
> こちらのサイトに、一時的にインストーラーを置いてありますので、自己責任で利用してください。

https://github.com/julienheinen/lobeAI/releases

# 1.5. 実行時のエラー対応策

この後の演習で用意されたコードを実行します。よくあるエラーとその対応策を以下に記載します。

## 1.5.1. そもそも何のエラーなのか分からない場合

Microsoft Copilot が使える場合は、以下のPromptを実行します。

***注意***
- 結果が100%正しいとは限りません。情報元のWebサイトを確認してください。また、それでも自身が無い場合には、専門とする方に相談してください。


Prompt:
```cmd
Pythonのコードを実行したのですが、以下のエラーメッセージが表示されて動作しません。
想定される可能な限り詳細な原因と、測定方法と、解決策をリストアップしてください。
問題解決の為の詳細な手順書も作成してください。

### エラーメッセージ
ModuleNotFoundError: No module named 'matplotlib'
```

ちなみに、このPromptの応用範囲は広く、様々なケースでガイダンスとして使えます。

例1: 建築現場での材料の破壊

Prompt:
```cmd
建築の現場で、材料の破壊がありました。

想定される可能な限り詳細な原因と、測定方法と、解決策をリストアップしてください。
問題解決の為の詳細な手順書も作成してください。
```

例2: 反応物と生成物のエンタルピーの実験

Prompt:
```cmd
反応物と生成物のエンタルピーの実験を行いましたが、期待していた結果と大きく異なりました。

想定される可能な限り詳細な原因と、測定方法と、解決策をリストアップしてください。
問題解決の為の詳細な手順書も作成してください。
```

## 1.5.2. `ModuleNotFoundError: No module named 'xxxx'`

`ModuleNotFoundError: No module named 'xxxx'` というエラーが出た場合は、そのモジュールがインストールされていない事が考えられます。

チャネルを追加する事で、そのモジュールをインストールする事が出来ます。

- **Anaconda Navigator** を起動します
- **Channels** を選択します
- 以下のチャネルを追加します

> conda-forge

> pytorch

[こちらの手順も再度ご確認ください: Anaconda の環境 (Environment) の作成](/README.md#131-anaconda-の環境-environment-の作成)

`plot_digits_classification.ipynb` で使っているモジュール:

- matplotlib
- sklearn

`mnist_pytorch.ipynb` で使っているモジュール:

- torch
- torchvision

Anaconda の Environment に、そのモジュールがインストールされているかを確認します。

- **Anaconda Navigator** を起動します
- **Environments** を選択します
- **mnist** などの Environment を選択します
- **Installed** を選択します
- 検索ボックスに、そのモジュール名を入力します

検索結果、チェックが入っているとインストールされています。

インストールされていない場合は、以下の手順でインストールを行います。

- **Not installed** を選択します
- 検索ボックスに、そのモジュール名を入力します
- 必要モジュールをチェックします
- 画面下部の**Apply** を押します



# 2. MNIST - 初めての機械学習モデルの作成
機械学習の入門として代表的なサンプルになります。

機械学習でとてもよく使われている scikit-learn というライブラリを使い、MNIST というデータセットを学習用に用いて、手書き数字の画像を分類するモデルを作成します。

ファイルの説明

```cmd
mnist --- ディレクトリ
        |--- plot_digits_classification.ipynb --- SVCでのモデル作成
        |--- mnist_pytorch.ipynb              --- CNNでのモデル作成
```

このサンプルコードは、sci-kit learn の公式サイトにあるものを使っています。

https://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html

> [!NOTE]
> `plot_digits_classification.ipynb` が使うのは、scikit-learnに含まれる8×8画像のDigitsデータセットです。28×28画像のMNISTデータセットとは画像サイズが異なります。

サンプルコードはこちらです。

[plot_digits_classification.ipynb](mnist/plot_digits_classification.ipynb)

## 目的
- 自分の環境でモデルの作成を行う
- 機械学習の学習のプログラムの概要を理解する

## Task
- Anaconda Navigator から、 Jupyter Notebookを起動します。
- `plot_digits_classification.ipynb` を開いて、セルを一つずつ、実行します
    - MNIST の写真の一部をみて、どんなデータセットなのかを理解する
    - プログラムの構造をリスト化します。どこで何をしているのか?
- `mnist_pytorch.ipynb` を開いて、セルを一つずつ、実行します
    - アルゴリズムが`SVC`と`Neural Network`の2つあります。それぞれの違いを調べてください
    - Neural Network のコードをどう修正すれば `Deep Neural Network` になるか調べてください
- `plot_digits_classification.ipynb` の全コードセルを上から実行し、最後のセルが学習したSVCモデルと評価指標をブラウザー推論用HTMLへ保存することを確認します

> [!TIP]
> Pythonのプログラムのコードの内容が不明瞭な場合は。CopilotやChatGPTに質問をしてみてください。例えば、以下の様なPromptです。

Prompt:
```cmd
私はPythonの初心者です。以下のコードで行っている内容を、具体的かつ明瞭かつ丁寧に説明してください。

<python のコードをここに貼り付け>
```

## ブラウザーで手書き数字を推論

`plot_digits_classification.ipynb` の全コードセルを上から実行すると、最後のセルが学習済みSVCモデルを [`mnist/plot_digits_predition.html`](mnist/plot_digits_predition.html) 内へ保存します。

1. `mnist/plot_digits_predition.html` をPCまたはMacのブラウザーで開きます。
2. 左側の描画エリアへ、マウス、タッチ、またはペンで0から9の数字を1つ描きます。
3. ストロークを離すと自動で推論され、右側に予測数字と統計情報が表示されます。
4. 描き直す場合は **クリア** を選びます。

このHTMLは、実行時にPython、Webサーバー、外部ネットワークを必要としません。描画は中央へ配置され、Notebookと同じ8×8・画素値0–16の64特徴へ変換されます。

表示されるクラス別投票数とSVCの決定値は、**確率ではありません**。Notebookの `SVC(gamma=0.001)` は確率推定を有効にしていません。

画面に「学習済みモデルを生成してください」と表示された場合は、案内に従ってNotebookの全コードセルを実行するか、[`SETUP.md`](SETUP.md) の全コード自動実行手順を実施してからHTMLを再読み込みしてください。

# 3. Deep Learning - Computer Vision

機械学習で画像を扱う処理を`Computer Vision` と称しています。

PyTorch でも実装できます。ですが、ここでは**ツール**を使って学習までの一連の流れを体験します。

## 目的
- ツールの存在を知る。そのツールで出来る事、出来ない事を知る

このテキストでは**Lobe.ai**というツールを使います。
https://github.com/lobe/lobe

Lobe.ai は、画像のアノテーション(ラベリング)と、学習の実行を行うツールです。無料で利用することが出来ます。

学習用の画像ファイル:

https://aka.ms/animal-images

> [!IMPORTANT]
> ファイルをダウンロードしたら、`C:\Work\images` に展開してください。**日本語のフォルダー名は避けてください**。画像ファイルのインポート時にエラーになる場合があります。

こちらの Blog post を参考に以下を行います。

- Lobe.ai のインストール
- Computer Vision モデルの作成
  - ダウンロードしたフォルダーのインポート
  - 任意の画像ファイルでの、モデルのテスト。画像ファイルが手元にない場合は、以下のファイルをダウンロードしてテストします。

    https://aka.ms/giraffe

> [!IMPORTANT]
> **手順 4 テスト** まで行ってください。その先の**5以降を行う必要はありません!**

Blog Post:

https://qiita.com/dahatake/items/05efc18eaf03605cb7d0


## Task
- 自分の手元の画像など**任意の画像**で、画像分類を行ってください。
- Computer Vision の主な処理には`分類 (Image Classification)` **以外** には何がありますか?
- Computer Vision の学習をする**前**に行うタスクは何ですか? それは、どこまで自動化が出来そうですか?


# (オプション 1) AutoML

> [!WARNING]
> この先は、通常の演習では行いません。機械学習そのものに興味がある方向けです。


金融機関での普通預金から定期預金になったというオープンデータを用いて、その予測のモデルを作成します。**分類**のタスクを行います。
ここでは、Python のコードは記載しません。Azure Machine Learning Studio の画面の中だけで、AutoML (自動機械学習) を実行します。

## 目的
- 機械学習の技術の進化の一端を知る

こちらのドキュメントの通り、実行してください。

https://learn.microsoft.com/ja-jp/azure/machine-learning/tutorial-first-experiment-automated-ml?view=azureml-api-2

変更点:
- [4.タスクの設定]で[制限]の設定を行います。
    - 最大ノード数: 3
    - タイムアウト(分): 20
    - 早期終了を有効にします: チェックする

![AutoML設定](/images/aml-automl-config.jpg)

注意点:
- ドキュメントと実際の画面に違いがある事がありえます。画面で設定する項目は、ドキュメントの別の章に記載がありますので、注意深く参照してください
- データセット(ファイル)を自分のPCにダウンロードをしてください。以下はドキュメントの中で指定しているファイルのURLです。
    - https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv
- ドキュメントにもありますが、学習のジョブを投入してから、全て完了するまで20分かかります

## Task
- AutoML の利点・欠点をリストアップしてください
- 幾つかのモデルの学習過程を調べてください
- 最も性能の良いモデルの**説明**と、**責任あるあるAI** では、何がわかるか調べてください


# (オプション 2) Azure for Student のセットアップ 推奨
- Microsoft Azure: 学生の方は無償で利用できる枠があります


# Azure 利用環境の作成

ここでは、大学・専門学校の皆様向けの方法を記載します。

## Azure for Students の取得

無料かつクレジットカードの登録無しで、100 USD 分/月 の Azure 利用などや各種ソフトウェアの利用もできる**学生向けの特典**を利用できます。

こちらのドキュメントに従って、学校のアカウント(メールアドレス)で、Azure Education Hub にログインします。

https://learn.microsoft.com/ja-jp/azure/education-hub/access-education-hub

[学習リソース]の[GitHub]に移動します。**GitHub Student Developer Pack にサインアップする** の [サインアップ]ボタンを押します。

![image](/images/eduhub-GitHub-Overview.jpg)

GitHub Student Developer Pack のサイトに移動します。
緑色の**Sign up for Student Developer Pack**のボタンを押します。

Individuals の Students から **Get student benefits** を押します。

![github-individuals](/images/github-individuals-students.jpg)


https://azure.microsoft.com/ja-jp/free/students/


### 利用する Azure の Service

- Azure AI Services : Custom Vision Service
    - プロジェクト 1つ以上

### サービスの作成できない場合:

ご利用のAzure Subscription で、リソース プロバイダーが有効化されていない事があります。

https://learn.microsoft.com/ja-jp/azure/azure-resource-manager/management/resource-providers-and-types


リソースプロバイダーの一覧:
https://learn.microsoft.com/ja-jp/azure/azure-resource-manager/management/azure-services-resource-providers?source=recommendations

- Custom Vision Service は Cognitive Service (旧名) になっています
