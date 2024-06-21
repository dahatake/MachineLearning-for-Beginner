# MachineLearning-for-Beginner
機械学習の原理や基礎をこれから学ぶ方向けのサンプルコードです。

ここでは以下の2つを体験を通して。機械学習について学びます。

1. MNIST
機械学習での Hello World ともいえる極めて有名なデータセットです。手書き画像の0-9の**分類**を行います。
画像処理ですからGPUがあった方が処理は早いです。ですが、小さな画像ですので、GPUの無いコンピューターでも、それ程時間はかかりません。

アルゴリズムは、SVCと、Neural Networkの双方を試します。

2. Computer Vision のモデル作成のタスクを行う
主に、アノテーション(ラベリング)と、学習の実行を行います。

# 対象者
- 一般的なPythonのプログラミングの基本コースを終了した方
- 理工学部だが、情報工学は専攻外の方
- Jupyter Notebook あるいは Jupyter Lab の一般的な使い方を知っている方
    - https://jupyter.org/try

# 利用ツール

- Jupyter Notebook
- Lobe.ai

# 0. 環境構築

機械学習のソフトウェア開発に必要な環境を整備します。

以下のアプリケーションをインストールします。
- Python
- Anaconda

## 0.1 作業フォルダの作成

このワークショップのための作業フォルダを作成します。例えば、以下の様な場所に作成します。

```shell
C:\Work
```

このテキストでは、`作業フォルダ`を **C:\Work** とします。他の場所に作成した場合は、適宜読み替えてください。


## 0.2. このワークショップで使うファイルのダウンロード

クローンとはダウンロードとほぼ同期です。実際には、Gitというツールを使って、複製を自分のPCあるいはMacに作成しします。その後の作業での変更点を大元に反映させるための準備をする事でもあります。


幾つも方法はあります。

- zip圧縮してダウンロード

![download-zip](/images/github-download-as-zip.jpg)

zip圧縮のファイルは、展開してください。

- (自分のPCもしくはMacにGitがインストール済みの場合) git clone コマンドでダウンロード

```shell
git clone https://github.com/dahatake/MachineLearning-for-Beginner.git
```


展開後のファイルを全て、先ほどの`作業フォルダ`にコピーします。

## 0.3. Anaconda のインストール

自分のPCあるいはMacにインストールがされていない場合はインストールを行ってください。
以下の公式サイトから**無料版 | Free** をダウンロードして、インストールします。

ダウンロード画面にメールアドレスでの登録を促して `Registration`の項目がありま。ここは、**スキップ**しても構いません。

![anaconda-download](/images/anaconda-skip-registration.jpg)

インストール時間は、環境にも寄りますが**5分程度**かかるかと思います。

https://www.anaconda.com/

### 0.3.1. Anaconda の環境 (Environment) の作成

既存の Anaconda の環境ファイルを取り込んで、自分のPCあるいはMacに同じ環境を作成します。環境は、特定のプログラムを動かす際に必要となるパッケージ(ライブラリとも言います)を一度にダウンロードするなりして、動作実行を行える状態を指します。まさに動作「環境」ですね。

環境ファイルは、このリポジトリの中にあります。**mnist.yml** です。

このテキストでは Anaconda の Environment 名を **mnist** としています。ご自身で作成したものに読み替えてください。

- **Anaconda Prompt** を起動します。
- 作業フォルダーまで移動します。以下は例です。ご自分の環境に合わせて変更してください。

```cmd
cd C:\Work
```

- 以下のコマンドを入力して、`Channel`を追加します。Anacondaでの多くのパッケージが、用意されている`defaults`のチャネル以外の`conda-forge`や `pytorch`にあります。

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

---------------------------------


- Anaconda Navigator を起動します
    - Sing in/Sing up はしなくていいです
- [Environment] に移動して、[Import] を選択します。
- Import Environment 画面で、各項目を設定します。


| 項目名 | 設定値 | 内容 |
| --- | --- | --- |
| local drive | mnist.yml のファイルパス | 環境ファイルを選択します |
| New environment name | mnist | 環境名を入力します |

![import-environment](/images/anaconda-import-environment.jpg)



### 0.3.2. (オプション) Jupyter Notebook のインストール

Anaconda Navigator の [Home] で、Jupyter Notebook が表示されていない場合は、インストールを行います。表示されている方は、スキップして、1 へ進んでください。

- [Environments] に移動します。
- 先ほど作成した Environment を選択します。ここでは、**mnist** です。
- 画面上部で、インストール済みかどうかなどを選択できるようになっています。ここでは **[Not installed]** を選択します。 

![install-notebook](/images/anaconda-install-notebook.jpg)


- 画面右上の検索ボックスに **notebook** と入力します
- 検索結果から**notebook**を選択します。
- 画面下の **Apply** を押します

![search-notebook](/images/anaconda-install-notebook-check-notebook.jpg)

- 依存関係などを調査した上で、関連するパッケージをインストールします。**Apply** を押します

![apply-dependency](/images/anaconda-install-notebook-apply-related-packages.jpg)

### 0.3.3. Jupyter Notebook の作業フォルダの変更

初期設定ですと、Jupyter Notebook は、ユーザーの**ホームディレクトリ**に作成されます。作業フォルダを変更するには、以下の手順を実行します:

- **Anaconda Prompt** を起動します。
- 以下のコマンドを入力して、設定ファイルを作成します。
```shell
jupyter notebook --generate-config
```

設定ファイルは、Windowsの場合は、以下の様な場所に出来ています。

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


### 0.3.4. Jupyter Notebook の起動

Jupyter Notebook が起動できるかを確認します。

- [Environment] - [mnist] を選択します
- 三角のアイコンをクリックして、**Open with Jupyter Notebook** を選択します

![Jupyter Notebook run](/images/anaconda-run-notebook.jpg)


# 1. MNIST - 初めての機械学習モデルの作成
機械学習の入門として代表的なサンプルになります。

機械学習でとてもよく使われている scikit-learn というライブラリを使って、MNIST というデータセットを使って、手書き数字の画像を分類します。

ファイルの説明

```cmd
mnist --- ディレクトリ
        |--- plot_digits_classification.ipynb --- SVCでのモデル作成
        |--- mnist_pytorch.ipynb --- CNNでのモデル作成
```

このサンプルコードは、sci-kit learn の公式サイトにあるものを使っています。

https://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html

サンプルコードはこちらです。

[plot_digits_classification.ipynb](mnist/plot_digits_classification.ipynb)

## 目的
- 自分の環境でモデルの作成を行う
- 機械学習の学習のプログラムの概要を理解する

## Task:
- `plot_digits_classification.ipynb` を開いて、実行してみてください
    - MNIST の写真の一部をみて、どんなデータセットなのかを理解する
    - プログラムの構造をリスト化します。どこで何をしているのか?
- `mnist_pytorch.ipynb` を開いて、実行してみてください
    - アルゴリズムが`SVC`と`Neural Network`の2つあります。それぞれの違いを調べてください
    - Neural Network のコードをどう修正すれば `Deep Neural Network` になるか調べてください
- 作成したモデルがファイルに保存をされていません。保存するためにプログラムを修正します

# 2. Deep Learning - Computer Vision

機械学習で画像を扱う処理を`Computer Vision` と称しています。

PyTorch でも実装できます。ですが、ここではツールを使って学習までの一連の流れを体験します。

## 目的
- ツールの存在を知る。そのツールで出来る事、出来ない事を知る

このテキストでは**Lobe.ai**というツールを使います。

https://www.lobe.ai/

Lobe.ai は、画像のアノテーション(ラベリング)と、学習の実行を行うツールです。無料で利用することが出来ます。

こちらの Blog post を参考に以下を行います。

- Lobe.ai のインストール
- Computer Vision モデルの作成

> [!IMPORTANT]
> **手順 4 テスト** まで行ってください。その先の**5以降を行う必要はありません!**

Blog Post:

https://qiita.com/dahatake/items/05efc18eaf03605cb7d0


## Task:
- 幾つかの画像をインターネットで検索して、テストをしてみてください。
- Computer Vision の主な処理には`分類 (Image Classification)` **以外** には何がありますか?
- Computer Vision の学習をする**前**に行うタスクは何ですか? それは、どこまで自動化が出来そうですか?


# 3. AutoML

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

# (オプション) Azure for Student のセットアップ 推奨
- Microsoft Azure: 学生の方は無償で利用できる枠があります

# Azure 利用環境の作成

ここでは、大学・専門学校の皆様向けの方法を記載します。

## (オプション) Azure for Students の取得

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
