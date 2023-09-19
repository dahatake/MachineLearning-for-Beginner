# MachineLearning-for-Beginner
機械学習の原理や基礎をこれから学ぶ方向けのサンプルコードです。

ここでは以下の2つを体験を通して。機械学習について学びます。

1. MNIST
機械学習での Hello World ともいえる極めて有名なデータセットです。手書き画像の0-9の**分類**を行います。Convolutional neural network (CNN) のモデルを作成します。
画像処理ですからGPUがあった方が処理は早いです。ですが、小さな画像ですので、GPUの無いコンピューターでも、それ程時間はかかりません。

2. Computer Vision のモデル作成のタスクを行う
主に、アノテーション(ラベリング)と、学習の実行を行います。


# Azure 利用環境の作成

ここでは、大学・専門学校の皆様向けの方法を記載します。

## a. Azure for Students の取得

無料かつクレジットカードの登録無しで、100 USD 分/月 の Azure 利用などや各種ソフトウェアの利用もできる学生向けの特典を利用できます。

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


# 0. 環境構築

予め出来上がった環境を使う方法もあります。ですが。

以下のアプリケーションをインストールします。
- Python
- Visual Studio Code

## 0.1. Python のインストール

以下の公式サイトからインストールを行います

https://www.python.org/downloads/

インストールが終わったら、コマンドプロンプトもしくはターミナルを起動します。以下のコマンドでインストールされている事を確認をします。

```shell
python3 --version
```

### 0.2. Visual Studio Code のインストール

以下のサイトから、Visual Studio Code をダウンロードして、Windows にインストールします

https://code.visualstudio.com/

Visual Studio Code を起動します。

### 0.2.1 extension のインストール

Visul Studio Code 上で、Python や Jupyter Notebook などを利用できるように、各種 Extension (拡張機能) をインストールします。

Extension のインストールは、[Marketplace で拡張機能を検索する] から行うと便利です。

![Visual Studio Code の extansion](/images/vscode-extensions.jpg)

- 日本語パック

![日本語化](/images/vscode-extensions-japanese.jpg)

- Python のExtension

![Python](/images/vscode-extensions-python.jpg)

- Jupyter Notebook の Extension

![notebook](/images/vscode-extensions-notebook.jpg)


# 1. MNIST - 初めての機械学習モデルの作成
機械学習の入門として代表的なサンプルになります。

## 目的
- 自分の環境でモデルの作成を行う
- 機械学習の学習のプログラムの概要を理解する
- 深層学習 (Deep Learning)のプログラムの概要を理解する


プログラムのコードはこちらです。

[コード](/1.mnist/mnist.py)

- ターミナルで [1.mnist] フォルダに移動
- 以下のコマンドを実行

```shell
python mnist.py
```

オリジナルコードはこちらです。
https://github.com/pytorch/examples/tree/main/mnist

実行結果は、以下の様になります。

![結果](/images/mnist-executed.jpg)

## Task:
- MNIST の写真の一部をみて、どんなデータセットなのかを理解します
- ニューラルネットワークの図を書いてみます
- プログラムの構造をリスト化します。どこで何をしているのか?
- 作成したモデルがファイルに保存をされていません。保存するためにプログラムを修正します

# 2. Deep Learning - Computer Vision

機械学習で画像を扱う処理をComputer Vision と称しています。
PyTorch でも実装できます。ですが、ここではクラウドサービスを使って学習までの一連の流れを体験します。クラウドサービスであれば、GPU搭載コンピューターがあったり、各種煩雑な作業が実装されている事が多いからです。

## 目的
- ツールの存在を知る。そのツールで出来る事、出来ない事を知る


Microsoft Learn のドキュメントの通り実行してください。

https://learn.microsoft.com/ja-jp/training/modules/classify-images-custom-vision/

- 演習から、英語のe-learning のサイトに行きます
- 英語の[AI-900-AIFundamentals]のサイトでは、前処理・モデル作成・推論環境作成・呼び出しアプリケーション作成までを行うテキストがあります。推論環境作成 **[Publish the image classification model]** まで実行してください。それ以降を行う必要はありません。

## Task:
- Computer Vision の主な処理には分類 (Image Classification)以外に何がありますか?
- Computer Vision の学習をする前に行うタスクは何ですか? それは、どこまで自動化が出来そうですか?

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

# 4. MNIST Update 1 - 学習状況と作成したモデルの保存
最初に作成した MNIST の学習用のコードの場合、学習の状況がわからないのと、モデルの保存は自分のPCです。つまり、他の人と共有が難しいですし、学習用のコードが動いている間は、そのPCは起動し続けさせる必要があります。

ここでは、学習用のコードと、それを制御するコードを分離させます。そして、学習状況や、モデルのファイルを一括して保存・管理してくれるサービスとして Azure Machine Learning を使います。

## 目的
- チームでのモデル開発
- 学習時の状況を記録して、比較検討を出来るようにする


### 4.1. Mminiconda 環境の作成

Visual Studio Code で **Ctl + @** キーを押して、ターミナルを開きます。


以下のコマンドで MNIST用の miniconda 環境を作成します。

```shell
conda create --name mnist-azureml
```

作成した環境に切り替えます。

```shell
conda activate mnist-azureml
```

Azure Machine Learning の Python SDK をインストールします。

```shell
pip install azure-ai-ml
pip install azure-identity
pip install mlflow azureml-mlflow
```

https://learn.microsoft.com/ja-jp/python/api/overview/azure/ai-ml-readme?view=azure-python