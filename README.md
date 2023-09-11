# MachineLearning-for-Beginner
機械学習の原理や基礎をこれから学ぶ方向けのサンプルコードです。

ここでは以下の3つを体験します。

1. MNIST
機械学習での Hello World ともいえる極めて有名なデータセットです。手書き画像の0-9の分類を行います。Convolutional neural network (CNN) のモデルを作成します。
画像処理ですからGPUがあった方が処理は早いです。ですが、小さな画像ですので、GPUの無いコンピューターでも、それ程時間はかかりません。

2. Computer Vision のモデル作成のタスクを行う
主に、アノテーション(ラベリング)と、学習の実行、推論用Web Server作成を行います。

3. AutoML
機械学習の学習のプロセスでは、データセットを変えたり、ハイパーパラメーターを変えたりといった試行錯誤が幾度も行われます。
AutoML (自動機械学習) では、データセットの作成や調整・ハイパーパラメーターの自動調整だけでなく、各種学習環境用の仮想マシンのセットアップ・起動・インストール・停止や、複数ジョブの同時実行でのアンサンブル学習。各種ログの保存とグラフ化などを行ってくれます。
今回は Azure Machine Learning の AutoML でそれを試してみます。

# Azure 利用環境の作成

ここでは、大学・専門学校の皆様向けの方法を記載します。

## a. Azure for Students の取得

無料かつクレジットカードの登録無しで、100 USD 分/月 の Azure 利用などや各種ソフトウェアの利用もできる学生向けの特典を利用できます。

こちらのサイトから、学校のアカウント(メールアドレス)で、サインアップします。

https://azure.microsoft.com/ja-jp/free/students/

## b. クォーター増加
Azure for Students の場合、Azureが使える状態になっていても、初期状態ですと、利用できるCPUの数などが少ない場合があります。事前にそのクォーター(上限数)の増加を行ってください。

クォーターの増加手順: https://learn.microsoft.com/ja-jp/azure/quotas/quickstart-increase-quota-portal


### 利用する Azure の Service

- Azure AI Services : Custom Vision Service
    - プロジェクト 1つ以上
- Azure Macnine Learning Services
    - 30コア程度

**Azure Machine Learning** のクォーター増加手順は、上記と**別**になっています。ご注意ください。

Azure Machine Learning 用のコンピューター:

https://learn.microsoft.com/ja-jp/azure/machine-learning/how-to-manage-quotas?view=azureml-api-2

- GPUは不要です
- どのコンピューターでも大丈夫です。
- 専用コア と 低優先度コア があります。大人数でハンズオンなど実施する場合は **専用コア** を選択してください。「低優先度」の場合は、コンピューター利用時のデータセンターの空きに依存するため、コンピューターが起動しない場合がありえます。専用の場合は、大丈夫です
- コア数としては30あれば十分です。4コアの場合は、同時に5台のコンピューターが利用できます

### サービス自体が作成できない場合:

ご利用のAzure Subscription で、リソース プロバイダーが有効化されていない事があります。

https://learn.microsoft.com/ja-jp/azure/azure-resource-manager/management/resource-providers-and-types


リソースプロバイダーの一覧:
https://learn.microsoft.com/ja-jp/azure/azure-resource-manager/management/azure-services-resource-providers?source=recommendations

- Custom Vision Service は Cognitive Service (旧名) になっています


# 0. 環境構築
Windows 利用の前提です。WSLが使える、Windows 10 バージョン 2004 以上 (ビルド 19041 以上) または Windows 11 が推奨です。

https://learn.microsoft.com/ja-jp/windows/wsl/install#prerequisites


## 0.1. 環境セットアップ

可能であれば OS や各種ソフトウェアを最新にしましょう。Windows Update や、winget コマンドなどで最新にします。

winget で各種アプリケーションを最新のバージョンに
```cmd
winget upgrade --all
```

以下の環境をインストールします。
- Visual Studio Code
- WSL
- Python
- miniconda

以下は、AutoML のために使います。
- Azure Machine Learning Python SDK


### 0.1.1. wsl のインストール

こちらのドキュメントを参考にして、wsl をインストールします。
https://learn.microsoft.com/ja-jp/windows/wsl/install

- Windows のターミナルを**管理者**として起動します
- 以下のコマンドを実行します
```cmd
wsl --install
```
- OSを再起動します
- Ubuntu に管理者賞のUNIXユーザー名とパスワードを設定します。任意のもので。

### 0.1.2. Python がインストールされている事を確認

wsl でインストールされる Ubuntu には、Python がデフォルトでインストールされています。
Windows のターミナルで**wsl**を実行して、wsl へログイン。その後、以下のコマンドで念のため確認をします。

```cmd
python3 --version
```

もしインストールされていない場合は、wslにて以下のコマンドを実行します。

```cmd
sudo apt-get install python3
```

### 0.1.3. miniconda のインストール
Python で複数のアプリケーションの実行環境を使い分けるために、miniconda をインストールします。

- wsl のターミナルを起動します。

- 以下のコマンドでファイルをダウンロードします。
```shell
wget -c https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.s
```

- 以下のコマンドで、ダウンロードしたファイルを実行します
```shell
bash Miniconda3-latest-Linux-x86_64.sh
```
- ライセンスアグリーメントに同意する必要があります。Enterキーを押します。その後、SpaceキーもしくはEnterキーを押して、全てのライセンスアグリーメントのドキュメントを表示させます
- ライセンスに同意するのかを[yes]か[no]で入力します。ここでは、**yes** を入力してライセンスに同意します。
- インストールの場所を選択できます。そのまま Enterキー を押してデフォルトの場所にインストールします。
- インストールが終了すると、初期化するかどうかを設定できます。ここでも**yes**を入力して、初期化を実行します。

miniconda インストール:
https://docs.conda.io/en/latest/miniconda.html#installing


miniconda 各種コマンド:
https://docs.conda.io/projects/conda/en/stable/commands.html

以下の画面の通り、ターミナル内の表示に **(base)** がある事を確認します。これで miniconda が動いている事がわかります。

![conda-installed](/images/vscode-conda-validation.jpg)

### 0.1.4. Visual Studio Code のインストール

wsl のターミナルから、以下のコマンドを実行して、Visual Studio Code をインストールします。

```shell
code .
```

公式ドキュメント:
https://learn.microsoft.com/ja-jp/windows/wsl/tutorials/wsl-vscode


Windows に戻ります。以下のサイトから、Visual Studio Code をダウンロードして、Windows にインストールします

https://code.visualstudio.com/

Visual Studio Code を起動します。

Visual Studio Code で **Ctl + @** キーを押して、ターミナルを開きます。

以下の画面の通り、ターミナルから**Ubuntu**を選択して、wsl上の ubuntu に接続します。

![conect-wsl](/images/vscode-connect-wsl.jpg)


### 0.1.5 extension のインストール

Visul Studio Code 上で、Python や Jupyter Notebook などを利用できるように、各種 Extension (拡張機能) をインストールします。

ローカルの Windows と wsl 上の Ubuntu の双方の Visual Studio Code にインスタンスします。
![extension-installed](/images/vscode-extensions-installed.jpg)

Extension のインストールは、[Marketplace で拡張機能を検索する] から行うと便利です。

![Visual Studio Code の extansion](/images/vscode-extensions.jpg)

- 日本語パック

![日本語化](/images/vscode-extensions-japanese.jpg)

- Python のExtension

![Python](/images/vscode-extensions-python.jpg)

- wsl

![wsl](/images/vscode-extensions-wsl.jpg)


これで Visual Studio Code から wsl へ接続する準備が出来ました。

Visual Studio Code 画面左下のボタンを押して、wsl に接続します。

![connect-wsl](/images/vscode-connect-wsl.jpg)

接続が完了すると、画面左下に **wsl** が表示されます。

![connected-wsl](/images/vscode-connect-wsl-done.jpg)

wsl から Windows のファイルを参照する際には **/mnt/c/** フォルダーから参照します。WindowsのC Driveが、/mnt/ に接続されています。

![mnt-windows](/images/vscode-wsl-change-directory.jpg)

wsl に接続した Visual Studio Code での Extension を確認してください。先ほどの Extension がインストールされていない場合には、インストールを行います。

ここからはVisual Studio Code を使って環境設定を進めます。

### 0.1.6. MNIST用の miniconda 環境の作成

Visual Studio Code で **Ctl + @** キーを押して、ターミナルを開きます。


以下のコマンドで MNIST用の miniconda 環境を作成します。

```shell
conda create --name mnist
```

作成した環境に切り替えます。

```shell
conda activate mnist
```

今回のサンプルでは、PyTorch を使います。

CPUのみの場合:
```shell
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

GPU搭載のマシンの場合:
```shell
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

Windows へのインストールのドキュメント:
https://learn.microsoft.com/ja-jp/windows/ai/windows-ml/tutorials/pytorch-installation


以下のコマンドでインストールが成功しているかを確認します。

```shell
python
```

```shell
import torch
x = torch.rand(5, 3)
print(x)
```

以下の様な出力がされていれば正常にインストールされています。

```shell
tensor([[0.3380, 0.3845, 0.3217],
        [0.8337, 0.9050, 0.2650],
        [0.2979, 0.7141, 0.9069],
        [0.1449, 0.1132, 0.1375],
        [0.4675, 0.3947, 0.1426]])
```

https://pytorch.org/get-started/locally/#windows-verification


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