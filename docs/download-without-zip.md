# zip を使わずにファイルを取得する手順

GitHub の **Code → Download ZIP** でダウンロードした際に、ブラウザーで「ダウンロードできませんでした - ウイルスが検出されました」と表示されることがあります。これは Microsoft Defender の機械学習による判定（`Trojan:Script/Wacatac.B!ml`）が、GitHub が生成した zip ファイルそのものに対して誤って反応したものです。リポジトリの内容にウイルスは含まれていません。

この場合は、以下のいずれかの方法で、このリポジトリーの全てのファイルを自分の PC に取得してください。

| 方法 | 事前のインストール | おすすめ度 |
|---|---|---|
| [方法 1: winget で Git をインストールしてクローンする](#方法-1-winget-で-git-をインストールしてクローンする-windows) | Git（この手順でインストール） | ◎ 最もおすすめ |
| [方法 2: Git を使わずに tar.gz 形式で取得する](#方法-2-git-を使わずに-targz-形式で取得する-windows) | 不要 | ○ |
| [macOS の場合](#macos-の場合) | Git（Command Line Tools） | ◎ |

> [!NOTE]
> このテキストでは`作業フォルダ`を **C:\Work** とします。他の場所を使う場合は、コマンド中の `C:\Work` を読み替えてください。

---

## 方法 1: winget で Git をインストールしてクローンする (Windows)

### 1-1. PowerShell を開く

1. キーボードの **Windows キー** を押します。
2. `PowerShell` と入力し、表示された **Windows PowerShell**（または **PowerShell**）をクリックします。

> [!TIP]
> 以降のコマンドは、コードブロック右上のコピーボタンでコピーし、PowerShell の画面で **右クリック**（または `Ctrl` + `V`）で貼り付けて、`Enter` キーを押して実行します。

### 1-2. winget が使えることを確認する

```powershell
winget --version
```

`v1.x.x` のようにバージョンが表示されれば OK です。

`winget : 用語 'winget' は、コマンドレット...として認識されません` と表示された場合は、Microsoft Store の **アプリ インストーラー** をインストール（または更新）してから、PowerShell を開き直してください。以下のコマンドで Microsoft Store の該当ページが開きます。

```powershell
Start-Process "ms-windows-store://pdp/?productid=9NBLGGH4NNS1"
```

### 1-3. Git をインストールする

```powershell
winget install --id Git.Git -e --source winget --accept-package-agreements --accept-source-agreements
```

- 途中で「このアプリがデバイスに変更を加えることを許可しますか?」（ユーザーアカウント制御）が表示されたら **はい** を選びます。
- `インストールが完了しました` と表示されれば OK です。
- `既存のパッケージが既にインストールされています` と表示された場合は、Git はインストール済みです。そのまま次へ進んでください。

### 1-4. Git が使えることを確認する

インストール直後の PowerShell では、`git` コマンドがまだ認識されないことがあります。以下のコマンドで、今開いている PowerShell に最新の PATH を読み込み直します。

```powershell
$env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [Environment]::GetEnvironmentVariable("Path", "User")
```

続いて、Git のバージョンを確認します。

```powershell
git --version
```

`git version 2.xx.x.windows.x` のように表示されれば OK です。表示されない場合は、PowerShell をいったん閉じて開き直してから、もう一度実行してください。

### 1-5. 作業フォルダを作成して移動する

```powershell
New-Item -ItemType Directory -Force -Path C:\Work
```

```powershell
Set-Location C:\Work
```

### 1-6. リポジトリーをクローンする

```powershell
git clone https://github.com/dahatake/MachineLearning-for-Beginner.git
```

`Cloning into 'MachineLearning-for-Beginner'...` と表示され、エラーなく終われば完了です。

### 1-7. ファイルを確認する

```powershell
Get-ChildItem C:\Work\MachineLearning-for-Beginner
```

`mnist`、`images`、`setup`、`README.md` などが表示されれば、全てのファイルの取得は完了です。エクスプローラーで開いて確認する場合は、以下を実行します。

```powershell
explorer.exe C:\Work\MachineLearning-for-Beginner
```

> [!NOTE]
> クローンしたファイルは、zip を展開した場合と違い「インターネットから取得したファイル」の印（Mark of the Web）が付かないため、[SETUP.md](../SETUP.md) にある zip の **ブロック解除** の操作は不要です。

### (参考) 後から最新の内容に更新する

リポジトリーの内容が更新された場合は、以下のコマンドで手元のファイルを最新にできます。

```powershell
git -C C:\Work\MachineLearning-for-Beginner pull
```

---

## 方法 2: Git を使わずに tar.gz 形式で取得する (Windows)

Git をインストールしたくない場合は、Windows 10 (1803 以降) / Windows 11 に標準で入っている `curl.exe` と `tar` を使って、zip ではなく **tar.gz 形式** で全てのファイルを取得できます。PowerShell を開いて、以下を順に実行します。

### 2-1. 作業フォルダを作成して移動する

```powershell
New-Item -ItemType Directory -Force -Path C:\Work
```

```powershell
Set-Location C:\Work
```

### 2-2. tar.gz ファイルをダウンロードする

```powershell
curl.exe -L -o MachineLearning-for-Beginner.tar.gz https://github.com/dahatake/MachineLearning-for-Beginner/archive/refs/heads/main.tar.gz
```

### 2-3. 展開する

```powershell
tar -xzf MachineLearning-for-Beginner.tar.gz
```

### 2-4. フォルダー名を変更し、不要になったファイルを削除する

展開すると `MachineLearning-for-Beginner-main` というフォルダーができます。方法 1 と同じフォルダー名に揃えます。

```powershell
Rename-Item -Path C:\Work\MachineLearning-for-Beginner-main -NewName MachineLearning-for-Beginner
```

```powershell
Remove-Item C:\Work\MachineLearning-for-Beginner.tar.gz
```

> [!NOTE]
> `Rename-Item` で「既に存在するファイルを作成することはできません」と表示された場合は、`C:\Work\MachineLearning-for-Beginner` が既にあります。古いフォルダーを別名に変更するか削除してから、もう一度実行してください。

### 2-5. ファイルを確認する

```powershell
Get-ChildItem C:\Work\MachineLearning-for-Beginner
```

`mnist`、`images`、`setup`、`README.md` などが表示されれば完了です。

---

## macOS の場合

**ターミナル**（Launchpad → その他 → ターミナル、または `Command` + `Space` で `ターミナル` と入力）を開いて、以下を順に実行します。

### Git の確認とインストール

```bash
git --version
```

バージョンが表示されれば Git はインストール済みです。「コマンドライン・デベロッパ・ツールが必要です」というダイアログが表示された場合は **インストール** をクリックし、完了後にもう一度 `git --version` を実行してください。ダイアログが表示されない場合は、以下でインストールを開始できます。

```bash
xcode-select --install
```

### 作業フォルダの作成とクローン

```bash
mkdir -p ~/Work
```

```bash
cd ~/Work
```

```bash
git clone https://github.com/dahatake/MachineLearning-for-Beginner.git
```

```bash
ls ~/Work/MachineLearning-for-Beginner
```

Git を使わない場合は、以下の 1 行で tar.gz 形式で取得・展開できます（`~/Work/MachineLearning-for-Beginner-main` に展開されます）。

```bash
curl -L https://github.com/dahatake/MachineLearning-for-Beginner/archive/refs/heads/main.tar.gz | tar -xz -C ~/Work
```

---

## うまくいかない場合

| 症状 | 対処 |
|---|---|
| `git` が認識されない | [1-4](#1-4-git-が使えることを確認する) の PATH の読み込み直しを実行するか、PowerShell を開き直します。 |
| `fatal: destination path 'MachineLearning-for-Beginner' already exists` | 既にクローン済みです。最新にする場合は `git -C C:\Work\MachineLearning-for-Beginner pull` を実行します。 |
| `Could not resolve host: github.com` | インターネットに接続できているか確認します。学校や会社のネットワークでは、プロキシ設定が必要な場合があります。 |
| tar.gz の取得でもウイルス検出と表示される | 方法 1（git clone）を使ってください。git clone は zip や tar.gz のようなアーカイブファイルを作らないため、この誤検知の対象になりません。 |
