[CmdletBinding()]
param(
    [ValidateSet("anaconda", "miniconda", "venv", "all")]
    [string]$Mode,
    [switch]$RunNotebooks,
    [switch]$SkipDataDownload,
    [switch]$Yes,
    [switch]$AcceptAnacondaTos,
    [string]$InstallRoot,
    [string]$CondaPath,
    [switch]$InitShell,
    [switch]$Check,
    [switch]$DryRun,
    [ValidateSet("anaconda", "miniconda", "venv", "all")]
    [string]$Remove,
    [switch]$Help
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
[Console]::OutputEncoding = [Text.UTF8Encoding]::new()
$env:PYTHONUTF8 = "1"
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

$SetupRoot = $PSScriptRoot
$RepositoryRoot = Split-Path -Parent $SetupRoot
$EnvironmentFile = Join-Path $SetupRoot "envs\environment.yml"
$ConfigFile = Join-Path $SetupRoot "config\versions.conf"
$EnvironmentName = "mlfb-mnist"
$LogDirectory = Join-Path $SetupRoot "logs"
$StateDirectory = Join-Path $SetupRoot ".state"
$TestIsolate = $env:MLFB_TEST_ISOLATE -eq "1"
$IsolatedHome = $null

function Enable-TestIsolation {
    if (-not $TestIsolate) { return }

    $temporaryRoot = if ($env:RUNNER_TEMP) { $env:RUNNER_TEMP } else { $env:TEMP }
    $script:IsolatedHome = Join-Path $temporaryRoot ("mlfb-test-home-{0}-{1}" -f $env:USERNAME, (Split-Path -Leaf $RepositoryRoot))
    New-Item -ItemType Directory -Force -Path $script:IsolatedHome | Out-Null
    $isolatedTemp = Join-Path $script:IsolatedHome "AppData\Local\Temp"
    New-Item -ItemType Directory -Force -Path $isolatedTemp | Out-Null
    $env:USERPROFILE = $script:IsolatedHome
    $env:HOME = $script:IsolatedHome
    $env:TEMP = $isolatedTemp
    $env:TMP = $isolatedTemp
    Remove-Item Env:CONDA_EXE, Env:CONDA_PREFIX, Env:CONDA_DEFAULT_ENV -ErrorAction SilentlyContinue
    Write-Host "テスト隔離モード: $script:IsolatedHome"
}

function Show-Usage {
    @"
使い方: setup\setup-windows.cmd [-Mode anaconda|miniconda|venv|all] [-RunNotebooks] [-Yes]

既定は Anaconda です。Anaconda の利用規約に同意する場合だけ
-AcceptAnacondaTos を指定してください。

診断のみ: -Check
実行内容のみ表示: -DryRun
作成した環境だけ削除: -Remove anaconda|miniconda|venv|all
"@ | Write-Host
}

function Read-Config {
    $values = @{}
    foreach ($line in Get-Content -LiteralPath $ConfigFile) {
        if ($line -match '^\s*#' -or $line -notmatch '=') { continue }
        $pair = $line.Split('=', 2)
        $values[$pair[0].Trim()] = $pair[1].Trim()
    }
    return $values
}

function Get-IsolatedPython {
    param([hashtable]$Config)
    $destination = Join-Path $env:USERPROFILE "python"
    $python = Join-Path $destination "python.exe"
    if ($DryRun) {
        Write-Host "[dry-run] $python を Python $($Config["python_version"]) のインストーラーで作成します。"
        return $python
    }
    if (Test-Path -LiteralPath $python -PathType Leaf) { return $python }

    $version = $Config["python_version"]
    $installer = Join-Path ([IO.Path]::GetTempPath()) "python-$version-amd64.exe"
    $url = "https://www.python.org/ftp/python/$version/python-$version-amd64.exe"
    $expectedHash = $Config["python_windows_x64_sha256"]
    Write-Host "テスト用 Python をダウンロード中: $url"
    Invoke-WebRequest -UseBasicParsing -Uri $url -OutFile $installer
    try {
        $actualHash = (Get-FileHash -LiteralPath $installer -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($actualHash -ne $expectedHash) {
            throw "Python インストーラーの SHA-256 が一致しません。期待値: $expectedHash / 実測値: $actualHash"
        }
        $process = Start-Process -FilePath $installer -ArgumentList @(
            "/quiet", "InstallAllUsers=0", "TargetDir=$destination", "PrependPath=0",
            "Include_pip=1", "Include_test=0"
        ) -Wait -PassThru
        if ($process.ExitCode -ne 0) { throw "Python インストーラーが失敗しました。終了コード: $($process.ExitCode)" }
    }
    finally {
        Remove-Item -LiteralPath $installer -Force -ErrorAction SilentlyContinue
    }
    if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
        throw "テスト用 Python が見つかりません: $python"
    }
    return $python
}

function Write-Step {
    param([int]$Number, [string]$Message, [string]$Estimate)
    Write-Host ("[{0}/7] {1}（{2}）" -f $Number, $Message, $Estimate)
}

function Invoke-SetupCommand {
    param([string]$Description, [scriptblock]$Command)
    if ($DryRun) {
        Write-Host "[dry-run] $Description"
        return
    }
    & $Command | Out-Host
    if ($LASTEXITCODE -and $LASTEXITCODE -ne 0) {
        throw "$Description に失敗しました。終了コード: $LASTEXITCODE"
    }
}

function Get-InstallParent {
    if ($InstallRoot) { return $InstallRoot }
    if ($TestIsolate) { return $env:USERPROFILE }
    if ($env:USERPROFILE -match '[\s\u0080-\uFFFF]') {
        Write-Host "ユーザー名の文字を避けるため、C:\mlfb をインストール先に使います。"
        return "C:\mlfb"
    }
    return $env:USERPROFILE
}

function Get-Conda {
    param([string]$SelectedMode)
    if ($CondaPath) {
        if (-not (Test-Path -LiteralPath $CondaPath -PathType Leaf)) {
            throw "指定した Conda が見つかりません: $CondaPath"
        }
        return $CondaPath
    }
    $parent = Get-InstallParent
    $candidates = @(
        (Join-Path $parent "anaconda3\Scripts\conda.exe"),
        (Join-Path $parent "miniconda3\Scripts\conda.exe"),
        (Join-Path $env:LOCALAPPDATA "anaconda3\Scripts\conda.exe"),
        (Join-Path $env:LOCALAPPDATA "miniconda3\Scripts\conda.exe"),
        "C:\ProgramData\anaconda3\Scripts\conda.exe"
    )
    if ($TestIsolate) {
        $candidates = @(
            (Join-Path $parent "anaconda3\Scripts\conda.exe"),
            (Join-Path $parent "miniconda3\Scripts\conda.exe")
        )
    }
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) { return $candidate }
    }
    return $null
}

function Install-Conda {
    param([string]$SelectedMode, [hashtable]$Config)
    $parent = Get-InstallParent
    $tool = if ($SelectedMode -eq "anaconda") { "anaconda3" } else { "miniconda3" }
    $destination = Join-Path $parent $tool
    $prefix = if ($SelectedMode -eq "anaconda") { "anaconda" } else { "miniconda" }
    $name = $Config["${prefix}_windows_x64_name"]
    $hash = $Config["${prefix}_windows_x64_sha256"]
    $url = if ($SelectedMode -eq "anaconda") {
        "https://repo.anaconda.com/archive/$name"
    } else {
        "https://repo.anaconda.com/miniconda/$name"
    }
    $installer = Join-Path ([IO.Path]::GetTempPath()) $name

    Write-Step 3 "$SelectedMode をインストールしています" "10〜20 分"
    if ($DryRun) {
        Write-Host "[dry-run] $url -> $destination"
        return (Join-Path $destination "Scripts\conda.exe")
    }
    Write-Host "ダウンロード中: $url"
    Invoke-WebRequest -UseBasicParsing -Uri $url -OutFile $installer
    $actual = (Get-FileHash -LiteralPath $installer -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $hash) {
        throw "ダウンロードの SHA-256 が一致しません。期待値: $hash / 実測値: $actual"
    }
    $process = Start-Process -FilePath $installer -ArgumentList @(
        "/InstallationType=JustMe", "/AddToPath=0", "/RegisterPython=0", "/S", "/D=$destination"
    ) -Wait -PassThru
    Remove-Item -LiteralPath $installer -Force -ErrorAction SilentlyContinue
    if ($process.ExitCode -ne 0) { throw "インストーラーが失敗しました。終了コード: $($process.ExitCode)" }
    $conda = Join-Path $destination "Scripts\conda.exe"
    if (-not (Test-Path -LiteralPath $conda)) { throw "インストール後に Conda が見つかりません: $conda" }
    return $conda
}

function Invoke-CondaSetup {
    param([string]$SelectedMode, [hashtable]$Config)
    $conda = Get-Conda $SelectedMode
    if (-not $conda) { $conda = Install-Conda $SelectedMode $Config }
    if ($SelectedMode -eq "anaconda") {
        if ($DryRun) {
            Write-Host "[dry-run] Anaconda ToS の同意確認と conda tos accept を表示します。"
        } elseif (-not $AcceptAnacondaTos) {
            throw "Anaconda の利用規約へ同意するには -AcceptAnacondaTos を明示してください。"
        } else {
            Invoke-SetupCommand "Anaconda 利用規約への同意" { & $conda tos accept }
        }
    }
    Write-Step 4 "mlfb-mnist 環境を作成しています" "5〜15 分"
    Invoke-SetupCommand "Conda 環境の作成または更新" {
        $condarc = Join-Path ([IO.Path]::GetTempPath()) "mlfb-condarc-$PID"
        @"
channels:
  - conda-forge
  - pytorch
channel_priority: flexible
"@ | Set-Content -LiteralPath $condarc -NoNewline
        $previousCondarc = $env:CONDARC
        $previousChannels = $env:CONDA_CHANNELS
        try {
            $env:CONDARC = $condarc
            $env:CONDA_CHANNELS = "conda-forge,pytorch"
            $env:CONDA_CHANNEL_PRIORITY = "flexible"
            & $conda env update --name $EnvironmentName --file $EnvironmentFile --prune
        }
        finally {
            $env:CONDARC = $previousCondarc
            $env:CONDA_CHANNELS = $previousChannels
            Remove-Item -LiteralPath $condarc -Force -ErrorAction SilentlyContinue
        }
    }
    if ($InitShell) { Invoke-SetupCommand "conda init" { & $conda init powershell } }
    return $conda
}

function Get-Python {
    param([hashtable]$Config)
    if ($TestIsolate) { return (Get-IsolatedPython $Config) }
    $py = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($py) {
        & $py.Source -3.12 -c "import sys" 2>$null
        if ($LASTEXITCODE -eq 0) { return $py.Source }
    }
    $python = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($python -and $python.Source -notmatch 'WindowsApps') { return @($python.Source) }
    throw "Python 3.10〜3.14 が見つかりません。venv モードには Python をインストールしてください。"
}

function Invoke-VenvSetup {
    param([hashtable]$Config)
    Write-Step 3 "Python 仮想環境を作成しています" "5〜15 分"
    $python = Get-Python $Config
    $venv = Join-Path $RepositoryRoot ".venv"
    $py = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($py -and $python -eq $py.Source) {
        Invoke-SetupCommand "venv の作成" { & $python -3.12 -m venv $venv }
    } else {
        Invoke-SetupCommand "venv の作成" { & $python -m venv $venv }
    }
    $venvPython = Join-Path $venv "Scripts\python.exe"
    Invoke-SetupCommand "pip の更新" { & $venvPython -m pip install --upgrade pip }
    Invoke-SetupCommand "Notebook 依存関係のインストール" {
        & $venvPython -m pip install -r (Join-Path $SetupRoot "requirements\venv-base.txt") -r (Join-Path $SetupRoot "requirements\venv-torch.txt")
    }
    Invoke-SetupCommand "Jupyter カーネルの登録" {
        & $venvPython -m ipykernel install --sys-prefix --name mlfb-venv --display-name "Python (mlfb-venv)"
    }
    return $venvPython
}

function Remove-SetupEnvironment {
    param([string]$SelectedMode)
    if ($SelectedMode -in @("venv", "all")) {
        $venv = Join-Path $RepositoryRoot ".venv"
        if (Test-Path -LiteralPath $venv) { Remove-Item -LiteralPath $venv -Recurse -Force }
    }
    if ($SelectedMode -in @("anaconda", "miniconda", "all")) {
        foreach ($conda in @((Get-Conda "anaconda"), (Get-Conda "miniconda"))) {
            if ($conda) { & $conda env remove --name $EnvironmentName --yes }
        }
    }
    if ($TestIsolate -and $IsolatedHome -and (Test-Path -LiteralPath $IsolatedHome)) {
        Remove-Item -LiteralPath $IsolatedHome -Recurse -Force
    }
}

if ($Help) { Show-Usage; exit 0 }
if (-not (Test-Path -LiteralPath $EnvironmentFile)) { throw "環境定義が見つかりません: $EnvironmentFile" }
if (-not (Test-Path -LiteralPath $ConfigFile)) { throw "設定が見つかりません: $ConfigFile" }
if ([Environment]::Is64BitOperatingSystem -eq $false) { throw "Windows x64 だけをサポートします。" }

Enable-TestIsolation
New-Item -ItemType Directory -Force -Path $LogDirectory, $StateDirectory | Out-Null
$log = Join-Path $LogDirectory ("setup-windows-{0}.log" -f (Get-Date -Format "yyyyMMdd-HHmmss"))
Start-Transcript -LiteralPath $log | Out-Null
try {
    Write-Step 1 "事前チェックをしています" "1 分"
    $freeGb = [math]::Round(((Get-Item -LiteralPath $RepositoryRoot).PSDrive.Free / 1GB), 1)
    Write-Host "空き容量: $freeGb GB"
    if ($Check) { exit 0 }
    if ($Remove) { Remove-SetupEnvironment $Remove; exit 0 }

    if (-not $Mode) {
        if ($Yes) { $Mode = "anaconda" }
        else {
            Write-Host "1: Anaconda（おすすめ） 2: Miniconda（軽い） 3: venv（Python 標準） 4: すべて"
            $choice = Read-Host "番号を入力してください"
            $Mode = @{"1"="anaconda";"2"="miniconda";"3"="venv";"4"="all"}[$choice]
            if (-not $Mode) { throw "有効な番号を入力してください。" }
        }
    }
    Write-Step 2 "モードを決めています: $Mode" "1 分"
    $config = Read-Config
    $modes = if ($Mode -eq "all") { @("anaconda", "miniconda", "venv") } else { @($Mode) }
    foreach ($selected in $modes) {
        $python = if ($selected -eq "venv") { Invoke-VenvSetup $config } else { Invoke-CondaSetup $selected $config }
        Write-Step 5 "環境を検証しています" "1 分"
        if (-not $DryRun) {
            if ($selected -eq "venv") {
                & $python (Join-Path $SetupRoot "tools\verify_env.py") --data-dir (Join-Path $RepositoryRoot "data") $(if (-not $SkipDataDownload) { "--download-mnist" })
            } else {
                & $python run --name $EnvironmentName python (Join-Path $SetupRoot "tools\verify_env.py") --data-dir (Join-Path $RepositoryRoot "data") $(if (-not $SkipDataDownload) { "--download-mnist" })
            }
        }
        if ($RunNotebooks -and -not $DryRun) {
            Write-Step 6 "Notebook を実行しています" "数分〜数十分"
            if ($selected -eq "venv") {
                & $python (Join-Path $SetupRoot "tools\run_notebooks.py") --root $RepositoryRoot --kernel-name mlfb-venv
            } else {
                & $python run --name $EnvironmentName python (Join-Path $SetupRoot "tools\run_notebooks.py") --root $RepositoryRoot
            }
        }
    }
    Set-Content -LiteralPath (Join-Path $StateDirectory "last-mode.txt") -Value $Mode
    Write-Step 7 "完了しました" "完了"
    Write-Host "次は setup\start-jupyter-windows.cmd を実行して Jupyter Notebook を開いてください。"
}
catch {
    Write-Error ("セットアップに失敗しました: {0}`n次にすること: SETUP.md の該当する OS の節を確認してください。`nログ: {1}" -f $_.Exception.Message, $log)
    exit 1
}
finally {
    Stop-Transcript -ErrorAction SilentlyContinue | Out-Null
}
