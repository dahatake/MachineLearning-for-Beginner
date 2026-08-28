#Requires -Version 7.0
<#
.SYNOPSIS
Sets up and optionally executes every notebook in this repository on Windows x64.

.DESCRIPTION
Uses an existing Conda installation when possible. Otherwise, installs the pinned
Miniforge release for the current user after verifying its SHA-256 digest.

Official sources:
- Miniforge installation and platform support:
  https://github.com/conda-forge/miniforge
- Miniforge release metadata and SHA-256 digest:
  https://api.github.com/repos/conda-forge/miniforge/releases/tags/26.5.3-0
- Conda environment management:
  https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-environments.html
- nbconvert notebook execution:
  https://nbconvert.readthedocs.io/en/latest/execute_api.html
- nbconvert execution configuration:
    https://nbconvert.readthedocs.io/en/latest/config_options.html
#>

[CmdletBinding()]
param(
    [switch]$RunNotebooks,
    [switch]$SkipDataDownload,
    [string]$CondaExecutable = "",
    [string]$MiniforgePrefix = (Join-Path $HOME "Miniforge3")
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$RepositoryRoot = $PSScriptRoot
$EnvironmentName = "mlfb-mnist"
$EnvironmentFile = Join-Path $RepositoryRoot "environment.yml"
$DataDirectory = Join-Path $RepositoryRoot "data"
$ExecutedNotebookDirectory = Join-Path $RepositoryRoot "executed-notebooks"

$MiniforgeVersion = "26.5.3-0"
$MiniforgeInstallerName = "Miniforge3-$MiniforgeVersion-Windows-x86_64.exe"
$MiniforgeInstallerUrl = "https://github.com/conda-forge/miniforge/releases/download/$MiniforgeVersion/$MiniforgeInstallerName"
$MiniforgeSha256 = "ac05d86a4dbf3094fe112e14d1547a07cb10c6ae04bca274f88f1c09a5549876"

function Resolve-CondaExecutable {
    param(
        [string]$RequestedExecutable,
        [string]$RequestedMiniforgePrefix
    )

    if ($RequestedExecutable) {
        if (Test-Path -LiteralPath $RequestedExecutable -PathType Leaf) {
            return (Resolve-Path -LiteralPath $RequestedExecutable).Path
        }

        $requestedCommand = Get-Command -Name $RequestedExecutable -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($null -ne $requestedCommand -and $requestedCommand.CommandType -in @("Application", "ExternalScript")) {
            return $requestedCommand.Source
        }

        throw "指定された Conda 実行ファイルが見つかりません: $RequestedExecutable"
    }

    if ($env:CONDA_EXE -and (Test-Path -LiteralPath $env:CONDA_EXE -PathType Leaf)) {
        return (Resolve-Path -LiteralPath $env:CONDA_EXE).Path
    }

    $condaCommand = Get-Command -Name "conda" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($null -ne $condaCommand -and $condaCommand.CommandType -in @("Application", "ExternalScript")) {
        return $condaCommand.Source
    }

    $candidates = @(
        (Join-Path $RequestedMiniforgePrefix "Scripts\conda.exe"),
        (Join-Path $HOME "Miniforge3\Scripts\conda.exe"),
        (Join-Path $HOME "miniforge3\Scripts\conda.exe"),
        (Join-Path $HOME "miniconda3\Scripts\conda.exe"),
        (Join-Path $HOME "anaconda3\Scripts\conda.exe"),
        (Join-Path $env:LOCALAPPDATA "Miniforge3\Scripts\conda.exe"),
        (Join-Path $env:LOCALAPPDATA "miniforge3\Scripts\conda.exe"),
        (Join-Path $env:LOCALAPPDATA "miniconda3\Scripts\conda.exe"),
        (Join-Path $env:LOCALAPPDATA "anaconda3\Scripts\conda.exe")
    )

    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
    }

    return $null
}

function Install-Miniforge {
    param([string]$Prefix)

    $architecture = [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture
    if ($architecture -ne [System.Runtime.InteropServices.Architecture]::X64) {
        throw "Miniforge の公式 Windows インストーラーが x86_64 向けのため、このスクリプトは Windows x64 のみをサポートします。検出値: $architecture"
    }

    $resolvedPrefix = [System.IO.Path]::GetFullPath($Prefix)
    if ($resolvedPrefix -notmatch '^[A-Za-z]:\\[A-Za-z0-9._\\-]+$') {
        throw "Miniforge の既知の制約を避けるため、ASCII の英数字・ピリオド・ハイフン・アンダースコアだけで構成されるインストール先を指定してください: $resolvedPrefix"
    }

    if (Test-Path -LiteralPath $resolvedPrefix) {
        $existingItem = Get-ChildItem -LiteralPath $resolvedPrefix -Force | Select-Object -First 1
        if ($null -ne $existingItem) {
            throw "Conda を含まない既存ディレクトリには上書きしません: $resolvedPrefix"
        }
        Remove-Item -LiteralPath $resolvedPrefix -Force
    }

    $installerPath = Join-Path ([System.IO.Path]::GetTempPath()) $MiniforgeInstallerName
    try {
        Write-Host "==> Miniforge $MiniforgeVersion をダウンロードしています"
        Remove-Item -LiteralPath $installerPath -Force -ErrorAction SilentlyContinue
        Invoke-WebRequest -Uri $MiniforgeInstallerUrl -OutFile $installerPath

        $actualSha256 = (Get-FileHash -LiteralPath $installerPath -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($actualSha256 -ne $MiniforgeSha256) {
            throw "Miniforge インストーラーの SHA-256 が公式値と一致しません。期待値: $MiniforgeSha256 / 実測値: $actualSha256"
        }
        Write-Host "==> Miniforge インストーラーの SHA-256 を検証しました"

        $arguments = @(
            "/InstallationType=JustMe",
            "/AddToPath=0",
            "/RegisterPython=0",
            "/S",
            "/D=$resolvedPrefix"
        )
        $process = Start-Process -FilePath $installerPath -ArgumentList $arguments -Wait -PassThru
        if ($process.ExitCode -ne 0) {
            throw "Miniforge のインストールに失敗しました。終了コード: $($process.ExitCode)"
        }
    }
    finally {
        Remove-Item -LiteralPath $installerPath -Force -ErrorAction SilentlyContinue
    }

    $installedConda = Join-Path $resolvedPrefix "Scripts\conda.exe"
    if (-not (Test-Path -LiteralPath $installedConda -PathType Leaf)) {
        throw "インストール完了後に Conda が見つかりません: $installedConda"
    }

    return $installedConda
}

function Invoke-Conda {
    param(
        [string[]]$Arguments,
        [string]$Description
    )

    Write-Host "==> $Description"
    & $script:ResolvedCondaExecutable @Arguments
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        throw "$Description に失敗しました。Conda 終了コード: $exitCode"
    }
}

function Get-RepositoryNotebooks {
    $executedPrefix = [System.IO.Path]::GetFullPath($ExecutedNotebookDirectory) + [System.IO.Path]::DirectorySeparatorChar

    return @(
        Get-ChildItem -LiteralPath $RepositoryRoot -Filter "*.ipynb" -File -Recurse |
            Where-Object {
                $fullPath = [System.IO.Path]::GetFullPath($_.FullName)
                -not $fullPath.StartsWith($executedPrefix, [System.StringComparison]::OrdinalIgnoreCase) -and
                $fullPath -notmatch '[\\/]\.ipynb_checkpoints[\\/]'
            } |
            Sort-Object -Property FullName
    )
}

if (-not $IsWindows) {
    throw "setup.ps1 は Windows 用です。macOS では setup.sh を実行してください。"
}
$architecture = [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture
if ($architecture -ne [System.Runtime.InteropServices.Architecture]::X64) {
    throw "このスクリプトの Windows 対応対象は x64 です。検出値: $architecture"
}
if (-not (Test-Path -LiteralPath $EnvironmentFile -PathType Leaf)) {
    throw "環境定義が見つかりません: $EnvironmentFile"
}

$script:ResolvedCondaExecutable = Resolve-CondaExecutable -RequestedExecutable $CondaExecutable -RequestedMiniforgePrefix $MiniforgePrefix
if (-not $script:ResolvedCondaExecutable) {
    $script:ResolvedCondaExecutable = Install-Miniforge -Prefix $MiniforgePrefix
}
Write-Host "==> Conda: $script:ResolvedCondaExecutable"

$condaInfoJson = & $script:ResolvedCondaExecutable info --json
if ($LASTEXITCODE -ne 0) {
    throw "Conda の情報を取得できませんでした。"
}
try {
    $condaInfo = ($condaInfoJson -join "`n") | ConvertFrom-Json -AsHashtable
}
catch {
    throw "Conda の JSON 情報を解析できませんでした: $($_.Exception.Message)"
}
if ($condaInfo["platform"] -ne "win-64") {
    throw "Windows x64 用 Conda が必要です。検出されたプラットフォーム: $($condaInfo["platform"])"
}
Write-Host "==> Conda version=$($condaInfo["conda_version"]) platform=$($condaInfo["platform"])"

& $script:ResolvedCondaExecutable run --name $EnvironmentName python -c "pass" *> $null
$environmentExists = $LASTEXITCODE -eq 0

$savedChannelPriority = Get-Item -LiteralPath Env:CONDA_CHANNEL_PRIORITY -ErrorAction SilentlyContinue
try {
    # Windows and macOS obtain PyTorch 2.3.1 from different listed channels.
    # Limit this override to dependency solving; do not modify the user's .condarc.
    $env:CONDA_CHANNEL_PRIORITY = "flexible"
    if ($environmentExists) {
        Invoke-Conda -Description "Conda 環境 '$EnvironmentName' を更新しています" -Arguments @(
            "env", "update", "--name", $EnvironmentName, "--file", $EnvironmentFile, "--prune"
        )
    }
    else {
        Invoke-Conda -Description "Conda 環境 '$EnvironmentName' を作成しています" -Arguments @(
            "env", "create", "--name", $EnvironmentName, "--file", $EnvironmentFile, "--yes"
        )
    }
}
finally {
    if ($null -ne $savedChannelPriority) {
        $env:CONDA_CHANNEL_PRIORITY = $savedChannelPriority.Value
    }
    else {
        Remove-Item -LiteralPath Env:CONDA_CHANNEL_PRIORITY -ErrorAction SilentlyContinue
    }
}

$verificationCode = 'import sys, matplotlib, sklearn, torch, torchvision; from sklearn.datasets import load_digits; digits = load_digits(); assert digits.data.shape == (1797, 64); mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available(); print(f"Python={sys.version.split()[0]} matplotlib={matplotlib.__version__} scikit-learn={sklearn.__version__} torch={torch.__version__} torchvision={torchvision.__version__} cuda={torch.cuda.is_available()} mps={mps}")'
Invoke-Conda -Description "Python 依存関係と scikit-learn Digits データを検証しています" -Arguments @(
    "run", "--no-capture-output", "--name", $EnvironmentName, "python", "-c", $verificationCode
)

if (-not $SkipDataDownload) {
    New-Item -ItemType Directory -Path $DataDirectory -Force | Out-Null
    $downloadCode = 'import sys; from torchvision import datasets; root = sys.argv[1]; train = datasets.MNIST(root, train=True, download=True); test = datasets.MNIST(root, train=False, download=True); assert len(train) > 0 and len(test) > 0; print(f"MNIST train={len(train)} test={len(test)} root={root}")'
    Invoke-Conda -Description "torchvision の MNIST データを準備しています" -Arguments @(
        "run", "--no-capture-output", "--name", $EnvironmentName, "python", "-c", $downloadCode, $DataDirectory
    )
}

if ($RunNotebooks) {
    $notebooks = @(Get-RepositoryNotebooks)
    if ($notebooks.Count -eq 0) {
        throw "実行対象の Notebook が見つかりません。"
    }

    foreach ($notebook in $notebooks) {
        $relativePath = [System.IO.Path]::GetRelativePath($RepositoryRoot, $notebook.FullName)
        $relativeDirectory = Split-Path -Path $relativePath -Parent
        $outputDirectory = if ($relativeDirectory) {
            Join-Path $ExecutedNotebookDirectory $relativeDirectory
        }
        else {
            $ExecutedNotebookDirectory
        }
        New-Item -ItemType Directory -Path $outputDirectory -Force | Out-Null
        $outputName = "$($notebook.BaseName).executed.ipynb"

        Push-Location $notebook.DirectoryName
        try {
            Invoke-Conda -Description "$relativePath の全セルを実行しています" -Arguments @(
                "run", "--no-capture-output", "--name", $EnvironmentName,
                "jupyter", "nbconvert",
                "--to", "notebook", "--execute", $notebook.Name,
                "--output", $outputName,
                "--output-dir", $outputDirectory,
                "--ExecutePreprocessor.timeout=-1",
                "--ExecutePreprocessor.kernel_name=python3",
                "--ExecutePreprocessor.shutdown_kernel=immediate"
            )
        }
        finally {
            Pop-Location
        }
    }

    Write-Host "==> 実行済みノートブック: $ExecutedNotebookDirectory"
}

Write-Host ""
Write-Host "セットアップが完了しました。"
Write-Host "Jupyter Notebook を起動する場合:"
Write-Host "  & '$script:ResolvedCondaExecutable' run --no-capture-output --name $EnvironmentName jupyter notebook '$RepositoryRoot'"
if (-not $RunNotebooks) {
    Write-Host "全ノートブックを自動実行する場合:"
    Write-Host "  pwsh.exe -NoLogo -NoProfile -File '$PSCommandPath' -RunNotebooks"
}
