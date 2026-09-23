[CmdletBinding()]
param(
    [string]$CondaExe = "conda"
)

$ErrorActionPreference = "Stop"
$setupRoot = Split-Path -Parent $PSScriptRoot
$environmentFile = Join-Path (Join-Path $setupRoot "envs") "environment.yml"
$baseRequirements = Join-Path (Join-Path $setupRoot "requirements") "venv-base.txt"
$downloadRoot = Join-Path ([IO.Path]::GetTempPath()) ("mlfb-pip-resolve-" + [guid]::NewGuid())

function Invoke-CondaDryRun {
    param([string]$Subdir)

    Write-Host "Resolving conda environment for $Subdir"
    $previousSubdir = $env:CONDA_SUBDIR
    $previousOsx = $env:CONDA_OVERRIDE_OSX
    $previousGlibc = $env:CONDA_OVERRIDE_GLIBC
    try {
        $env:CONDA_SUBDIR = $Subdir
        Remove-Item Env:CONDA_OVERRIDE_OSX -ErrorAction SilentlyContinue
        Remove-Item Env:CONDA_OVERRIDE_GLIBC -ErrorAction SilentlyContinue
        switch ($Subdir) {
            "osx-64" { $env:CONDA_OVERRIDE_OSX = "12.0" }
            "osx-arm64" { $env:CONDA_OVERRIDE_OSX = "11.0" }
            { $_ -like "linux-*" } { $env:CONDA_OVERRIDE_GLIBC = "2.17" }
        }
        & $CondaExe env create --dry-run --file $environmentFile --prefix (Join-Path $downloadRoot "conda-$Subdir")
        if ($LASTEXITCODE -ne 0) { throw "Conda resolution failed for $Subdir." }
    }
    finally {
        $env:CONDA_SUBDIR = $previousSubdir
        $env:CONDA_OVERRIDE_OSX = $previousOsx
        $env:CONDA_OVERRIDE_GLIBC = $previousGlibc
    }
}

function Invoke-PipResolve {
    param(
        [string]$Platform,
        [string]$PythonVersion,
        [string]$Abi,
        [string[]]$TorchPackages,
        [string]$IndexUrl
    )

    $destination = Join-Path $downloadRoot "$Platform-py$PythonVersion"
    Write-Host "Resolving pip wheels for $Platform / Python $PythonVersion"
    $baseArguments = @(
        "-m", "pip", "download", "--dest", $destination, "--only-binary=:all:", "--platform", $Platform,
        "--implementation", "cp", "--python-version", $PythonVersion, "--abi", $Abi,
        "--requirement", $baseRequirements
    )
    & python @baseArguments
    if ($LASTEXITCODE -ne 0) { throw "Base pip wheel resolution failed for $Platform / Python $PythonVersion." }

    $torchArguments = @(
        "-m", "pip", "download", "--dest", $destination, "--only-binary=:all:", "--no-deps", "--platform", $Platform,
        "--implementation", "cp", "--python-version", $PythonVersion, "--abi", $Abi
    )
    if ($IndexUrl) {
        $torchArguments += @("--index-url", $IndexUrl)
    }
    $torchArguments += $TorchPackages
    & python @torchArguments
    if ($LASTEXITCODE -ne 0) { throw "pip wheel resolution failed for $Platform / Python $PythonVersion." }
}

if (-not (Get-Command $CondaExe -ErrorAction SilentlyContinue)) {
    throw "Conda executable was not found: $CondaExe"
}

try {
    foreach ($subdir in @("win-64", "osx-64", "osx-arm64", "linux-64", "linux-aarch64")) {
        Invoke-CondaDryRun $subdir
    }

    $pipTargets = @(
        @{ Platform = "win_amd64"; PythonVersion = "310"; Abi = "cp310"; TorchPackages = @("torch==2.3.1+cpu", "torchvision==0.18.1+cpu"); IndexUrl = "https://download.pytorch.org/whl/cpu" },
        @{ Platform = "win_amd64"; PythonVersion = "312"; Abi = "cp312"; TorchPackages = @("torch==2.3.1+cpu", "torchvision==0.18.1+cpu"); IndexUrl = "https://download.pytorch.org/whl/cpu" },
        @{ Platform = "macosx_10_15_x86_64"; PythonVersion = "310"; Abi = "cp310"; TorchPackages = @("torch==2.2.2", "torchvision==0.17.2"); IndexUrl = "" },
        @{ Platform = "macosx_12_0_arm64"; PythonVersion = "312"; Abi = "cp312"; TorchPackages = @("torch==2.3.1", "torchvision==0.18.1"); IndexUrl = "" },
        @{ Platform = "manylinux_2_17_x86_64"; PythonVersion = "313"; Abi = "cp313"; TorchPackages = @("torch==2.9.1+cpu", "torchvision==0.24.1+cpu"); IndexUrl = "https://download.pytorch.org/whl/cpu" },
        @{ Platform = "manylinux_2_17_aarch64"; PythonVersion = "314"; Abi = "cp314"; TorchPackages = @("torch==2.9.1", "torchvision==0.24.1"); IndexUrl = "" }
    )
    foreach ($target in $pipTargets) {
        Invoke-PipResolve @target
    }
}
finally {
    if (Test-Path $downloadRoot) {
        Remove-Item -Recurse -Force $downloadRoot
    }
}

Write-Host "Cross-platform dependency resolution passed."
