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
    try {
        $env:CONDA_SUBDIR = $Subdir
        & $CondaExe env create --dry-run --file $environmentFile --prefix (Join-Path $downloadRoot "conda-$Subdir")
        if ($LASTEXITCODE -ne 0) { throw "Conda resolution failed for $Subdir." }
    }
    finally {
        $env:CONDA_SUBDIR = $previousSubdir
    }
}

function Invoke-PipResolve {
    param(
        [string]$Platform,
        [string]$PythonVersion,
        [string]$Abi,
        [string[]]$TorchPackages
    )

    $destination = Join-Path $downloadRoot "$Platform-py$PythonVersion"
    Write-Host "Resolving pip wheels for $Platform / Python $PythonVersion"
    & python -m pip download --dest $destination --only-binary=:all: --platform $Platform `
        --implementation cp --python-version $PythonVersion --abi $Abi `
        --requirement $baseRequirements @TorchPackages
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
        @{ Platform = "win_amd64"; PythonVersion = "310"; Abi = "cp310"; TorchPackages = @("torch==2.3.1", "torchvision==0.18.1") },
        @{ Platform = "win_amd64"; PythonVersion = "312"; Abi = "cp312"; TorchPackages = @("torch==2.3.1", "torchvision==0.18.1") },
        @{ Platform = "macosx_10_15_x86_64"; PythonVersion = "310"; Abi = "cp310"; TorchPackages = @("torch==2.2.2", "torchvision==0.17.2") },
        @{ Platform = "macosx_11_0_arm64"; PythonVersion = "312"; Abi = "cp312"; TorchPackages = @("torch==2.3.1", "torchvision==0.18.1") },
        @{ Platform = "manylinux_2_17_x86_64"; PythonVersion = "313"; Abi = "cp313"; TorchPackages = @("torch==2.9.1", "torchvision==0.24.1") },
        @{ Platform = "manylinux_2_17_aarch64"; PythonVersion = "314"; Abi = "cp314"; TorchPackages = @("torch==2.9.1", "torchvision==0.24.1") }
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
