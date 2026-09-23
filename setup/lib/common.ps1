[CmdletBinding()]
param([switch]$StartJupyter, [switch]$Navigator)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if (-not $StartJupyter) { return }

$SetupRoot = Split-Path -Parent $PSScriptRoot
$RepositoryRoot = Split-Path -Parent $SetupRoot
$EnvironmentName = "mlfb-mnist"
$venvPython = Join-Path $RepositoryRoot ".venv\Scripts\python.exe"

if ($venvPython -and (Test-Path -LiteralPath $venvPython)) {
    & $venvPython -m notebook --notebook-dir $RepositoryRoot
    exit $LASTEXITCODE
}

$candidates = @(
    (Join-Path $env:USERPROFILE "anaconda3\Scripts\conda.exe"),
    (Join-Path $env:USERPROFILE "miniconda3\Scripts\conda.exe"),
    (Join-Path $env:LOCALAPPDATA "anaconda3\Scripts\conda.exe"),
    (Join-Path $env:LOCALAPPDATA "miniconda3\Scripts\conda.exe")
)
foreach ($conda in $candidates) {
    if (Test-Path -LiteralPath $conda) {
        if ($Navigator) {
            & $conda run --name base anaconda-navigator
        } else {
            & $conda run --name $EnvironmentName jupyter notebook --notebook-dir $RepositoryRoot
        }
        exit $LASTEXITCODE
    }
}
throw "mlfb-mnist または .venv が見つかりません。先に setup-windows.cmd を実行してください。"
