[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$setupRoot = Split-Path -Parent $PSScriptRoot
$root = Split-Path -Parent $setupRoot

foreach ($path in @(
    "$setupRoot\setup-windows.ps1",
    "$setupRoot\lib\common.ps1"
)) {
    $tokens = $null
    $errors = $null
    [void][System.Management.Automation.Language.Parser]::ParseFile($path, [ref]$tokens, [ref]$errors)
    if ($errors) { throw ($errors | Out-String) }
}

$scriptAnalyzer = Get-Command Invoke-ScriptAnalyzer -ErrorAction SilentlyContinue
if (-not $scriptAnalyzer) {
    throw "PSScriptAnalyzer が必要です。Install-Module PSScriptAnalyzer -Scope CurrentUser を実行してください。"
}
foreach ($path in @("$setupRoot\setup-windows.ps1", "$setupRoot\lib\common.ps1")) {
    $issues = Invoke-ScriptAnalyzer -Path $path -Severity Error
    if ($issues) { throw ($issues | Out-String) }
}

$bom = [byte[]](0xEF, 0xBB, 0xBF)
foreach ($path in @("$setupRoot\setup-windows.ps1", "$setupRoot\lib\common.ps1")) {
    $bytes = [IO.File]::ReadAllBytes($path)
    if ($bytes.Length -lt 3 -or -not (($bytes[0..2] -join ",") -eq ($bom -join ","))) {
        throw "UTF-8 BOM がありません: $path"
    }
}

foreach ($path in @(
    "$setupRoot\setup-mac.sh",
    "$setupRoot\setup-linux.sh",
    "$setupRoot\start-jupyter.sh",
    "$setupRoot\lib\common.sh"
)) {
    if ([IO.File]::ReadAllBytes($path) -contains 13) {
        throw "LF 改行である必要があります: $path"
    }
}

$bashFiles = @(
    "setup/setup-mac.sh",
    "setup/setup-linux.sh",
    "setup/start-jupyter.sh",
    "setup/lib/common.sh"
)
Push-Location $root
try {
    foreach ($path in $bashFiles) {
        & bash -n $path
        if ($LASTEXITCODE -ne 0) { throw "Bash 構文検査に失敗しました: $path" }
    }
    & shellcheck -e SC1091 @bashFiles
    if ($LASTEXITCODE -ne 0) { throw "ShellCheck に失敗しました。" }
}
finally {
    Pop-Location
}
& python "$setupRoot/tests/check_links.py"
if ($LASTEXITCODE -ne 0) { throw "check_links.py に失敗しました。" }

Write-Host "Static checks passed."
