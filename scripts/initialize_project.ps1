[CmdletBinding()]
param([string]$ProjectRoot = '')

$ErrorActionPreference = 'Stop'
if (-not $ProjectRoot) { $ProjectRoot = Split-Path -Parent $PSScriptRoot }
$ProjectRoot = [IO.Path]::GetFullPath($ProjectRoot)
$directories = @(
    'data/raw', 'data/processed', 'data/interim', 'notebooks',
    'src/data', 'src/models', 'src/training', 'src/utils',
    'configs', 'outputs/logs', 'outputs/figures', 'outputs/predictions',
    'scripts', 'docs'
)
foreach ($relativePath in $directories) {
    $directoryPath = Join-Path $ProjectRoot $relativePath
    New-Item -ItemType Directory -Path $directoryPath -Force | Out-Null
    if ($relativePath.StartsWith('data/') -or $relativePath.StartsWith('outputs/')) {
        $keepPath = Join-Path $directoryPath '.gitkeep'
        if (-not (Test-Path -LiteralPath $keepPath)) {
            New-Item -ItemType File -Path $keepPath | Out-Null
        }
    }
}
foreach ($package in @('src', 'src/data', 'src/models', 'src/training', 'src/utils')) {
    $initPath = Join-Path (Join-Path $ProjectRoot $package) '__init__.py'
    if (-not (Test-Path -LiteralPath $initPath)) {
        Set-Content -LiteralPath $initPath -Value '"""Satellite classification project package."""' -Encoding utf8
    }
}
Write-Output "Project directories ready: $ProjectRoot"
