[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot

function Invoke-Checked {
    param([string]$Executable, [string[]]$Arguments)
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Executable failed with exit code $LASTEXITCODE"
    }
}

Push-Location $projectRoot
try {
    & (Join-Path $PSScriptRoot 'initialize_project.ps1')
    $condaInfo = (& conda info --envs --json | Out-String | ConvertFrom-Json)
    if ($LASTEXITCODE -ne 0) { throw 'Could not inspect conda environments.' }
    $existing = @($condaInfo.envs | Where-Object { (Split-Path -Leaf $_) -eq 'sat_clas' })
    if ($existing.Count -gt 0) {
        throw 'sat_clas already exists. Use it as documented in README.md; this script will not replace or alter it.'
    }
    Invoke-Checked conda @('create', '--name', 'sat_clas', '--override-channels', '--channel', 'conda-forge', 'python=3.11', 'pip', '--yes')
    Invoke-Checked conda @('run', '--no-capture-output', '-n', 'sat_clas', 'python', '-m', 'pip', 'install', 'torch', 'torchvision', '--index-url', 'https://download.pytorch.org/whl/cu130')
    Invoke-Checked conda @('run', '--no-capture-output', '-n', 'sat_clas', 'python', '-m', 'pip', 'install', '-r', 'requirements.txt')
    Invoke-Checked conda @('run', '--no-capture-output', '-n', 'sat_clas', 'python', '-m', 'pip', 'install', '--no-deps', '-e', '.')
    Invoke-Checked conda @('run', '--no-capture-output', '-n', 'sat_clas', 'python', '-m', 'ipykernel', 'install', '--user', '--name', 'sat_clas', '--display-name', 'Python (sat_clas)')
    Invoke-Checked conda @('run', '--no-capture-output', '-n', 'sat_clas', 'python', '-m', 'src.utils.verify_environment')
    Invoke-Checked conda @('run', '--no-capture-output', '-n', 'sat_clas', 'python', '-m', 'src.models.sam3_loader', '--inspect')
    Invoke-Checked conda @('run', '--no-capture-output', '-n', 'sat_clas', 'python', 'scripts/export_environment.py')
} finally {
    Pop-Location
}
