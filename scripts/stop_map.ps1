$ErrorActionPreference = 'Stop'
$projectDirectory = Split-Path -Parent $PSScriptRoot
$pidFile = Join-Path $projectDirectory 'outputs/logs/map_server.pid'
if (-not (Test-Path -LiteralPath $pidFile)) {
    Write-Output 'No recorded background map server. For a terminal server, press Ctrl+C there.'
    exit 0
}
$mapServerProcessId = [int](Get-Content -LiteralPath $pidFile -Raw).Trim()
$mapServerProcess = Get-CimInstance Win32_Process -Filter "ProcessId = $mapServerProcessId"
if ($null -eq $mapServerProcess) {
    Write-Output 'The recorded map server is already stopped.'
    exit 0
}
if ($mapServerProcess.CommandLine -notmatch '-m\s+src\.cli\s+serve' -or
    $mapServerProcess.ExecutablePath -notlike '*\envs\sat_clas\python.exe') {
    throw 'The recorded process ID belongs to another command. It was not stopped.'
}
Stop-Process -Id $mapServerProcessId
Remove-Item -LiteralPath $pidFile
Write-Output 'Stopped the background sat_clas map server.'
