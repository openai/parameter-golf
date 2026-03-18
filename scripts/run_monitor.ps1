param(
    [string]$LogPath = ""
)

$ErrorActionPreference = "Stop"

$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$projectDir = Join-Path $root "tools\RunMonitor"
$project = Join-Path $projectDir "RunMonitor.csproj"
$exe = Join-Path $projectDir "bin\Debug\net9.0-windows\RunMonitor.exe"

if (-not (Test-Path $project)) {
    throw "RunMonitor project not found at $project"
}

Get-Process -Name "RunMonitor" -ErrorAction SilentlyContinue | Stop-Process -Force

$needsBuild = -not (Test-Path $exe)
if (-not $needsBuild) {
    $latestSource = Get-ChildItem -Path $projectDir -Recurse -File -Include *.cs,*.xaml,*.csproj |
        Sort-Object LastWriteTimeUtc -Descending |
        Select-Object -First 1
    if ($null -ne $latestSource) {
        $needsBuild = $latestSource.LastWriteTimeUtc -gt (Get-Item $exe).LastWriteTimeUtc
    }
}

if ($needsBuild) {
    & dotnet build $project | Out-Host
    if ($LASTEXITCODE -ne 0) {
        throw "RunMonitor build failed."
    }
}

$arguments = @()
if (-not [string]::IsNullOrWhiteSpace($LogPath)) {
    if (-not (Test-Path $LogPath)) {
        throw "Log path not found: $LogPath"
    }
    $arguments += (Resolve-Path $LogPath).Path
}

$process = if ($arguments.Count -gt 0) {
    Start-Process -FilePath $exe -ArgumentList $arguments -WorkingDirectory $root -PassThru
}
else {
    Start-Process -FilePath $exe -WorkingDirectory $root -PassThru
}
Write-Output ("RunMonitor PID={0}" -f $process.Id)
