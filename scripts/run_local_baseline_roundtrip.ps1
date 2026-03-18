param(
    [string]$RunId = "",
    [switch]$Background
)

$ErrorActionPreference = "Stop"

$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$launcher = Join-Path $root "scripts\run_local_3090.ps1"
if (-not (Test-Path $launcher)) {
    throw "Launcher not found at $launcher"
}

$args = @(
    "-ExecutionPolicy", "Bypass",
    "-File", $launcher,
    "-RunId", $(if ([string]::IsNullOrWhiteSpace($RunId)) { "baselinert3090_" + (Get-Date -Format "yyyyMMdd_HHmmss") } else { $RunId }),
    "-MaxWallclockSeconds", "180",
    "-TrainBatchTokens", "32768",
    "-ValBatchSize", "32768",
    "-ValMaxTokens", "524288",
    "-RoundtripValMaxTokens", "262144",
    "-TrainLogEvery", "10",
    "-ValLossEvery", "50",
    "-WarmupSteps", "0",
    "-NumLayers", "12",
    "-NumUniqueBlocks", "12",
    "-ModelDim", "384",
    "-EmbedDim", "0",
    "-NumHeads", "6",
    "-NumKvHeads", "3",
    "-MlpMult", "2",
    "-CompressionRegWeight", "0",
    "-CompressionRegInterval", "1",
    "-CompressionRegWarmupSteps", "0",
    "-CompressionRegSampleTensors", "0",
    "-CompressionRegMaxCols", "128",
    "-TernaryRegWeight", "0",
    "-OutlierRegWeight", "0",
    "-EvalCacheMixWeight", "0",
    "-EvalBigramMixWeight", "0",
    "-EvalCacheSize", "0",
    "-SaveRawCheckpoint", "0",
    "-FinalRoundtripEval", "1"
)

if ($Background) {
    $args += "-Background"
}

& powershell @args
