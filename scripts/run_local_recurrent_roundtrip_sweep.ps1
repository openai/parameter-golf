param(
    [string]$SweepId = ""
)

$ErrorActionPreference = "Stop"

$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$launcher = Join-Path $root "scripts\run_local_3090.ps1"
if (-not (Test-Path $launcher)) {
    throw "Launcher not found at $launcher"
}

if ([string]::IsNullOrWhiteSpace($SweepId)) {
    $SweepId = "recurtsweep_" + (Get-Date -Format "yyyyMMdd_HHmmss")
}

$controllerLog = Join-Path $root "logs\$SweepId.controller.txt"
New-Item -ItemType Directory -Force -Path (Join-Path $root "logs") | Out-Null

$experiments = @(
    @{ Suffix = "l16_u8_e000"; NumLayers = "16"; NumUniqueBlocks = "8"; EmbedDim = "0" },
    @{ Suffix = "l18_u6_e000"; NumLayers = "18"; NumUniqueBlocks = "6"; EmbedDim = "0" },
    @{ Suffix = "l16_u8_e256"; NumLayers = "16"; NumUniqueBlocks = "8"; EmbedDim = "256" },
    @{ Suffix = "l18_u6_e256"; NumLayers = "18"; NumUniqueBlocks = "6"; EmbedDim = "256" }
)

foreach ($experiment in $experiments) {
    $runId = "{0}_{1}" -f $SweepId, $experiment.Suffix
    Add-Content -Path $controllerLog -Value ("[{0}] START {1} layers={2} unique={3} embed={4}" -f (Get-Date -Format s), $runId, $experiment.NumLayers, $experiment.NumUniqueBlocks, $experiment.EmbedDim)
    $args = @(
        "-ExecutionPolicy", "Bypass",
        "-File", $launcher,
        "-RunId", $runId,
        "-MaxWallclockSeconds", "180",
        "-TrainBatchTokens", "32768",
        "-ValBatchSize", "32768",
        "-ValMaxTokens", "524288",
        "-RoundtripValMaxTokens", "262144",
        "-TrainLogEvery", "10",
        "-ValLossEvery", "50",
        "-WarmupSteps", "0",
        "-NumLayers", $experiment.NumLayers,
        "-NumUniqueBlocks", $experiment.NumUniqueBlocks,
        "-ModelDim", "384",
        "-EmbedDim", $experiment.EmbedDim,
        "-NumHeads", "6",
        "-NumKvHeads", "3",
        "-MlpMult", "2",
        "-WindowSize", "0",
        "-CompressionRegWeight", "0.005",
        "-CompressionRegInterval", "4",
        "-CompressionRegWarmupSteps", "32",
        "-CompressionRegSampleTensors", "4",
        "-CompressionRegMaxCols", "128",
        "-TernaryRegWeight", "0",
        "-OutlierRegWeight", "0",
        "-EvalCacheMixWeight", "0",
        "-EvalBigramMixWeight", "0",
        "-EvalCacheSize", "0",
        "-SaveRawCheckpoint", "0",
        "-FinalRoundtripEval", "1"
    )
    & powershell @args
    if ($LASTEXITCODE -ne 0) {
        Add-Content -Path $controllerLog -Value ("[{0}] FAIL {1}" -f (Get-Date -Format s), $runId)
        throw "Sweep run failed: $runId"
    }
    Add-Content -Path $controllerLog -Value ("[{0}] DONE {1}" -f (Get-Date -Format s), $runId)
}

Add-Content -Path $controllerLog -Value ("[{0}] SWEEP_DONE {1}" -f (Get-Date -Format s), $SweepId)
