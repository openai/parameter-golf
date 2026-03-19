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
    $SweepId = "isobyte_fixedstep_" + (Get-Date -Format "yyyyMMdd_HHmmss")
}

$controllerLog = Join-Path $root "logs\$SweepId.controller.txt"
New-Item -ItemType Directory -Force -Path (Join-Path $root "logs") | Out-Null

function Test-RoundtripRunComplete {
    param(
        [string]$LogPath
    )

    if (-not (Test-Path $LogPath)) {
        return $false
    }

    return [bool](Select-String -Path $LogPath -Pattern '^final_int8_zlib_roundtrip_exact ' -Quiet)
}

$experiments = @(
    @{ Suffix = "b10"; Target = "10MB"; Layers = "12"; Dim = "480"; Heads = "8";  KvHeads = "4" },
    @{ Suffix = "b12"; Target = "12MB"; Layers = "12"; Dim = "528"; Heads = "12"; KvHeads = "6" },
    @{ Suffix = "b14"; Target = "14MB"; Layers = "12"; Dim = "576"; Heads = "12"; KvHeads = "6" },
    @{ Suffix = "b155"; Target = "15.5MB"; Layers = "12"; Dim = "592"; Heads = "8"; KvHeads = "4" }
)

foreach ($experiment in $experiments) {
    $runId = "{0}_{1}" -f $SweepId, $experiment.Suffix
    $runLog = Join-Path $root "logs\$runId.txt"
    Add-Content -Path $controllerLog -Value ("[{0}] START {1} target={2} layers={3} dim={4} heads={5} kv={6}" -f (Get-Date -Format s), $runId, $experiment.Target, $experiment.Layers, $experiment.Dim, $experiment.Heads, $experiment.KvHeads)
    $args = @(
        "-ExecutionPolicy", "Bypass",
        "-File", $launcher,
        "-RunId", $runId,
        "-MaxWallclockSeconds", "0",
        "-Iterations", "300",
        "-TrainBatchTokens", "32768",
        "-ValBatchSize", "32768",
        "-ValMaxTokens", "524288",
        "-RoundtripValMaxTokens", "262144",
        "-TrainLogEvery", "10",
        "-ValLossEvery", "50",
        "-WarmupSteps", "0",
        "-NumLayers", $experiment.Layers,
        "-NumUniqueBlocks", $experiment.Layers,
        "-ModelDim", $experiment.Dim,
        "-EmbedDim", "0",
        "-NumHeads", $experiment.Heads,
        "-NumKvHeads", $experiment.KvHeads,
        "-MlpMult", "2",
        "-WindowSize", "0",
        "-Int8AxisMode", "auto",
        "-Int8ResidualRank", "1",
        "-Int8ResidualBudgetBytes", "65536",
        "-CompressionRegWeight", "0.005",
        "-CompressionRegInterval", "4",
        "-CompressionRegWarmupSteps", "32",
        "-CompressionRegSampleTensors", "4",
        "-CompressionRegMaxCols", "128",
        "-CompressionGridRegWeight", "0.10",
        "-CompressionScaleRegWeight", "0.0",
        "-CompressionRank1RegWeight", "0.0",
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
    if (-not (Test-RoundtripRunComplete -LogPath $runLog)) {
        Add-Content -Path $controllerLog -Value ("[{0}] FAIL {1} missing_final_roundtrip_metric" -f (Get-Date -Format s), $runId)
        throw "Sweep run missing final roundtrip metric: $runId"
    }
    Add-Content -Path $controllerLog -Value ("[{0}] DONE {1}" -f (Get-Date -Format s), $runId)
}

Add-Content -Path $controllerLog -Value ("[{0}] SWEEP_DONE {1}" -f (Get-Date -Format s), $SweepId)
