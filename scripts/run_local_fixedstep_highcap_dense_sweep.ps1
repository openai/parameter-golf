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
    $SweepId = "highcapdense_fixedstep_" + (Get-Date -Format "yyyyMMdd_HHmmss")
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

function Start-LauncherAndWait {
    param(
        [string[]]$LauncherArgs
    )

    $output = & powershell @($LauncherArgs + "-Background")
    if ($LASTEXITCODE -ne 0) {
        throw "Launcher failed before background start"
    }
    $pidLine = $output | Where-Object { $_ -match '^PID=' } | Select-Object -First 1
    if (-not $pidLine) {
        throw "Launcher did not report a PID"
    }
    $pid = [int]($pidLine -replace '^PID=', '')
    try {
        Wait-Process -Id $pid -ErrorAction Stop
    } catch {
        $proc = Get-Process -Id $pid -ErrorAction SilentlyContinue
        if ($proc) {
            throw
        }
    }
}

$experiments = @(
    @{ Suffix = "w608_l12"; Label = "width_14p4MB"; Layers = "12"; Dim = "608"; Heads = "8"; KvHeads = "4" },
    @{ Suffix = "w624_l12"; Label = "width_15p0MB"; Layers = "12"; Dim = "624"; Heads = "8"; KvHeads = "4" },
    @{ Suffix = "d576_l14"; Label = "depth_15p0MB"; Layers = "14"; Dim = "576"; Heads = "8"; KvHeads = "4" },
    @{ Suffix = "w640_l12"; Label = "width_15p7MB"; Layers = "12"; Dim = "640"; Heads = "8"; KvHeads = "4" }
)

foreach ($experiment in $experiments) {
    $runId = "{0}_{1}" -f $SweepId, $experiment.Suffix
    $runLog = Join-Path $root "logs\$runId.txt"
    if (Test-RoundtripRunComplete -LogPath $runLog) {
        Add-Content -Path $controllerLog -Value ("[{0}] SKIP {1} already_complete" -f (Get-Date -Format s), $runId)
        continue
    }
    Add-Content -Path $controllerLog -Value ("[{0}] START {1} label={2} layers={3} dim={4} heads={5} kv={6}" -f (Get-Date -Format s), $runId, $experiment.Label, $experiment.Layers, $experiment.Dim, $experiment.Heads, $experiment.KvHeads)
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
    try {
        Start-LauncherAndWait -LauncherArgs $args
    } catch {
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
