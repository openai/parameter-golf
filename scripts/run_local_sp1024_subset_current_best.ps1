param(
    [string]$RunId = "",
    [switch]$Background
)

$ErrorActionPreference = "Stop"

$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$dataPath = Join-Path $root "data\datasets\fineweb10B_sp1024_local120k"
$tokenizerPath = Join-Path $root "data\tokenizers\fineweb_1024_bpe.model"

if (-not (Test-Path $dataPath)) {
    throw "SP1024 local subset dataset not found at $dataPath"
}
if (-not (Test-Path $tokenizerPath)) {
    throw "SP1024 tokenizer not found at $tokenizerPath"
}

$args = @(
    "-ExecutionPolicy", "Bypass",
    "-File", (Join-Path $root "scripts\run_local_3090.ps1"),
    "-RunId", $(if ([string]::IsNullOrWhiteSpace($RunId)) { "sp1024subsetbest_" + (Get-Date -Format "yyyyMMdd_HHmmss") } else { $RunId }),
    "-DataPath", $dataPath,
    "-TokenizerPath", $tokenizerPath,
    "-VocabSize", "1024",
    "-TieEmbeddings", "1",
    "-MaxWallclockSeconds", "0",
    "-Iterations", "300",
    "-TrainBatchTokens", "32768",
    "-ValBatchSize", "32768",
    "-ValMaxTokens", "524288",
    "-RoundtripValMaxTokens", "262144",
    "-TrainLogEvery", "10",
    "-ValLossEvery", "50",
    "-WarmupSteps", "0",
    "-NumLayers", "14",
    "-NumUniqueBlocks", "14",
    "-ModelDim", "576",
    "-EmbedDim", "0",
    "-NumHeads", "8",
    "-NumKvHeads", "4",
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

if ($Background) {
    $args += "-Background"
}

& powershell @args
