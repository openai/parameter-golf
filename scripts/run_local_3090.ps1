param(
    [string]$RunId = "",
    [int]$MaxWallclockSeconds = 420,
    [int]$TrainBatchTokens = 65536,
    [int]$ValBatchSize = 65536,
    [int]$ValMaxTokens = 4194304,
    [int]$RoundtripValMaxTokens = 2097152,
    [int]$TrainLogEvery = 10,
    [int]$ValLossEvery = 50,
    [int]$WarmupSteps = 0,
    [int]$NumLayers = 16,
    [int]$NumUniqueBlocks = 4,
    [int]$ModelDim = 512,
    [int]$EmbedDim = 0,
    [int]$NumHeads = 8,
    [int]$NumKvHeads = 4,
    [int]$MlpMult = 2,
    [double]$CompressionRegWeight = 0.02,
    [int]$CompressionRegInterval = 4,
    [int]$CompressionRegWarmupSteps = 10,
    [int]$CompressionRegSampleTensors = 4,
    [int]$CompressionRegMaxCols = 128,
    [double]$TernaryRegWeight = 0.15,
    [double]$OutlierRegWeight = 0.01,
    [double]$EvalCacheMixWeight = 0.03,
    [double]$EvalBigramMixWeight = 0.0,
    [int]$EvalCacheSize = 8,
    [int]$SaveRawCheckpoint = 0,
    [int]$FinalRoundtripEval = 0,
    [switch]$Background
)

$ErrorActionPreference = "Stop"

$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Python venv not found at $python"
}

if ([string]::IsNullOrWhiteSpace($RunId)) {
    $RunId = "local3090_" + (Get-Date -Format "yyyyMMdd_HHmmss")
}

$env:RUN_ID = $RunId
$env:DATA_PATH = (Join-Path $root "data\datasets\fineweb10B_sp1024")
$env:TOKENIZER_PATH = (Join-Path $root "data\tokenizers\fineweb_1024_bpe.model")
$env:VOCAB_SIZE = "1024"
$env:TIE_EMBEDDINGS = "1"
$env:NUM_LAYERS = $NumLayers.ToString()
$env:NUM_UNIQUE_BLOCKS = $NumUniqueBlocks.ToString()
$env:MODEL_DIM = $ModelDim.ToString()
$env:EMBED_DIM = $EmbedDim.ToString()
$env:NUM_HEADS = $NumHeads.ToString()
$env:NUM_KV_HEADS = $NumKvHeads.ToString()
$env:MLP_MULT = $MlpMult.ToString()
$env:ENABLE_TORCH_COMPILE = "0"
$env:SDP_BACKEND = "math"
$env:INT8_AXIS_MODE = "auto"
$env:INT8_RESIDUAL_RANK = "1"
$env:INT8_RESIDUAL_BUDGET_BYTES = "65536"
$env:TRAIN_BATCH_TOKENS = $TrainBatchTokens.ToString()
$env:VAL_BATCH_SIZE = $ValBatchSize.ToString()
$env:VAL_MAX_TOKENS = $ValMaxTokens.ToString()
$env:ROUNDTRIP_VAL_MAX_TOKENS = $RoundtripValMaxTokens.ToString()
$env:TRAIN_LOG_EVERY = $TrainLogEvery.ToString()
$env:VAL_LOSS_EVERY = $ValLossEvery.ToString()
$env:WARMUP_STEPS = $WarmupSteps.ToString()
$env:MAX_WALLCLOCK_SECONDS = $MaxWallclockSeconds.ToString()
$env:SAVE_RAW_CHECKPOINT = $SaveRawCheckpoint.ToString()
$env:FINAL_ROUNDTRIP_EVAL = $FinalRoundtripEval.ToString()
$env:COMPRESSION_REG_WEIGHT = $CompressionRegWeight.ToString([System.Globalization.CultureInfo]::InvariantCulture)
$env:COMPRESSION_REG_INTERVAL = $CompressionRegInterval.ToString()
$env:COMPRESSION_REG_WARMUP_STEPS = $CompressionRegWarmupSteps.ToString()
$env:COMPRESSION_REG_SAMPLE_TENSORS = $CompressionRegSampleTensors.ToString()
$env:COMPRESSION_REG_MAX_COLS = $CompressionRegMaxCols.ToString()
$env:TERNARY_REG_WEIGHT = $TernaryRegWeight.ToString([System.Globalization.CultureInfo]::InvariantCulture)
$env:OUTLIER_REG_WEIGHT = $OutlierRegWeight.ToString([System.Globalization.CultureInfo]::InvariantCulture)
$env:EVAL_CACHE_MIX_WEIGHT = $EvalCacheMixWeight.ToString([System.Globalization.CultureInfo]::InvariantCulture)
$env:EVAL_BIGRAM_MIX_WEIGHT = $EvalBigramMixWeight.ToString([System.Globalization.CultureInfo]::InvariantCulture)
$env:EVAL_CACHE_SIZE = $EvalCacheSize.ToString()

$scriptPath = Join-Path $root "train_gpt.py"
$stdoutPath = Join-Path $root "logs\$RunId.stdout.txt"
$stderrPath = Join-Path $root "logs\$RunId.stderr.txt"
New-Item -ItemType Directory -Force -Path (Join-Path $root "logs") | Out-Null

if ($Background) {
    $proc = Start-Process `
        -FilePath $python `
        -ArgumentList $scriptPath `
        -WorkingDirectory $root `
        -RedirectStandardOutput $stdoutPath `
        -RedirectStandardError $stderrPath `
        -PassThru
    Write-Output ("RUN_ID={0}" -f $RunId)
    Write-Output ("PID={0}" -f $proc.Id)
    Write-Output ("STDOUT={0}" -f $stdoutPath)
    Write-Output ("STDERR={0}" -f $stderrPath)
    exit 0
}

& $python $scriptPath
