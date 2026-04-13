param(
    [int]$Seed = 2025,
    [string]$RunSuffix = "",
    [int]$TrainShards = 125
)

$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
Set-Location $RepoRoot

$Python = if (Test-Path ".\.venv\Scripts\python.exe") {
    (Resolve-Path ".\.venv\Scripts\python.exe").Path
} else {
    "python"
}

if ([string]::IsNullOrWhiteSpace($RunSuffix)) {
    $RunSuffix = "seed$Seed"
}

& $Python ".\data\cached_challenge_fineweb.py" --variant sp1024 --train-shards $TrainShards

$env:RUN_ID = "saliency_24h_30gb_legacy_$RunSuffix"
$env:SEED = "$Seed"
$env:DATA_PATH = ".\data\datasets\fineweb10B_sp1024"
$env:TOKENIZER_PATH = ".\data\tokenizers\fineweb_1024_bpe.model"
$env:VOCAB_SIZE = "1024"
$env:ARTIFACT_BUDGET_BYTES = "15950000"
$env:USE_COMPILE = "0"
$env:LOCAL_PROXY_EVAL = "0"
$env:SALIENCY_ENABLE = "1"
$env:SALIENCY_TOKEN_PRIOR = "1"
$env:SALIENCY_BIGRAM = "0"
$env:SALIENCY_DYNAMIC = "1"
$env:SALIENCY_PHRASE = "1"
$env:SALIENCY_ATTN_BIAS = "1"
$env:SALIENCY_VALUE_SCALE = "0"
$env:SALIENCY_MIN = "0.90"
$env:SALIENCY_MAX = "1.12"
$env:SALIENCY_LAMBDA = "0.10"
$env:SALIENCY_GAMMA = "0.20"
$env:SALIENCY_BIGRAM_BUCKETS = "4096"
$env:SALIENCY_TOKEN_SCALE = "1.0"
$env:SALIENCY_BIGRAM_SCALE = "0.0"
$env:SALIENCY_DYNAMIC_SCALE = "0.10"
$env:SALIENCY_PHRASE_SCALE = "0.05"
$env:SALIENCY_REG_WEIGHT = "0.0"
$env:TRAIN_BATCH_TOKENS = "262144"
$env:TRAIN_SEQ_LEN = "1024"
$env:VAL_BATCH_SIZE = "65536"
$env:VAL_MAX_TOKENS = "0"
$env:ITERATIONS = "50000"
$env:WARMDOWN_ITERS = "3488"
$env:WARMUP_STEPS = "0"
$env:VAL_LOSS_EVERY = "2000"
$env:TRAIN_LOG_EVERY = "200"
$env:MAX_WALLCLOCK_SECONDS = "86400"

& $Python ".\records\track_non_record_16mb\2026-04-13_SaliencyGuidedLocal5090\train_gpt.py"
