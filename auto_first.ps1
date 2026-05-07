 [CmdletBinding()]
param(
    [string]$RepoRoot = (Get-Location).Path,
    [int]$NProc = 1,
    [int[]]$Seeds = @(42, 1337, 2024),
    [string[]]$Candidates = @("stride16_candidate", "sota_int5_10l", "sota_int6_9l"),
    [int]$TrainShards = 10,
    [switch]$SkipDataDownload,
    [switch]$DryRun,
    [switch]$PrepareSubmission,
    [string]$SubmissionAuthor = "YOUR_NAME",
    [string]$SubmissionGithubId = "YOUR_GITHUB_ID",
    [Alias("?")][switch]$Help
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Resolve-RepoRoot {
    param([string]$Path)
    if (Test-Path (Join-Path $Path "README.md")) {
        return (Resolve-Path $Path).Path
    }
    $nested = Join-Path $Path "parameter-golf"
    if (Test-Path (Join-Path $nested "README.md")) {
        return (Resolve-Path $nested).Path
    }
    throw "Could not find parameter-golf repo root from '$Path'."
}

function Require-Command {
    param([string]$Name)
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Required command '$Name' is not available in PATH."
    }
}

function Get-Launcher {
    param([bool]$AllowNoTorch = $false)

    $torchrun = Get-Command torchrun -ErrorAction SilentlyContinue
    if ($torchrun) {
        return [PSCustomObject]@{
            kind = "torchrun"
            command = "torchrun"
        }
    }

    $pythonCmd = if (Get-Command python -ErrorAction SilentlyContinue) { "python" } elseif (Get-Command py -ErrorAction SilentlyContinue) { "py" } else { $null }
    if (-not $pythonCmd) {
        throw "Neither 'torchrun' nor Python launcher ('python'/'py') is available in PATH."
    }

    if (-not $AllowNoTorch) {
        # Validate torch exists in Python env before attempting distributed launch fallback.
        & $pythonCmd -c "import torch" *> $null
        if ($LASTEXITCODE -ne 0) {
            throw "PyTorch is not installed in the active environment. Install dependencies first (e.g., pip install -r requirements.txt)."
        }
    }

    return [PSCustomObject]@{
        kind = "python_torch_distributed"
        command = $pythonCmd
    }
}

function Test-CudaAvailable {
    param([string]$PythonCmd)
    if (-not $PythonCmd) {
        return $false
    }
    & $PythonCmd -c "import torch; print('1' if torch.cuda.is_available() else '0')" 2>$null | Out-String | ForEach-Object { $_.Trim() } | ForEach-Object {
        return ($_ -eq "1")
    }
    return $false
}

function Parse-RunLog {
    param([string]$LogFile)
    $lines = Get-Content -Path $LogFile -ErrorAction Stop

    $exact = $lines | Where-Object { $_ -match "final_int8_zlib_roundtrip_exact" } | Select-Object -Last 1
    $size = $lines | Where-Object { $_ -match "Total submission size int8\+zlib:" } | Select-Object -Last 1

    $valLoss = $null
    $valBpb = $null
    $totalBytes = $null

    if ($exact -and $exact -match "val_loss:([0-9]+\.[0-9]+)\s+val_bpb:([0-9]+\.[0-9]+)") {
        $valLoss = [double]$matches[1]
        $valBpb = [double]$matches[2]
    }
    if ($size -and $size -match "Total submission size int8\+zlib:\s*([0-9]+)\s*bytes") {
        $totalBytes = [int64]$matches[1]
    }

    [PSCustomObject]@{
        val_loss = $valLoss
        val_bpb = $valBpb
        total_bytes = $totalBytes
        valid_16mb = ($null -ne $totalBytes -and $totalBytes -le 16000000)
    }
}

function New-SubmissionFolder {
    param(
        [string]$Repo,
        [string]$CandidateKey,
        [string]$CandidateScript,
        [object[]]$CandidateRuns,
        [string]$Author,
        [string]$GithubId
    )

    $dateTag = Get-Date -Format "yyyy-MM-dd"
    $folderName = "${dateTag}_AutoFast_${CandidateKey}"
    $destDir = Join-Path $Repo (Join-Path "records/track_10min_16mb" $folderName)
    New-Item -ItemType Directory -Path $destDir -Force | Out-Null

    Copy-Item -Path (Join-Path $Repo $CandidateScript) -Destination (Join-Path $destDir "train_gpt.py") -Force

    foreach ($run in $CandidateRuns) {
        if (Test-Path $run.log_path) {
            Copy-Item -Path $run.log_path -Destination (Join-Path $destDir ("train_seed{0}.log" -f $run.seed)) -Force
        }
    }

    $vals = $CandidateRuns | Where-Object { $_.val_bpb -ne $null } | Select-Object -ExpandProperty val_bpb
    if ($vals.Count -eq 0) {
        throw "Cannot create submission metadata: no parsed val_bpb values."
    }
    $mean = ($vals | Measure-Object -Average).Average
    $name = "AutoFast $CandidateKey"
    $blurb = "Auto-generated record scaffold from scripted multi-seed run of $CandidateKey."

    $submissionJson = [ordered]@{
        name = $name
        val_loss = [math]::Round($mean, 6)
        bytes_total = 15900000
        blurb = $blurb
        author = $Author
        github_id = $GithubId
        date = (Get-Date -Format "yyyy-MM-dd")
    } | ConvertTo-Json -Depth 5
    Set-Content -Path (Join-Path $destDir "submission.json") -Value $submissionJson -Encoding UTF8

    $rows = @()
    foreach ($run in $CandidateRuns) {
        $rows += "| $($run.seed) | $($run.val_bpb) | $($run.total_bytes) | $($run.log_path) |"
    }
    $readme = @"
# $name

Automated scaffold generated by auto_first.ps1.

## Results

| Seed | val_bpb | total_bytes | log |
|------|---------|-------------|-----|
$($rows -join "`n")

Mean val_bpb: $([math]::Round($mean, 8))
"@
    Set-Content -Path (Join-Path $destDir "README.md") -Value $readme -Encoding UTF8

    return $destDir
}

if ($Help) {
    Write-Host "auto_first.ps1 - fast automated multi-seed runner for parameter-golf"
    Write-Host ""
    Write-Host "Usage examples:"
    Write-Host "  powershell -File .\\auto_first.ps1 -DryRun"
    Write-Host "  powershell -File .\\auto_first.ps1 -NProc 1 -TrainShards 1"
    Write-Host "  powershell -File .\\auto_first.ps1 -NProc 8 -PrepareSubmission -SubmissionAuthor 'Your Name' -SubmissionGithubId 'yourid'"
    Write-Host ""
    Write-Host "Key switches: -SkipDataDownload -DryRun -PrepareSubmission"
    return
}

$RepoRoot = Resolve-RepoRoot -Path $RepoRoot
Push-Location $RepoRoot
try {
    $launcher = Get-Launcher -AllowNoTorch:$DryRun

    $pythonCmd = if (Get-Command python -ErrorAction SilentlyContinue) { "python" } elseif (Get-Command py -ErrorAction SilentlyContinue) { "py" } else { $null }
    if (-not $pythonCmd -and -not $SkipDataDownload) {
        throw "Python is required for dataset download. Install Python or pass -SkipDataDownload."
    }

    if (-not $DryRun) {
        $hasCuda = Test-CudaAvailable -PythonCmd $pythonCmd
        if (-not $hasCuda) {
            throw "CUDA is not available in the active Python environment. This challenge requires GPU execution; run this script on an NVIDIA CUDA machine (e.g., RunPod H100)."
        }
    }

    $dataRoot = Join-Path $RepoRoot "data/datasets/fineweb10B_sp1024"
    $tokenizerPath = Join-Path $RepoRoot "data/tokenizers/fineweb_1024_bpe.model"
    if ((-not (Test-Path $dataRoot)) -or (-not (Test-Path $tokenizerPath))) {
        if ($DryRun) {
            Write-Host "[auto] DryRun: dataset/tokenizer not found locally; continuing without download."
        }
        elseif ($SkipDataDownload) {
            throw "Dataset/tokenizer missing and -SkipDataDownload was specified."
        }
        else {
            Write-Host "[auto] Downloading dataset/tokenizer (train-shards=$TrainShards)..."
            & $pythonCmd "data/cached_challenge_fineweb.py" "--variant" "sp1024" "--train-shards" "$TrainShards"
        }
    }

    $candidateMap = @{
        "sota_int5_10l" = @{
            script = "records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/train_gpt.py"
            desc = "Current leaderboard #1 baseline"
        }
        "sota_int6_9l" = @{
            script = "records/track_10min_16mb/2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA/train_gpt.py"
            desc = "Current leaderboard #2 baseline"
        }
        "stride16_candidate" = @{
            script = "records/track_10min_16mb/2026-03-22_10L_Int5MLP_MuonWD04_SWA50_Stride16/train_gpt.py"
            desc = "Current leaderboard #1 with tighter stride-16 sliding-window eval"
        }
    }

    $missing = @($Candidates | Where-Object { -not $candidateMap.ContainsKey($_) })
    if ($missing.Count -gt 0) {
        throw "Unknown candidate(s): $($missing -join ', '). Valid keys: $($candidateMap.Keys -join ', ')"
    }

    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $runDir = Join-Path $RepoRoot "runs/auto_first_$stamp"
    New-Item -ItemType Directory -Path $runDir -Force | Out-Null

    $results = @()
    foreach ($candidateKey in $Candidates) {
        $spec = $candidateMap[$candidateKey]
        $scriptPath = Join-Path $RepoRoot $spec.script
        if (-not (Test-Path $scriptPath)) {
            throw "Candidate script not found: $scriptPath"
        }

        foreach ($seed in $Seeds) {
            $runId = "auto_${candidateKey}_seed${seed}_$stamp"
            $logFile = Join-Path $runDir ("{0}_seed{1}.log" -f $candidateKey, $seed)

            Write-Host "[auto] Running $candidateKey seed=$seed"

            $oldSeed = $env:SEED
            $oldRunId = $env:RUN_ID
            $oldData = $env:DATA_PATH
            $oldTok = $env:TOKENIZER_PATH
            try {
                $env:SEED = "$seed"
                $env:RUN_ID = $runId
                $env:DATA_PATH = "./data/datasets/fineweb10B_sp1024"
                $env:TOKENIZER_PATH = "./data/tokenizers/fineweb_1024_bpe.model"

                if ($DryRun) {
                    if ($launcher.kind -eq "torchrun") {
                        "DRYRUN: torchrun --standalone --nproc_per_node=$NProc $scriptPath" | Tee-Object -FilePath $logFile
                    }
                    else {
                        "DRYRUN: $($launcher.command) -m torch.distributed.run --standalone --nproc_per_node=$NProc $scriptPath" | Tee-Object -FilePath $logFile
                    }
                }
                elseif ($launcher.kind -eq "torchrun") {
                    & torchrun --standalone "--nproc_per_node=$NProc" $scriptPath 2>&1 | Tee-Object -FilePath $logFile
                }
                else {
                    $oldLibuv = $env:USE_LIBUV
                    try {
                        # Windows wheels may lack libuv support; forcing 0 avoids rendezvous failure.
                        $env:USE_LIBUV = "0"
                        & $launcher.command -m torch.distributed.run --standalone "--nproc_per_node=$NProc" $scriptPath 2>&1 | Tee-Object -FilePath $logFile
                    }
                    finally {
                        $env:USE_LIBUV = $oldLibuv
                    }
                }
            }
            finally {
                $env:SEED = $oldSeed
                $env:RUN_ID = $oldRunId
                $env:DATA_PATH = $oldData
                $env:TOKENIZER_PATH = $oldTok
            }

            if ($DryRun) {
                $parsed = [PSCustomObject]@{ val_loss = $null; val_bpb = $null; total_bytes = $null; valid_16mb = $false }
            }
            else {
                $parsed = Parse-RunLog -LogFile $logFile
            }
            $results += [PSCustomObject]@{
                candidate = $candidateKey
                seed = $seed
                val_loss = $parsed.val_loss
                val_bpb = $parsed.val_bpb
                total_bytes = $parsed.total_bytes
                valid_16mb = $parsed.valid_16mb
                log_path = $logFile
                script = $spec.script
            }
        }
    }

    $resultsSorted = $results | Sort-Object val_bpb
    $resultsJsonPath = Join-Path $runDir "results.json"
    $resultsCsvPath = Join-Path $runDir "results.csv"
    $resultsSorted | ConvertTo-Json -Depth 5 | Set-Content -Path $resultsJsonPath -Encoding UTF8
    $resultsSorted | Export-Csv -Path $resultsCsvPath -NoTypeInformation -Encoding UTF8

    $grouped = @($results | Group-Object candidate)
    $summary = @()
    foreach ($g in $grouped) {
        $vals = @($g.Group | Where-Object { $_.val_bpb -ne $null } | Select-Object -ExpandProperty val_bpb)
        if ($vals.Count -eq 0) {
            continue
        }
        $mean = ($vals | Measure-Object -Average).Average
        $std = 0.0
        if ($vals.Count -gt 1) {
            $acc = 0.0
            foreach ($v in $vals) { $acc += [math]::Pow(($v - $mean), 2) }
            $std = [math]::Sqrt($acc / ($vals.Count - 1))
        }
        $summary += [PSCustomObject]@{
            candidate = $g.Name
            n = $vals.Count
            mean_val_bpb = [math]::Round($mean, 8)
            std_val_bpb = [math]::Round($std, 8)
            best_val_bpb = ($vals | Measure-Object -Minimum).Minimum
        }
    }
    $summarySorted = $summary | Sort-Object mean_val_bpb
    $summaryPath = Join-Path $runDir "summary.csv"
    $summarySorted | Export-Csv -Path $summaryPath -NoTypeInformation -Encoding UTF8

    Write-Host ""
    Write-Host "[auto] Finished. Top runs:"
    $resultsSorted | Select-Object -First 10 | Format-Table candidate, seed, val_bpb, total_bytes, valid_16mb -AutoSize
    Write-Host "[auto] Artifacts:"
    Write-Host "  $resultsJsonPath"
    Write-Host "  $resultsCsvPath"
    Write-Host "  $summaryPath"

    if ($PrepareSubmission) {
        if ($summarySorted.Count -eq 0) {
            throw "No valid parsed runs to build submission scaffold."
        }
        $winner = $summarySorted[0].candidate
        $winnerSpec = $candidateMap[$winner]
        $winnerRuns = $results | Where-Object { $_.candidate -eq $winner }
        $submissionDir = New-SubmissionFolder -Repo $RepoRoot -CandidateKey $winner -CandidateScript $winnerSpec.script -CandidateRuns $winnerRuns -Author $SubmissionAuthor -GithubId $SubmissionGithubId
        Write-Host "[auto] Submission scaffold created: $submissionDir"
    }
}
finally {
    Pop-Location
}
