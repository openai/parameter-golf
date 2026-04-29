# AGENTS.md — Parameter Golf Contest Rules & Codebase Guide

## Contest Overview

**OpenAI Model Craft Challenge: Parameter Golf** — train the best language model that fits in a **16 MB artifact = 16,000,000 bytes in decimal including code and compressed model** and for the record track trains in **under 10 minutes on 8×H100s** and for the non-record track can train longer, scored by **bits per byte (val_bpb)** on the FineWeb validation set (tokenizer-agnostic).

This is an L(N) optimization challenge: minimize loss given a fixed parameter budget, unconstrained by data, compute, steps, or architecture.

## work directories

You must do all work under /hpfs/scratch/gpfs/mcclec07/parameter_golf2 . The only other directory (and subdirectories) you are allowed to read from are under /hpfs/scratch/gpfs/mcclec07/code/parameter_golf/. 

## Hard Rules

1. **16 MB artifact cap** — Code bytes (`train_gpt.py` + deps) plus compressed model bytes ≤ 16,000,000 bytes (decimal, not MiB).
2. **10-minute training wall-clock** on 8×H100 SXM. `MAX_WALLCLOCK_SECONDS=600`.
3. **10-minute evaluation wall-clock** (separate from training).
4. **Self-contained artifact** — No external downloads, no network calls, no training-data access during evaluation.
5. **No validation-set cheating** — You CANNOT access validation data during training. Test-time training is allowed ONLY on validation tokens you have already evaluated (already graded).
6. **Statistical significance** — New SOTA records must beat existing SOTA by ≥ 0.005 nats at p < 0.01. Provide enough run logs (typically ≥ 3 runs).
7. **No brute-force seed search** or other schemes that smuggle in extra compute unfairly.
8. **Tokenizer changes** require extra proof that val_bpb is correctly calculated; these are scrutinized more heavily.
9. **External packages are allowed** (e.g. FlashAttention) as long as they don't smuggle extra compute, capabilities, or inflated code size. Include a `requirements.txt` and setup instructions.
10. **Non-reproducible results can be disqualified.**
11. Make sure to include code and artifacts for all claims when preparing PR's — if you claim a certain val_bpb, you must provide the corresponding `train.log` and `train_gpt.py` that achieves it within the rules.
12. Review https://matotezitanka.github.io/parameter-golf/ for more details on rules clarifications and potential strategies.
13. Make sure that data is downloaded from runpod jobs before they terminate.`

## Scoring

- Primary metric: **val_bpb** (bits per byte) on the full `fineweb_val_*` split.
- Lower is better.
- Evaluation can use any sequence length.
- Sliding-window / overlapping evaluation is allowed and encouraged.

## BPB Calculation Safety

**Treat BPB accounting as a critical correctness surface, not a cosmetic reporting detail.** A model can be trained correctly while still reporting invalid `val_bpb` if byte normalization or token-byte accounting is wrong.

- **Byte accounting must match the tokenizer semantics exactly.** For SentencePiece tokenizers, be especially careful with leading-space markers such as `▁`. Do not count the same space byte twice via both LUT construction and per-token evaluation logic.
- **Normalization must be by true byte count, not token count.** Always verify that `bits per byte = total_nll_bits / total_raw_bytes` uses the correct denominator over the evaluated target bytes only.
- **Keep training-time validation and standalone eval consistent.** If there are duplicated helpers (for example LUT builders or sliding-window evaluators), update and verify all copies together.
- **Validate against known-good implementations.** Before trusting a new BPB method, compare its logic and outputs against merged PRs or previously validated evaluation paths in this repository.
- **When changing tokenizer handling, eval stride/windowing, or TTT accounting, assume BPB can silently drift.** Re-audit the denominator and confirm that any conditional byte additions are neither omitted nor double-counted.
- **Never claim an improvement from a new evaluation method until the accounting has been checked.** If a result looks surprisingly good, first suspect metric-accounting bugs before assuming a modeling breakthrough.
- **When possible, report both the formula and the implementation path used to compute BPB.** This makes later audits much easier.

## Submission Format

Every submission is a PR adding a new folder under `records/track_10min_16mb/` (or `records/track_non_record_16mb/` for non-record / unlimited-compute runs). Required files:

1. `README.md` — Explain your approach in detail.
2. `submission.json` — Name, GitHub ID, val_bpb, metadata.
3. `train.log` — Training log demonstrating statistical significance.
4. `train_gpt.py` — Must compile and run from inside the records folder.
5. `requirements.txt` (if extra deps needed).

## Repository Structure

```
parameter_golf2/
├── train_gpt.py          # Training script (keep ≤ 1500 lines)
├── train_gpt_mlx.py      # Apple Silicon local dev script
├── data/                  # Dataset download & tokenizer utilities
├── records/
│   ├── track_10min_16mb/         # Official leaderboard submissions
│   └── track_non_record_16mb/    # Non-record / unlimited-compute submissions
├── plans/                # Science plans and reviews
├── requirements.txt
├── AGENTS.md             # This file
├── README.md
└── THIRD_PARTY_NOTICES.md
```

## Key Technical Details

- **Baseline**: 9-layer transformer, dim=512, 8 attn heads / 4 KV heads (GQA), 2× MLP, vocab 1024, seq 1024, tied embeddings.
- **Optimizer**: Muon (for matrix params) + Adam (for scalars/embeddings). See `train_gpt.py`.
- **Quantization**: INT8 zlib round-trip is default compression. INT6/INT5/mixed-precision QAT and zstd explored by top submissions.
- **Data**: FineWeb 10B tokens, 1024-token BPE vocabulary (`fineweb_1024_bpe.model`). 80 training shards, fixed validation split.
- **Distributed**: `torchrun --standalone --nproc_per_node=8 train_gpt.py` for 8-GPU runs.

## Winning Strategies (from leaderboard)

- Depth recurrence / parameter tying
- Aggressive test-time training (LoRA TTT on already-evaluated tokens)
- INT5/INT6 mixed-precision QAT + zstd compression
- 3× MLP expansion, SmearGate, BigramHash embeddings
- Sliding-window evaluation (stride < seq_len)
- Longer training sequences (2048–4096)
- SWA (Stochastic Weight Averaging), weight decay tuning
- Orthogonal initialization, spectral embedding init

---

## RunPod Operational Guardrails

### RunPod Security & Data Policy

- API keys MUST be set via environment variables only (`RUNPOD_API_KEY`). **NEVER** write API keys to files, scripts, or commit them.
- Do not assume the system `python3` on this HPC can import the RunPod SDK. Use `runpodctl` or an explicit virtual environment/interpreter with `runpod` installed before relying on SDK-based scripts.
- Only the following files/directories may be uploaded to RunPod pods:
  - `train_gpt.py` (training script)
  - `data/cached_challenge_fineweb.py` (data download script)
  - `data/tokenizer_specs.json`
  - `requirements.txt`
  - Compressed model artifacts (`.ptz`, `.pt` files) for eval-only runs
- **NEVER upload:** `.git/`, `plans/`, `records/` (except the specific record being tested), personal notes, API keys, credentials.
- All RunPod pods must be terminated after use. No idle pods.

### RunPod Proven Vs Unproven Paths

**Consistently working from this HPC:**

- RunPod GraphQL API — pod creation, status polling, listing, termination — all reliable over HTTPS.
- HTTP-bootstrap launch (`scripts/runpod_http_rehearsal.py`): bundle allowed files into a base64 env var, inject via `dockerArgs` at pod creation, unpack on boot, run the job, serve results via `python3 -m http.server 30000`, and retrieve artifacts through the RunPod HTTPS proxy (`https://{pod_id}-30000.proxy.runpod.net/`). This path has been validated end-to-end for environment probes, short training, stability runs, and short eval — all with successful artifact retrieval back to this HPC.
- Pod-side self-termination via a background shell that sleeps for `PGOLF_HARD_DEADLINE_SEC` and then calls the `podTerminate` GraphQL mutation (falling back to `kill 1`). This fires regardless of whether the HPC session is still alive.
- Budget-gated launches: balance check before creation, cost estimation, and deterministic teardown in a `try/finally` block.

**Consistently failing — do not use as a primary path:**

- **SSH from this HPC to RunPod pods.** Outbound SSH is blocked or unreliable on this cluster. Do not rely on `ssh`, `scp`, `rsync`, or `runpodctl ssh` for file transfer or command execution.
- **Jupyter contents-API file uploads (`PUT /api/contents/...`).** XSRF token management through the RunPod HTTPS proxy is fragile; uploads have silently failed or produced empty files. The `runpod_fetch_*` directories at repo root are artifacts of these failures.
- **Jupyter kernel execution for running commands.** Depends on Jupyter being fully started and correctly proxied; has been unreliable for orchestrating jobs from this HPC.

These may work from other networks or with manual browser interaction, but they are not dependable for automated, unattended control from this HPC.

**Recommended primary path — HTTP-bootstrap fallback:**

The HTTP-bootstrap pattern (`scripts/runpod_http_rehearsal.py`) works because it avoids SSH and Jupyter entirely:

1. **Bundle injection at pod creation.** Allowed files are tar-gzipped, base64-encoded, and passed as the `PGOLF_BUNDLE_B64` environment variable in the `create_pod` call. The pod's `dockerArgs` boot command decodes and extracts the bundle. No post-creation file upload is needed.
2. **Job execution in the boot command.** The user command runs as part of the pod's startup script. Output and artifacts are written to a known directory (`/root/rehearsal_out/`).
3. **Result serving via HTTP.** After the job finishes, `python3 -m http.server 30000` serves the output directory. The HPC polls `https://{pod_id}-30000.proxy.runpod.net/status.txt` until it reads `DONE` or `FAIL`.
4. **Artifact retrieval via HTTPS proxy.** Individual files are downloaded through the same proxy URL. No SSH tunnel or Jupyter API required.
5. **Deterministic teardown.** The launcher's `finally` block always calls `terminate_and_wait`. The pod-side self-termination timer runs in parallel as a hard backstop.

**Best practices for bounded launches, retrieval, and teardown:**

- **Only bundle allowed files.** The whitelist is defined in `FILES_TO_BUNDLE` inside `runpod_http_rehearsal.py` and must match the allowed-upload list in this document. Never include `.git/`, `plans/`, `records/`, credentials, or personal notes.
- **Set `--max-minutes` conservatively.** The job wallclock must leave room for the retrieval buffer (default 2 min) within the pod hard deadline (default 12 min).
- **Poll for completion, then download, then terminate.** Always follow this sequence. Never terminate before confirming `status.txt` is `DONE` or `FAIL` and all expected artifacts are downloaded.
- **Verify retrieved artifacts.** Check file sizes are non-zero and sensible before treating a retrieval as successful.
- **Use the validation ladder.** Start with a 1-GPU environment probe, then short training, then stability run, then short eval — all via the HTTP-bootstrap path — before committing to an 8×H100 production run.
- **Do not mix control paths.** If a run is launched via HTTP-bootstrap, retrieve via the same HTTP proxy. Do not attempt SSH or Jupyter as a fallback mid-run.
- Do not assume any community checkpoint server exists or is required. The documented default path is public GitHub clone plus Hugging Face dataset download.
- Do not introduce a GitHub personal access token into the default workflow unless a concrete step is blocked without it.

### Training Runs on RunPod

- **Hardware:** 8×H100 80GB SXM (pod type: `gpu-8x-h100-sxm`)
- **Docker image:** `matotezitanka/proteus-pytorch:community` (has PyTorch 2.9+, CUDA 12.8, FA3)
- **Data download on-pod:**
  ```bash
  MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp8192
  ```
- **Training command:**
  ```bash
  SEED=42 QK_GAIN_INIT=5.25 TTT_ENABLED=1 TTT_LR=0.005 TTT_EPOCHS=3 \
    torchrun --standalone --nproc_per_node=8 train_gpt.py
  ```
- **BEFORE terminating a training pod:** retrieve all checkpoints, logs, and artifacts.
- **Budget discipline:** track all spending, do not exceed allocated budget per experiment.

### Mandatory Validation Ladder

- CPU smoke test training locally before any RunPod training.
- CPU smoke test eval locally before any RunPod eval.
- 1×H100 startup smoke before any longer GPU run.
- 1×H100 short training test before production training:
  - startup and environment validation
  - short training run (for example ~10 steps)
  - artifact/log retrieval back to this HPC
- 1×H100 training stability run before production training:
  - approximately 100 training steps
  - confirm training stability and successful retrieval
- 1×H100 short eval test before production eval:
  - startup and environment validation
  - short eval run (about 1 minute)
  - result retrieval back to this HPC
- Only after all lower-cost stages succeed may you run production 8×H100 training or 8×H100 eval.
- Treat training and eval as separate staged workflows. Do not skip the eval rehearsal because training succeeded.

### Eval Runs on RunPod (separate from training)

- Can use 1×H100 or 8×H100 depending on eval parallelism.
- Upload only: compressed model artifact + eval code.
- Download SP8192 val data only (not training shards).
- Eval is separate from training — different pod, different budget.
- Retrieve all eval logs before teardown.

### Checkpoint Retrieval Protocol

**ALWAYS retrieve before pod termination:**

1. Training logs (`train_seed*.log`)
2. Model checkpoints (`.pt`, `.ptz` files)
3. Quantized artifacts (compressed model)
4. Any generated results/metrics

**Gate condition:** no 8×H100 production run is allowed until a 1×H100 rehearsal has successfully retrieved at least one log and one output artifact back to this HPC.

**Transfer procedure:**

- Use `runpodctl send` or `rsync` to copy files to local storage.
- If using a Jupyter-proxy or named-file download path, treat it as provisional until it has been demonstrated on a 1×H100 rehearsal.
- Verify file integrity after transfer (check file sizes).
- Only after verification: terminate the pod.

### Timed Shutdown Requirement

- **NEVER** launch a RunPod pod without a timed shutdown path that does not depend on this HPC remaining connected.
- A local watchdog on this HPC is only a secondary safeguard, not the primary shutdown guarantee.
- The primary shutdown guarantee must be pod-side or otherwise server-independent for the launched job.
- Every job must have an explicit wallclock budget and an explicit teardown path on timeout or completion.
- If automatic shutdown cannot be guaranteed for a workflow, do not run that workflow on paid GPUs.

### Budget Controls

- Total RunPod budget: as specified per session (default $100).
- Log every pod creation with: pod ID, type, start time, purpose.
- Log every pod termination with: end time, total cost.
- Estimate costs BEFORE launching: 8×H100 SXM ≈ $25–30/hr.
- Kill pods immediately after retrieving results.
- **Never leave pods running overnight or unattended.**
- Leave explicit wallclock buffer before any hard cutoff to retrieve artifacts and verify integrity.
- Minimum buffer: 5 minutes. Safer default for unproven retrieval paths: 10-15 minutes.

---

## Agent Guidelines

- When modifying `train_gpt.py`, keep it under 1500 lines.
- Always verify compressed model size ≤ 16 MB after changes.
- Run validation to confirm val_bpb improvement before claiming progress.
- Before trusting any BPB improvement, audit the byte denominator, tokenizer-byte mapping, and any special-token / leading-space handling.
- When implementing a new eval path, compare the result against a merged PR or previously validated evaluator before treating it as authoritative.
- If BPB logic is duplicated across files, patch and verify every copy together.
- Track experiments in `plans/` directory.
- The non-record track (`track_non_record_16mb/`) has no 10-minute constraint but still requires the 16 MB artifact limit.
- Hyperparameters are controlled via environment variables — prefer this over code edits for experimentation.
- **For RunPod runs:** always create a run plan BEFORE launching, estimate cost, and get budget approval.
