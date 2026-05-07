# DominationV2 + BOS-Reset Entropy-Gated Bigram Cache

This is the current best candidate to run next on 8xH100.

It keeps the upstream `DominationV2` training and export stack and adds one evaluation-time modification:

- final validation uses flat sliding-window scoring
- an online bigram cache is built during evaluation and reset at each BOS boundary
- cache interpolation is capped by `CACHE_ALPHA`
- the effective interpolation weight grows with seen bigram counts via `total / (total + CACHE_TAU)`
- that weight is multiplied by normalized model entropy raised to `CACHE_ENTROPY_POWER`
- the cache is applied after the quantized roundtrip and after optional TTT

The cache does not change training weights or the exported model. It only changes how the final validation pass is scored.

## Status

Decision: keep this candidate as the main bet.

Why:

- the integrated evaluator was audited and corrected before finalization
- the corrected implementation now has local parity checks against the research evaluator
- prior 1xH100 proxy evidence still shows post-TTT gains from this cache
- a final narrow replacement search failed to find a better alternative quickly enough to justify swapping it out

## What Is Verified

### Local 4080 verification

The following command passes:

```bash
conda run --no-capture-output -n torch-wsl python -u research/verify_domv2_cache.py
```

What it verifies:

- exact single-rank parity between the integrated final evaluator and the research evaluator for plain sliding-window scoring
- exact single-rank parity for cache scoring
- exact multi-rank aggregation back to the single-rank answer for `world_size in {2,3,8,32}`
- exact equivalence between the research `run_ttt` loop and the final script's TTT loop in the `world_size=1` subset setting used for the 1xH100 probes

### 1xH100 proxy evidence

On a strong 1xH100 checkpoint, after 3 epochs of TTT on matched 1M-token validation slices, the cache still improved bpb:

| `start_token` | no cache | cache (`alpha=0.20`) | delta |
| --- | ---: | ---: | ---: |
| `0` | `1.61700285` | `1.61190207` | `-0.00510078` |
| `20000000` | `1.58458543` | `1.57923265` | `-0.00535278` |
| `40000000` | `1.56495388` | `1.56022034` | `-0.00473354` |

On the middle slice, the tested `alpha` sweep was:

- `0.10`: `1.58013994`
- `0.20`: `1.57923535`
- `0.30`: `1.57976811`
- `0.40`: `1.58128413`

## What Is Not Yet Verified

- Full end-to-end performance on the official full validation run under the 8xH100 time budget.
- Multi-rank TTT parity against the research evaluator. Multi-rank evaluation aggregation is verified; multi-rank TTT is still a reasonable extrapolation.
- Final artifact margin on the target run. `train_gpt.py` is currently `55,299` bytes, which is about `9.8 KB` larger than the upstream control this was built from.
- Whether the best cache strength on the final 8xH100 checkpoint stays exactly at `alpha=0.20`. It is the current best default, not a theorem.

## Included Files

- `train_gpt.py`
  - packaged candidate. Trains, exports `final_model.pt`, quantizes and compresses to `final_model.int8.ptz`, reloads the round-tripped weights, optionally runs TTT, then performs the final evaluation.
- `train_8xH100.sh`
  - intended launcher for the next 8xH100 run.
- `eval_checkpoint.py`
  - wrapper around `research/eval_doc_cache.py` for probing a saved checkpoint with or without the cache.

## Intended 8xH100 Workflow

Run these commands from the repo root unless stated otherwise.

### 1. Environment setup

If you are using the OpenAI/Runpod challenge image, most dependencies may already be present. On a fresh Python environment, install:

```bash
python3 -m pip install --upgrade pip
pip install -r requirements.txt zstandard
```

`zstandard` matters. Without it, the script falls back to `zlib-9`, which is functional but not the intended compressed-model path.

### 2. Auth requirements

- GitHub auth is not required to run the method.
- Hugging Face auth was not required in the locally verified path for the default published dataset repo.
- If your machine cannot access the default dataset repo anonymously, authenticate with Hugging Face before downloading data.

### 3. Dataset and tokenizer preparation

Download the default `sp1024` cached challenge data:

```bash
python3 data/cached_challenge_fineweb.py --variant sp1024
```

Expected paths after download:

- `data/datasets/fineweb10B_sp1024/`
- `data/tokenizers/fineweb_1024_bpe.model`

The launcher expects those paths by default.

### 4. Training command

```bash
cd final/dominationv2_cache
bash train_8xH100.sh
```

Important default settings in `train_8xH100.sh`:

- `NPROC_PER_NODE=8`
- `MAX_WALLCLOCK_SECONDS=600`
- `TRAIN_SEQ_LEN=2048`
- `TRAIN_BATCH_TOKENS=524288`
- `TTT_ENABLED=1`
- `TTT_EPOCHS=3`
- `TTT_LR=1e-4`
- `CACHE_ENABLED=1`
- `CACHE_ALPHA=0.20`
- `CACHE_TAU=8`
- `CACHE_ENTROPY_POWER=1.0`
- `EVAL_STRIDE=64`
- `EVAL_BATCH_SEQS=32`

Useful overrides:

```bash
cd final/dominationv2_cache
RUN_ID=dominationv2_cache_trial NPROC_PER_NODE=8 bash train_8xH100.sh
```

### 5. What outputs to expect

Outputs are written in `final/dominationv2_cache/`:

- `logs/<RUN_ID>.txt`
- `final_model.pt`
- `final_model.int8.ptz`

### 6. What to verify in the training log

Before trusting the run, inspect these markers in `logs/<RUN_ID>.txt`:

- `cache_enabled:True` or `cache_enabled:1`
- `ttt: done in ...`
- `final_eval_mode:sliding_window_cache stride:64 batch_seqs:32 ...`
- `final_int8_zlib_roundtrip compressed_model_bytes:... code_bytes:... total_artifact_bytes:...`
- `final_roundtrip_exact val_loss:... val_bpb:...`

The minimum trust checks are:

- `total_artifact_bytes < 16000000`
- the final evaluation mode is `sliding_window_cache`, not plain sliding
- `final_model.int8.ptz` was actually written
- the number you compare against other runs is `final_roundtrip_exact val_bpb`

If the log shows `compression:zlib-9` or the compressed artifact is unexpectedly large, install `zstandard` and rerun.

### 7. Evaluation and checkpoint probing

To probe an existing checkpoint with the same evaluation path:

```bash
cd /path/to/repo
python final/dominationv2_cache/eval_checkpoint.py \
  --checkpoint experiments/domv2_ctrl3600/final_model.pt \
  --quant-roundtrip \
  --tokenizer data/tokenizers/fineweb_1024_bpe.model \
  --val-files "data/datasets/fineweb10B_sp1024/fineweb_val_*.bin" \
  --mode flat_cache_bigram_adaptive_entropy \
  --reset-cache-on-bos \
  --alphas 0 0.20 \
  --tau 8 \
  --entropy-power 1.0 \
  --seq-len 2048 \
  --stride 64 \
  --batch-seqs 32 \
  --max-tokens 1048576 \
  --start-token 20000000 \
  --ttt-epochs 3 \
  --ttt-lr 1e-4 \
  --ttt-batch-seqs 4
```

This is the main tool for narrow alpha sweeps or for checking whether a newly trained checkpoint still likes the cache.

## Recommended Decision Rule After the 8xH100 Run

Back this candidate if all of the following hold:

- the artifact fits comfortably under `16,000,000` total bytes
- the final log confirms `sliding_window_cache`
- the run finishes cleanly inside the challenge wallclock budget
- the final `val_bpb` is at least competitive with the control you are comparing against

If the checkpoint is clearly stronger or weaker than expected, do one narrow post-hoc sweep with `eval_checkpoint.py` around `alpha in {0.15, 0.20, 0.25, 0.30}` before changing anything larger.

## Alternatives Not Promoted

- A smoothed bigram plus unigram-backoff cache variant was tested in the last focused search and was worse than the current cache on the local proxy checkpoint.
- Broader architecture or tokenizer changes were not promoted in this final phase because they would require a fresh evidence cycle and would not beat the current artifact-quality priority.

## Main Risks

- The single-H100 subset gains may not transfer one-for-one to the full 8xH100 run.
- The extra code bytes reduce artifact margin relative to the upstream control.
- Multi-rank TTT is still an extrapolation rather than a separately parity-tested path.
- If the final checkpoint has materially different entropy calibration, `alpha=0.20` may stop being the exact optimum even if the cache family remains good.
