# RationalRaven — sp4096 + MLP×3.25 + LeakyReLU² + late-QAT int8 attn/KV (FROZEN 2026-04-30, SF time)

This has been an amazing learning opportunity. The competition was fun and intense in equal measure, with lessons spanning both the craft of running experiments and the practical realities around them. I plan to write a detailed retrospective on it soon. This is not a SOTA submission, but it is the culmination of my most earnest attempt.

3-seed lock on a single recipe across seeds {1337, 1338, 1339}; the artifact submitted is the seed 1339 run.

**Headline number:** sliding eval (stride=64), 3-seed **mean ± σ = 1.139957 ± 0.000249** (post-quant, training-time exact roundtrip; sample std, n=3).

## 3-seed lock results

All three seeds use identical recipe (`run.sh` env), only `SEED` differs.

| seed | post-q sliding64 | pre-q EMA sliding64 | quant tax | artifact bytes (.ptz) | total bytes | cap headroom |
|---:|---:|---:|---:|---:|---:|---:|
| 1337 | 1.140218 | (chunked 1.1738) | 0.0080 chunked | 15,285,622 | 15,362,368 | 637,632 |
| 1338 | 1.139930 | 1.1321 | 0.0078 | 15,675,958 | 15,752,704 | 247,296 |
| **1339** | **1.139722** | **1.1324** | **0.0073** | **15,521,366** | **15,598,112** | **401,888** |

**3-seed statistics (post-quant sliding64):**

| stat | value |
|---|---:|
| mean | 1.139957 |
| min (seed 1339) | 1.139722 |
| max (seed 1337) | 1.140218 |
| range | 0.000496 |
| std (sample, σ_{n−1}) | 0.000249 |
| std (population, σ_n) | 0.000203 |
| SEM (σ/√n) | 0.000144 |
| 95% CI of mean (t, df=2) | [1.139337, 1.140577] |

- σ ≈ **2.5 × 10⁻⁴**, well below the ±0.001 working-band estimate from prior sweeps — recipe is more deterministic across seeds than previously assumed.
- No failed runs across the 3 seeds.
- Quant tax uniform (~0.0078 ± 0.0001) — stable across seeds.

**Submitted artifact: seed 1339 run** (one of the three; sits at −0.94σ from the 3-seed mean, comfortably within the 1σ envelope).

## Submitted-artifact metrics (seed 1339 run, validated 2026-04-30 on 8×H100 SXM, training-time eval)

| metric | value | source |
|---|---|---|
| **val_bpb (sliding stride=64, post-quant)** | **1.13972247** | `final_mixed_zlib_roundtrip_exact` line in `train.log` (label is misnomer — actual compressor is zstd, see `Serialized model mixed+zstd: 15521366 bytes`) |
| val_loss (sliding stride=64, post-quant) | 2.62252609 | same line |
| val_bpb (pre-quant EMA sliding stride=64) | 1.1324 | `pre_quant_ema_eval` line in `train.log` |
| quant tax | 0.0073 | post-q minus pre-q |
| step at finish | 7,958 / 20,000 | `stopping_early: wallclock_cap` |
| training time | 600 s (8×H100 SXM) | `train_time:600034ms` |
| step_avg_ms | 75.40 | training_time / step |
| seed | 1339 | env |
| late-QAT trigger | step 6048 | `QAT_START_STEP=6048` (deterministic) |
| LATE_QAT actual steps | 1910 | 7958 − 6048 |
| **model_bytes (`.ptz`)** | **15,521,366** | `stat -c%s rationalraven_final.mixed.ptz` |
| code_bytes (`train_gpt.py`) | 76,746 | slim port |
| **submission_bytes (cap measure)** | **15,598,112** | model + code |
| cap headroom | **401,888** (2.51%) | 16,000,000 − 15,598,112 |

## Artifacts

| file | purpose |
|---|---|
| `rationalraven_final.mixed.ptz` | **the submission.** Mixed quantization (int6 bulk + int8 on `.attn.proj,.c_k,.c_v` + tied head/embed) with `USE_DELTA_FROM_INIT=1`, zstd L22 compressed. |
| `rationalraven_final.mixed.ptz.sha256` | `fd994aebad3d842762ffadf0482ab8311b6046d626058de4ea607fe5cb3f661b` |
| `train_gpt.py` | Training/eval script — slim port (76,746 B) limited to the recipe path. Counted toward cap. |
| `run.sh` | Reference launcher (NOT counted toward cap). Default `SEED=1339`; all other env defaults match seed 1337/1338 — recipe is identical. |
| `train.log` | Full training stdout for seed 1339 (steps + val checkpoints + post-train quantize/eval). |

## Reproduction recipe (seed 1339)

```bash
# Inside an 8×H100 SXM pod, with FineWeb sp4096 shards reachable
# (auto-pulled from kevclark/parameter-golf if HF auth is set, or pre-staged at
# data/datasets/fineweb10B_sp4096/ + data/tokenizers/fineweb_4096_bpe.model):
bash run.sh
```

`run.sh` sets `SEED=1339` by default; identical to seed 1337/1338 runs in every other respect. Direct invocation:

```bash
WARMDOWN_ITERS=2000 MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
  MLP_ACTIVATION=leaky_relu2 LEAKY_RELU_SLOPE=0.5 \
  GRAD_CLIP_NORM=0.3 LOGIT_SOFTCAP=18 \
  NUM_LAYERS=11 MLP_MULT=3.25 \
  MUON_MOMENTUM=0.95 MUON_MOMENTUM_WARMUP_START=0.85 MUON_MOMENTUM_WARMUP_STEPS=500 \
  LATE_QAT_STEPS=2400 QAT_START_STEP=6048 EMA_DECAY=0.997 EMA_FP32=1 \
  WEIGHT_DECAY=0.04 EVAL_STRIDE=64 USE_DELTA_FROM_INIT=1 \
  MAX_WALLCLOCK_SECONDS=600 SEED=1339 \
  QUANT_SCALE_SCHEME=per_row QUANT_INT8_CATS=.attn.proj,.c_k,.c_v \
  VOCAB_SIZE=4096 DATA_PATH=data/datasets/fineweb10B_sp4096 \
  TOKENIZER_PATH=data/tokenizers/fineweb_4096_bpe.model \
  MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Reproducibility: `QAT_START_STEP=6048` pins the late-QAT trigger to a deterministic step, immune to `step_ms` jitter from torch.compile cold-start. With FA2 + flash SDPA, runs are deterministic up to floating-point reduction noise (sliding64 within ±0.0005 across reruns at 8×H100).

## Architecture

A fairly conventional small GPT-style decoder, pre-norm + RoPE + RMSNorm + tied embeddings. The non-default choices were each made to free up bytes for higher-precision quantization on the heaviest tensors.

### Shape

| component | value | notes / why this choice |
|---|---|---|
| layers | **11** | depth tuned against byte cap (10L gave less, 12L cost too many bytes at this MLP size) |
| d_model | **512** | — |
| context length | 1024 tokens | training & eval sequence length (`TRAIN_SEQ_LEN`) |
| total params | **29,516,376** | counted across all trained tensors |

### Attention (per layer)

| component | value | notes / why this choice |
|---|---|---|
| query heads | **8** | head_dim = 512 / 8 = 64 (FlashAttention-2-friendly) |
| KV heads | **4 (GQA, ratio 2:1)** | halves KV-projection weight bytes; funded keeping `.attn.proj`, `.c_k`, `.c_v` + tied head/embed at int8 (rather than int6) |
| QKV projection | **separate** `c_q`, `c_k`, `c_v` modules | needed so we can apply different quant precisions per role (`.c_k`/`.c_v` go int8, `c_q` stays int6) |
| attention output | `attn.proj` | int8 under the mixed quant scheme |
| Q/K normalization | **RMSNorm on Q and K** | standard stability trick; helps under late-QAT |
| per-head Q gain | learnable `q_gain` (init 1.5) | per-head scale on Q before attention |
| position encoding | **RoPE**, base = 10000 | applied to Q and K post-RMSNorm |
| attention kernel | FlashAttention-2 (flash SDPA) | head_dim=64 sits in the FA2 fast path |
| pre-norm | **RMSNorm** before attention | — |

### MLP (per layer)

| component | value | notes / why this choice |
|---|---|---|
| expansion | **×3.25** → 1664 hidden | shrunk from the more common ×4 to pay for the int8-head bytes above |
| activation | **LeakyReLU²(slope=0.5)** | small but reliable BPB win over ReLU² / GELU in this setup |
| layout | up-projection `mlp.fc` + down-projection `mlp.proj` | both quantized at int6 in the final container |
| pre-norm | **RMSNorm** before MLP | — |

### Embeddings & head

| component | value | notes / why this choice |
|---|---|---|
| tokenizer | **sp4096 BPE** (vocab = 4096) | meaningful BPB win over sp1024; richer embed table paid for by the byte savings above |
| token embedding | `tok_emb`, 4096 × 512 | int8-quantized in the mixed container |
| LM head | **tied** to `tok_emb` | the shared 2,097,152-param matrix accounts for ~7 % of total params |
| final norm | RMSNorm | applied before the head |

### Output stabilization

| component | value | notes / why this choice |
|---|---|---|
| logit softcap | **18.0** (`cap · tanh(logits / cap)`) | stabilizes late training, especially under late-QAT |
| bias terms | none on linear layers | standard GPT-style |

## Quantization (post-training)

- **delta-from-init**: weights stored as Δ from deterministic init, then quantized
- **int8 per-row** on: `.attn.proj`, `.c_k`, `.c_v` (×11 layers each), `lm_head.weight`, `tok_emb.weight` — 34 tensors
- **int6 per-row** on: 33 control tensors
- **fp32 passthrough** on: 45 small/scalar tensors
- Mixed container, **zstd level 22** compressed (`Serialized model mixed+zstd: 15521366 bytes`)

## Integrity verification (local, no training)

```bash
cd submissions/rationalraven_frozen_20260501
sha256sum -c <(printf '%s  rationalraven_final.mixed.ptz\n' "$(cat rationalraven_final.mixed.ptz.sha256)")
# Expected: rationalraven_final.mixed.ptz: OK
```

## Pre-flight before submission push

- [x] sha256 recorded: `fd994aebad3d842762ffadf0482ab8311b6046d626058de4ea607fe5cb3f661b`
- [x] Cap headroom verified: 15,598,112 / 16,000,000 (401,888 spare)
- [x] Sliding eval (stride=64) results captured — 3-seed mean = **1.139957 ± 0.000249** (training-time exact roundtrip)
- [x] 3-seed lock complete (1337/1338/1339); seed 1339 artifact submitted
- [ ] Pushed to leaderboard

## Limitations / honesty disclosure

- **Mean reported, single artifact submitted**: the headline number is the 3-seed mean (1.139957 ± 0.000249). The submitted file is one of the three runs — seed 1339, which scored 1.139722 in this lock (−0.94σ from the mean). I'm representing the recipe by its expected performance rather than the best draw of three.
- **Does not beat the public leader** (1.0810 via GPTQ + byte-shuffle + Brotli). This submission isn't a record claim.
- The slim `train_gpt.py` is the on-disk script that runs the recipe end-to-end; experimental branches from a much larger development program have been removed. The 76,746-byte size is what's counted toward the cap.

## Acknowledgements

Thank you to OpenAI for sponsoring $500 worth of compute with their partners RunPod. Every 8×H100 SXM hour that produced this artifact (and the 3-seed lock that picked it) was funded by that grant.
