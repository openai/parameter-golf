# Record: SP8192 + 3GDN/8FA 4096 + Attention Sink

**val_bpb = 1.07849** (3-seed mean, std 0.00058) | **~15.934 MB** | 8xH100 80GB SXM | PyTorch 2.9.1+cu129

This is a size-compliant 3-seed submission for `track_10min_16mb`. The final score is the quantized model after legal score-first test-time training (TTT). It improves on the 2026-04-09 SP8192 3-layer recurrence / QK5.25 / legal TTT baseline record (1.0810 bpb) by **0.00251 bpb** / **0.00652 nats per token**, which clears the **0.005 nats** threshold.

Baseline reference: <https://github.com/openai/parameter-golf/blob/main/records/track_10min_16mb/2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT/README.md>

## 3-Seed Results

| Seed | Steps | Pre-quant bpb | Quantized bpb | **TTT bpb** | TTT gain | Train | Eval total | Artifact |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 4350 | 1.08126 | 1.09077 | **1.07910** | -0.01167 | 591.020s | 312.305s | 15,935,382 |
| 777 | 4351 | 1.07994 | 1.08945 | **1.07795** | -0.01150 | 591.025s | 311.997s | 15,932,823 |
| 1337 | 4348 | 1.08058 | 1.09007 | **1.07842** | -0.01165 | 591.074s | 312.645s | 15,934,105 |
| **Mean** | 4350 | **1.08059** | **1.09010** | **1.07849** | **-0.01161** | **591.040s** | **312.316s** | **15,934,103** |

All three artifacts are below the 16,000,000 byte cap. Training and total eval both stay below 600 seconds on every seed.

## Key Changes From Baseline

| Component | Baseline stack | This submission |
| --- | --- | --- |
| Context | 2048 train/eval | 4096 train/eval |
| Mixers | 11 full-attention layers | 3 GDN layers (0, 1, 10) + 8 full-attention layers (2-9) |
| Recurrence | Loop layers 3-5 | Same loop segment, activated at 40% wallclock |
| Parallel residuals | Layers 7+ | Layers 8+ |
| Attention | QK gain, partial RoPE, XSA | Adds learned attention sink (see gpt-oss) |
| Training knobs | Warmdown 0.72, softcap 30 in the local baseline code | Warmdown 0.85, softcap 20, min LR 0.1 |
| Quantization | GPTQ int6 matrices, int8 embeddings | Hadamard V/O, `attn.proj` int7, embedding top4025 int8/tail int7, matrix clip 13.0 |
| Quantization robustness | - | Late QAT noise on `attn.c_k.weight` and `attn.proj.weight` after 95% wallclock |
| TTT | Legal score-first, broad trainable set in the baseline record | Legal score-first, MLP-only SGD, lr 0.017, 5 epochs, 64K-token chunks |

## Architecture

The model keeps the 11-block, 512d, 8 query head / 4 KV head SP8192 base shape with tied embeddings, partial RoPE over 16 of 64 head dimensions, QK gain 5.25, skip gates, layerwise LN scaling, and LeakyReLU(0.5)^2 MLPs with multiplier 4.125.

Layers 0, 1, and 10 use a GDN mixer based on `chunk_gated_delta_rule`: kernel-2 depthwise causal convolution over Q/K/V, ELU+1 Q/K, SiLU V/Z, two Q heads, four value heads, and an output projection with no RMS. The selected kernel-2 GDN gave the best quality/time/size trade-off. Several more aggressive GDN changes were unstable during training; this was the strongest stable variant, so it stayed while the rest were dropped. The goal of this work was to make a viable GDN-like model under the challenge's rules, which I kind of achieved, even though it's not a full GDN model. Due to compilation graph breaks this architecture tends to be slower, so a lot of other architectural changes and experiments were needed to even achieve the baseline's quality, not to mention the **0.005 nats** clearance.

Layers 2-9 are full-attention layers. Each full-attention layer prepends a learned `k_sink` key per KV head with a zero value vector, giving attention a learned null/sink key without injecting future-token content. XSA is enabled on every full-attention layer. The full-attention middle is compiled as a separate fullgraph segment. The outer path stays compatible with the GDN kernels.

## Quantization

Quantization uses full-Hessian GPTQ with 8 calibration batches of 720,896 tokens at sequence length 4096. Before GPTQ, all full-attention V/O pairs are Hadamard-rotated.

Weights are stored as:

- `tok_emb.weight`: frequency split, top 4025 rows int8, tail rows int7, counted over 100M training tokens.
- Attention/MLP matrices: int6 by default.
- `blocks.*.attn.proj.weight`: int7.
- Small controls and recurrent parameters: float16 passthrough.

The serialized weight payload is byte-shuffled and Brotli-compressed at quality 11. The submitted runner code is 22,112 bytes in the final logs.

## TTT

TTT follows the legal score-first ordering used by accepted legal-TTT submissions such as the 2026-04-06 SP8192 QK5 entry:
<https://github.com/openai/parameter-golf/blob/main/records/track_10min_16mb/2026-04-06_SP8192_QK5_LegalTTT_1.0828/README.md>

For each contiguous 65,536-token chunk:

1. Score all validation windows assigned to that chunk with no parameter update.
2. Then train only on tokens from that already-scored chunk.
3. Move to the next chunk and repeat.

The final TTT trainable set is `mlp` only, using SGD with momentum 0.9, gradient clip 1.0, learning rate 0.017, cosine chunk LR, and 5 epochs per non-final chunk. No token is trained on before it is scored, and no token is re-scored after adaptation.

## Context Chunking / Compression

This mechanism was important during development but is intentionally not in the final pruned `train_gpt.py`. Earlier candidates used an 8192-token train/eval window with `CONTEXT_COMPRESS=2` or `4`. At `CONTEXT_COMPRESS=4`, the model split each sequence into four 2048-token chunks, added a causal left halo (usually 128 tokens), ran the full-attention middle independently per chunk, and then stitched the outputs back together. This kept the quadratic full-attention segment affordable while preserving strict causality, but it also meant most tokens only had local chunk context plus the halo.

Several attempts tried to recover the lost cross-chunk signal: larger halos, memory-prefix tokens, K/V summary prefixes, raw-token prefixes, and a late global mixer. None was worth keeping. The final submission instead uses `CONTEXT_COMPRESS=1` with 4096-token train/eval windows, so the full-attention middle sees the whole 4096-token causal window directly and the now-unused chunking code was removed for code size.

## What Was Tried And Rejected
- Because FLA's GDN kernels break fullgraph compilation, several specialized kernels and implementations were tested around the model: Liger embedding/loss (not really a speedup-focused module, more of a memory saver, which is not needed, but still an interesting try), Quack RMSNorm/MLP, etc. None produced a substantial end-to-end gain; several slowed training because they introduced extra boundaries inside the `torch.compile`d path.
- An inter-head mixer on the GDN output significantly improved per-step losses, including the step-4000 validation loss, but it slowed Muon steps (due to a non-traditional shape of [4, 4]) enough to reduce total training steps. Moving that mixer into the control-parameter group did not preserve the quality gain.
- The earlier FA29/8192-token chunked-context line did not produce sufficient results. Increasing the halo to 512 was slower and worse; halo 256 was effectively neutral. Larger context windows with higher compression factors kept similar throughput but produced worse quality.
- Cross-chunk memory and summary mechanisms were not promoted. Prefix memory produced NaNs in the tested implementation, K/V summary reuse improved no-TTT base only marginally and did not compound through TTT, raw-token prefixes regressed, and the late global mixer was worse and oversized.
- AWQ-style output absorption, halo teacher/student distillation, low-rank residual GPTQ patches, and E2E-TTT meta-training were all tried. They either moved less than the noise floor or improved the apparent TTT delta by damaging the base model.
- Quantization-aware noise had to be very late and light. Earlier or stronger QAT noise hurt base quality; the useful setting was the final 5% of wallclock, multiplier 0.05, only on `attn.c_k.weight` and `attn.proj.weight`.
- Mixed-bit quantization was helpful only in narrow places. Int7 on attention projection and a frequency split for embeddings survived; broader int7 attention variants were too large, while int5 compensation and low-rank GPTQ residuals lost too much quality.
- TTT sweeps saturated before the final architecture and training changes. All-parameter SGD over learning rate, momentum, epochs, and chunk size clustered near 1.0802 bpb on the pre-sink candidate; momentum 0.99 and Adam-on-MLP were clear regressions. The final MLP-only, 5-epoch setting was the best stable speed/quality point.
- Attention sink was useful but noisy. One sink key per KV head beat the no-sink candidate; four sink keys were essentially tied.
- Other probes such as MTP auxiliary loss, z-loss, Muon modifications (including Triton and SVD-driven Newton-Schulz variants), and several attempts to apply XSA-style corrections to GDN were worse or inconsistent.

## Compliance

- **Causality:** All normal eval and TTT scoring paths are causal. Full attention uses causal FA3; GDN uses a causal convolution and causal delta-rule scan.
- **Normalized distribution:** The score comes from the standard full-vocabulary softmax/cross-entropy path with logit softcap only. No cache or normalization shortcut is used.
- **Score before update:** Every TTT chunk is scored before any SGD update on that chunk.
- **Single pass:** Each token/window contribution is scored once; there is no rescoring after adaptation.
- **No eval-time shortcuts:** No SLOT, no ETLB/logit bias, no n-gram cache, no two-pass selection, and no pre-quant TTT on validation data.
- **Budgets:** 3 distinct seeds, all artifacts under 16 MB, all training runs under 600 seconds, all eval runs under 600 seconds.

## Reproduction

```bash
pip install brotli causal-conv1d==1.6.1 fla-core==0.4.2 flash-linear-attention==0.4.2 gram-newton-schulz==0.1.3
pip install flash-attn-3 --no-deps --find-links https://windreamer.github.io/flash-attention3-wheels/cu129_torch291/
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp8192

for SEED in 42 777 1337; do
  RUN_ID=final_seed${SEED}_v2 \
  SEED=${SEED} \
  MODEL_PATH=models/seed${SEED}.pt \
  QUANTIZED_MODEL_PATH=models/seed${SEED}.int6.ptz \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
done
```

The final logs used Python 3.12.3, PyTorch 2.9.1+cu129, CUDA driver 570.133.20, and 8x NVIDIA H100 80GB HBM3 GPUs. Runtime dependencies include `flash_attn_interface`, `fla`, `causal_conv1d`, `sentencepiece`, `numpy`, `torch.distributed`, and `brotli`.

## Included Files

- `README.md`
- `submission.json`
- `train_gpt.py`
- `train_gpt_human.py`
- `logs/final_seed42_v2.txt`, `logs/final_seed777_v2.txt`, `logs/final_seed1337_v2.txt`
