# Non-record: Systems-Fusion Ceiling Investigation on the SP8192 Stack + H-Net Tokenization Proposal

**Author**: [@diaslmb](https://github.com/diaslmb)  •  **Track**: `track_non_record_16mb`  •  **Hardware**: 1× H100 80GB SXM (RunPod)  •  **Date**: 2026-04-14

## tl;dr

Two hypothesized systems speedups were tested on top of PR #1493 (bigbag, SP8192 + 3-layer depth recurrence + parallel residuals + QK-Gain 5.25 + legal TTT, val_bpb 1.0810):

1. **Custom Triton kernel replacing `_xsa_efficient`** (XSA decomposes into `bmm + sum + elementwise` under Inductor; a single Triton kernel reads v once, normalizes, and writes `y - <y, v̂>·v̂` for every GQA head in one pass).
2. **Fused QKV projection** (stacking `c_q / c_k / c_v` into one `(D + 2·D_kv, D)` GEMM).

**200-step training pilot (1× H100, matched seed, `torch.compile(mode="max-autotune-no-cudagraphs")`)**:

| variant | step_avg (ms) | steady-state step (100→200, ms) | tok/s | train_loss at 200 |
|---|---:|---:|---:|---:|
| baseline (3-linear, torch-XSA) | 1020 | 1140 | 769,855 | 3.7650 |
| +fused QKV | 1020 | 1140 | 772,679 | 3.7752 |
| +fused QKV + Triton-XSA | **1080** | **1260** | 734,418 | 3.7738 |

**Neither helps. Triton-XSA regresses 6 %** because the `torch.autograd.Function` wrapper creates a graph break that prevents Inductor from fusing around the kernel, and the fusion-barrier cost exceeds the kernel's isolated-forward advantage.

The broader finding: **at the D=512 scale of this contest, Inductor + max-autotune + FA3 is very close to the block-level kernel-fusion ceiling**. Future systems PRs should either (a) ship Triton kernels via `torch.library.custom_op` with `register_fake` so Inductor can fuse around them, or (b) target cross-op patterns Inductor cannot identify — cross-layer, cross-component, or operator-semantics outside Inductor's pattern-matcher. Simple op-local fusion is already taken.

This PR also carries **H-Net Milestone 1: hierarchical byte-level stack with a fixed chunker**, run end-to-end as a pilot and showing clear signs of life. val_bpb drops from **4.49 → 2.51** as we fix architecture and scale training data (factor-of-14 data increase, single skip-connection fix responsible for 1.25 bpb of the improvement). Grant-funded M2 adds the learned chunker (the unclaimed piece of the Requests-for-PRs entry).

---

## Part 1 — Systems investigation

### 1.1 Target and method

**Baseline**: PR #1493's `train_gpt.py` (LZMA-compressed self-extracting source, ~16.6 KB packed). Architecture: 11 layers × D=512 × 8 heads / 4 KV (GQA) × MLP_MULT=4, SP8192 tokenizer, FA3 attention, XSA (vector rejection) on all layers, parallel residuals on layers 7+, 3-layer loop on layers 3–5 enabled at frac=0.35, Muon + AdamW optimizers, EMA 0.9965, GPTQ+SDClip export, Brotli-11.

**Two interventions**:
- A Triton forward + backward kernel for the XSA op (`_xsa_efficient`), wrapped in `torch.autograd.Function`.
- A fused QKV linear (three `CastedLinear` attributes replaced by one `c_qkv`; forward method rebound to split/reshape the stacked output).

**Protocol**:
- FA3 available on both sides of the comparison (installed via the `cu128_torch280` wheel at `windreamer.github.io/flash-attention3-wheels/`, which is the only bundle that ABI-matched our torch 2.8.0 + CUDA 12.8 image).
- Block-level microbench: `(B, T, D, H, KVH) = (8, 2048, 512, 8, 4)`, bf16, `torch.compile(..., mode="max-autotune-no-cudagraphs")`. Timed with CUDA events over 300 samples per variant after a 30-step warm-up + 2-second GEMM thermal prime.
- End-to-end training pilot: bigbag's full recipe, `ITERATIONS=200`, `SEED=42`, `WARMUP_STEPS=10`, `VAL_LOSS_EVERY=0`, baseline vs. patched variants with everything else held constant.

### 1.2 Kernel numerics (forward + backward)

The XSA kernel was validated across 4 shape configs and both dtypes. Max elementwise error on forward and gradients:

| dtype | shape (B,T,H,Hkv,D) | fwd max err | grad_y max err | grad_v max err |
|---|---|---:|---:|---:|
| fp32 | (8, 2048, 8, 4, 64) | 6.0e-7 | 5.1e-7 | 8.3e-7 |
| fp32 | (4, 1024, 4, 4, 64) | 4.8e-7 | 4.2e-7 | 4.8e-7 |
| bf16 | (8, 2048, 8, 4, 64) | 3.1e-2 | 3.1e-2 | 2.3e-2 |
| bf16 | (2, 512, 16, 4, 64) | 3.1e-2 | 1.6e-2 | 2.3e-2 |

bf16 errors are at the representable-precision floor for a length-64 dot product; fp32 errors are at single-precision float noise. Numerical parity is solid.

QKV fusion: **0.000e+00** max forward error vs. the 3-linear path (weights stacked exactly, splits at matching offsets).

### 1.3 Isolated-op microbench (XSA, fwd + bwd, bf16, B=8, T=2048, H=8, KVH=4, D=64)

Measured with CUDA events; numbers below from phase 2b after adding the Triton backward kernel:

| impl | fwd μs | fwd + bwd μs |
|---|---:|---:|
| torch eager | 153 | 751 |
| torch compiled (Inductor, max-autotune) | 106 | 342 |
| Triton (ours) | **94** | 745 |

The Triton kernel **wins the forward by ~12 %** against Inductor's decomposition. It loses fwd+bwd (745 vs 342) — even with a hand-rolled Triton backward, in isolation Inductor is extremely efficient on this op because every primitive (normalize / sum / elementwise) maps to a well-tuned Inductor template.

### 1.4 Block-level measurement

`torch.compile(mode="max-autotune-no-cudagraphs")` applied to the full `Block` forward + backward, 300 samples per variant, CUDA-event timing, GPU clock stabilized by 2-second thermal prime, `torch._dynamo.reset()` between variants to avoid the recompile-limit trap:

| backend | XSA | QKV | p50 ms/iter | min ms/iter |
|---|---|---|---:|---:|
| SDPA | torch | 3-linear | 3.08 | 1.87 |
| SDPA | torch | fused | 3.87 | **1.62** |
| SDPA | Triton | 3-linear | 1.80 | 1.77 |
| SDPA | Triton | fused | 2.92 | 1.72 |
| FA3 | torch | 3-linear | 2.87 | 1.94 |
| FA3 | torch | fused | 2.08 | 1.66 |
| FA3 | Triton | 3-linear | 3.60 | 2.36 |
| FA3 | Triton | fused | 4.72 | 2.43 |

Distribution is bimodal on several rows. The GPU's SM boost clock oscillates under short-burst microbenchmark load even with thermal priming; `min ms/iter` is the most reliable proxy for steady-state cost. **Triton XSA wins when paired with 3-linear QKV and SDPA**, but loses in combination with FA3 and with fused-QKV. This is the signal that graph-break cost matters more than kernel quality at this scale.

### 1.5 Training-pilot (the actual arbiter)

200-step pilots at matched seed 42, all hyperparameters as in PR #1493, `ITERATIONS=200`, `VAL_LOSS_EVERY=0`, `WARMUP_STEPS=10`, `MAX_WALLCLOCK_SECONDS=0`, on one H100 80GB SXM. Sustained workload keeps the GPU at turbo for the duration — noise is much lower than the microbench.

Training logs (step-by-step, cumulative `train_time` and `tok/s`):

```
# baseline (no patches)
100/200 train_loss: 4.5012 train_time: 1.5m tok/s: 885388
120/200 train_loss: 4.2465 train_time: 1.9m tok/s: 843300
140/200 train_loss: 4.0483 train_time: 2.3m tok/s: 815321
160/200 train_loss: 3.8626 train_time: 2.6m tok/s: 795856
180/200 train_loss: 3.8147 train_time: 3.0m tok/s: 781165
200/200 train_loss: 3.7650 train_time: 3.4m tok/s: 769855

# +fused QKV (c_qkv replacing c_q/c_k/c_v, identical weights at init)
100/200 train_loss: 4.5012 train_time: 1.5m tok/s: 885388   [matches baseline until layer-loop kicks in]
180/200 train_loss: 3.8258 train_time: 3.0m tok/s: 783607
200/200 train_loss: 3.7752 train_time: 3.4m tok/s: 772679

# +fused QKV + Triton XSA
180/200 train_loss: 3.8252 train_time: 3.2m tok/s: 745316
200/200 train_loss: 3.7738 train_time: 3.6m tok/s: 734418
```

**Δ vs baseline** (200-step `train_time`):
- +fused QKV: **0 ms / step** (no change; Inductor already does the equivalent fusion when it sees the 3 linears share input `x`).
- +fused QKV + Triton-XSA: **+60 ms / step → −5 % throughput**. Graph-break overhead from the `autograd.Function` exceeds the kernel-forward advantage.

Steady-state (steps 100 → 200, after the 3-layer loop has kicked in at frac=0.35): baseline 1140 ms/step, fused-QKV 1140 ms/step, full bundle 1260 ms/step. Same conclusion.

`train_loss` diverges by **~0.01 nats across variants** at step 200. This is within bf16 step-to-step noise but is a *real* effect for fused-QKV — see §1.6 for the Muon-equivalence subtlety.

### 1.6 Why fused-QKV is not a "free" systems change

Under Muon, the 2-D weight matrices of `c_q`, `c_k`, `c_v` are each orthogonalized independently via the Newton-Schulz-5 iteration on their gradients. Fusing them into a single `c_qkv` of shape `(D + 2·D_kv, D)` and applying Muon to that stacked weight runs the Newton-Schulz polynomial on the *joint* gradient matrix, which orthogonalizes its spectrum differently. The forward output is bit-identical at init (we verified `0.000e+00` max elementwise error) but training trajectories diverge — hence the 0.01-nat `train_loss` gap. A correct "systems-only" fused QKV would need to either (a) re-split the stacked weight before each Muon step, or (b) derive a Muon variant that respects a Kronecker / block-diagonal structure. Neither is addressed in this PR.

### 1.7 What would rescue these interventions

- **`torch.library.custom_op` + `register_fake`** for the Triton XSA instead of `autograd.Function`. This registers the kernel as a leaf op Inductor understands, so Inductor can continue fusing around it across `block` boundaries. Pre-registered to try in a follow-up if the dev grant comes through.
- **Stacked-gradient Muon** for fused QKV (or equivalent post-hoc split of the fused weight's NS iteration).
- Moving out of the "per-op fusion" corner entirely, toward patterns Inductor cannot identify — e.g., a fused **Muon Newton-Schulz** kernel for the optimizer (5 matmuls + polynomial, one launch), or a **byte-level tokenization** redesign that changes what Inductor has to compile in the first place. The latter is the focus of Part 2 of this PR.

### 1.8 Scope limitations and reproducibility

- All experiments on 1× H100 SXM with `world_size=1` and `grad_accum_steps=8` (global batch 786 432 tokens matches the 8× H100 target).
- FA3 wheel pinned to `cu128_torch280`. SDPA fallback shim verified numerically equivalent up to softmax-order effects.
- Block-microbench noise: high enough that p50 is unreliable; minima are the trustworthy figure. Training-pilot numbers are reliable because the GPU is at sustained turbo for minutes.
- `main()` segfaults in the post-quantization final validation pass on 1× H100 for all three pilots. The segfault is after training + log output and does not affect the measurements. We have not root-caused it but it looks like a torch 2.8 vs. torch 2.9.1 (bigbag's target) mismatch in `base_model.load_state_dict(dequantize(...))` — possibly related to the GPT module being re-constructed after quantization.

---

## Part 2 — H-Net Milestone 1 pilot (done) + M2–M4 proposal (grant-funded)

The repo's Requests-for-PRs list in `README.md` still has **H-net tokenization** as an unchecked entry. This PR carries a working implementation of Milestone 1 (hierarchical byte-level stack with a fixed chunker) and proposes M2–M4 (learned chunker, full 16 MB recipe, ablations) as the target for an OpenAI dev grant.

### 2.1 M1 pilot results

Four runs on 1× H100 SXM, all at BYTE_SEQ_LEN=4096, CHUNK_STRIDE=4, BATCH_SIZE=8, AdamW (LR=1.5e-4, WD=0.01, bf16, `torch.compile(mode="default")`). All runs produced by the same code path (`hnet_m1/train_hnet_m1.py`) differing only in whether the byte-encoder→byte-decoder skip connection is present and how many training steps are executed.

| run                          | steps | tokens (train) | skip | final train_loss | **val_bpb** | wallclock | tok/s |
|------------------------------|-----:|-------------:|:----:|---:|---:|---:|---:|
| `hnet_m1_pilot`              |   300 |         10 M | ✗    | 3.13 | 4.49 |  11 s | 942 k |
| `hnet_m1_long`               | 1 500 |         49 M | ✗    | 3.04 | 4.40 |  54 s | 919 k |
| `hnet_m1_skip`               | 1 500 |         49 M | ✓    | 2.16 | 3.15 |  55 s | 904 k |
| **`hnet_m1_final`**          | **4 500** | **147 M** | ✓    | **1.76** | **2.51** | **2.6 min** | 950 k |

Random-byte baseline: `ln(256)/ln(2) = 8.00 bpb`. SP8192 baseline (`bigbag` PR #1493 full training, 4 550 steps × 786 K tokens ≈ 3.6 B tokens): 1.0810 bpb.

Two decisive findings in the pilot:

1. **Byte-encoder → byte-decoder skip connection is load-bearing.** Without it, the decoder has no per-byte fine-grained information — every 4 adjacent bytes share the same upsampled main-network output and must differentiate themselves from just that shared vector. Adding one `x_dec = x_dec + x_enc_final` line dropped val_bpb from **4.40 → 3.15** at matched training data (−1.25 bpb, 28% relative).
2. **Loss continues to decrease.** From 49 M → 147 M tokens (3× more data) val_bpb dropped 3.15 → 2.51 (−0.64 bpb). No plateau observed through step 4500. The curve suggests additional training data at the grant-funded scale would push further; a rough extrapolation at the observed decay rate projects ~1.5–1.7 bpb at 3.6 B tokens, which is within range of SP8192 baselines even with a fixed chunker. Learned chunking (M2) is the expected path to close the remaining gap.

**Model size**: 33.9 M parameters total, dominated by the main network at D=512 × 11 layers:

| component                                 | params          |
|-------------------------------------------|----------------:|
| byte_emb (256 × 256)                      | 65 536          |
| byte_encoder (2 blocks × D=256)           | 1 181 712       |
| enc_to_main projection                    | 131 072         |
| **main_blocks (11 × D=512)**              | **31 742 040** |
| main_to_dec projection                    | 131 072         |
| byte_decoder (1 block × D=256)            | 590 856         |
| final_norm                                | 0              |
| byte_head (256 × 256)                     | 65 536          |
| **total**                                 | **33 907 824** |

At 16 MB int6 quantization the main-network budget is roughly 20 M params; M3 will need to trim either main depth (11 → 7–8) or main width (512 → 384). The SP8192 → byte-level move frees ~2 MB of quantized-tokenizer budget that currently lives in the artifact, mitigating some of the main-network shrinkage.

### 2.2 Why H-Net fits parameter-golf

**H-Net** (Hwang et al., _Dynamic Chunking for End-to-End Hierarchical Sequence Modeling_, arXiv:2507.07955, Jul 2025) trains a byte-level language model end-to-end with a *learned* chunker — two linear projections compute cosine similarity between adjacent encoder outputs, producing a boundary probability `p_t` per position. Boundaries become the compressed representation passed to a main network; the routing decision is made differentiable via a straight-through estimator + EMA smoothing of chunk representations. A compression-ratio loss (with α = 0.03) regularizes the chunker toward a target rate.

Three structural reasons this fits parameter-golf:

1. **Tokenizer model becomes free code.** The baseline bundles `fineweb_8192_bpe.model` (363 KB) into the 16 MB artifact as a frozen SentencePiece file. An H-Net chunker is *~130 K parameters* (two `(D, D_k)` linear projections at the encoder width) which live in the standard quantized-weights payload; the tokenization policy ships as a few tens of lines of code. At D=256 and int6 quant, total chunker cost is <100 KB of the artifact, vs. 363 KB for SP8192. That's ~270 KB of budget freed for the main network.
2. **Compute-adaptive inference.** At eval, H-Net only activates the main network at predicted boundaries, so each step can choose how much compute to spend on a byte. Paper reports 3.5–4× effective compression on English (matching BPE rates) without the hard-coded tokenizer. For parameter-golf's sliding-window eval, this maps directly to a per-step compute ratio we can tune.
3. **Stronger data efficiency at weak-tokenization substructures.** The paper shows ~4× better data efficiency on DNA / code / non-Latin languages vs. BPE. FineWeb is web text with heavy code fragments — a regime where BPE arguably leaves compression on the table.

### 2.3 Minimal viable design (used in the M1 pilot above)

Detailed spec in `hnet_scope.md`. Outline:

- **Byte encoder**: 2 thin layers × D=256 with partial RoPE + GQA (like current baseline, 1/4 width).
- **Dynamic chunker**: Wq, Wk ∈ ℝ^{256×256}, cosine-similarity boundary predictor + EMA smoothing + STE. ~130 K params.
- **Compressed representation**: select encoder outputs at boundary positions (paper's "direct vector selection" — beats mean/max/attn in their ablations).
- **Main network**: 7 layers × D=512 like the current baseline, operating on the compressed chunk stream. Inherits bigbag's stack (parallel residuals, depth recurrence, GPTQ+SDClip, Muon).
- **Byte decoder / upsampler**: 1 layer to project main-network outputs back to per-byte logits.
- **Losses**: byte-level autoregressive CE + α=0.03 compression-ratio regularizer targeting 3.5× compression (≈ SP8192 effective rate on FineWeb).
- **Sliding-window eval** at byte granularity, scoring per-byte NLL.

Parameter budget (rough):
- Byte encoder: 2 × (4d² ≈ 256K + 2 × 256·1024 MLP ≈ 512K) ≈ 1.5 M params.
- Chunker: 130 K.
- Main network: 7 × 12d²  ≈ 22 M params (similar to current main backbone).
- Byte decoder: 1 layer × similar-to-encoder ≈ 0.8 M.
- Byte embedding: 256 × 256 ≈ 66 K. (vs. 8192 × 256 ≈ 2.1 M for SP8192.)
- **Total ≈ 25–28 M params** — fits in the 16 MB int6 budget with headroom for the upsampler.

### 2.4 Risk

- **Byte-level context length bump.** Byte sequences are ~3.7× longer than SP8192 tokens (measured in our preprocessing: 3.73 bytes per SP8192 token on the FineWeb train shards). At BYTE_SEQ_LEN=4096 the main network sees 1024 chunks per sample — less context than bigbag's 2048-token baseline. For M3 we scale BYTE_SEQ_LEN to 8192 so the main network sees 2048 chunks at stride-4, matching baseline context, at the cost of 2× byte-encoder compute.
- **Two-stage joint optimization in M2.** M1 bypassed this risk with a fixed chunker. M2 adds Wq/Wk boundary projections + EMA smoothing + STE + ratio loss — all from the H-Net paper. Paper reports no collapse at 680 M params; at our 34 M (pilot) or 20-25 M (post-trim for M3) we may need α warmup or boundary-init tricks. Our M2 plan includes an explicit early-abort criterion if the chunker collapses.
- **Grad through straight-through estimator** can be noisy. Mitigated by the EMA smoothing path through chunk representations.

### 2.5 Grant-funded milestone plan (M1 done, M2–M4 open)

_Each milestone is gated — if signs of life fail at the gate, the grant pivots._

**Milestone 1 — DONE (self-funded, ~$2 of our quickstart credits)**
- Byte encoder + static chunker + main network + skip-to-decoder.
- Validated signs of life: val_bpb 4.49 → 2.51 over 300 → 4500 training steps. Skip connection responsible for 1.25 bpb of that gap. All data in §2.1.

**Milestone 2 (≈ $120 grant GPU)**: Replace the static stride-4 chunker with the H-Net learned chunker (Wq + Wk cosine-similarity boundary predictor + EMA smoothing + STE + ratio loss targeting r≈3.5). Verify non-degenerate boundaries emerge (F ∈ [0.2, 0.4]) and val_bpb drops below M1's 2.51. Expected outcome: meaningfully below 2.0 bpb by matched-data-budget, demonstrating that content-aware chunking beats fixed-stride at this scale.

**Milestone 3 (≈ $200 grant GPU)**: Full 16 MB submission. Trim the main network (11L → 7–8L or D=512 → 384) to fit the int6 budget, carry over bigbag's SP8192-stack fusions (parallel residuals, depth recurrence, MuonEq-R, GPTQ+SDClip, Brotli-11). 3-seed mean on 8× H100 SXM. Target: land as a creative non-record submission at < 1.5 bpb, or (aspirationally) as a record if the learned chunker compensates for the smaller main-network budget.

**Milestone 4 (≈ $180 grant GPU)**: Ablations — compression ratio sweep (r ∈ {2, 3, 3.5, 4, 5}), byte encoder depth, chunker variants (cosine-sim vs small MLP). Published as an update to this PR.

Total grant-funded spend **≈ $500** — matches the OpenAI dev grant amount. Explicit abort criteria in `hnet_scope.md`.

---

## Artifacts in this PR

| file | purpose |
|---|---|
| `README.md` | this writeup |
| `xsa_triton.py` | forward + backward Triton kernels for XSA, autograd.Function wrapper, torch reference. Numerical parity verified |
| `qkv_fuse.py` | QKV weight-stacking patch for `CausalSelfAttention`. Applies at instance level via monkey-patch. Muon-equivalence caveat documented inline |
| `phase3_run.py` | training-pilot wrapper. `exec()`s bigbag's baseline, intercepts `GPT.__init__` to apply patches post-construction, runs `main()` |
| `bench_scripts/phase1b.sh` | FA3 fallback patch + block microbench at target shape |
| `bench_scripts/phase2a.sh` | FA3 install probe + XSA kernel correctness + isolated microbench |
| `bench_scripts/phase2b.sh` | Block-level bench, FA3 baseline vs FA3 + Triton XSA |
| `bench_scripts/phase2c.sh` | Expanded grid over backend × XSA-impl × QKV-impl |
| `bench_scripts/phase2d.sh` | Drift-controlled grid with pre-compile + dual measurement |
| `bench_scripts/phase2e.sh` | `dynamo.reset()` between variants + thermal prime + CUDA-event timing + 300-sample distribution |
| `bench_scripts/phase3.sh` | Three 200-step training pilots |
| `bench_scripts/phase3b.sh` | Segfault-tolerant rerun + headline summary |
| `hnet_scope.md` | H-Net M2–M4 implementation sketch and milestone budget |
| `hnet_m1/hnet_m1.py` | HNetM1 model factory: byte_emb + encoder + fixed stride-4 chunker + main network + upsampler + byte-encoder-to-decoder skip + byte_decoder + head |
| `hnet_m1/make_byte_shards.py` | decodes SP8192 cached shards back to UTF-8 bytes (observed 3.73 bytes/SP8192-token on FineWeb) and writes byte shards in the baseline's on-disk format |
| `hnet_m1/train_hnet_m1.py` | M1 pilot training loop: AdamW, cosine warmdown, bf16+compile, per-byte val_bpb at end. All four logs in §2.1 came from this script |
| `hnet_m1/phase4.sh` | M1 pilot orchestration (preprocess once, then train) |

## Reproducing

On RunPod `pytorch:1.0.2-cu1281-torch280-ubuntu2404` (or equivalent torch 2.8 + CUDA 12.8), one-time bootstrap + both parts takes ~30 minutes and ≲ $2 of 1×H100 time.

```bash
# one-time bootstrap: repo + deps + SP8192 data + FA3 wheel + unpack baseline
bash bootstrap.sh                     # ~5 min

# Part 1 benchmarks (investigation of the systems-fusion ceiling)
bash bench_scripts/phase1b.sh         # SDPA patch + block microbench
bash bench_scripts/phase2e.sh         # robust drift-controlled benchmark (all 8 variants)
bash bench_scripts/phase3.sh          # 3 × 200-step training pilots (≈30 min)

# Part 2 M1 pilot (H-Net with fixed stride-4 chunker)
bash hnet_m1/phase4.sh                # preprocess byte shards once, then 300-step default pilot

# Reproduce the specific Part 2 runs in §2.1:
ITERATIONS=300  LR=3e-4   RUN_ID=hnet_m1_pilot   bash hnet_m1/phase4.sh   # no skip (earlier code)
ITERATIONS=1500 LR=1.5e-4 RUN_ID=hnet_m1_long    bash hnet_m1/phase4.sh   # no skip, 1500 steps
ITERATIONS=1500 LR=1.5e-4 RUN_ID=hnet_m1_skip    bash hnet_m1/phase4.sh   # with skip (current code)
ITERATIONS=4500 LR=1.5e-4 WARMDOWN_FRAC=0.2 RUN_ID=hnet_m1_final bash hnet_m1/phase4.sh
```

Logs land in `/workspace/logs/${RUN_ID}.txt` with step-level train_loss and tok/s, final per-byte `val_nll` and `val_bpb`.

## Credits and prior art

- Baseline: PR #1493 (@bigbag) — SP8192 + 3-layer recurrence + parallel residuals + QK-Gain 5.25 + legal TTT stack.
- FlashAttention-3 binaries from [`windreamer.github.io/flash-attention3-wheels`](https://windreamer.github.io/flash-attention3-wheels/) (cu128_torch280).
- H-Net: Hwang et al., _Dynamic Chunking for End-to-End Hierarchical Sequence Modeling_, arXiv:2507.07955.
- Scaling-law context: Kaplan et al., _Scaling Laws for Neural Language Models_, arXiv:2001.08361 — the L(N) framing of the parameter-golf challenge.
- Prior systems-PR precedent: PR #1105 (@abaybektursun) Fused MLP (Triton+CUTLASS EVT); PR #1447 (@shram86) FlashMuon.
