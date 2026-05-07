# JEPA Hybrid: Joint-Embedding Predictive Architecture for Language Modeling

**Non-Record Submission (Novel Architecture)**
**Author:** butbutt42 ([@butbutt42](https://github.com/butbutt42))
**Final:** 1.7622 bpb | **Artifact:** 7.5MB (int8 + zlib) | **Steps:** 5908 in 600s on 8xH100
**Crashes survived:** 6 | **Pods burned:** 5 | **Budget burned before first successful run:** ~$22

---

## TL;DR

This is the first JEPA (Joint-Embedding Predictive Architecture) submission to Parameter Golf — directly from the [OpenAI wishlist](https://github.com/openai/parameter-golf/issues/2). Nobody else has submitted a model that trains by predicting in latent space rather than token space.

The BPB (1.76) is not competitive with SOTA (~1.10). That's expected: JEPA is a fundamentally different training paradigm being compared against an AR-optimized evaluation metric. The contribution is demonstrating that JEPA can be trained at all under the 16MB / 10min constraint, and documenting the 6 crashes it took to get there.

| Run | GPU | Steps | BPB | Status |
|-----|-----|-------|-----|--------|
| #1 small | 8xH100 | ~4000 | 1.89 | OOM rank 6 |
| #2 scaled | 8xH100 | ~3800 | 1.71 | Pod EXITED |
| #3 scaled | 2xH100 | ~2000 | — | inference_mode crash |
| #4 scaled | 2xH100 | ~2100 | 1.90 | torch.compile crash |
| #5 scaled | 2xH100 | 2500 | 1.84 | SUCCESS (first completion) |
| #6 scaled | 8xH100 | — | — | Disk full, pod EXITED |
| **#7 final** | **8xH100** | **5908** | **1.76** | **SUCCESS** |

---

## What Is JEPA and Why Does It Matter

JEPA (Yann LeCun, 2022) learns by predicting representations of masked inputs in latent space, rather than predicting discrete tokens. The core idea: instead of "what is the next token?", JEPA asks "what does this region of the sequence *look like* in embedding space?"

This matters for Parameter Golf because:

1. **It's on the wishlist.** OpenAI explicitly asked for "JEPA-style" submissions as novel architecture explorations.
2. **Different inductive bias.** AR transformers learn token-level statistics. JEPA learns structural relationships between sequence regions. Under a 16MB size constraint, this *could* be more parameter-efficient for capturing high-level patterns.
3. **Nobody has tried it.** As of April 17, 2026 — 0 JEPA submissions out of 1600+ PRs.

The fundamental problem: JEPA's training objective (latent MSE) is disconnected from the evaluation metric (autoregressive BPB). We bridge this with hybrid training.

---

## Architecture

```
Encoder: 9-layer Transformer (384d, 6 heads, MLP×2)
Predictor: 2-layer MLP (384 → 128 → 384)
EMA Target: Exponential moving average of encoder (momentum=0.996)
Vocab: 8192 BPE (sp8192 tokenizer)
Params: 13,861,248 total
Artifact: 7,488,519 bytes (int8 + zlib)
```

The model has three forward modes:

1. **JEPA forward** (`forward_jepa`): Mask 15% of tokens → encode unmasked context → predict masked latent representations → MSE loss against EMA target encoder output.

2. **AR forward** (`forward`): Standard causal autoregressive next-token prediction → cross-entropy loss. Uses the same transformer backbone + tied embedding projection.

3. **Hybrid training**: Alternates between JEPA and AR objectives on consecutive steps. Even steps = JEPA, odd steps = AR.

### Why alternating, not combined?

We originally did combined (JEPA + AR in a single step). This required two full forward passes per step, doubling peak GPU memory. On 8xH100, this OOM'd at ~step 3000-4000 every time. Alternating halves peak memory while maintaining both training signals.

### Training schedule

- Steps 0–70% of training: Hybrid (alternating JEPA/AR), `jepa_weight=1.0, ar_weight=0.3`
- Steps 70%–100%: Pure AR (JEPA off, AR only). This fine-tunes the model for the BPB evaluation metric.

---

## The Crash Log: 6 Failures Before Success

This is the part of the submission I think is most useful to others. Each crash had a distinct root cause, and each fix was non-obvious from the error message alone.

### Crash #1: Combined Forward OOM (8xH100, step ~4000)

```
RuntimeError: CUDA out of memory on rank 6
```

**What happened:** `forward_hybrid()` called both `forward_jepa()` and `forward()` in the same step. Two full encoder forward passes = 2× peak memory. On 8 GPUs with DDP, this meant ~80GB+ per GPU under gradient accumulation.

**Why it wasn't caught earlier:** Smoke tests of 30-60 seconds (<1000 steps) never hit OOM because the memory fragmentation hadn't accumulated yet. It took 5-8 minutes of continuous training for CUDA's memory allocator to fragment enough.

**Fix:** Alternate JEPA and AR on consecutive steps instead of combining them. Peak memory went from ~70GB to ~25GB per GPU.

```python
def forward_hybrid(self, input_ids, target_ids, jepa_weight, ar_weight):
    if not hasattr(self, '_hybrid_step'): self._hybrid_step = 0
    self._hybrid_step += 1
    if self._hybrid_step % 2 == 0 and jepa_weight > 0:
        return jepa_weight * self.forward_jepa(input_ids)
    else:
        return ar_weight * self.forward(input_ids, target_ids)
```

**Lesson:** Two forward passes in one step = doubled memory. Obvious in retrospect, invisible when writing the code.

### Crash #2: Same OOM, Different Machine (8xH100, step ~3800)

Moved to a different RunPod machine to rule out hardware. Same crash, same step range. Confirmed: it was the code, not the GPU.

**Cost:** $5.33 in RunPod credits for two pods that both died.

### Crash #3: inference_mode vs torch.compile (2xH100, step ~2000)

```
RuntimeError: Inference tensors cannot be saved for backward
```

**What happened:** Validation uses `torch.inference_mode()` for eval. But `torch.compile()` caches tensor metadata across train/eval boundaries. An inference-mode tensor got cached, then the next training step tried to compute gradients through it.

**Why it wasn't caught earlier:** Validation runs every 1000 steps. Smoke tests of <1000 steps never triggered it.

**Fix:** Replace `torch.inference_mode()` with `torch.no_grad()` everywhere. `no_grad()` doesn't set the inference flag on tensors, so `torch.compile()` cache stays consistent.

### Crash #4: Post-val mystery crash (2xH100, step ~2100)

Crash #3 fix worked — val at step 2000 passed. Then the model crashed at ~step 2100 with no traceback. Memory was stable. `inference_mode` was fixed.

**Best theory:** `torch.compile()` triggers a re-trace/recompilation after the val eval changes the model state. Something in the recompilation path breaks silently.

**Fix (nuclear):** Disable `torch.compile()` entirely. Also skip periodic validation (`val_loss_every=99999`), only eval at the very end.

**Cost:** This made training ~50% slower per step, but the model actually finishes instead of crashing.

### Crash #5: First successful 2xH100 run

With all four fixes applied (alternating, no_grad, no compile, no periodic val), the model trained for 10 full minutes without crashing. **BPB: 1.8373, artifact: 7.6MB, 2500 steps.**

This was the first JEPA model to complete training in Parameter Golf. But 2xH100 is only 2500 steps. We needed 8xH100 for more steps and better BPB.

### Crash #6: Disk full (8xH100, step 0)

Created an 8xH100 pod with 20GB container disk. The dataset download (`snapshot_download` without `allow_patterns`) pulled the entire HuggingFace repo (~30GB of sp1024 + sp8192 + other datasets) and filled the disk before training started. Pod EXITED.

**Fix:** Use `allow_patterns=['datasets/fineweb10B_sp8192/*', 'tokenizers/fineweb_8192*']` to download only the 8GB we need. Also: create pods with 50GB disk minimum.

**Cost:** ~$3 in idle GPU time while 20GB of wrong data downloaded at full speed.

### Run #7: Final successful 8xH100 run

50GB disk. `allow_patterns` download (30 seconds). Training started immediately.

```
Steps completed: 5908 in 600 seconds
Memory: FLAT 0.2GB allocated / 2.4GB reserved (entire run)
Peak: 15,760 MiB (out of 80GB per GPU)
Final val_bpb: 1.7579 (pre-quant) → 1.7622 (int8+zlib roundtrip)
```

Every previous crash point was passed cleanly:
- Step 2000 (crash #3 point): clean
- Step 2100 (crash #4 point): clean
- Step 3800 (crash #2 point): clean
- Step 4000 (crash #1 point): clean

---

## Results

### Training curve (8xH100 final run, logged every 200 steps)

| Step | Train Loss | Wall Time | GPU Mem |
|------|-----------|-----------|---------|
| 0 | — | 0s | 0.2 / 2.4 GB |
| 200 | 0.5466 | 20.6s | 0.2 / 2.4 GB |
| 1000 | 0.5834 | 101.8s | 0.2 / 2.4 GB |
| 2000 | 0.6014 | 203.4s | 0.2 / 2.4 GB |
| 3000 | 0.6069 | 305.0s | 0.2 / 2.4 GB |
| 4000 | 0.6331 | 406.5s | 0.2 / 2.4 GB |
| 5000 | 0.6281 | 507.9s | 0.2 / 2.4 GB |
| 5800 | 0.6233 | 589.0s | 0.2 / 2.4 GB |
| 5908 | — | 600.1s | **WALLCLOCK** |

**Final val_bpb: 1.7622** (int8+zlib roundtrip)

Note: train_loss alternates between JEPA loss (~0.5-0.6, MSE in latent space) and AR loss (~0.6, cross-entropy). The logged values are from whichever mode was active at the logging step.

### 2xH100 vs 8xH100

| Metric | 2xH100 | 8xH100 |
|--------|--------|--------|
| Steps | 2500 | 5908 |
| BPB | 1.8373 | 1.7622 |
| Artifact | 7.6 MB | 7.5 MB |
| Time | 600s | 600s |

8xH100 gave 2.4× more steps and 4.1% better BPB. The improvement is sublinear because the AR loss plateau is already visible by step 2000.

---

## Why the BPB Is Not Competitive (And Why That's Fine)

Current SOTA is ~1.10 bpb. We're at 1.76. The gap (~0.66 bpb) comes from fundamental architectural choices, not engineering mistakes:

1. **Objective mismatch.** JEPA trains on latent MSE; evaluation is token-level cross-entropy. Even with 30% AR training mixed in, the model's representations are shaped by JEPA's objective. The AR head is essentially fine-tuned on top of JEPA features rather than being the primary training signal.

2. **Alternating halves the effective AR steps.** Only odd-numbered steps train AR. The model gets ~2950 AR steps out of 5908 total. A pure AR model would get all 5908.

3. **No torch.compile.** We disabled it for stability. A pure AR model with compile would be 50-100% faster per step, getting 8000-10000 steps in the same wallclock.

4. **Predictor overhead.** The JEPA predictor (384→128→384) adds parameters that don't help BPB. They only help JEPA learn better latent representations.

This submission is not trying to win on BPB. It's demonstrating that JEPA — a fundamentally different learning paradigm — can be trained at all within Parameter Golf's constraints, and documenting the practical engineering required to make it work.

---

## What We Tried (Abbreviated Negative Results)

- **Combined JEPA+AR loss in single step:** 2× memory, OOM. Don't do this.
- **torch.compile with JEPA:** Crashes after val eval. Incompatible with model state changes between train/eval.
- **torch.inference_mode() for eval:** Poisons torch.compile cache. Use no_grad().
- **20GB container disk:** Not enough for dataset + HF cache. Use 50GB.
- **RotorQuant (block-diagonal rotation in GPTQ):** Artifact exceeded 16MB by 40KB. Dropped.

---

## Reproducing

```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Requirements: PyTorch 2.4+, sentencepiece, brotli. No external packages beyond standard PyTorch + scipy.

Dataset: `sproos/parameter-golf-tokenizers` (sp8192 split, ~8GB).

Expected: ~5900 steps in 600s, val_bpb ~1.76, artifact ~7.5MB.

---

## Key Fixes Checklist (For Anyone Trying Hybrid Architectures)

If you're building a non-standard architecture for Parameter Golf, here's what we learned the hard way:

- [ ] **One forward pass per step.** If your architecture has two loss terms, alternate them. Don't combine.
- [ ] **Use `torch.no_grad()`, not `torch.inference_mode()`.** The latter breaks `torch.compile()` cache.
- [ ] **Disable `torch.compile()` if your model has multiple forward modes.** Compile retraces after state changes and may crash silently.
- [ ] **Log GPU memory every N steps.** Memory leaks are invisible without logging.
- [ ] **Run full 10-minute tests, not 60-second smoke tests.** OOM and compile bugs often appear after minutes, not seconds.
- [ ] **50GB container disk minimum.** Dataset + HF cache + model + artifacts need room.
- [ ] **Use `allow_patterns` in `snapshot_download`.** Don't download entire repos.

---

## Attribution

- JEPA concept: Yann LeCun, "A Path Towards Autonomous Machine Intelligence" (2022)
- EMA target encoder: inspired by BYOL (Grill et al., 2020) and I-JEPA (Assran et al., 2023)
- Base training infrastructure: adapted from openai/parameter-golf baseline
- Tokenizer: sp8192 from sproos/parameter-golf-tokenizers
- Training compute: RunPod 8xH100 SXM (~$25 total across all attempts)

---

## Total Cost Breakdown

| What | Cost |
|------|------|
| Grant account crashes (#1, #2) | ~$13 |
| Personal 2xH100 tests (#3, #4, #5) | ~$4 |
| Personal 8xH100 disk crash (#6) | ~$3 |
| Personal 8xH100 final run (#7) | ~$3 |
| Idle time between runs | ~$2 |
| **Total** | **~$25** |

Seven runs. Six crashes. One successful model. The first JEPA in Parameter Golf.
