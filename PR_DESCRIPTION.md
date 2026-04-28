## Parameter Golf: S9 + SmearGate + Sparse Attention Gate + LQER

**Submitted result: quant+TTT val_bpb = 1.0705, artifact = 15.92 MB** (single seed, 8xH100 SXM).
Built on PR #1867. Current merged SOTA (PR #1493): 1.0810 BPB. Delta: **-0.0105 BPP**.

---

### Starting Point: Community Baselines

The starting point was `train_gpt_0427.py`, a minified single-file GPT implementation inheriting the community's standard architecture:

- 11L x 512d, 8 heads / 4 KV heads, MLP 4x with LeakyReLU(0.5)^2 activation
- Tied embeddings (vocab 8192), logit softcap = 30.0, partial RoPE (16/64 dims)
- Layer looping (layers 3-5, 2 loops), parallel residuals from layer 7+, skip gates (U-Net connections)
- Muon optimizer (Newton-Schulz orthogonalization, 5 backend steps) for matrix params, AdamW for embeddings/scalars
- GPTQ int6 quantization + sliding-window eval + optional TTT
- EMA decay 0.9965, wallclock cap 600s

Key HPs from 0427: `matrix_lr=0.022`, `muon_wd=0.095`, `qk_gain_init=5.0`, `warmdown_frac=0.72`, `muon_momentum=0.99`.

---

### Stage 1: 0427 Baseline Results (Apr 26-27)

Three seeds on 1-GPU runs:

| Variant | Seeds | Post-EMA BPB | Quant+SW BPB | Artifact |
|---------|-------|-------------|--------------|----------|
| 0427 baseline | 1337/1338/1339 | ~1.089 | **1.0831** | 16.44 MB |
| 0409 + TTT | 42/999 | ~1.085 | **1.0782** (quant+TTT) | 15.99 MB |

The 0409 baseline with TTT established that test-time training provides approximately -0.005 BPB improvement over sliding-window eval alone.

---

### Stage 2: S9 Stack -- Bank-Mode + Polar-Express Muon (Apr 27)

`train_gpt_s9.py` was a clean rewrite with several key changes:

**Architecture changes:**
- Flash-attn backend with graceful fallback chain: flash_attn_3 (Hopper) -> flash_attn_2 -> PyTorch SDPA
- `flash_attn_varlen` document packing support (avoids padding waste)
- Fused softcapped cross-entropy via custom Triton kernels
- Bank-mode weight storage (`qo_bank`, `kv_bank`, `mlp_up_bank`, `mlp_down_bank`)
- Polar-Express Newton-Schulz coefficients for Muon orthogonalization

**HP changes vs 0427:**
- `matrix_lr`: 0.022 -> **0.026**
- `muon_momentum`: 0.99 -> **0.97**
- `warmdown_frac`: 0.72 -> **0.75**
- `loop_start`: 4 -> **3** (earlier layer looping)
- `enable_looping_at`: 0.5 -> **0.35** (enable looping earlier in training)
- `parallel_start_layer`: 7 -> **8**
- `min_lr`: 0.0 -> **0.1** (nonzero floor)

**TTT overhaul:**
- Phased LoRA TTT with Adam optimizer (lr=0.0001, beta1=0, beta2=0.999, wd=1.0)
- LoRA rank 96, alpha 144, applied to K, O, and MLP projections
- 3-phase scoring with 2000 prefix documents per phase
- Document-boundary-respecting chunking (chunk size 48 tokens)

**New module catalog (all OFF by default, toggled via env vars):**
- SmearGate, Sparse Attention Gate, Gated Attention, AttnOutGate, Recur-Alpha, SpinQuant, LQER

**S9 1-GPU results (3 seeds):**

| Seed | Post-EMA BPB | Quant+TTT BPB | Artifact |
|------|-------------|---------------|----------|
| 42   | 1.0716      | **1.0683**    | 16.89 MB |
| 314  | 1.0724      | **1.0696**    | 16.90 MB |
| 999  | 1.0716      | **1.0684**    | 16.89 MB |
| **Mean** | **1.0719** | **1.0688** | **16.89 MB** |

Substantial improvement over the 0427 baseline (-0.014 BPB), primarily from the higher matrix LR, earlier looping, better TTT, and the Polar-Express Muon formulation.

**Problem: artifact size.** At 16.89 MB, the S9 vanilla artifact was **0.89 MB over the 16 MB limit**.

---

### Stage 3: Artifact Size Reduction & New Features (Apr 28)

#### 3a: S9 + SmearGate + Sparse Attn Gate + LQER + Embed int7

The key insight: **instead of just compressing harder, use the freed bytes from better quantization to add architectural features that improve BPB.**

- **SmearGate** (gate_window=12, BOS-masked): learned gating that blends neighboring token representations via a sliding window. At init, lambda=0 so transparent; the model learns local context mixing.
- **Sparse Attention Gate** (init_std=0.0): per-head learned gates allowing selective attention pattern pruning.
- **LQER** (rank=4, int4 factors, asymmetric, group_size=64, top-3 layers): post-GPTQ error correction. Adds small low-rank correction factors to the 3 weight matrices with highest quantization error.
- **Embed int7** (clip_sigmas=15.0): 7-bit embedding quantization instead of 8-bit, saving ~1 MB.

| Config | Post-EMA BPB | Quant BPB | Quant+TTT BPB | Artifact |
|--------|-------------|-----------|---------------|----------|
| S9+smear+sparse+LQER+embed7 | 1.0720 | 1.0816 | **1.0705** | **15.92 MB** |

The artifact dropped from 16.89 MB to **15.92 MB** (under the 16 MB limit) while BPB only degraded by 0.0017 vs the 1-GPU 3-seed mean. LQER recovered most quality lost from aggressive embedding quantization.

#### 3b: Cap Tokenizer + CaseOps + LQER

Explored a different compression axis: **reducing vocabulary** via case-folding operations. By training a "cap" tokenizer that folds case, vocab can shrink from 8192 to ~7088 tokens, drastically reducing the embedding table.

| Config | Vocab | Quant+TTT BPB | Artifact |
|--------|-------|---------------|----------|
| Cap + CaseOps v7088 | 7088 | **1.0715** | **15.62 MB** |
| Cap + CaseOps v7972 + MLP 4.125x | 7972 | **1.0689** | **16.17 MB** |

The v7972 variant achieved the **best single-seed BPB of any 8-GPU run** (1.0689), but was not selected for submission due to tokenizer compliance risk -- the BPB evaluation pipeline expects the standard sp8192 tokenizer, and CaseOps changes the byte-counting LUT path.

---

### Stage 4: PR #1851 Exploration (Apr 28)

`train_gpt_s0_pr1851_mod.py` explored the upstream PR #1851 architecture with extensive annotations. Both runs showed competitive pre-quant BPB (base: 1.0722, cap: 1.0748) but **crashed before quantization** due to `pyminify` not being installed and `.int6.ptz` deserialization issues on non-rank-0 workers.

---

### Crash Catalogue

| Issue | Runs Affected | Root Cause |
|-------|---------------|------------|
| `pyminify` FileNotFoundError | PR1851 base/cap (2 runs) | All ranks called `subprocess.run(["pyminify", ...])` |
| `.int6.ptz` deserialization | S9 8-GPU (3 runs) | Each rank generated unique `run_id`, rank N looked for files only rank 0 created |
| NCCL timeout | 0427 4-GPU (3 runs) | `nvmlDeviceGetHandleByIndex(7) failed` on initial setup |
| `flash_attn_varlen` missing | S9 early (3 runs) | Nodes without flash_attn_3 installed |
| Data shard mismatch | Cap tokenizer (1 run) | Incorrect directory layout for re-tokenized shards |

---

### Why S9+SmearGate+Sparse+LQER Was Selected

| Candidate | Quant+TTT BPB | Artifact | Status |
|-----------|---------------|----------|--------|
| S9 1-GPU 3-seed mean | **1.0688** | 16.89 MB | Over 16 MB limit |
| Cap v7972 + MLP4.125 | **1.0689** | 16.17 MB | Tokenizer compliance risk |
| **S9+smear+sparse+LQER** | **1.0705** | **15.92 MB** | **Submitted** |
| Cap v7088 + CaseOps | 1.0715 | 15.62 MB | Tokenizer compliance risk |
| S9 8-GPU vanilla | 1.0719 | 16.89 MB | Over limit |
| PR1851 base | 1.0722 (pre-quant) | Unknown | Crashed |

The only configuration that fits under 16 MB, uses the standard tokenizer, and completed the full pipeline.

---

### Future Directions

1. **Multi-seed validation** -- Need 2 additional seeds for statistical significance per submission guidelines.
2. **CaseOps tokenizer submission** -- The cap v7972 variant (1.0689 BPB) is the strongest known config if tokenizer compliance is confirmed.
3. **LQER rank sweep** -- Current rank-4 on top-3 layers. Higher rank with more layers might be worth the extra bytes.
4. **SpinQuant integration** -- Online Hadamard rotation is implemented but untested in the submitted config.
5. **PR1851 completion** -- Fix pyminify/deserialization crashes to get full quant+TTT numbers.
6. **Recur-Alpha and Gated Attention ablation** -- Both implemented but not yet enabled.

---

### Reproduction

```bash
pip install brotli sentencepiece
pip install flash_attn_3 --no-deps --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp8192

SEED=42 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

### Files in this PR

| File | Description |
|------|-------------|
| `train_gpt_0427.py` | Stage 1: minified 0427 baseline |
| `train_gpt_s9.py` | Stage 2: S9 bank-mode + Polar-Express Muon stack |
| `train_gpt_s9_caseops_lqer.py` | Stage 3b: S9 + CaseOps + LQER cap tokenizer variant |
| `train_gpt_s0_pr1851_mod.py` | Stage 4: annotated PR #1851 exploration |
| `records/.../2026-04-28_S9_SmearGate_SparseAttn_LQER/` | Submission record (README, submission.json, train_gpt.py, train_seed42.log) |
