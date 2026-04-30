# V22: V21 base + PR #1953's 7 levers + EVAL_SEQ_LEN=2816 — val_bpb 1.05877 (3-seed mean, all strict <600s)

> **V22 update (2026-05-01)** layers PR #1953 (@andrewbaggio1)'s 7 hparam levers on top of V21's PR #1908+AWQ-lite+AsymLogit+WD=2.0 base, with `EVAL_SEQ_LEN` raised from PR #1953's 2560 to **2816** (longer eval context). All 3 seeds strict <600s train wallclock (596.087-596.152s) and 475-522s eval (well under 600s cap).

## V22 results (3-seed)

| Seed | Stop step | Train wallclock | Eval time | Pre-quant | Quantized | **Post-TTT** | Artifact |
|------|----------:|----------------:|----------:|----------:|----------:|-------------:|---------:|
| 42   | 4,984 | 596.152s ✅ | 522.21s | 1.05952 | 1.06791 | **1.057334** | 15,981,259 |
| 0    | 4,934 | 596.103s ✅ | 479.95s | 1.06204 | 1.07029 | **1.059588** | 15,981,985 |
| 1234 | 4,935 | 596.087s ✅ | 475.58s | 1.06149 | 1.07015 | **1.059375** | 15,982,315 |
| **Mean** | **4,951** | **596.11s** | **492.58s** | **1.06102** | **1.06945** | **1.058769** | **15,981,853** |

**3-seed mean val_bpb: 1.05877** (std 0.00102) | **~15.98 MB** | 8×H100 SXM5 80GB (Hyperbolic eu-north-4) | full TTT eval

## V22 vs leaderboard (2026-04-30)

| | V22 mean | Δ vs V22 |
|---|---:|---:|
| PR #1967 ndokutovich (N-gram Tilt) | 1.05851 | +0.00026 |
| PR #1953 andrewbaggio (7 levers) | 1.05855 | +0.00022 |
| **V22 (this submission)** | **1.05877** | — |
| PR #1965 himanshudongre | 1.05875 | -0.00002 |
| PR #2007 elubrazione | 1.05899 | -0.00022 |
| **V21 v2 alertcat (this PR's prior version)** | **1.05943** | **-0.00066** ✅ |
| PR #1908 romeerp (AWQ-lite frontier) | 1.06081 | -0.00204 |
| PR #1855 codemath3000 (cocohearts-merged #1) | 1.06108 | -0.00231 |
| MERGED SOTA bigbag PR #1493 | 1.0810 | -0.02223 |

**V22 improves over V21 v2 by −0.00066 BPB** (within the community's 0.0006 floor for meaningful improvement). V22 falls 0.00022 BPB short of PR #1953/1967 — within seed noise but technically behind on 3-seed mean. The +66µ delta from V21 came primarily from seed 42's pre-quant dropping to 1.05952 (vs PR #1953's 1.06163 at the same seed), made possible by the longer eval context (EVAL_SEQ_LEN=2816 vs 2560).

## V22 stack (in addition to V21)

7 hparam levers from [PR #1953](https://github.com/openai/parameter-golf/pull/1953) by **@andrewbaggio1**, with EVAL_SEQ_LEN raised:

```
EVAL_SEQ_LEN=2816          # V22 raised from PR #1953's 2560
TTT_EVAL_SEQ_LEN=2816      # matched
TTT_MASK=no_qv             # K/MLP/O LoRA active, Q/V LoRA disabled at TTT
TTT_Q_LORA=0
TTT_V_LORA=0
TTT_LOCAL_LR_MULT=0.75     # local LR multiplier for per-doc adapter
QK_GAIN_INIT=5.25          # init for QK gain scalar
```

All other V21 settings (PR #1908 base + AWQ-lite + AsymLogit + WD=2.0) carried over verbatim.

## V22 revisions

- **v3 (2026-05-01)**: V22 = V21 v2 stack + 7 PR #1953 levers + EVAL_SEQ_LEN=2816. 3-seed mean 1.05877. All 3 seeds strict <600s. Run on Hyperbolic eu-north-4 Iceland VM (8×H100 SXM5 80GB).

---

# Original V21 submission (preserved below for context)

# V21: PR #1855 stack + AWQ-lite + Asymmetric Logit Rescale — val_bpb 1.05943 (3-seed mean, all strict <600s)

**3-seed mean val_bpb: 1.05943** (std 0.00064) | **~15.98 MB** | 8×H100 SXM | full TTT eval

**All 3 seeds strict <600s wallclock (596.045-596.102s)** — addressing community feedback from @aquariouseworkman + @romeerp on initial v1 submission.

**Improvement over current MERGED SOTA (bigbag PR #1493 at 1.0810): −0.02157 BPB / −0.0498 nats**
**Improvement over current open frontier (PR #1908 romeerp at 1.06081): −0.00138 BPB** (Welch t≈2.18, p≈0.045)
**Improvement over current cocohearts-merged #1 (PR #1855 codemath3000 at 1.06108): −0.00165 BPB**

## Results

| Seed | Stop step | Train wallclock | Pre-quant BPB | Quantized BPB | **Post-TTT BPB** | Artifact |
|------|----------:|----------------:|--------------:|--------------:|-----------------:|---------:|
| 42   | 4,908 | 596.102s ✅ | 1.064267 | 1.072599 | **1.058675** | 15,981,148 |
| 0    | 4,880 | 596.057s ✅ | 1.065056 | 1.073377 | **1.059394** | 15,977,881 |
| 1234 | 4,870 | 596.045s ✅ | 1.065740 | 1.074314 | **1.060243** | 15,986,941 |
| **Mean** | **4,886** | **596.07s** | **1.065021** | **1.073430** | **1.059434** | **15,981,990** |

**3-seed std: 0.00064 BPB / 0.00141 nats.** Each individual seed beats the merged 1.0810 leaderboard by ≥0.0207 BPB / ≥0.0478 nats.

**Note on revisions**: Initial v1 submission used `FORCE_STOP_STEP=4920` + `GPTQ_RESERVE_SECONDS=0.5` for seed 42 which produced 602.048s wallclock (borderline, matching PR #1908 seed 42 at 601.153s). Per @aquariouseworkman + @romeerp review (the latter being PR #1908 author who confirmed his own step-matched runs were ablation-only, not record-grade), seed 42 was re-run with `GPTQ_RESERVE_SECONDS=4.0` and no `FORCE_STOP_STEP` (identical config to seeds 0 and 1234). v2 mean 1.05943 vs v1 mean 1.05932 (+0.00011, well within the tighter v2 std of 0.00064). All 3 seeds now strict <600s.

## Stack: PR #1855 (codemath3000) + PR #1908 quantization + V21 innovation

This submission follows the architectural lineage that cocohearts merged into the official leaderboard chain on 2026-04-28 (via PR #1902, listing PR #1855 as the new top row). On top of that base, this submission applies:

1. **AWQ-lite mixed-precision GPTQ** from PR #1908 (romeerp)
   - Activation-aware salient-group selection
   - Top-1 group of 64 columns promoted to int8 inside the same Hessian-based GPTQ solve
   - Net: ~−0.0002 BPB on the PR #1855 base (verified by PR #1908)

2. **Asymmetric Logit Rescale** from PR #1923 (jorge-asenjo) — V21's only architectural addition
   - Replaces the single `logit_softcap` scalar with two learnable scalars (`softcap_pos`, `softcap_neg`) on the eval path
   - Acts via `where(logits>0, sp*tanh(logits/sp), sn*tanh(logits/sn))` in `forward_logits` and `forward_ttt`
   - Both scalars init to `LOGIT_SOFTCAP=30.0` (identity at step 0)
   - Eval-only — train path keeps the single fused softcap unchanged
   - 8-byte artifact cost (2 × fp16 passthrough scalars)
   - **Empirical TTT recovery boost: +0.00128 BPB consistent across 3 seeds**

3. **All other components** inherited verbatim from PR #1855:
   - 11L XSA + LQER + Sparse Attn Gate + BOS-fixed SmearGate
   - Polar-Express Newton-Schulz Muon
   - Phased TTT 3 phases at boundaries [833, 1666, 2500]
   - Per-group lrzip ZPAQ compression + L1 similarity-sort

## Key innovation: Asymmetric Logit Rescale on PR #1908 base

PR #1923 (jorge-asenjo) reported the asymmetric softcap as **+0.00469 BPB negative** on the PR #1855 base alone (1.06577 vs 1.06108). PR sunnypatneedi's 2026-04-29 frontier-scan flagged this as "empirical NEGATIVE result, regresses ~0.005 vs #1855 — Don't try this."

**This submission falsifies that conclusion** when the asymmetric softcap is combined with PR #1908's AWQ-lite mixed-precision quantization:

| Configuration | Pre-quant | Quantized | Post-TTT | TTT recovery |
|---|---|---|---|---|
| PR #1908 seed 42 (no AsymLogit) | 1.06384 | 1.07226 | 1.05957 | 0.01269 |
| **V21 seed 42 (AsymLogit on)** | **1.06393** | **1.07232** | **1.05834** | **0.01398** |

The asymmetric logit head **does not change pre-quant or quantized values** (within numerical noise) but **improves TTT recovery by +0.00129 BPB**. This pattern holds across all 3 seeds (recovery 0.01398 / 0.01398 / 0.01407). The likely mechanism: during 3-phase TTT, the per-doc LoRA adapter learns to push asymmetric logit distributions that the symmetric softcap cannot capture, but the asymmetric softcap can.

## Compliance (Issue #1017 Track A)

- [x] **Causality**: VarLen attention with per-doc cu_seqlens, strict causal mask (inherited from PR #1855)
- [x] **Normalized softmax**: full SP8192 vocab via lossless CaseOps tokenizer, softcap then standard softmax
- [x] **Score-before-update**: Phased TTT 3-phase, prefix docs scored under no_grad (gd:0) before LoRA grad steps; suffix docs scored with adapted LoRA (gd:1) — each val token scored exactly once
- [x] **Single pass**: each val token scored exactly once across all 3 phases (verified in train logs)
- [x] **No SLOT, no pre-quant TTT, no n-gram cache, no ETLB**
- [x] **3-seed validation**: seeds 42 / 0 / 1234 (matching PR #1908 / PR #1855 convention), std 0.00078
- [x] **Artifact size**: max 15,986,941 bytes (under 16,000,000 cap)
- [x] **Eval wallclock**: 414-460s (well under 600s cap)
- [x] **Train wallclock**: seeds 0 + 1234 strict <600s; seed 42 borderline 602.048s (matches PR #1908 borderline status accepted by cocohearts)

## Reproduction

### System setup (one time)

```bash
# Install lrzip (system binary required for COMPRESSOR=pergroup, same as PR #1855)
apt-get install -y lrzip

# Python deps
pip install --break-system-packages sentencepiece brotli huggingface_hub numpy python-minifier hf_transfer
pip install --break-system-packages --no-deps flash_attn_3 --find-links \
  https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/

# Dataset (CaseOps-tokenized FineWeb 10B, ~16 GB)
HF_HUB_ENABLE_HF_TRANSFER=1 python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='romeerp/parameter-golf-caseops-v1',
    repo_type='dataset',
    local_dir='/workspace/caseops_data',
    max_workers=16,
)"
# IMPORTANT: chmod 644 all files (RunPod FUSE bug prevention)
find /workspace/caseops_data -type f -exec chmod 644 {} +
```

### Run 3-seed validation

```bash
ENV_VARS="DATA_DIR=/workspace/caseops_data/datasets/ \
  DATA_PATH=/workspace/caseops_data/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved \
  TOKENIZER_PATH=/workspace/caseops_data/datasets/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model \
  CASEOPS_ENABLED=1 VOCAB_SIZE=8192 \
  ITERATIONS=20000 MAX_WALLCLOCK_SECONDS=600 \
  WARMUP_STEPS=20 WARMDOWN_FRAC=0.85 BETA2=0.99 \
  GRAD_CLIP_NORM=0.3 MIN_LR=0.1 MATRIX_LR=0.026 \
  GLOBAL_TTT_MOMENTUM=0.9 \
  SPARSE_ATTN_GATE_ENABLED=1 SPARSE_ATTN_GATE_SCALE=0.5 \
  SMEAR_GATE_ENABLED=1 GATE_WINDOW=12 GATED_ATTN_QUANT_GATE=1 \
  FUSED_CE_ENABLED=1 EMBED_BITS=7 \
  MLP_CLIP_SIGMAS=11.5 ATTN_CLIP_SIGMAS=13.0 EMBED_CLIP_SIGMAS=14.0 \
  GPTQ_RESERVE_SECONDS=4.0 GPTQ_CALIBRATION_BATCHES=16 COMPRESSOR=pergroup \
  LQER_ENABLED=1 LQER_ASYM_ENABLED=1 LQER_RANK=4 LQER_FACTOR_BITS=4 \
  LQER_ASYM_GROUP=64 LQER_TOP_K=3 \
  AWQ_LITE_ENABLED=1 AWQ_LITE_BITS=8 AWQ_LITE_GROUP_TOP_K=1 AWQ_LITE_GROUP_SIZE=64 \
  PHASED_TTT_ENABLED=1 PHASED_TTT_PREFIX_DOCS=2500 PHASED_TTT_NUM_PHASES=3 \
  TTT_CHUNK_SIZE=48 TTT_BETA2=0.99 TTT_WEIGHT_DECAY=0.5 TTT_LORA_RANK=80 \
  MUON_BACKEND_STEPS=5 NCCL_NET=Socket VAL_LOSS_EVERY=0 \
  ASYM_LOGIT_RESCALE=1"

for SEED in 42 0 1234; do
  env SEED=$SEED $ENV_VARS \
    torchrun --standalone --nproc_per_node=8 train_gpt.py \
    > train_seed${SEED}.log 2>&1
done
```

**Note on seed 42**: this submission's seed 42 was originally run with `FORCE_STOP_STEP=4920` and `GPTQ_RESERVE_SECONDS=0.5` (which produced 602.048s wallclock — borderline). Reproducers should use the standard env vars above (which all 3 of our seeds 0+1234 used) and all 3 seeds will finish strictly under 600s.

## Code changes vs PR #1908

5 surgical edits to `train_gpt.py` (+26 lines, all eval-only). Train numerics are bit-identical to PR #1908.

1. Line ~299 — `TTT_WEIGHT_DECAY` default 1.0 → 2.0 (sunnypatneedi 2026-04-28 finding for fused-CE + warm-start LoRA-A stability; we override to 0.5 via env to match PR #1855)

2. Line ~1259 — `nn.Parameter` additions in `GPT.__init__`:
   ```python
   self.asym_logit_enabled = bool(int(os.environ.get("ASYM_LOGIT_RESCALE", "0")))
   if self.asym_logit_enabled:
       self.softcap_pos = nn.Parameter(torch.tensor(float(h.logit_softcap), dtype=torch.float32))
       self.softcap_neg = nn.Parameter(torch.tensor(float(h.logit_softcap), dtype=torch.float32))
   ```

3. Line ~1419 — `_apply_asym_softcap` helper method:
   ```python
   def _apply_asym_softcap(self, logits):
       sp = self.softcap_pos.to(logits.dtype)
       sn = self.softcap_neg.to(logits.dtype)
       return torch.where(logits > 0, sp * torch.tanh(logits / sp), sn * torch.tanh(logits / sn))
   ```

4. Line ~1431 — `forward_logits` eval path branch:
   ```python
   if self.asym_logit_enabled:
       return self._apply_asym_softcap(logits_proj)
   return self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
   ```

5. Line ~1533 — `forward_ttt` eval path branch (same conditional)

The training-path `forward()` and the fused softcapped CE Triton kernel are **unchanged** — train numerics match PR #1908 exactly.

## Files

- `train_gpt.py` — full training script (PR #1908 base + 5 V21 edits, ~3,998 lines, 170 KB)
- `requirements.txt` — Python deps reference
- `submission.json` — structured 3-seed metadata
- `V21_README.md` — this writeup
- `train_seed42.log`, `train_seed0.log`, `train_seed1234.log` — full per-seed run logs
- Auxiliary scripts:
  - `run_v21_full_stack_scout.sh` — single-seed scout (initial verification, 1.05829 BPB at FSS=4945)
  - `run_v21_3seeds.sh` — historical 3-seed runner (FSS=4920, used for seed 42)
  - `run_v21_seeds_0_1234_optimized.sh` — strict <600s 2-seed runner (used for seeds 0 + 1234)

## Credits

V21's stack stacks decisions from a long sequence of community PRs, layered exactly as cocohearts has been merging:

- [PR #1908](https://github.com/openai/parameter-golf/pull/1908) by **@romeerp** — AWQ-lite mixed-precision GPTQ on PR #1855 base. V21's quantization path is bit-identical.
- [PR #1855](https://github.com/openai/parameter-golf/pull/1855) by **@codemath3000** — base architecture. cocohearts listed as official #1 on 2026-04-28 via PR #1902.
- [PR #1923](https://github.com/openai/parameter-golf/pull/1923) by **@jorge-asenjo** — Asymmetric Logit Rescale conceptual contribution.
- [PR #1797](https://github.com/openai/parameter-golf/pull/1797) by **@dexhunter** — Smear Gate + LQER asymmetric rank-4.
- [PR #1787](https://github.com/openai/parameter-golf/pull/1787) by **@nprime06** — Polar Express NS, MIN_LR=0.1, sparse attention gate, fused softcapped CE.
- [PR #1729](https://github.com/openai/parameter-golf/pull/1729) by **@romeerp** — sp8192 lossless caps caseops v1 tokenizer + per-token byte sidecar.
- [PR #1493](https://github.com/openai/parameter-golf/pull/1493) by **@bigbag** — current merged SOTA baseline (1.0810).
- [PR #1394](https://github.com/openai/parameter-golf/pull/1394) by **@clarkkev** — SP8192 + GPTQ + SDClip foundation.
- [PR #1530](https://github.com/openai/parameter-golf/pull/1530) by **@samacqua** — VarLen attention, fused LeakyReLU² MLP Triton kernel, parallel residuals, doc-based LoRA TTT.
- [PR #1344](https://github.com/openai/parameter-golf/pull/1344) — Polar-Express Newton-Schulz coefficients + depth recurrence.
- [PR #1626](https://github.com/openai/parameter-golf/pull/1626) by **@dexhunter** — Multi-phase global SGD phased-TTT.
- [PR #1610](https://github.com/openai/parameter-golf/pull/1610) — VarLenAttn + originator of phased TTT.

V21's only original contribution is integrating the asymmetric softcap (PR #1923) on top of PR #1908's quantization stack. The empirical observation that this combination is **net positive** (despite PR #1923's standalone result being negative on PR #1855 base) is the novel finding presented here.

This PR follows the contribution norm established by cocohearts on 2026-04-28: incremental wins on the leading chain are accepted via the p<0.25 statistical-significance bar (Welch one-sided t-test). V21 vs PR #1908: **t≈2.18, p≈0.045 (one-sided)** — well below the 0.25 threshold.
