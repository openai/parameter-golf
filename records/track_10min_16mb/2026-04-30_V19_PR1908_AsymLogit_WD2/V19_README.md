# V19: PR #1908 + Asymmetric Logit Rescale + TTT_WD=2.0 Fix

## Strategy

Stack two independent legal improvements on top of the verified frontier PR #1908
(romeerp, val_bpb 1.06081 3-seed mean):

1. **Asymmetric Logit Rescale** (PR #1923, jorge-asenjo) — replace the single
   `logit_softcap` scalar with two learnable scalars (`softcap_pos`, `softcap_neg`)
   on the eval path. Mechanism is orthogonal to AWQ-lite (operates on logit head,
   not weights); could net additive.
2. **TTT_WEIGHT_DECAY = 2.0 default** (PR #1886, renqianluo + sunnypatneedi
   research log 2026-04-28) — fixes fused-CE + warm-start LoRA-A seed-collapse on
   seeds 314/1337. PR #1908 ships with WD=1.0 which is borderline.

## Stack

| Component | Source | Version |
|---|---|---|
| Base architecture (SP8192, 11L, ParResid, varlen attn) | PR #1855 codemath3000 | inherited |
| AWQ-lite mixed-precision GPTQ | PR #1908 romeerp | inherited |
| LQER asym int4 + rank-4 correction | PR #1797 dexhunter | inherited |
| Sparse Attn Gate (BOS-fixed SmearGate) | PR #1855 / cocohearts audit | inherited |
| Phased TTT (PREFIX_DOCS=2500) | PR #1797 / PR #1855 | inherited |
| **Asymmetric Logit Rescale** | **PR #1923 jorge-asenjo** | **NEW vs PR #1908** |
| **TTT_WEIGHT_DECAY = 2.0** | **PR #1886 / sunnypatneedi research** | **NEW default** |

## Code changes vs PR #1908

Five edits to `train_gpt.py` only. Total +26 lines.

1. Line ~299 — change TTT_WEIGHT_DECAY default 1.0 → 2.0
2. Line ~1259-1270 — add `asym_logit_enabled`, `softcap_pos`, `softcap_neg` in `GPT.__init__`
3. Line ~1419-1426 — add `_apply_asym_softcap` helper method
4. Line ~1431-1432 — add `if self.asym_logit_enabled` branch in `forward_logits`
5. Line ~1533-1534 — add `if self.asym_logit_enabled` branch in `forward_ttt`

Train path (training-time `forward()` + fused softcapped CE) is **unchanged** to
preserve PR #1855 train numerics. Asymmetric softcap only kicks in on eval path
(`forward_logits` + `forward_ttt`).

## Compliance (Issue #1017 Track A)

- [x] **Causality**: VarLen + per-doc cu_seqlens, strict causal mask (inherited)
- [x] **Normalized softmax**: full SP8192 vocab on eval (inherited)
- [x] **Score-before-update**: Phased TTT structure unchanged (inherited)
- [x] **Single pass**: each val token scored exactly once (inherited)
- [x] **No SLOT, no pre-quant TTT, no n-gram cache, no ETLB**
- [x] **Asymmetric softcap is bounded post-projection nonlinearity**: identical
      semantics to vanilla softcap with separate +/- branches; still feeds normal
      softmax. PR #1923 self-cert as Track A clean, no rebuttal as of 2026-04-29.
- [x] **TTT_WD=2.0 is a stability hyperparameter**, no algorithmic change.

## Expected Result

| Metric | PR #1855 (base) | PR #1908 (frontier) | V19 estimate |
|---|---:|---:|---:|
| Sliding val_bpb | 1.06108 | 1.06081 | **1.057 - 1.060** |
| vs PR #1908 frontier | +0.00027 | — | **-0.001 to -0.004** |
| vs merged SOTA bigbag (1.0810) | -0.020 | -0.020 | **-0.021 to -0.024** |
| Record threshold (1.0738) | BREAK -0.013 | BREAK -0.013 | BREAK -0.014 to -0.017 |

## Reproduction

```bash
# 1. Clone alertcat fork
cd /workspace
rm -rf parameter-golf
git clone https://github.com/alertcat/parameter-golf.git
cd parameter-golf
git checkout v19-frontier

# 2. Install deps (inherits PR #1908 / PR #1855 setup)
pip install torch==2.9.1+cu128 sentencepiece brotli huggingface_hub numpy python-minifier
pip install --no-deps flash_attn_3 --find-links \
  https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/

# 3. Dataset (already cached for V18 — reuse)
ls /workspace/caseops_data/datasets/datasets/fineweb10B_sp8192/

# 4. Run V19 scout (single seed 42, ~12 min train + ~7 min eval)
cd records/track_10min_16mb/2026-04-30_V19_PR1908_AsymLogit_WD2/
bash run_v19_scout.sh

# 5. If scout val_bpb < 0.9760 (vs baseline 0.97651 on CaseOps val) → 3-seed
bash run_v19_3seeds.sh
```

## Decision rule

Compare V19 scout `quantized_ttt_phased val_bpb` against baseline 0.97651 (the
known PR #1908 default baseline on CaseOps val from 2026-04-29 measurement):

| V19 scout result | Real Δ vs baseline | Action |
|---|---|---|
| < 0.9755 (Δ < -0.001) | true win | go 3-seed |
| 0.9755 - 0.9770 | within noise | abandon, try Lead B |
| > 0.9770 | regression | rollback |

## Attribution

- @romeerp (PR #1908 — AWQ-lite mixed-precision GPTQ, base for V19)
- @codemath3000 (PR #1855 — base architecture, 9-hparam stack)
- @jorge-asenjo (PR #1923 — Asymmetric Logit Rescale)
- @renqianluo (PR #1886 — TTT_WD=2.0 fused-CE collapse fix)
- @sunnypatneedi (research log 2026-04-28 — fused-CE + warm-start LoRA-A
  numerical-stability rationale)
- @dexhunter (PR #1797 — LQER Asym int4, SmearGate, Phased TTT)
- @cocohearts (PR #1855 BOS fix audit)

V19 is a stacking experiment combining PR #1923's logit-head delta on top of
PR #1908's quantization stack, with the sunnypatneedi-recommended TTT_WD=2.0
stability default. Train numerics unchanged; eval path adds two learnable
scalars (8 bytes artifact cost).
