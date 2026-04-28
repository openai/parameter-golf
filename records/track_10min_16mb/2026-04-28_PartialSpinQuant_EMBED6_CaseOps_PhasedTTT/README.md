# Record: Partial SpinQuant (start_layer=5) + EMBED_BITS=6 + PR#1855 Hparams + PR#1851 Base

**val_bpb = 1.06614** (3-seed mean, std 0.00131) | **~15.63 MB** | 8×H100 SXM

## 3-Seed Results

| Seed | Pre-quant BPB | Post-GPTQ BPB | **TTT BPB** | Artifact | Eval time |
|------|--------------|---------------|-------------|----------|-----------|
| 42   | —            | —             | **1.06484** | 15,627,137 | 500.4s |
| 2024 | 1.06747      | 1.07929       | **1.06611** | 15,623,946 | 493.8s |
| 1337 | 1.06758      | 1.08050       | **1.06746** | 15,626,137 | 492.5s |
| **Mean** | | | **1.06614** | **15,625,740** | |
| **Std**  | | | **0.00131** | | |

Merged SOTA (PR #1413 @dexhunter): **1.0810**. Delta: **−0.01486 BPB**.
Previous self-PR #1695: **1.07590**. Delta: **−0.00976 BPB**.

## Key Techniques

All techniques below are from prior community PRs. The single new contribution in this PR is item 1.

1. **Partial SpinQuant (`SPINQUANT_START_LAYER=5`)** ← *new in this PR* — Hadamard pre-rotation applied to layers 5–10 only (6/11 layers, 12 weight modules). Full SpinQuant rotates all 66 modules adding ~1MB brotli entropy overhead; partial rotation reduces this to ~200KB, making EMBED_BITS=6 viable within the 16MB cap. Zero serialized bytes — rotation matrix is regenerated from seed at eval. Code: `install_spinquant_rotations(..., start_layer=5)` skips `layer_idx < start_layer`. (@X-Abhishek-X, this PR, building on PR #1695)

2. **PR#1851 base** — SmearGate BOS-token fix + LQER Asymmetric (rank-4) + 3-phase Phased TTT. (@aquariouseworkman, PR #1851)

3. **CaseOps SP8192 tokenizer** — case-preserving sentencepiece tokenizer, 8192 vocab. (@romeerp, PR #1729)

4. **SparseAttnGate + PolarNS + MIN_LR** — sparse attention gating, polar Newton-Schulz optimizer, minimum LR floor. (@nprime06, PR #1787)

5. **SmearGate + LQER Asymmetric** — gated residual smearing, low-rank quantization error reduction with asymmetric init. (@dexhunter, PR #1797; BOS audit @cocohearts)

6. **3-Phase Phased TTT** — post-quantization test-time training in 3 phases over 50k docs (2500 prefix + 47500 suffix). Score-first ordering, LoRA rank 80. (@abaybektursun, PR #549)

7. **GPTQ + SDClip** — full-Hessian GPTQ int6 quantization with sigma-based weight clipping. (@clarkkev, PR #1394)

8. **PR#1855 hparam greedy** — 9 env-var-only overrides validated by community at 1.06108 3-seed: `MLP_CLIP_SIGMAS=11.5`, `EMBED_CLIP_SIGMAS=14.0`, `WARMDOWN_FRAC=0.85`, `BETA2=0.99`, `TTT_BETA2=0.99`, `TTT_WEIGHT_DECAY=0.5`, `TTT_LORA_RANK=80`, `SPARSE_ATTN_GATE_SCALE=0.5`, `PHASED_TTT_PREFIX_DOCS=2500`. (PR #1855 authors)

## Training Config

```
Hardware:        8xH100 80GB SXM
PyTorch:         2.9.1+cu128
Steps:           ~4860–4876 (wall-clock cap ~596s)
SPINQUANT_ENABLED=1      SPINQUANT_SEED=20260416   SPINQUANT_START_LAYER=5
EMBED_BITS=6
CASEOPS_ENABLED=1        SPARSE_ATTN_GATE_ENABLED=1
SMEAR_GATE_ENABLED=1     LQER_ENABLED=1            LQER_ASYM_ENABLED=1
MIN_LR=0.1               PHASED_TTT_NUM_PHASES=3
MLP_CLIP_SIGMAS=11.5     EMBED_CLIP_SIGMAS=14.0    WARMDOWN_FRAC=0.85
BETA2=0.99               TTT_BETA2=0.99            TTT_WEIGHT_DECAY=0.5
TTT_LORA_RANK=80         SPARSE_ATTN_GATE_SCALE=0.5  PHASED_TTT_PREFIX_DOCS=2500
```

## Reproduction

```bash
pip install python-minifier brotli sentencepiece

# Download CaseOps dataset (~16GB)
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('romeerp/parameter-golf-caseops-v1', repo_type='dataset', local_dir='/workspace/parameter-golf/data/datasets')
"

SPINQUANT_ENABLED=1 SPINQUANT_SEED=20260416 SPINQUANT_START_LAYER=5 \
EMBED_BITS=6 CASEOPS_ENABLED=1 SPARSE_ATTN_GATE_ENABLED=1 \
SMEAR_GATE_ENABLED=1 LQER_ENABLED=1 LQER_ASYM_ENABLED=1 \
MIN_LR=0.1 PHASED_TTT_NUM_PHASES=3 \
MLP_CLIP_SIGMAS=11.5 EMBED_CLIP_SIGMAS=14.0 WARMDOWN_FRAC=0.85 \
BETA2=0.99 TTT_BETA2=0.99 TTT_WEIGHT_DECAY=0.5 \
TTT_LORA_RANK=80 SPARSE_ATTN_GATE_SCALE=0.5 PHASED_TTT_PREFIX_DOCS=2500 \
SEED=42 DATA_DIR=/workspace/parameter-golf/data \
torchrun --nproc_per_node=8 train_gpt.py
```

## Compliance

Per competition rules (track_10min_16mb):

- **Training under 600s:** ✅ All seeds stopped at wall-clock cap (~596s, ~4860–4876 steps)
- **Artifact under 16,000,000 bytes:** ✅ All seeds ~15.63MB (374KB headroom)
- **Eval under 600s:** ✅ Seeds 492–500s
- **No pre-quant TTT:** ✅ TTT runs post-quantization only
- **Score-first TTT:** ✅ Phased TTT scores before updating
- **No SLOT / no ETLB / no n-gram cache:** ✅
- **3 seeds:** ✅ Seeds 1337, 42, 2024

## Credits

- **@aquariouseworkman** — PR#1851 base: SmearGate BOS fix, LQER Asymmetric, 3-phase Phased TTT
- **@romeerp** — CaseOps SP8192 tokenizer (PR #1729)
- **@nprime06** — SparseAttnGate, PolarNS, MIN_LR (PR #1787)
- **@dexhunter** — SmearGate + LQER Asymmetric implementation (PR #1797)
- **@cocohearts** — SmearGate BOS-token audit (PR #1797)
- **@abaybektursun** — Phased TTT framework (PR #549)
- **@clarkkev** — GPTQ + SDClip quantization (PR #1394)
- **PR #1855 authors** — hparam greedy search (9 overrides)
- **@X-Abhishek-X** — Partial SpinQuant `SPINQUANT_START_LAYER` (this PR, built on PR #1695)
