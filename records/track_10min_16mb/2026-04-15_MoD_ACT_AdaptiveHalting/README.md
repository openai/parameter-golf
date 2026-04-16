# WIP: Relaxed Recursive Transformer + Full SOTA Stack

**Status: Implementation complete — compute grant pending. Ready to run.**

Target: sub-1.076 val_bpb | 8xH100 SXM | ≤16MB artifact

---

## Novel Contribution: RRT-LoRA

Current depth recurrence submissions (PR #1394, #1437, #1493) loop layers 3-5
with **identical shared weights** on every pass. This has two fundamental problems:

1. **No specialization** — every recurrence pass is identical, so the model can't
   learn coarse-to-fine refinement across steps
2. **Quantization error amplification** — int6 quantization error compounds
   multiplicatively across recurrence steps, limiting stable loop depth to ~3

**RRT-LoRA** solves both. Each recurrence step applies a tiny learned delta
(rank=4) to the shared Q and V projections:

```python
Q_step_i = W_q(x) + alpha * lora_B_i(lora_A_i(x))
V_step_i = W_v(x) + alpha * lora_B_v_i(lora_A_v_i(x))
```

Key design decisions:
- **Zero init**: LoRA B matrices initialize to zero — RRT-LoRA is
  identical to the SOTA baseline at step 0, ensuring stable training
- **Alpha warmup**: LoRA contribution ramps from 0→1 over 500 steps
  after recurrence activates at 35% of training — smooth curriculum
- **Parameter cost**: ~4K params per adapter pair × 3 layers × 3 steps
  = ~36K total LoRA params. Negligible against the 16MB artifact budget
- **Quantization stability**: Low-rank deltas are far less sensitive to
  int6 quantization than full weight matrices, reducing error amplification

Reference: Bae et al. "Relaxed Recursive Transformers" (ICLR 2025)

---

## Full SOTA Stack

All proven techniques from the current leaderboard, combined:

| Technique | Source | Expected gain |
|-----------|--------|---------------|
| SP8192 tokenizer | PR #1394 @clarkkev | ~0.12 bpb vs SP1024 |
| GPTQ SDClip int6/int8 | PR #1394 @clarkkev | Fits 16MB budget |
| MuonEq-R optimizer | arXiv:2603.28254 | ~0.005 bpb |
| Exact SOTA recurrence schedule | PR #1493 @bigbag | ~0.03 bpb |
| Parallel residuals (GPT-J) | PR #1412 @Robby955 | ~0.008 bpb + speed |
| QK-Gain 5.25 | PR #1413 @dexhunter | ~0.005 bpb |
| EMA decay 0.9965 | PR #1445 @X-Abhishek-X | ~0.003 bpb |
| Legal score-first TTT | PR #549, #1413 | ~0.018 bpb |
| Post-TTT temp calibration T=0.98 | PR #576 | ~0.003 bpb |
| Bit-packed int6 storage | novel | ~25% artifact space savings |
| Brotli-11 + LZMA compression | PR #1493 | ~43KB savings |
| **RRT-LoRA (novel)** | **this submission** | **~0.005–0.015 bpb** |

---

## Architecture

```
Physical layers:         11
Model dim:               512
Heads / KV heads:        8 / 4
MLP expansion:           4×  (LeakyReLU(0.5)²)
Partial RoPE:            16 / 64 head dims
Tied embeddings:         yes

Recurrence schedule (exact SOTA):
  encoder: [0, 1, 2, 3, 4, 5, 3, 4]       → 8 encoder steps
  decoder: [5, 3, 4, 5, 6, 7, 8, 9, 10]   → 9 decoder steps
  virtual layers: 17  (vs 11 physical)

RRT-LoRA:
  Applied to layers:  {3, 4, 5}
  Rank:               4
  Steps:              3 per layer
  LoRA params:        ~36K total
  Alpha schedule:     0 → 1 over 500 steps (after recurrence activates)

Parallel residuals:  layers 7–10 (GPT-J style)
QK-Gain init:        5.25 (learned per-head)
Logit softcap:       30.0
```

---

## Training

```
Optimizer (matrices):    MuonEq-R  lr=0.022, momentum=0.95, NS5 steps
Optimizer (embed/head):  AdamW     lr=0.05 (tied) / 0.008 (head)
Optimizer (scalars):     AdamW     lr=0.04
Weight decay:            0.095
Grad clip:               1.0
Warmup:                  20 steps
Warmdown:                72% of training (linear to 0)
EMA decay:               0.9965
Batch tokens:            524,288 / step
Max wallclock:           600s (8×H100 SXM)
```

---

## Quantization

```
Weight matrices:    int6 SDClip  k=12.85  (bit-packed: 4 values → 3 bytes)
Embeddings:         int8 SDClip  k=20.0
Compression:        Brotli-11 (model) + LZMA (code)
Estimated artifact: ~15.8 MB
```

---

## Legal Score-First TTT

Compliance per Issue #1017 (Track B):

- **Causality**: Sliding window eval is strictly causal
- **Score-first**: Each chunk fully scored under `torch.inference_mode()`
  BEFORE any SGD update. Graded predictions never see future tokens.
- **Single pass**: Each token scored exactly once — no rescoring
- **Normalized**: Standard softmax only, no logit biasing
- Freeze layers 0–1 during TTT
- Cosine LR decay across 3 epochs per 32K-token chunk
- Gradient clipping at 1.0
- Post-TTT temperature calibration T=0.98 to correct overconfidence
- Model weights restored after TTT eval — does not affect training state

---

## Ablation Design

The implementation includes a built-in ablation flag for the first run:

```bash
# Run 1: Pure depth recurrence baseline (no LoRA)
SEED=42 ABLATE=1 ABLATE_BOTH=1 \
  torchrun --standalone --nproc_per_node=8 train_gpt_rrt_v3.py

# Run 2: Full RRT-LoRA
SEED=42 ABLATE=0 \
  torchrun --standalone --nproc_per_node=8 train_gpt_rrt_v3.py
```

This gives an immediate apples-to-apples comparison of LoRA gain on the
first experiment, without needing separate code paths.

---

## Why This Should Work

The leaderboard has been stuck on naive depth recurrence since PR #1394. Every
submission since has improved quantization, hyperparameters, or TTT — but nobody
has addressed the fundamental limitation that recurrence passes are identical.

RRT-LoRA is architecturally motivated: the first recurrence pass should do
coarse feature extraction, the second should refine, the third should finalize.
Identical weights prevent this specialization. Adding 36K LoRA params (0.4% of
total parameter budget) to enable per-step specialization is arguably the most
parameter-efficient improvement available in the current regime.

The zero-init + alpha warmup design means worst case this is exactly the SOTA
baseline. Best case it's 0.005–0.015 bpb better, which clears the 0.005
threshold for a leaderboard record.

---

## Background

- Solo ML researcher / founder — VectaBind (AI drug discovery platform,
  vectabind.com)
- Built SE(3)-equivariant EGNN + ESM2-3B cross-attention model from scratch,
  trained on BindingDB (3M), PDBBind 2020, CrossDocked2020, TDC, GDSC, CTRP on
  a single NVIDIA L4. Val MAE 0.20 pKd on PDBBind 2020 validation split.
- Adaptive computation background: implemented PonderNet/ACT-style architecture
  for multi-game reasoning (Othello, MiniGrid, drone navigation) with learned
  halting over latent recurrence steps — directly relevant to this submission
- GitHub: ChipGlitch

---

## Progress

- [x] Full leaderboard review: PRs #1394, #1412, #1413, #1437, #1445, #1493
- [x] RRT-LoRA architecture designed and implemented
- [x] Full training script: `train_gpt_rrt_v3.py`
- [x] Exact SOTA recurrence schedule: enc=[0,1,2,3,4,5,3,4] dec=[5,3,4,5,6,7,8,9,10]
- [x] MuonEq-R optimizer with row normalization
- [x] Legal score-first TTT with full Track B compliance
- [x] Post-TTT temperature calibration (T=0.98)
- [x] GPTQ SDClip: bit-packed int6 matrices + int8 embeddings
- [x] Brotli-11 model compression + LZMA code compression
- [x] EMA decay 0.9965
- [x] Three-way optimizer param separation (Muon / AdamW-embed / AdamW-scalar)
- [x] LoRA alpha warmup curriculum
- [x] Built-in ablation flag (ABLATE=1) for baseline comparison
- [ ] H100 validation run (pending compute grant)
- [ ] Ablation: LoRA vs pure recurrence
- [ ] LoRA rank sweep: {2, 4, 8}
- [ ] 3-seed statistical validation (seeds 42, 314, 999)
- [ ] Final submission

---

## Reproduction

```bash
pip install brotli sentencepiece
pip install flash_attn_3 --no-deps \
  --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/

MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf \
  python3 data/cached_challenge_fineweb.py --variant sp8192

# Full RRT-LoRA run
SEED=42 torchrun --standalone --nproc_per_node=8 train_gpt_rrt_v3.py

# Ablation baseline (no LoRA)
SEED=42 ABLATE=1 torchrun --standalone --nproc_per_node=8 train_gpt_rrt_v3.py
```

---

## References

- Bae et al. (ICLR 2025). Relaxed Recursive Transformers
- MuonEq-R: arXiv:2603.28254 (March 2026)
- Raposo et al. (2024). Mixture of Depths: arXiv:2404.02258
- clarkkev — SP8192 + GPTQ SDClip + MuonEq-R + depth recurrence (PR #1394)
- dexhunter — 3-layer depth recurrence + legal TTT (PR #1437, #1413)
- abaybektursun — Score-first TTT framework (PR #549)
- Robby955 — Parallel residuals on SP8192 (PR #1412)
- X-Abhishek-X — Hyperparameter tuning WD=0.095, MLR=0.022, EMA=0.9965 (PR #1445)
- bigbag — 3-layer recurrence + parallel residuals + QK-Gain 5.25 (PR #1493)

---

## Planned Files (final submission)

- `README.md` (this file)
- `submission.json`
- `train_gpt_rrt_v3.py`
- `train_seed42.log`
- `train_seed314.log`
- `train_seed999.log`
