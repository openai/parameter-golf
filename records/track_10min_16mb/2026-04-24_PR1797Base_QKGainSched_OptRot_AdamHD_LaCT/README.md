# Record attempt: PR #1797 base + QK-Gain Schedule + OptRot + AdamHD + LaCT

**val_bpb: TBD** (3-seed mean, std TBD) | **~TBD MB** | 8×H100 SXM

*Placeholder — to be filled after 3-seed 8×H100 validation run.*

Win bar: ≤1.0566 BPB (frontier #1797 at 1.06157 − 0.005).

## Mechanism stack

| Layer | Mechanism | Env var | Default |
|---|---|---|---|
| 0 (base) | PR #1797 full stack (Smear gate + LQER asym + Phased TTT on #1787 base) | — | — |
| 1 | Per-layer QK-Gain init schedule | `QK_GAIN_INIT_SCHEDULE` | off (uniform 5.0) |
| 2 | OptRot Hadamard pre-GPTQ rotation | `OPTROT_ENABLED=1` | off |
| 3 | AdamHD Huber weight decay (Muon) | `MUON_HUBER_WD=1` | off |
| 4 | LaCT — Muon as global TTT optimizer | `GLOBAL_TTT_OPTIMIZER=muon` | sgd |

All levers are off-by-default and additive — the base PR #1797 behavior is unchanged when no lever env vars are set.

## Layer 1: QK-Gain init schedule

Per-layer initialization of the learnable `q_gain` parameter. The PR #1797 base uses a uniform init of 5.0 across all 11 physical layers. This submission replaces that with a schedule:

```
Layer:   0    1    2    3    4    5    6    7    8    9   10
Init:  2.0  2.5  3.0  3.5  4.0  4.5  4.5  4.0  3.5  3.0  2.5
```

`q_gain` remains trainable — the schedule is an initialization hint, not a constraint. Validated on the #1394 SP8192 base at 1.07060 BPB (3-seed mean, 1×H100).

Env var: `QK_GAIN_INIT_SCHEDULE="2.0,2.5,3.0,3.5,4.0,4.5,4.5,4.0,3.5,3.0,2.5"`

## Layer 2: OptRot (Hadamard pre-GPTQ rotation)

Applies normalized Fast Walsh-Hadamard Transform (FWHT) to paired weight matrices before GPTQ quantization. Pairs: (W_up, W_down) per MLP layer; (W_v, W_o) per attention layer (per kv-head/query-head group). The rotation is orthogonal and self-inverse (R² = I), so model outputs are preserved exactly. GPTQ sees redistributed weight distributions, reducing quantization error.

Technical: for MLP pair with hidden_dim=2048 (power of 2):
- `W_up' = R @ W_up`
- `W_down' = W_down @ R`
- `W_down' @ W_up' = W_down @ R² @ W_up = W_down @ W_up` ✓

Hessian calibration is run on the rotated model to match rotated activations.

Paper: arxiv 2512.24124. Claims 30-50% reduction in int6 quantization error.

Env var: `OPTROT_ENABLED=1`

## Layer 3: AdamHD (Huber weight decay for Muon)

Replaces Muon's L2 weight decay with a Huber regularizer. Gradient of the Huber regularizer L_δ(w):
- `w` if `|w| ≤ δ` (quadratic region — same as L2)
- `δ * sign(w)` if `|w| > δ` (linear region — bounded magnitude)

Bounds the decay rate on large-weight outliers, preventing excessive L2 suppression of weights that are large-but-useful. The tradeoff: outliers that ARE harmful to quantization are suppressed more slowly, but the model can maintain outliers it actually learned to use.

Default δ = 0.1 (`MUON_HUBER_DELTA=0.1`). Applied to all Muon parameter groups (matrix weights).

Paper: arxiv 2511.14721.

Env vars: `MUON_HUBER_WD=1`, `MUON_HUBER_DELTA=0.1`

## Layer 4: LaCT (Muon as global TTT optimizer)

Replaces the SGD global TTT optimizer with Muon (Newton-Schulz orthogonalized gradient updates). Muon is already used for training in this architecture, so its gradient scale is compatible with the learned weight norms. LaCT (arxiv 2505.23884, ICLR 2026 Oral) achieves 70% GPU utilization during TTT vs <5% for per-token TTT by using large document chunks and efficient gradient computation.

Implementation: `GLOBAL_TTT_OPTIMIZER=muon` routes global TTT through a lightweight Muon instance (matrix params) + SGD (scalar/embedding params). LR and momentum from `GLOBAL_TTT_LR`, `GLOBAL_TTT_MOMENTUM`.

Env var: `GLOBAL_TTT_OPTIMIZER=muon`

## Run command

```bash
DATA_DIR=./data \
CASEOPS_ENABLED=1 \
PHASED_TTT_PREFIX_DOCS=2000 PHASED_TTT_NUM_PHASES=3 \
MATRIX_CLIP_SIGMAS=12.85 ATTN_CLIP_SIGMAS=13.0 MLP_CLIP_SIGMAS=12.0 \
EMBED_BITS=7 EMBED_CLIP_SIGMAS=15.0 \
MATRIX_LR=0.026 MIN_LR=0.1 \
FUSED_CE_ENABLED=1 SPARSE_ATTN_GATE_ENABLED=1 \
SMEAR_GATE_ENABLED=1 GATE_WINDOW=12 \
LQER_ENABLED=1 LQER_RANK=4 LQER_TOP_K=3 LQER_FACTOR_BITS=4 \
LQER_ASYM_ENABLED=1 LQER_ASYM_GROUP=64 \
TTT_WARM_START_A=1 \
GPTQ_RESERVE_SECONDS=0.5 GPTQ_CALIBRATION_BATCHES=16 \
QK_GAIN_INIT_SCHEDULE="2.0,2.5,3.0,3.5,4.0,4.5,4.5,4.0,3.5,3.0,2.5" \
OPTROT_ENABLED=1 \
MUON_HUBER_WD=1 MUON_HUBER_DELTA=0.1 \
GLOBAL_TTT_OPTIMIZER=muon \
SEED=314 \
torchrun --standalone --nproc_per_node=8 train_gpt_human.py
```

## Incremental ablation plan (Task 5, 7, 9, 11 from PRD)

Each layer is tested incrementally on 1×H100 (seed 42, grad_accum=8) against the #1797 1×H100 baseline established in Task 3:

| Test | Stack | Expected cumulative gain vs #1797 |
|---|---|---|
| Task 5 | #1797 + Layer 1 | -0.001 to -0.003 |
| Task 7 | #1797 + L1 + L2 | -0.003 to -0.008 |
| Task 9 | #1797 + L1+L2+L3 | -0.005 to -0.013 |
| Task 11 | Full stack | -0.008 to -0.023 |

Abort thresholds per PRD: drop any layer that is neutral or negative on its single-seed test.

## Lineage

- PR #549, #1019, #1394, #1530 — merged base stack
- PR #1729 (@romeerp) — CaseOps bijective transform
- PR #1787 (@nprime06) — SparseAttnGate + PolarNS + MIN_LR + FusedCE
- PR #1797 (@dexhunter) — Smear gate + LQER asymmetric + Phased TTT — **direct base**
- arxiv 2512.24124 — OptRot
- arxiv 2511.14721 — AdamHD
- arxiv 2505.23884 — LaCT

## Files

- `train_gpt_human.py` — human-readable source with all 4 levers (run directly)
- `train_gpt.py` — LZMA+b85 compressed wrapper (to be built with `build_submission.py`)
- `submission.json`
- `README.md`
- `prepare_caseops_data.py` — from PR #1797 (BOS_ID=1 fix applied)
- `lossless_caps.py` — from PR #1797
- `tokenizers/` — from PR #1797
- `train_seed314.log`, `train_seed42.log`, `train_seed1234.log` — to be added after runs
