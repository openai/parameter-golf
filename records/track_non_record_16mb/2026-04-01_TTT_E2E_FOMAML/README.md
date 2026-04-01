# TTT-E2E: Meta-Learned Test-Time Training (FOMAML)

Non-record submission exploring end-to-end meta-learned test-time training
(TTT-E2E, arxiv 2512.23675) adapted for the Parameter Golf setting.

## Motivation

SLOT showed -0.0037 BPB of test-time adaptation headroom on our PR 1105 model
(1.1125 -> 1.1088) but violates causality. All 25 prior naive TTT attempts in
Parameter Golf failed because the model was never trained to be adapted.

TTT-E2E is the legal mechanism to capture this headroom: meta-learn the model so
it *expects* gradient-based adaptation at eval time, using score-first evaluation
to satisfy causality (Condition 3).

## Method

**Architecture:** Add small "prime MLPs" (rank-256, LeakyReLU(0.5)^2) to the last
3 transformer blocks (layers 8-10). Each prime MLP has its own RMSNorm and runs
sequentially *before* the main MLP:

```
h = h + attn(norm(h))
h = h + prime_MLP(prime_norm(h))   # adapted at test time
h = h + MLP(mlp_norm(h))           # frozen at test time
```

Prime MLP params: 786K (3 layers x 262K). Down projections zero-initialized so
model starts identical to baseline.

**Two-phase training:**
- Phase 1: Standard pretraining (prime MLPs zero-init, no effect)
- Phase 2: FOMAML meta-fine-tuning (base model frozen, only prime MLPs trained)

**FOMAML inner loop (Phase 2):**
1. Clone prime weights (detached)
2. K=1 inner SGD step on first half of mini-batch
3. Outer loss on second half with adapted weights
4. `retain_grad()` on adapted tensors; copy gradients back to prime init params
5. AdamW update on prime init

**Eval-time TTT (score-first, legal):**
1. Forward pass on chunk -> score (record BPB) 
2. Compute CE loss from same forward pass
3. Backward through prime MLPs only -> SGD update
4. Next chunk (score locked before any adaptation that could use it)

## Key Implementation Detail

PR 1105 uses parameter banking (3D contiguous tensors for batched Muon). Prime MLP
weights are stored separately as regular parameters — they're small, need independent
gradient handling for FOMAML, and are the only weights updated during eval.

FA3/CUTLASS EVT not available on L40S dev machine — used FA2 + unfused MLP fallback.

## Results (1x L40S, 1 train shard — dev run)

| Stage | val_bpb | Notes |
|-------|---------|-------|
| Phase 1 baseline (7200 steps) | 1.3872 | Standard training, no prime MLPs |
| Post-Phase2 FOMAML (no TTT) | 1.5465 | Expected: W_0 optimized for adaptation, not standalone |
| **TTT eval (score-first)** | **1.4707** | Recovers 0.076 BPB from Phase 2 degradation |

**TTT adaptation works** — it recovers most of the Phase 2 degradation — but doesn't
beat the Phase 1 baseline on this limited setup. Root causes:

1. **1 train shard** (5% of data): meta-learning needs diversity to learn "how to adapt"
2. **Weak base model** (1.387 vs 1.1125 BPB): less headroom for adaptation
3. **Prime-only Phase 2**: freezing base model means the base can't co-adapt with prime MLPs

## What Would Be Different on 8xH100

- Full dataset (20 shards) for Phase 2 meta-learning diversity
- Phase 1: ~4500 steps at 88ms/step, Phase 2: ~1500 steps at ~175ms/step
- Joint base+prime training in Phase 2 (base at 0.1x LR)
- torch.compile + fused kernels for ~2x throughput
- Expected: TTT should provide -0.001 to -0.003 BPB over baseline

## Config

```
PRIME_RANK=256  PRIME_LAYERS=8,9,10
PHASE2_STEPS=1500  PHASE2_OUTER_LR=0.003  PHASE2_INNER_LR=0.01
TTT_LR=0.01  TTT_CHUNK=1024  SEQ_LEN=2048
```

## Running

```bash
# Requires: Phase 1 checkpoint as final_model.pt
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
CHECKPOINT=final_model.pt \
PHASE2_STEPS=1500 \
python -u train_ttt_e2e.py
```

## References

- TTT-E2E paper: arxiv.org/abs/2512.23675
- FOMAML: arxiv.org/abs/1803.02999
- Our PR 1105 (base model): 1.1125 BPB, 3-seed mean
