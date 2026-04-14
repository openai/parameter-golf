# PR 1418 — Autoresearch-Guided Hyperparameter Optimization (100+ Experiments, Negative Results)

**Author:** taehwan
**Claimed BPB:** null / not stated (non-record submission)
**Artifact size:** not stated
**Seeds:** not stated

## Files retrieved
- `records__track_10min_16mb__2026-04-06_MuonEqR_SLOT_QKGain3__README.md`
- `records__track_10min_16mb__2026-04-06_MuonEqR_SLOT_QKGain3__train_gpt.py`
- `records__track_10min_16mb__2026-04-06_MuonEqR_SLOT_QKGain3__submission.json`

## Environment variables (from run command)
SEED=42, QK_GAIN_INIT=3.0

## Claimed changes (from README, verbatim)
> Systematic hyperparameter exploration via autoresearch methodology: 100+ automated experiments (80+ on Mac, 10 on H100) to validate training improvements, combined with SLOT logit-bias test-time adaptation and MuonEq-R optimizer. Non-record submission: Documenting findings and negative results.
>
> What Works (validated on H100): QK-Gain=3.0 (-0.002 BPB vs default 1.5).
>
> What Doesn't Work: QK-Gain=5.0 (+0.035 BPB worse), Matrix LR=0.03 or 0.05 (worse than 0.04), Muon momentum=0.95 (worse than 0.99), Warmdown=4000 (much worse than 3500), Batch 1M tokens (worse), MuonEq-R with Parallel Muon (~40% regression due to row-normalization interacting badly with banked/sharded weights), Int4 quantization (25x worse MSE than int6), Hadamard rotation (only 1.16x improvement), init_std=0.02 (helps AdamW, hurts Muon).
>
> Novel Techniques Explored (negative results):
> 1. HadaGPTQ: Hadamard rotation before GPTQ. Only 1.16x MSE improvement.
> 2. Int4 GPTQ + larger model: QAT at int4 causes +27.7% val loss degradation.
> 3. MuonEq-R with Parallel Muon: row-normalizing momentum in the banked optimizer causes training regression when weights are sharded across GPUs.
