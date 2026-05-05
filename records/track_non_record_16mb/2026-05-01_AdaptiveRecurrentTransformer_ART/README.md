# Adaptive Recurrent Transformer (ART)

*Not every token deserves equal compute.*

Standard language models allocate depth uniformly. Yet intuitively, some tokens are easier to predict than others. Take the following fragment:

“After lunch, he was looking forward…”

As humans, we easily recognize the next word to likely be “to”. Accordingly, I approached this challenge with a singular thesis: not every token deserves equal compute – and dynamic recurrence can provide this adaptability.

## Summary

ART adds routing MLPs which determine when/how to recurse within transformer architectures.

This is a non-record research submission for `records/track_non_record_16mb/`. It packages the Adaptive Recurrent Transformer line of experiments, and the strongest recent ART edits on other Parameter Golf stacks.

The core example is **PR1855-family + Simple ART**, which scored **1.19991555 BPB** after quantization and post-quant TTT on **1xH100**. That run is intentionally included as a research result rather than a leaderboard claim: its artifact was over the 16 MB cap and its full TTT evaluation exceeded the strict leaderboard runtime. The point of this PR is to demonstrate the architectural idea of ART.

Note that although PR1855 + Simple ART is the upfront example, it is not the most interesting. The purpose of this PR, rather, is to demonstrate the architectural idea of Adaptive Recurrent Transformers and the variety of ART techniques – changing attention heads and MoE blocks, multiple-block and single-block recursion, etc.


## Selected Results

| Variation | Best BPB | Hardware |
|---|---:|---|
| PR1855 + Soft ART | 1.0643 | 8xH100 |
| PR1812 + ART | 1.1183 | 8xH100 |
| PR1855 + Simple ART | 1.1999 | 1xH100 |
| Triple-ART Small Batch | 1.2410 | 1xH100 |
| ART Saved-RoPE + TTT | 1.2413 | 1xH100 |
| Adaptive Recurrent Transformer 1 | 1.3168 | 1xH100 |
| ART-2 | 1.3207 | 1xH100 |
| ART-3 | 1.4849 | 1xH100 |
| ART-2.5 | 1.5285 | 1xH100 |

## Included Files

| File | Purpose |
|---|---|
| `train_gpt.py` | Canonical runnable Simple ART script for the 1.1999 BPB core example. |
| `training_files/pr1855_simple_art.py` | Same Simple ART script, kept with the selected evidence bundle. |
| `training_files/pr1855_soft_art.py` | Soft ART / delta-gated recurrence variant that reached 1.0643 BPB on 8xH100. |
| `training_files/pr1812_art.py` | PR1812 + ART 8xH100 attempt that reached 1.1183 BPB. |
| `training_files/triple_art_small_batch.py` | Triple-ART small-batch 1xH100 pre-TTT launcher. Uses `training_files/run_variant.py` and `training_files/train_gpt_base_prettt.py`. |
| `training_files/run_variant.py` | Helper for the Triple-ART pre-TTT suite variants. |
| `training_files/train_gpt_base_prettt.py` | Base Triple-ART pre-TTT script used by the small-batch launcher. |
| `training_files/art_saved_rope_ttt.py` | Triple-ART saved-RoPE fix with post-quant TTT. |
| `training_files/art_1.py` | First standalone Adaptive Recurrent Transformer implementation. |
| `training_files/art_2.py` | ART-2 record-style training implementation. |
| `training_files/art_3.py` | ART-3 sparse-head / MoE-router implementation. |
| `training_files/art_2_5.py` | ART-2.5 outer-router shared-MoE implementation. |
| `logs/` | One log for each row in the selected-results table. |

## Evidence Logs

| Variation | Script | Log |
|---|---|---|
| PR1855 + Soft ART | `training_files/pr1855_soft_art.py` | `logs/pr1855_soft_art_dashboard_20260430_204203_abf0ec8e.stdout.log` |
| PR1812 + ART | `training_files/pr1812_art.py` | `logs/pr1812_art_dashboard_20260429_224456_a9eaabed.stdout.log` |
| PR1855 + Simple ART | `training_files/pr1855_simple_art.py` | `logs/pr1855_simple_art_dashboard_20260430_224429_ad4a36d5.stdout.log` |
| Triple-ART Small Batch | `training_files/triple_art_small_batch.py` | `logs/triple_art_small_batch_dashboard_20260429_192817_de3b33e8.stdout.log` |
| ART Saved-RoPE + TTT | `training_files/art_saved_rope_ttt.py` | `logs/art_saved_rope_ttt_dashboard_20260429_162729_57cac6b0.stdout.log` |
| Adaptive Recurrent Transformer 1 | `training_files/art_1.py` | `logs/art_1_dashboard_20260417_013525_a3d8aea8.stdout.log` |
| ART-2 | `training_files/art_2.py` | `logs/art_2_dashboard_20260419_161439_9b946387.stdout.log` |
| ART-3 | `training_files/art_3.py` | `logs/art_3_dashboard_20260419_225908_cf9443b3.stdout.log` |
| ART-2.5 | `training_files/art_2_5.py` | `logs/art_2_5_dashboard_20260420_004715_e49b5f70.stdout.log` |

## Simple ART Mechanism

The submitted `train_gpt.py` keeps the proven record-family transformer mostly unchanged. After the normal recurrence phase begins and a delayed ART enable point is reached, a tiny router samples one whole-batch action:

```text
short: skip the two additional recurrent 3-4-5 cycles
full:  run the original recurrent path with the additional six blocks
```

Both branches are compiled separately for training, logits, and TTT. In distributed runs, rank 0 samples the branch and broadcasts it so all GPUs execute the same static path. This is less expressive than token-level ART, but it avoids the packed-active dynamic routing overhead that made earlier ART variants too slow.

## Reproducing The Core Example

The exact data paths depend on the local Parameter Golf setup. This is the shape of the run used for the Simple ART 1xH100 result:

```bash
DATA_DIR=./data \
VOCAB_SIZE=8192 \
DATA_PATH=./data/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved \
TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model \
CASEOPS_ENABLED=1 \
ITERATIONS=20000 MAX_WALLCLOCK_SECONDS=600 \
ART_SIMPLE_ENABLED=1 ART_SIMPLE_ENABLE_AT=0.45 ART_EVAL_MODE=sample \
ART_ROUTER_ENTROPY_START=0.02 ART_ROUTER_ENTROPY_END=0.002 ART_CYCLE_PENALTY=0.002 \
EMA_ACTIVATE_AT=0.55 \
PHASED_TTT_ENABLED=1 PHASED_TTT_PREFIX_DOCS=2500 PHASED_TTT_NUM_PHASES=3 \
EMBED_BITS=7 MATRIX_LR=0.026 MIN_LR=0.1 \
MLP_CLIP_SIGMAS=11.5 ATTN_CLIP_SIGMAS=13.0 EMBED_CLIP_SIGMAS=14.0 \
GRAD_CLIP_NORM=0.3 TTT_CHUNK_SIZE=48 WARMUP_STEPS=20 MUON_BACKEND_STEPS=5 \
GLOBAL_TTT_MOMENTUM=0.9 WARMDOWN_FRAC=0.85 BETA2=0.99 \
TTT_BETA2=0.99 TTT_WEIGHT_DECAY=0.5 TTT_LORA_RANK=80 \
SPARSE_ATTN_GATE_SCALE=0.5 \
GPTQ_RESERVE_SECONDS=8.0 GPTQ_CALIBRATION_BATCHES=16 VAL_LOSS_EVERY=0 \
GATED_ATTN_QUANT_GATE=1 SPARSE_ATTN_GATE_ENABLED=1 GATE_WINDOW=12 \
SMEAR_GATE_ENABLED=1 \
LQER_ENABLED=1 LQER_ASYM_ENABLED=1 LQER_RANK=4 LQER_FACTOR_BITS=4 LQER_ASYM_GROUP=64 LQER_TOP_K=3 \
FUSED_CE_ENABLED=1 COMPRESSOR=pergroup NCCL_NET=Socket \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

## Non-Record Notes

- This PR is an architecture-research package.
- The included scripts do not use network calls during scoring.
- TTT variants use score-first TTT.

## Acknowledgments

The 1.1999 Simple ART script builds on the current SmearGateBOSFix / LQER / CaseOps / phased-TTT record-family stack documented locally at `records/track_10min_16mb/2026-04-29_SmearGateBOSFix_3Seed_1.06141/README.md`. That README identifies the stack as support for PR #1851 by @aquariouseworkman, with lineage from PR #1787 by @nprime06, PR #1797 by @dexhunter, PR #1729 by @romeerp, PR #1394 by @clarkkev, PR #549 by @abaybektursun, and the BOS bug identification by @cocohearts.

The ART mechanisms in this folder are the new contribution: learned recurrent-depth control, soft recurrent-delta weighting, static-graph Simple ART branch selection, etc.
