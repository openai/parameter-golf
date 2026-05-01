# Independent PR #2014 Prefix-2400 Reproduction Support

This is an independent seed-42 reproduction/support package for
[PR #2014](https://github.com/openai/parameter-golf/pull/2014). It uses the
same PR #2014 SP8192 CaseOps progressive-3k training stack, then reruns the
saved quantized artifact with a slightly smaller phased-TTT prefix budget:

- `PHASED_TTT_PREFIX_DOCS=2400`
- `PHASED_TTT_NUM_PHASES=1`
- `TTT_MASK=no_qv`
- `TTT_LOCAL_LR_MULT=0.75`
- `TTT_SHORT_SCORE_FIRST_STEPS=256:8,2000:24`

This package is intended as support evidence for the late-April PR #2014
frontier line, not as a separate three-seed architecture claim.

## Result

| Seed | Train steps | Train ms | Pre-quant BPB | Post-TTT BPB | TTT eval s | Total artifact bytes |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 4,703 | 596,137 | 1.06066112 | **1.05803761** | 591.4 | 15,989,499 |

The TTT-only eval log reports:

```text
val_tokens: 47853343
ttt_phased: total_docs:50000 prefix_docs:2400 suffix_docs:47600 num_phases:1 boundaries:[2400] target_tokens:47853343
quantized_ttt_phased val_loss:2.31538446 val_bpb:1.05803761 eval_time:591446ms
total_eval_time:591.4s
```

The train/artifact log reports:

```text
stopping_early: wallclock_cap train_time: 596137ms step: 4703/20000
diagnostic pre-quantization post-ema val_loss:2.32112568 val_bpb:1.06066112 eval_time:18334ms
Serialized model quantized+pergroup: 15951365 bytes
Total submission size quantized+pergroup: 15989499 bytes
```

Relative to the merged PR #1855 leaderboard record metadata used by PR #2014
(`1.06107587` BPB, `2.32202732` nats), this single-seed reproduction is:

- `-0.00303826` BPB
- `-0.00664286` nats

Because this is only one seed, it should be read as reproduction/support
evidence rather than standalone statistical proof.

## What Changed vs PR #2014

The training stack and code are the PR #2014 package. The local reproduction
artifact trained to step 4,703 on an 8xH100 SXM RunPod node under the 600s cap.
The primary eval rerun changed only the phased-TTT prefix budget from the PR
#2014 default `2500` to `2400`, keeping the eval under the 600s cap on this
machine while preserving full validation target coverage.

The same local artifact was also tested with a 2500-doc prefix and reached a
very similar BPB, but that local eval pass took about 603s. The 2400-doc eval
is the compliant support run included here.

## Compliance Notes

- **Artifact cap:** `15,989,499` total bytes, including compressed code.
- **Training wallclock:** `596.137s`.
- **Eval wallclock:** `591.446s` for the validation-data TTT timer.
- **Full validation targets:** `val_tokens == target_tokens == 47,853,343`.
- **Score-before-update:** uses the PR #2014 quantized phased LoRA TTT path;
  each chunk is scored before that chunk's update is applied.
- **No validation data in training:** training uses only the CaseOps SP8192
  training shards. The TTT pass is left-to-right over validation docs.
- **Original-byte BPB:** preserves the CaseOps byte-sidecar accounting.

## Reproduction

Install the dependencies in `requirements.txt`. FlashAttention 3 and `lrzip`
are noted there because they use separate install paths.

Prepare or reuse the same CaseOps SP8192 data used by PR #1855 and PR #2014,
then train seed 42 with the PR #2014 command. The included
`train_seed42_repro.log` contains the full command-time hyperparameter dump and
the source code snapshot.

After the compressed artifact is written, run an eval-only pass with the saved
artifact and `PHASED_TTT_PREFIX_DOCS=2400`:

```bash
NCCL_NET=Socket \
DATA_DIR=./data \
DATA_PATH=./data/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved \
TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model \
CASEOPS_ENABLED=1 \
VOCAB_SIZE=8192 \
EVAL_INCLUDE_TAIL=1 \
TRAIN_SEQ_LEN=3072 \
ROPE_TRAIN_SEQ_LEN=3072 \
EVAL_SEQ_LEN=3072 \
EVAL_STRIDE=1536 \
TTT_ENABLED=1 \
TTT_EVAL_ONLY=1 \
LOAD_ARTIFACT_PATH=./final_model.int6.ptz \
TTT_EVAL_SEQ_LEN=3072 \
TTT_BATCH_SIZE=24 \
TTT_CHUNK_SIZE=48 \
TTT_SHORT_SCORE_FIRST_ENABLED=1 \
TTT_SHORT_DOC_LEN=2000 \
TTT_SHORT_CHUNK_SIZE=24 \
TTT_SHORT_SCORE_FIRST_STEPS=256:8,2000:24 \
TTT_LORA_RANK=80 \
TTT_LORA_LR=0.0001 \
TTT_LOCAL_LR_MULT=0.75 \
TTT_MASK=no_qv \
TTT_Q_LORA=0 \
TTT_V_LORA=0 \
TTT_WEIGHT_DECAY=0.5 \
TTT_BETA2=0.99 \
PHASED_TTT_PREFIX_DOCS=2400 \
PHASED_TTT_NUM_PHASES=1 \
QK_GAIN_INIT=5.25 \
SPARSE_ATTN_GATE_ENABLED=1 \
SPARSE_ATTN_GATE_SCALE=0.5 \
GATED_ATTN_QUANT_GATE=1 \
SMEAR_GATE_ENABLED=1 \
GATE_WINDOW=12 \
FUSED_CE_ENABLED=1 \
EMBED_BITS=7 \
LQER_ENABLED=1 \
LQER_RANK=4 \
LQER_TOP_K=3 \
LQER_FACTOR_BITS=4 \
LQER_ASYM_ENABLED=1 \
LQER_ASYM_GROUP=64 \
AWQ_LITE_ENABLED=1 \
AWQ_LITE_BITS=8 \
AWQ_LITE_GROUP_TOP_K=1 \
AWQ_LITE_GROUP_SIZE=64 \
ASYM_LOGIT_RESCALE=1 \
COMPRESSOR=pergroup \
VAL_LOSS_EVERY=0 \
SEED=42 \
torchrun --standalone --nproc_per_node=8 train_gpt.py \
    > eval_seed42_prefix2400.log 2>&1
```

## Included Files

- `train_gpt.py` - PR #2014 training/eval script used for this reproduction.
- `train_seed42_repro.log` - seed-42 training and artifact creation log.
- `eval_seed42_prefix2400.log` - TTT-only prefix-2400 eval log.
- `submission.json` - structured metadata for this support package.
- `requirements.txt` - Python dependencies plus FA3/lrzip notes.
- `prepare_caseops_data.py`, `lossless_caps.py`, and `tokenizers/` - CaseOps
  data preparation and tokenizer files from the PR #2014 package.

## Lineage and Credits

This is a reproduction/support package for PR #2014 by @simonbissonnette.
The stack builds on the public CaseOps/SP8192/LQER/SparseAttnGate/BOS-fixed
SmearGate lineage credited in PR #2014, including PR #1855, PR #1953, PR #1797,
PR #1787, PR #1736, PR #1729, PR #1667, PR #1626, PR #1610, and Issue #1017.
