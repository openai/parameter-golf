# Record: CaseOps Tokenizer + Recurrence Depth Curriculum + Base Arch Stack — val_bpb 1.06505

**val_bpb: 1.06505** (3-seed mean, std=0.00081) | **val_loss: 2.33073 nats/token** (std=0.00178) | **~15.98 MB** | 8xH100 SXM | Phased TTT

## Results (8xH100 80GB SXM, PyTorch 2.9.1+cu128, phased TTT, 10-min train / 10-min eval budgets)

### Core table (phased TTT)

| Seed | Steps | Pre-TTT BPB | Post-TTT BPB | TTT gain | TTT time | Artifact (bytes) |
|------|------:|------------:|-------------:|---------:|---------:|-----------------:|
| 0    | 4599  | 1.07689     | 1.06417      | -0.01272 | 470.6s   | 15,984,426       |
| 42   | 4603  | 1.07792     | 1.06521      | -0.01271 | 513.8s   | 15,986,579       |
| 1234 | 4604  | 1.07836     | 1.06578      | -0.01258 | 470.6s   | 15,982,914       |
| **Mean** | **4602** | **1.07772** | **1.06505** | **-0.01267** | **485.0s** | **15,984,640** |
| **Std**  |          | 0.00076     | **0.00081** |          | 24.9s    | 1,842            |

### Supplemental diagnostics

| Seed | Post-EMA BPB (pre-quant) | Quantized BPB (no TTT) | Sliding/TTT BPB | val_loss (nats) | Train time | Eval time |
|------|-------------------------:|-----------------------:|----------------:|----------------:|-----------:|----------:|
| 0    | 1.06779                  | 1.07689                | 1.06417         | 2.32880         | 596.10s    | 470.6s    |
| 42   | 1.06872                  | 1.07792                | 1.06521         | 2.33108         | 596.15s    | 513.8s    |
| 1234 | 1.06934                  | 1.07836                | 1.06578         | 2.33231         | 596.14s    | 470.6s    |

Compared with PR #1736's 3-seed mean of **1.06549**, this curriculum improves the final endpoint by **0.00043 BPB** while staying under both the 600s eval budget and the 16,000,000-byte decimal artifact cap.

## Specific contribution in this record

The core new idea here is **curriculum recurrence depth**.

The base stack already existed:

- SP8192 base architecture / looped stack from earlier merged work
- CaseOps tokenizer + original-byte sidecar accounting from PR #1729
- phased TTT from the prior stack
- gated attention / quant-gate components from earlier work

This record's contribution is to change how recurrence depth is used during training and evaluation.

Instead of training with one fixed recurrent depth after loop activation, this submission uses a **deterministic equal-thirds recurrence curriculum**:

- once the loop path is enabled, train at total recurrence depth `1`
- then switch to total recurrence depth `3`
- then switch to total recurrence depth `4`
- evaluate and run phased TTT at fixed depth `4`

Depth here is counted as the **total number of passes through the recurrent loop block**. So `1` is the shallowest loop-enabled path, `3` is the standard middle-depth path, and `4` is one extra refinement pass at the endpoint.

The intended mechanism is:

- early in the loop-enabled regime, force the recurrent block to learn a useful shallow refinement operator
- then expand to the normal depth so the model keeps strong baseline behavior
- only in the final phase ask the same shared recurrent block to support a deeper refinement chain
- at eval / phased TTT, cash in that extra learned depth by running the model at depth `4`

So the hypothesis is not "train deeper everywhere." It is "teach the recurrent block to scale its refinement depth over training, then evaluate at the deepest trained depth." Empirically, that improves the final phased-TTT endpoint even though one seed (`1234`) is slightly worse than PR #1736; the mean improves because seeds `0` and `42` improve more strongly.

## CaseOps tokenizer and legality

CaseOps (`lossless_caps_caseops_v1`) is a **bijective**, character-level text transform applied before SentencePiece training. It removes English capitalization from the body of the text and records it as four operator tokens that become part of the BPE vocabulary as SentencePiece `user_defined_symbols`:

- `TITLE` — next word is TitleCase
- `ALLCAPS` — next word or region is UPPERCASE
- `CAPNEXT` — next letter is capitalized
- `ESC` — escape for a literal operator-looking sequence

Because the transform is fully invertible, no information is lost. Reconstruction is exact by replaying these capitalization operators over the lowercase lexical stream.

**BPB is still charged on the original raw UTF-8 bytes**, not on the transformed representation. The validation export emits a per-token byte sidecar (`fineweb_val_bytes_XXXXXX.bin`) parallel to the transformed token stream. Eval sums those byte counts for the scored positions, so the denominator remains the original FineWeb byte count.

That means:

- extra CaseOps control tokens are **not free**
- they still contribute prediction loss
- but the BPB denominator stays anchored to the original corpus bytes

So the submission remains legality-preserving: it changes representation, not the underlying text being compressed.

## Rule compliance

- **Artifact <= 16,000,000 bytes DECIMAL**: all 3 seeds <= 15,986,579 bytes.
- **train_time <= 600s**: all 3 seeds are 596.10-596.15s.
- **total_eval_time <= 600s**: all 3 seeds are 470.6-513.8s.
- **Score-first TTT**: phased TTT snapshots the pre-update score on each chunk before the LoRA adapter step.
- **BPB on original bytes**: per-token byte sidecar encodes the canonical UTF-8 byte count of each val position.
- **Reversibility**: `decode_lossless_caps_v2(encode_lossless_caps_v2(x)) == x`.
- **No val data in training**: training uses only `fineweb_train_*.bin` shards.
- **No external network during eval**: self-contained; tokenizer + transform ship with the submission.

## Requirements

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install flash-attn-interface sentencepiece triton numpy
```

Python >= 3.12 is recommended.

Run all commands below from this record directory.

## Data setup (run once)

The submission ships with the trained CaseOps SentencePiece model and the bijective transform module. Train/val shards and the byte sidecar are rebuilt from the canonical FineWeb-10B doc stream:

```bash
# 1. Ensure docs_selected.jsonl exists (standard repo setup step).
python3 ../../data/download_hf_docs_and_tokenize.py

# 2. Build CaseOps-transformed shards + val byte sidecar.
#    This reproduces the original CaseOps export format:
#    one BOS token per doc, and a matching leading 0 byte-count entry.
python3 prepare_caseops_data.py \
    --docs ./fineweb10B_raw/docs_selected.jsonl \
    --out  ./data/datasets/fineweb10B_sp8192_caseops/datasets \
    --sp   ./tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model \
    --val-docs 50000
```

## Run command (3-seed reproduction)

```bash
for SEED in 42 0 1234; do
  NCCL_NET=Socket \
  DATA_DIR=./data \
  CASEOPS_ENABLED=1 \
  PHASED_TTT_ENABLED=1 PHASED_TTT_PREFIX_DOCS=2000 PHASED_TTT_NUM_PHASES=3 \
  MLP_CLIP_SIGMAS=12.0 ATTN_CLIP_SIGMAS=13.0 \
  EMBED_BITS=7 EMBED_CLIP_SIGMAS=15.0 \
  MATRIX_LR=0.026 \
  GPTQ_RESERVE_SECONDS=4 GPTQ_CALIBRATION_BATCHES=16 \
  GATED_ATTN_ENABLED=1 GATED_ATTN_INIT_STD=0.005 GATED_ATTN_QUANT_GATE=1 \
  TRAIN_LOOP_PHASE_DEPTHS=1,3,4 \
  TRAIN_LOOP_PREWARM_DEPTHS=3,4 \
  EVAL_LOOP_DEPTH=4 \
  SEED=$SEED \
  torchrun --standalone --nproc_per_node=8 train_gpt.py \
      > train_seed${SEED}.log 2>&1
done
```

## Lineage

- **PR #1530** contributed the core SP8192 base architecture / looped-stack foundation.
- **PR #1626** contributed the phased-TTT schedule that this stack continues to use.
- **PR #1729** contributed the **CaseOps tokenizer**, lossless capitalization transform, and original-byte sidecar BPB accounting.
- **PR #1667** contributed the attention out-gate pattern used in this family of runs.
- **PR #1736** assembled those ingredients into one competitive stack.
- **This record's novel change** is the deterministic `1 -> 3 -> 4` recurrence-depth curriculum with fixed-depth-`4` eval.

## Credits

- @romeerp — CaseOps tokenizer, byte-sidecar accounting, and this recurrence-curriculum contribution.
- @samacqua — SP8192 base architecture / looped-stack foundation from PR #1530.
- @MarioPaerle — attention gate pattern.
- prior phased-TTT contributors in the PR #1626 line.

## Included files

- `train_gpt.py` — main training script.
- `submission.json` — metadata.
- `README.md` — this file.
- `train_seed42.log`, `train_seed0.log`, `train_seed1234.log` — 3-seed run logs.
- `tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model` — CaseOps SentencePiece model.
- `lossless_caps.py` — bijective CaseOps transform.
- `prepare_caseops_data.py` — one-time data prep script that emits the per-token byte sidecar.
