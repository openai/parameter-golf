# Non-record: SP8192 MP-SGD TTT + QK-Gain 5.25 - Seed 42 Reproduction

This is a non-record single-seed reproduction of the open PR #1727 / PR #1700 SP8192 multi-phase SGD + phased LoRA TTT lineage. It is submitted as corroborating evidence and fallback documentation, not as an independent method or a new leaderboard record.

The measured seed-42 score beats the currently accepted PR #1493 leaderboard entry in the local checkout, but it trails PR #1727 itself and the current open frontier. The claim here is intentionally narrow: one full-validation seed reproduced under the 16 MB artifact cap with internal train/eval timers under 600 seconds.

## Result

| Seed | val_bpb | val_loss_nats | Train timer | Eval timer | Artifact bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| 42 | 1.07383884 | 2.77383724 | 596.160s | 453.812s | 15,929,546 |

Relevant log evidence:

- `train_shards: 80`
- `val_tokens: 40540160`
- `stopping_early: wallclock_cap train_time: 596160ms step: 4841/20000`
- `Total submission size quantized+brotli: 15929546 bytes`
- `quantized_ttt_phased val_loss:2.77383724 val_bpb:1.07383884 eval_time:453812ms`
- `total_eval_time:453.8s`

Important caveat: the controller-level command span was longer than 20 minutes because it included data/runtime overhead and TTT compile warmup. The script's internal train and eval timers pass the stated budget, but this is a reason to avoid presenting this as a new record claim.

## Comparison

| Entry | Status | BPB | Notes |
| --- | --- | ---: | --- |
| PR #1493 | accepted | 1.0810 | current accepted leaderboard entry in this checkout |
| This reproduction | local non-record | 1.07383884 | one seed, full validation |
| PR #1700 | open | 1.07219 | same base lineage, 3-seed mean |
| PR #1727 | open | 1.07217 | same code lineage plus QK-Gain 5.25 / 4 phases, 3-seed mean |
| PR #1855 | open | 1.06108 | stronger open neural/CaseOps-family frontier |
| PR #1925 | open | 1.06049 | newer open CaseOps/phased-TTT frontier claim by title |
| PR #1908 | open | 1.06081 | newer open frontier claim as of 2026-04-30 |

Some open byte-level PPM / casefold / SLOT PRs claim much lower BPB, but those are rule-sensitive and are not treated here as clean comparable neural submissions.

## Reproduction

Run environment:

- Hardware: 8x NVIDIA H100 SXM 80GB
- PyTorch: 2.9.1+cu128
- Process count: `torchrun --standalone --nproc_per_node=8`
- Data: `sp8192` cached FineWeb challenge data from `kevclark/parameter-golf`
- Tokenizer: `data/tokenizers/fineweb_8192_bpe.model`

Data prep:

```bash
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf \
python3 data/cached_challenge_fineweb.py --variant sp8192
```

Run command:

```bash
SEED=42 \
DATA_DIR=/workspace/parameter-golf-pr1727/data \
QK_GAIN_INIT=5.25 \
PHASED_TTT_ENABLED=1 \
PHASED_TTT_PREFIX_DOCS=2000 \
PHASED_TTT_NUM_PHASES=4 \
MLP_CLIP_SIGMAS=12.0 \
ATTN_CLIP_SIGMAS=13.0 \
EMBED_BITS=7 \
EMBED_CLIP_SIGMAS=15.0 \
MATRIX_LR=0.026 \
GPTQ_RESERVE_SECONDS=4 \
GPTQ_CALIBRATION_BATCHES=16 \
torchrun --standalone --nproc_per_node=8 train_gpt.py \
  2>&1 | tee train_seed42.log
```

## Files

- `train_gpt.py`: copied from the PR #1727 record folder; sha256 `9802de706b5d71b319d25da01b93642c4253bd67318315f0965c075232d43dbd`
- `train_seed42.log`: seed-42 run evidence; sha256 `6791b62dc21a70ca3995d120050e8bd75cb456d994826acd19140752aa1d1987`
- `controller_seed42.log`: controller-level run capture; sha256 `5d6d33a3b9255907c9a4989e9a9cd6e4c339c32de15e3c21688172013b1744f8`
- `requirements.txt`: declared Python dependency surface for reproduction

## Compliance Notes

- Track label: non-record 16 MB; adaptive compression / score-first TTT.
- Artifact size: 15,929,546 bytes, under the decimal 16,000,000-byte cap.
- Internal train timer: 596.160s, under 600s.
- Internal eval timer: 453.812s, under 600s.
- Full validation: log reports `val_tokens: 40540160` and `val_doc_fraction: 1.0`.
- No casefold or byte-level PPM is used in this reproduction.
- Evaluation does not require network access or external data downloads.
- Code compression invokes `pyminify` during training to account for final compressed code size; this is disclosed in `requirements.txt`.

## Attribution

- Full base stack: PR #1700 by @jorge-asenjo, reused by PR #1727 by @yahya010.
- QK-Gain 5.25 setting: PR #1493 by @bigbag.
- Additional upstream lineage as documented by PR #1727: PR #1530, PR #1394, PR #1626, PR #549, and PR #1413.
