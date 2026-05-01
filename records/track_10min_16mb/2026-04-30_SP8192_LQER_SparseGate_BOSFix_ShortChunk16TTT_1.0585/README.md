# SP8192 LQER SparseGate BOSFix + ShortChunk16 TTT

3-seed record candidate for `track_10min_16mb`.

Mean result: `val_bpb=1.05924613` and `val_loss=2.31802318`.
Standard deviation: `val_bpb_std=0.00025710` and
`val_loss_std=0.00056262`.

## Results

| Seed | Steps | Train time | Quant BPB | TTT BPB | Eval time | Artifact bytes | Total bytes | Logs |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | 4918 | 599.519s | 1.07315022 | 1.05942388 | 590.9s | 15,869,183 | 15,909,199 | `train_seed42.log`, `eval_seed42.log` |
| 0 | 5042 | 599.505s | n/a | 1.05895134 | 526.4s | 15,864,395 | 15,904,411 | `train_seed0.log`, `eval_seed0.log` |
| 1234 | 5039 | 599.559s | 1.07314675 | 1.05936318 | 559.5s | 15,875,788 | 15,915,804 | `train_seed1234.log`, `eval_seed1234.log` |

All reported eval times are below 600 seconds. All reported/inferred total
submission sizes are below 16,000,000 decimal bytes.

Seed 0 note: the artifact was produced and the TTT eval completed, but the train
wrapper was later interrupted after per-group compression during a cancelled
rerun. The preserved seed-0 train log contains the completed training loop,
pre-quant diagnostic, code size, GPTQ hessian collection, and compression
completion, but not the final quantized diagnostic lines. The seed-0 artifact
bytes above were measured before cleanup from `final_model.int6.ptz`; total
bytes are artifact bytes plus the logged compressed code size.

## Method

Training stack:

- SP8192 CaseOps data with the lossless caps sidecar byte accounting.
- 11-layer 512d model with 8 attention heads and 4 KV heads.
- SparseGate, BOSFix, gated attention quant gate, layer looping, parallel final
  lane mean, and fused CE.
- Muon momentum `0.97`, row normalization enabled, EMA `0.9965`.
- Int6 matrix quantization, int7 embedding quantization, per-group compression.
- LQER asymmetric recovery with top-k `3`, rank `4`, factor bits `4`.

Eval stack:

- Eval-only score-first TTT from a fresh quantized artifact.
- `EVAL_SEQ_LEN=2560` and `TTT_EVAL_SEQ_LEN=2560`.
- LoRA rank `224`, LR `7e-5`, alpha `144`, Adam beta1 `0`, beta2 `0.99`.
- `TTT_LOCAL_LR_MULT=0.875`.
- Short-doc specialization: for documents shorter than `2048` tokens, use
  `SHORT_TTT_CHUNK_SIZE=16`.
- `PHASED_TTT_PREFIX_DOCS=0`, so there is no prefix warmup pass in this
  candidate.

The TTT path is intended to remain score-first: validation tokens are scored
before they are used for any test-time update.

## Reproduction

From this record folder:

```bash
bash run_all_seeds.sh 42 0 1234
```

Run one seed:

```bash
bash run_all_seeds.sh 42
```

The launcher trains first with `TTT_ENABLED=0`, then starts a fresh eval process
with `TTT_EVAL_ONLY=1`. Set `FORCE_TRAIN=1`, `FORCE_EVAL=1`, or `FORCE=1` to
ignore cache hits.

CaseOps data must be available at the path used by `run_all_seeds.sh`, or can be
prepared with:

```bash
python3 prepare_caseops_data.py \
  --docs /path/to/docs_selected.jsonl \
  --out ./data/datasets/fineweb10B_sp8192_caseops/datasets \
  --sp ./tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model
```

## Included Files

- `train_gpt.py`: training snapshot.
- `train_gpt_eval.py`: eval-only TTT snapshot.
- `run_all_seeds.sh`: single public train/eval launcher.
- `summarize_results.py`: local result summarizer.
- `prepare_caseops_data.py`, `lossless_caps.py`, and `tokenizers/...model`:
  CaseOps data/tokenizer dependencies.
- `train_seed*.log` and `eval_seed*.log`: preserved run evidence.
- `submission.json`: structured metadata and seed metrics.
