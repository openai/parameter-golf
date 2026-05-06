# Gated XSA + CaseOps + LQER g32/top4 + In-Timer N-gram Tilt

The default run is:

- V21/Gated-XSA transformer with in-timer token-only n-gram tilt.
- LQER/GPTQ retune: `MATRIX_LR=0.028`, `LQER_RANK=2`, `LQER_ASYM_GROUP=32`, `LQER_TOP_K=4`.
- CaseOps SP8192 tokenizer and byte sidecar scoring.
- One-phase score-first TTT with 1000 prefix docs, keeping eval under the 600s cap.

## Results

The folder includes three successful 8xH100 logs. 

**Final result:** `1.05933439` mean TTT BPB across three logs.  
**Final file size:** max observed submission size `15,991,624 B` (`8,376 B` under the 16,000,000-byte cap).

| Seed | Config | Post-EMA BPB | Quant BPB | TTT BPB | TTT loss | TTT eval time | Train stop | Total submission size |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 42 | g32/top4, rank2, prefix1000 | 1.06357301 | 1.07110057 | 1.05872644 | 2.31688588 | 521.1s | 596.047s | 15,984,321 B |
| 0 | g32/top4, rank2, prefix1000 | 1.06531335 | 1.07291783 | 1.06035228 | 2.32044383 | 475.6s | 596.146s | 15,987,537 B |
| 1234 | g32/top4, rank2, prefix1000 | 1.06335328 | 1.07134445 | 1.05892444 | 2.31731918 | 467.9s | 598.091s | 15,991,624 B |
| **Mean** | all three logs | **1.06407988** | **1.07178762** | **1.05933439** | **2.31821630** | **488.2s** | **596.761s** | **15,987,827 B** |


## What Changed

### Architecture and training

The architecture is an integrated CaseOps/Gated-XSA stack:

- 11-layer, 512-dim, 8-head GQA transformer with 4 KV heads.
- CaseOps SP8192 lossless-caps tokenizer and byte sidecar validation scoring.
- Gated XSA on all layers with zero-init per-head gates.
- Looping layers 3-5 after 35% of training.
- Parallel final lane from layer 8.
- SmearGate and Sparse Attention Gate enabled.
- Fused CE training path, FA3 attention, partial RoPE, logit softcap.
- EMA applied before quantization.

The training loop and optimizer routing are kept conservative. The run stops on the 600s wallclock cap.

### Quantization

The final quantization path uses GPTQ/LQER/AWQ-lite with the g32/top4 LQER retune:

| Setting | Value |
|---|---:|
| `MATRIX_BITS` | 6 |
| `EMBED_BITS` | 7 |
| `LQER_RANK` | 2 |
| `LQER_ASYM_GROUP` | 32 |
| `LQER_TOP_K` | 4 |
| `AWQ_LITE_ENABLED` | 1 |
| `AWQ_LITE_GROUP_SIZE` | 64 |
| `COMPRESSOR` | `pergroup` |

The largest included g32/top4 run was 15,987,537 bytes, leaving 12,463 bytes of headroom under the 16,000,000-byte cap.

### Evaluation

Evaluation uses the conservative score-first TTT timing recipe:

| Setting | Value |
|---|---:|
| `EVAL_SEQ_LEN` | 2560 |
| `TTT_EVAL_SEQ_LEN` | 2560 |
| `PHASED_TTT_NUM_PHASES` | 1 |
| `PHASED_TTT_PREFIX_DOCS` | 1000 |
| `TTT_LORA_RANK` | 80 |
| `TTT_LORA_LR` | 0.0001 |
| `TTT_CHUNK_SIZE` | 48 |
| `NGRAM_TILT_ENABLED` | 1 |
| `NGRAM_HINT_PRECOMPUTE_OUTSIDE` | 0 |
| `TOKEN_ORDER` | 16 |
| `TOKEN_THRESHOLD` | 0.800 |
| `TOKEN_BOOST` | 2.625 |
| `WITHIN_BOOST` | 0 |
| `WORD_BOOST` | 0 |

The n-gram helper is inlined into `train_gpt.py`, so the evaluation logic is self-contained and counted with the submitted code. The helper is token-only, prefix-only, and its hint precompute happens inside the measured TTT eval timer.

## Compliance Notes

- Train time: all logs stop under the 600s training cap.
- Eval time: all logs finish under the 600s eval cap.
- Artifact size: all logs are below 16,000,000 bytes.
- No network during train/eval: runtime uses local dataset shards, local tokenizer, and local Python/C helpers only.
- Validation data is not accessed during training.
- N-gram tilt is causal and score-time only; it reads the validation stream left-to-right and emits each hint from prefix state before observing the target.
- TTT is score-first: each chunk is scored before its loss is used for adapter updates.

## Reproduction

Run from this record folder or from the repository root. The run script sets the final default hyperparameters.

```bash
# optional: install Python dependencies
pip install -r requirements.txt
```

The CaseOps dataset can be downloaded with:

```bash
huggingface-cli download romeerp/parameter-golf-caseops-v1 \
  --repo-type dataset \
  --local-dir ./data/datasets/fineweb10B_sp8192_caseops/
```

Then run the final configuration:

```bash
SEED=42 ./run.sh
```


The expected data root is:

```bash
./data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved
```

`run.sh` also supports overriding:

```bash
DATA_PATH=/path/to/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved \
TOKENIZER_PATH=/path/to/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model \
SEED=42 ./run.sh
```

## Files

- `train_gpt.py`: self-contained training, quantization, serialization, TTT eval, and in-timer n-gram tilt logic.
- `run.sh`: final record-candidate command wrapper.
- `submission.json`: leaderboard metadata.
- `requirements.txt`: Python dependencies beyond the base environment.
- `tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model`: tokenizer used by the CaseOps dataset.
- `lossless_caps.py` and `prepare_caseops_data.py`: CaseOps transform and dataset-prep helpers.
- `train_seed42.log`, `train_seed0.log`, `train_seed1234.log`
## Lineage and Credits

This submission combines public Parameter Golf components into a single final recipe:

- Gated XSA stack and in-timer token-only n-gram tilt implementation by `simon-marcus`.
- [PR #2060](https://github.com/openai/parameter-golf/pull/2060) by `S0urC10ud`: LQER/GPTQ retune ported here where supported.
- [PR #2007](https://github.com/openai/parameter-golf/pull/2007) by `Elubrazione`: long-context/QK/asymmetric-logit lineage used by later stacks.
- [PR #1967](https://github.com/openai/parameter-golf/pull/1967) by `ndokutovich`: V21 + n-gram tilt + LeakyReLU base lineage.
- [PR #1948](https://github.com/openai/parameter-golf/pull/1948) by `TimS-ml`: LeakyReLU squared slope sweep.
- [PR #1953](https://github.com/openai/parameter-golf/pull/1953) by `andrewbaggio1`: TTT/QK tuning lineage.
- [PR #1923](https://github.com/openai/parameter-golf/pull/1923) by `jorge-asenjo`: asymmetric logit rescale.
- [PR #1908](https://github.com/openai/parameter-golf/pull/1908) by `romeerp`: AWQ-lite mixed precision.
- [PR #1855](https://github.com/openai/parameter-golf/pull/1855) by `codemath3000`: per-group compression and strong SP8192 stack.
- [PR #1514](https://github.com/openai/parameter-golf/pull/1514) by `codemath3000`: strict token-only n-gram precedent.
- [PR #1145](https://github.com/openai/parameter-golf/pull/1145) by `AnirudhRahul`: n-gram tilt with closed-form probability renormalization.
