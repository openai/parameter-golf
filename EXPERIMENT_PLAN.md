# Experiment Plan

This plan is optimized for limited budget and the challenge rules.

## Goals

- Improve `final_int8_zlib_roundtrip_exact val_bpb`
- Improve `final_int8_ttt_lora val_bpb`
- Stay under the `16,000,000` byte artifact cap
- Avoid risky dataset changes until the safe path is exhausted

## Model Ablations First

Run these in order on remote GPUs:

1. `base10l`
2. `zloss_low`
3. `zloss_med`
4. `twice_low`
5. `zloss_twice`
6. `eval2048`

Use the named launcher:

```bash
NPROC_PER_NODE=1 bash scripts/run_remote_profile.sh base10l
NPROC_PER_NODE=1 bash scripts/run_remote_profile.sh zloss_low
```

Then repeat the best 2-3 profiles on `8x H100 SXM`.

## Dataset And Tokenizer Work

The challenge allows tokenizer or dataset changes, but the repo says they will be examined carefully and you must prove the `val_bpb` calculation is correct. See [README.md](/Users/deividasmataciunas/Desktop/research/openai_golf/README.md#L168).

Safest path:

- Rebuild tokenizers from the published docs cache only
- Re-export shards from the same selected docs
- Keep validation on the fixed first `50k` docs

Use:

```bash
bash scripts/rebuild_tokenizer_export.sh
```

Default ablation config:

- `sp_bpe_768`
- `sp_bpe_1024`
- `sp_bpe_1280`
- `sp_bpe_1536`
- `pure_byte_260`

## Dataset Ideas That Look Safe

- Vary tokenizer vocab size on the same published docs
- Compare pure-byte vs SentencePiece BPE
- Train on a prefix of shards, then do a short final stage on a higher-quality subset from the same docs
- Filter obviously low-value docs from the training side only
- Keep document boundaries clean during training and eval

## Risky Ideas

- External corpora
- Changing validation docs
- Any data use at eval time beyond what the rules allow
- Tokenizer changes without exact byte-accounting validation

## Success Metrics

For each run, record:

- `val_bpb`
- `final_int8_zlib_roundtrip_exact val_bpb`
- `final_int8_ttt_lora val_bpb`
- `Total submission size int8+zlib`
- `step_avg`

If a tokenizer change helps pre-quant quality but hurts artifact bytes, reject it early.
