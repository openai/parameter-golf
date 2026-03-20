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
2. `twice_low`
3. `twice_eval2048_ttt1024`
4. `twice_layerwise`
5. `drope_eval`
6. `yarn_eval`
7. `mtp_low`
8. `muon_balance`
9. `hybrid_delta`

Use the named launcher:

```bash
NPROC_PER_NODE=1 bash scripts/run_remote_profile.sh base10l
NPROC_PER_NODE=1 bash scripts/run_remote_profile.sh twice_eval2048_ttt1024
```

Then repeat the best 2-3 profiles on `8x H100 SXM`.

Interpretation order:

- If `drope_eval` or `yarn_eval` beats `twice_eval2048_ttt1024`, keep the better rope scaling and discard the other.
- If `mtp_low` wins, sweep `MTP_DEPTH=3` and `MTP_LOSS_WEIGHT` around `0.05-0.2`.
- If `hybrid_delta` wins even slightly, open a dedicated hybrid branch before changing more optimizer knobs.

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

After the model-side shortlist settles, do these data sweeps:

1. Rebuild `sp_bpe_768`, `sp_bpe_1280`, and `pure_byte_260`
2. Rerun the current best profile on `TRAIN_SHARDS=1`
3. Only promote tokenizer changes that help `final_int8_ttt_lora` without pushing artifact bytes in the wrong direction

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
