# Experiment Plan

This plan is optimized for limited budget and the challenge rules.

## Goals

- Improve `final_int8_zlib_roundtrip_exact val_bpb`
- Improve `final_int8_ttt_lora val_bpb`
- Stay under the `16,000,000` byte artifact cap
- Avoid risky dataset changes until the safe path is exhausted

## 5-Run Moonshot Sequence

Run these in order on remote GPUs, using the current branch and `TRAIN_SHARDS=1`:

1. `drope_eval`
2. `yarn_eval`
3. `mtp_low`
4. `muon_balance`
5. `hybrid_delta`

Run the entire sequence:

```bash
NPROC_PER_NODE=1 bash scripts/run_moonshot5.sh
```

This prints each run tail and then a ranked JSON summary against the control run `twice_eval2048_ttt1024_clean2`.

Ranking priority:

1. Lowest `final_int8_ttt_lora val_bpb`
2. Lowest `final_int8_zlib_roundtrip_exact val_bpb`
3. Smallest artifact
4. Fastest step time

Promotion rules:

- Promote any run that beats the control on at least one final metric without exceeding the artifact cap.
- Promote `hybrid_delta` if it beats the control on either final metric, even slightly.

Next-step rules:

- If `drope_eval` beats `yarn_eval`, keep DRoPE and drop YaRN.
- If `yarn_eval` beats `drope_eval`, keep YaRN and drop DRoPE.
- If `mtp_low` wins, sweep `MTP_DEPTH=3` and `MTP_LOSS_WEIGHT` in `0.05`, `0.1`, `0.2`.
- If `muon_balance` wins, sweep `MUON_UPDATE_BALANCE` in `0.25`, `0.5`, `0.75`.
- If `hybrid_delta` wins even slightly, open a dedicated hybrid branch next.

## Next Moonshot

New architecture branch:

1. `shared_depth`

Idea:

- reuse `4` unique blocks across `10` logical layers
- keep tiny per-pass learned output scales so reused blocks can still specialize
- preserve the existing optimizer, export, and TTT paths

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
