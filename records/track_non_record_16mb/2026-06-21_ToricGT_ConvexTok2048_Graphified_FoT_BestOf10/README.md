# ToricGT ConvexTok-2048 Graphified FoT

This is packaged as a non-record submission by default. The run uses ConvexTok-2048 and a graphified OAI-baseline adaptation, so the BPB path needs extra independent verification before any SOTA claim. The record folder includes the tokenizer and the exact minimal ToricGT modules required by `train_gpt.py`, and the README reports a self-contained code/tokenizer/model size estimate.

## Result

- selected run: `tg-bpb-bestof10-convextok2048-det-20260621T040717Z-r008-gate2500_experimental_family_selective-20260621T154110Z`
- selected profile: `gate2500_experimental_family_selective`
- checkpoint step: `1000`
- train BPB: `0.5036`
- validation BPB: `0.4924`
- int8+zlib round-trip BPB: `0.49533264`
- compressed model bytes: `12017502`
- self-contained code/tokenizer/dependency bytes: `1107863`
- total estimated artifact bytes: `13125365` / `16000000`
- final model SHA256: `3d22314ef66ce11b8851851795ab7bf23fa728c0e1f4b2c29e9d8132869c863e`
- checkpoint SHA256: `0178065f3d79f15d31332c9cf4149269d853ba3d6d28c56499b664a18c994b80`

## Technique Summary

This submission adapts the OpenAI Parameter Golf baseline rather than replacing it with the full ToricGT research model. The byte LM remains the BPB objective, while the input/output stream is augmented with first-class graph structure.

- **ConvexTok-2048 deterministic tokenizer**: byte-boundary tokenization DAG with LP/token-rank/price/byte-length features.
- **TokenGT-style FineWeb graphification**: token nodes, causal one-dimensional edge tokens, distance/endpoint/identifier channels, and toric phase features are injected into the baseline hidden stream.
- **OAI-FineWeb-only flattening**: graph-output states are flattened back to the tokenizer sequence for BPB scoring; non-FineWeb graph data remains graph structured.
- **Tropical/toric tokenization features**: min-plus path structure and toric vocabulary-face regularization expose tokenizer active paths as trainable geometry.
- **Embedding-space GFlowNet and Forest-of-Thought heads**: training-only trajectory objectives encourage useful reasoning forests and memory retrieval without increasing the compressed inference artifact.
- **Low-weight advanced sidecar losses**: full-rank GraphCG, analogical memory, TokenGT graph supervision, toric geometry, vector-bundle 1D-cone/sheaf, Toric BGG category-O, Koszul persistence, combinatorial toric commutative algebra, and derived signatures are active with BPB-first staging.

## Best-Of-10 Sweep

| run | profile | train BPB | val BPB | int8 BPB | self-contained bytes | status |
|---|---|---:|---:|---:|---:|---|
| tg-bpb-bestof10-convextok2048-det-20260621T040717Z-r001-convextok2048_det_tropical_toric_bpb-20260621T040718Z | convextok2048_det_tropical_toric_bpb | 1.005800 | 0.564000 | 0.565125 | 15874408 | candidate |
| tg-bpb-bestof10-convextok2048-det-20260621T040717Z-r002-convextok2048_bias_ood_probe-20260621T101421Z | convextok2048_bias_ood_probe | 4.463500 | n/a | n/a | n/a | candidate |
| tg-bpb-bestof10-convextok2048-det-20260621T040717Z-r003-gate1500_high_batch_toric_bgg_memory-20260621T101558Z | gate1500_high_batch_toric_bgg_memory | 4.463100 | n/a | n/a | n/a | candidate |
| tg-bpb-bestof10-convextok2048-det-20260621T040717Z-r004-gate1500_fast_main_lr_light_graphcg-20260621T101734Z | gate1500_fast_main_lr_light_graphcg | 0.902600 | 0.668900 | 0.669255 | 15981888 | candidate |
| tg-bpb-bestof10-convextok2048-det-20260621T040717Z-r005-gate1500_fast_main_lr_aux_conflict_recovery-20260621T120339Z | gate1500_fast_main_lr_aux_conflict_recovery | 0.829500 | 0.718100 | 0.718665 | 16219014 | candidate |
| tg-bpb-bestof10-convextok2048-det-20260621T040717Z-r006-gate1500_structural_toric_heavy-20260621T135410Z | gate1500_structural_toric_heavy | 0.873800 | 0.657400 | 0.658505 | 15094053 | candidate |
| tg-bpb-bestof10-convextok2048-det-20260621T040717Z-r007-gate1500_bgg_memory_probe-20260621T144834Z | gate1500_bgg_memory_probe | 0.831600 | 0.687900 | 0.689378 | 15283086 | candidate |
| tg-bpb-bestof10-convextok2048-det-20260621T040717Z-r008-gate2500_experimental_family_selective-20260621T154110Z | gate2500_experimental_family_selective | 0.503600 | 0.492400 | 0.495333 | 12896710 | selected |
| tg-bpb-bestof10-convextok2048-det-20260621T040717Z-r009-gate2500_fast_bpb_low_aux-20260621T161926Z | gate2500_fast_bpb_low_aux | 0.505600 | 0.495200 | 0.497956 | 12893602 | candidate |
| tg-bpb-bestof10-convextok2048-det-20260621T040717Z-r010-large_batch_lower_lr_sidecar_light-20260621T164928Z | large_batch_lower_lr_sidecar_light | 0.531600 | 0.521200 | 0.536639 | 10819607 | candidate |
| tg-bpb-bestof10-convextok2048-det-202606-r100-gate2500_experimental_family_selective-20260621T171946Z | gate2500_experimental_family_selective | 0.521800 | 0.502200 | 0.505831 | 12489361 | candidate |

## Reproduction

The record folder is self-contained with the minimal ToricGT modules required by the adapted `train_gpt.py`. It still expects the Parameter Golf FineWeb binary shards to be available through `DATA_PATH`; no network access is used during evaluation.

```bash
PYTHONPATH=. \
TOKENIZER_PATH=./fineweb_convextok_2048_det.convextok.json \
VOCAB_SIZE=2048 \
FINEWEB_GRAPHIFY=1 TOKENGT_FIRST_CLASS=1 GRAPH_OUTPUT_FLATTENING=1 OAI_FINEWEB_OUTPUT_FLATTENING=1 \
OAI_GFLOWNET=1 OAI_EMBEDDING_FOT=1 OAI_MTP=1 \
TORICGT_SIDECAR=1 REQUIRE_TORICGT_SIDECAR=1 \
DATA_PATH=/path/to/fineweb10B_convextok2048_det \
python train_gpt.py
```

## Rule Notes

- The official README states the cap is `16,000,000` decimal bytes for code plus compressed model. This folder reports the stricter self-contained size including the tokenizer and dependency modules.
- Tokenizer changes require extra proof that BPB is correct. The campaign logs include native validation BPB and int8 round-trip BPB; ConvexTok regret/tokenization-DAG analyses are retained in the ToricGT training notes.
- This local campaign was not an 8xH100 10-minute record run. It is submitted as a unique non-record graph/tropical/toric OAI-baseline adaptation unless later rerun under official record conditions.
- Score-first test-time adaptation is configured as non-committing in these short runs; no validation tokens are used for training before scoring.
