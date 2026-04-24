# Techniques Index

Each leaderboard submission is in `records/track_10min_16mb/<date>_<slug>/` (or `records/track_non_record_16mb/` for unlimited-compute experiments). Read the README inside each record for technique detail. **Do not copy code** — only learn categories of techniques. This file is your idea menu, not a substitute for reading the source.

## How to use this file

- Filter by score to see what's working at different levels.
- Filter by tag to find prior work in a technique family.
- Open the linked record directory. Read the README. Then close it before designing your experiment.
- Some entries below are in `records/` but are not in the root `README.md` leaderboard table (either superseded, unofficial, or draft). They're still worth skimming — especially the Scylla 0.9485 entry, which is the best-known score but not in the README table yet.

## Records (sorted by score ascending — best on top)

| Score  | Date       | Slug                                                                    | Tags                                                              |
| ------ | ---------- | ----------------------------------------------------------------------- | ----------------------------------------------------------------- |
| 0.9485 | 2026-03-31 | 2026-03-31_Scylla_FullGPTQ_XSA11_FA3_0.9485                             | scylla-tokenizer, full-gptq, xsa, fa3                             |
| 1.0810 | 2026-04-09 | 2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT                   | sp8192, 3-layer-recur, parallel-resid, qk-gain, ttt               |
| 1.0822 | 2026-04-08 | 2026-04-08_SP8192_ParallelResid_ScoreFirstTTT                           | sp8192, parallel-resid, ttt                                       |
| 1.0828 | 2026-04-06 | 2026-04-06_SP8192_QK5_LegalTTT_1.0828                                   | sp8192, qk-gain, ttt                                              |
| 1.0835 | 2026-04-06 | 2026-04-06_SP8192_HessianSDClip_ProgressiveRecurrence                   | sp8192, hessian-clip, progressive-recur                           |
| 1.0856 | 2026-04-05 | 2026-04-05_SP8192_GPTQ-Embeddings_SDClip_Loop45x2                       | sp8192, gptq, depth-recur, sdclip                                 |
| 1.0897 | 2026-04-04 | 2026-04-04_SP4096_DepthRecurrence_ParallelResid_MuonEqR                 | sp4096, depth-recur, parallel-resid, muon-eqr, qk-gain            |
| 1.0912 | 2026-04-03 | 2026-04-03_MuonEqR_DepthRecurrence_WD090_AllInt6                        | muon-eqr, depth-recur, wd, int6                                   |
| 1.0979 | 2026-04-01 | 2026-04-01_Vocab4096_MLPMult4_WD085                                     | sp4096, mlp4x, wd                                                 |
| 1.1063 | 2026-03-31 | 2026-03-31_ParallelResiduals_MiniDepthRecurrence                        | parallel-resid, depth-recur, gptq                                 |
| 1.1122 | 2026-03-29 | 2026-03-29_Loader_FullGPTQ_XSA11_BigramHash2816                         | coprime-stride-loader, full-gptq, xsa, bigram-hash                |
| 1.1147 | 2026-03-25 | 2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072                             | gptq, xsa, bigram-hash                                            |
| 1.1179 | 2026-03-28 | 2026-03-28_MuonTTT_EntropyAdaptive_11L_8xH100                           | 11L, muon-ttt, entropy-adaptive                                   |
| 1.1194 | 2026-03-23 | 2026-03-23_LeakyReLU_LegalTTT_ParallelMuon                              | leakyrelu², ttt, parallel-muon                                    |
| 1.1228 | 2026-03-22 | 2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233                 | 11L, ema, gptq, qat                                               |
| 1.1248 | 2026-03-21 | 2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248                      | 11L, xsa, ema, partial-rope, qat                                  |
| 1.1271 | 2026-03-20 | 2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271                          | 11L, xsa, ema, int6, mlp3x, wd                                    |
| 1.1307 | 2026-03-20 | 2026-03-20_11L_EfficientPartialXSA_FA3_SWA120                           | 11L, partial-xsa, swa                                             |
| 1.1428 | 2026-03-20 | 2026-03-20_10L_Int5MLP_MuonWD04_SWA50                                   | 10L, int5, muon-wd, swa, bigram-hash                              |
| 1.1458 | 2026-03-20 | 2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA                   | int6, mlp3x, smear-gate, bigram-hash, muon-wd                     |
| 1.1502 | 2026-03-19 | 2026-03-19_MLP3x_QAT_Int6_SlidingWindow                                 | 11L, mlp3x, qat, int6, sliding-window                             |
| 1.1556 | 2026-03-19 | 2026-03-19_smeargate_orthoinit_muonwd                                   | smear-gate, ortho-init, muon-wd, bigram-hash, mlp3x, qat          |
| 1.1570 | 2026-03-24 | 2026-03-24_74M_Ternary_UNet_FP8_10L_8192BPE_YaRN_NeoMuon                | ternary-quant, 10L, unet, fp8, sp8192, yarn, neomuon              |
| 1.1586 | 2026-03-19 | 2026-03-19_Seq2048_FP16Emb_TunedLR                                      | 10L, int6, qat, mlp2.6x, sliding-window                           |
| 1.1630 | 2026-03-19 | 2026-03-19_MixedQuant_Int6Int8_SlidingWindow                            | int6, int8, mlp3x, sliding-window                                 |
| 1.1748 | 2026-03-19 | 2026-03-19_SlidingWindow_FP16Emb_10L_MuonWD_OvertoneInit                | sliding-window, fp16-emb, 10L, muon-wd, ortho-init                |
| 1.1925 | 2026-03-19 | 2026-03-19_SlidingWindowEval                                            | sliding-window-eval                                               |
| 1.1928 | 2026-03-17 | 2026-03-17_LoRA_TTT                                                     | lora, ttt                                                         |
| 1.2014 | 2026-03-19 | 2026-03-19_TrainingOptSeq4096                                           | seq4096                                                           |
| 1.2060 | 2026-03-18 | 2026-03-18_LongContextSeq2048                                           | seq2048                                                           |
| 1.2147 | 2026-03-19 | 2026-03-19_10L_MixedPrecision                                           | 10L, int6, int8                                                   |
| 1.2154 | 2026-03-19 | 2026-03-19_WarmdownQuantization                                         | warmdown-quant, early-baseline                                    |
| 1.2197 | 2026-03-18 | 2026-03-18_FP16Embed_WD3600                                             | fp16-emb, wd                                                      |
| 1.2230 | 2026-03-18 | 2026-03-18_LowerLR                                                      | lr-tune, early-baseline                                           |
| 1.2244 | 2026-03-17 | 2026-03-17_NaiveBaseline                                                | baseline, 9L, sp1024                                              |
| —      | 2026-03-19 | `2026-03-19_int6_STE QAT_ MLP_bigram _U_Net`                            | (directory exists but README is empty — skip unless browsing)     |

## Non-record / unlimited-compute track

| Score  | Date       | Slug                                                                         | Tags                                               |
| ------ | ---------- | ---------------------------------------------------------------------------- | -------------------------------------------------- |
| 1.1239 | 2026-03-24 | 2026-03-24_106M_Binary_Asymmetric_UNet_FP8_15L_8192BPE_YaRN_NeoMuon_Smear    | 1bit-quant, asym, unet, fp8, 15L, sp8192, yarn     |
| 1.2074 | 2026-03-18 | 2026-03-18_Quasi10Bfrom50B_SP1024_9x512_KV4_4h_pgut3                         | sp1024, 9L, 4h-baseline                            |
| —      | 2026-03-19 | 2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090                            | swiglu, warmdown-fix, single-gpu-5090              |
| —      | 2026-03-21 | 2026-03-21_DepthRecurrence_MixedPrecisionQuant                               | depth-recur, mixed-precision-quant                 |

## Tag glossary (one line each)

These are anchors for searching. Not technique definitions — read the source.

- `sp1024` / `sp4096` / `sp8192` / `scylla-tokenizer` — tokenizer vocab size or family variant
- `9L` / `10L` / `11L` / `15L` — number of unique transformer layers
- `mlp2x` / `mlp2.6x` / `mlp3x` / `mlp4x` — MLP hidden-dim multiplier
- `depth-recur` / `3-layer-recur` / `progressive-recur` — depth recurrence variants (loop layers, sometimes increasing R during training)
- `parallel-resid` — parallel attention/MLP residual lanes (vs sequential)
- `qk-gain` / `qk-norm` — multiplicative gain on Q (and K) before SDPA
- `gptq` / `gptq-lite` / `gptq-embeddings` / `full-gptq` — post-training int quantization with calibration data
- `qat` / `qat015` / `late-qat` / `warmdown-quant` — quantization-aware training (sometimes only late in training, or integrated with LR warmdown)
- `int5` / `int6` / `int8` — quantization bit-width
- `ternary-quant` / `1bit-quant` / `asym` — extreme low-bit quantization (often paired with mixed-precision)
- `ema` — exponential moving average of weights for the eval checkpoint
- `xsa` / `xsa4` / `partial-xsa` / `efficient-partial-xsa` — exponential / sparse / partial attention variants
- `swa` / `sliding-window` / `sliding-window-eval` — sliding window attention (in train and/or eval)
- `ttt` / `legal-ttt` / `score-first-ttt` / `lora-ttt` / `muon-ttt` / `entropy-adaptive` — test-time training (TTT) flavors
- `muon-wd` / `muon-eqr` / `cautious-decay` — Muon optimizer variants (weight decay, equivariant, etc.)
- `parallel-muon` / `neomuon` — Muon kernel/timing variants
- `sdclip` / `hessian-clip` — gradient clipping schemes
- `partial-rope` / `yarn` — RoPE variants (rotate fewer dims, extended context)
- `unet` — U-Net-style encoder-decoder skip structure (the baseline already has one; explicit label = stronger variant)
- `bigram-hash` / `coprime-stride-loader` — hashed bigram embedding, structured data loading
- `smear-gate` / `swiglu` / `relu²` / `leakyrelu²` — activation / gating variants
- `fp16-emb` / `fp8` — embedding/weight precision tweaks
- `wd` / `wd040` / `wd085` / `wd090` — weight-decay tuning
- `ortho-init` / `overtone-init` — initialization schemes
- `warmdown3500` / `warmdown-fix` — extended or corrected LR warmdown
- `seq2048` / `seq4096` — longer training sequence length
- `fa3` — FlashAttention-3 kernel
- `lr-tune` / `early-baseline` — LR tuning on the naive baseline (early-game material, likely subsumed by later stacks)
