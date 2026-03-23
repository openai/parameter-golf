=== Smoke Test Results (120s, 1 GPU, seed=42) ===
Date: 公曆 20廿六年 三月 廿三日 週一 十三時五分廿四秒
Baseline SOTA target: 1.1428 BPB (full 8xH100 10min run)

| Rank | Experiment | val_bpb | vs baseline | Status |
|------|-----------|---------|-------------|--------|
| 1 | idea16_embed_scales | 3.15109912 | -0.0090 | ✅ BETTER |
| 2 | idea07_bigram16k | 3.15438649 | -0.0057 | ✅ BETTER |
| 3 | baseline | 3.16013399 | +0.0000 | ❌ |
| 4 | idea19_grad_central | 3.16467439 | +0.0045 | ❌ |
| 5 | idea12_zloss | 3.16780107 | +0.0077 | ❌ |
| 6 | idea01_byte_weighted_loss | 3.18665442 | +0.0265 | ❌ |
| 7 | idea11_swiglu | 3.18892855 | +0.0288 | ❌ |
| 8 | idea14_layerwise_lr | 3.20030272 | +0.0402 | ❌ |
| 9 | idea13_label_smooth | 3.25258968 | +0.0925 | ❌ |
| 10 | idea20_ema | 4.64555590 | +1.4854 | ❌ |

### Not yet completed:
- idea02_factorized_bigram (not started)
- idea03_entropy_reg (not started)
- idea04_conditional_resid (not started)
- idea05_embed_factorize (not started)
- idea06_adaptive_ns (not started)
- idea08_11th_layer (not started)
- idea09_trigram (not started)
- idea10_combined_best (not started)
- idea15_eval_stride32 (not started)
- idea17_gqa2kv (not started)
- idea18_lzma (not started)
- idea21_stochastic_depth (running/crashed)
- idea22_wsd_schedule (running/crashed)
- idea23_batch_warmup (not started)
- idea24_deepnorm_init (not started)
- idea25_sandwich_norm (not started)
- idea26_agc (not started)
- idea27_multi_token_pred (not started)
- idea28_diff_attention (not started)
- idea29_asymmetric_quant (not started)
- idea30_groupwise_quant (not started)
