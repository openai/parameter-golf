# PR 1385 — Compressor-Aware Training (CAT)

**Author:** Tomas Korenblit (korentomas)
**Claimed BPB:** 1.4465 (int8+zlib roundtrip), pre-quant 1.4461
**Artifact size:** 11.48 MB (11,480,222 bytes)
**Seeds:** not stated (590 steps, single run)
**Track:** non_record_16mb
**Hardware:** 1xH100 80GB HBM3, 600s

## Files retrieved
- `records__track_non_record_16mb__2026-04-05_CompressorAwareTraining_CAT__README.md`
- `records__track_non_record_16mb__2026-04-05_CompressorAwareTraining_CAT__submission.json`
- `records__track_non_record_16mb__2026-04-05_CompressorAwareTraining_CAT__train_golf.py`

## Claimed changes (from README, verbatim)
"Compressor-Aware Training (CAT): differentiable LZ77 autocorrelation + entropy proxy regularizer for zlib-friendly weights. Two differentiable loss terms: L_total = L_language_model + lambda_lz * L_dictionary_match + lambda_h * L_entropy. Dictionary matching proxy: computes soft match score at power-of-2 lag distances (1..512 bytes): for lag in [1,2,4,8,16,32,64,128,256,512]: diff_sq = (byte_stream[lag:] - byte_stream[:-lag]).square(); match_score += torch.exp(-diff_sq / temperature).mean(). Uses raw byte values [0-255] and temperature=50. Entropy proxy: soft histogram of byte values via Gaussian kernel, Shannon entropy. Both losses backprop through quantization via STE. Results (5 runs, 600s 1xH100): Control no CAT 1.4374 BPB 12.32 MB; Dict match only (lz=0.01) 1.4463 12.15 MB; Entropy only (h=0.1) 1.4465 11.52 MB; Combined (lz=0.01 h=0.1) 1.4465 11.48 MB; Entropy strong (h=1.0) 1.5044 9.81 MB. Architecture: 4 physical transformer layers looped 3 times (12 effective), per-loop LoRA rank 16, dim=896, 14 heads, 2 KV heads GQA, QAT fused with LR cooldown. Based on Relaxed Recursive Transformers (Bae 2024)."
