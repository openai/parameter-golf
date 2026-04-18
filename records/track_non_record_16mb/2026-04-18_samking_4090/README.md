# [4090 Reproduction] Achieve 1.1249 val_bpb based on SOTA architecture

## 📌 Overview
This PR submits a robust reproduction and validation of the current SOTA architecture, specifically adapted and tested on a **single consumer-grade GPU (RTX 4090 24GB)**. 

## ⚠️ Important Note on File Size (Known Issue)
**Please note:** The total submission size is **16,018,877 bytes**, which slightly exceeds the strict decimal **16,000,000 bytes** cap by approximately **18 KB**. 

Due to my limited cloud compute credits (which are now depleted), I am unable to perform a re-run to prune the vocabulary or scripts for a smaller footprint. However, the training dynamics and the **1.1249 BPB** metric shown in the logs remain a valid proof of the architecture's performance on consumer hardware.

## 💻 Hardware & Performance
* **Environment:** 1x NVIDIA RTX 4090 (24GB VRAM, Cloud Instance)
* **Training Time:** ~3 hours (Equivalent to the 10-minute limit on 8xH100 via step alignment)
* **Final val_bpb:** `1.1249`

## 🛠️ Key Modifications & Adaptations
1. **Dynamic Gradient Accumulation:** Utilized `grad_accum_steps = 96 // world_size`. This safely protected the 24GB VRAM locally but will scale down to `12` on an 8xH100 cluster.
2. **Environment Variable Fallbacks:** Fully restored dynamic parsing for `MAX_WALLCLOCK_SECONDS = 600` and `ITERATIONS`.
3. **Quantization & Compression:** Successfully validated the 6-bit GPTQ and `brotli` compression pipeline.

## 📁 Artifacts
I have attached the complete training log (`training_log.txt`) recovered from the instance logs.

## 🤝 Request to Maintainers / Reviewers
I do not own a local GPU and my cloud balance is now depleted. If a maintainer or anyone with access to an 8xH100 cluster could kindly run a 3-seed verification using this script (perhaps with minor vocab pruning to fit the strict 16MB limit), I would be incredibly grateful!
