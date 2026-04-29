# PR #1855 base + AWQ-lite mixed-precision GPTQ — val_bpb 1.06086 (3-seed mean)

Applies activation-aware mixed-precision GPTQ (from PR #1908 / romeerp) on top of codemath3000 PR #1855 stack.

## Results

| Seed | val_bpb (post-TTT) | artifact bytes | steps | eval time |
|------|--------------------|----------------|-------|-----------|
| 42   | 1.06118            | 15,978,503     | 4989  | 392.8s    |
| 314  | 1.06005            | 15,976,469     | 4986  | 395.8s    |
| 1234 | 1.06135            | 15,976,673     | 4977  | 395.5s    |
| **mean** | **1.06086**    | —              | —     | —         |

3-seed std: 0.00069. Beats codemath3000 PR #1855 (1.06108) by 0.00022 BPB.

## Technique

Training is identical to PR #1855. The only change is post-training quantization:

**AWQ-lite (activation-aware GPTQ):**
1. Collect per-input-channel activation RMS during GPTQ calibration
2. Score column groups: `saliency = act_rms * mean(abs(weight))`
3. Select top-1 most salient 64-column group per matrix
4. Quantize that group at int8 inside the same full-tensor GPTQ solve (rest stays int6)

Env vars: `AWQ_LITE_ENABLED=1 AWQ_LITE_BITS=8 AWQ_LITE_GROUP_TOP_K=1 AWQ_LITE_GROUP_SIZE=64`

## Setup
1. `pip install -r requirements.txt`
2. `apt-get install -y lrzip`
3. Install FA3: `pip install --no-deps flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/`
4. Run `prepare_caseops_data.py` to build the dataset
5. `AWQ_LITE_ENABLED=1 AWQ_LITE_BITS=8 AWQ_LITE_GROUP_TOP_K=1 AWQ_LITE_GROUP_SIZE=64 torchrun --standalone --nproc_per_node=8 train_gpt.py`

## Environment
- 8xH100 80GB SXM (RunPod)
- PyTorch 2.9.1+cu128
- FlashAttention 3.0.0
- Triton 3.5.1
