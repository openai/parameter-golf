# Record: PR #1953 stack — no_qv TTT + AWQ-lite + AsymLogit + long-context eval

**val_bpb = 1.05847** (3-seed mean, std 0.00063) | **max artifact 15,985,934 bytes** | 8x H100 SXM | strict 600s train + eval

## Results

| Seed | Stop step | Train time | Pre-quant BPB | Quantized BPB | **Post-TTT BPB** | Eval time | Artifact bytes |
|------|-----------|------------|---------------|---------------|------------------|-----------|----------------|
| 42   | 4892      | 595.97s ✅ | 1.06126       | 1.06962       | **1.05788**      | 493.2s ✅ | 15,979,342     |
| 0    | 4884      | 595.97s ✅ | 1.06181       | 1.07019       | **1.05840**      | 420.5s ✅ | 15,979,187     |
| 1234 | 4894      | 596.14s ✅ | 1.06232       | 1.07093       | **1.05914**      | 428.4s ✅ | 15,985,934     |
| **Mean** | **4890** | **596.03s** | **1.06180** | **1.07025** | **1.05847** | **447.4s** | **15,981,488** |

vs merged PR #1855 (1.06108): **-0.00261 BPB / -0.00571 nats**

## Stack

Inherits the full PR #1855 base (codemath3000) and layers:

1. **AWQ-lite mixed-precision GPTQ** (PR #1908, romeerp) — activation-aware salient-group int8 promotion
2. **Asymmetric Logit Rescale** (PR #1923, jorge-asenjo) — learnable pos/neg softcap during TTT eval
3. **no_qv TTT mask** (PR #1953, himanshudongre) — disable Q/V LoRA in TTT, keep K/MLP/O
4. **TTT_LOCAL_LR_MULT=0.75** — scaled TTT optimizer LR
5. **QK_GAIN_INIT=5.25** — per-head Q-gain initialization
6. **EVAL_SEQ_LEN=2560** — extended eval context
7. **PHASED_TTT_PREFIX_DOCS=3000** — larger global-TTT prefix
8. **TTT_LORA_RANK=56** — reduced LoRA rank (compute reallocation)

## Compliance

- [x] Artifact under 16,000,000 bytes (max 15,985,934)
- [x] Train wallclock under 600s (max 596.14s)  
- [x] Eval wallclock under 600s (max 493.2s)
- [x] No PPM, no SLOT, no pre-quant TTT, no n-gram cache
- [x] Single left-to-right pass, score-before-update
- [x] Full normalized softmax distribution

## Reproduction

```bash
apt-get install -y lrzip
pip install sentencepiece brotli huggingface_hub numpy python-minifier
pip install --no-deps flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/

# Dataset
HF_HUB_ENABLE_HF_TRANSFER=1 python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='romeerp/parameter-golf-caseops-v1', repo_type='dataset', local_dir='/workspace/caseops_data')
"

# Run
for SEED in 42 0 1234; do
  SEED=$SEED \
  DATA_PATH=/workspace/caseops_data/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved \
  TOKENIZER_PATH=/workspace/caseops_data/datasets/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model \
  CASEOPS_ENABLED=1 VOCAB_SIZE=8192 MAX_WALLCLOCK_SECONDS=600 VAL_LOSS_EVERY=0 \
  SMEAR_GATE_ENABLED=1 GATE_WINDOW=12 SPARSE_ATTN_GATE_ENABLED=1 SPARSE_ATTN_GATE_SCALE=0.5 \
  GATED_ATTN_QUANT_GATE=1 FUSED_CE_ENABLED=1 QK_GAIN_INIT=5.25 \
  EMBED_BITS=7 MATRIX_CLIP_SIGMAS=12.85 ATTN_CLIP_SIGMAS=13.0 MLP_CLIP_SIGMAS=11.5 EMBED_CLIP_SIGMAS=14.0 \
  GPTQ_RESERVE_SECONDS=4.0 GPTQ_CALIBRATION_BATCHES=16 COMPRESSOR=pergroup \
  LQER_ENABLED=1 LQER_ASYM_ENABLED=1 LQER_RANK=4 LQER_FACTOR_BITS=4 LQER_ASYM_GROUP=64 LQER_TOP_K=3 \
  AWQ_LITE_ENABLED=1 AWQ_LITE_BITS=8 AWQ_LITE_GROUP_TOP_K=1 AWQ_LITE_GROUP_SIZE=64 \
  ASYM_LOGIT_RESCALE=1 \
  TTT_ENABLED=1 PHASED_TTT_ENABLED=1 PHASED_TTT_NUM_PHASES=3 PHASED_TTT_PREFIX_DOCS=3000 \
  TTT_LORA_RANK=56 TTT_MASK=no_qv TTT_Q_LORA=0 TTT_V_LORA=0 TTT_LOCAL_LR_MULT=0.75 \
  TTT_CHUNK_SIZE=48 TTT_BETA2=0.99 TTT_WEIGHT_DECAY=0.5 \
  EVAL_SEQ_LEN=2560 TTT_EVAL_SEQ_LEN=2560 \
  WARMDOWN_FRAC=0.85 BETA2=0.99 GRAD_CLIP_NORM=0.3 MIN_LR=0.1 MATRIX_LR=0.026 \
  NCCL_NET=Socket GLOBAL_TTT_MOMENTUM=0.9 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py > train_seed${SEED}.log 2>&1
done
```
