## 11L EMA + GPTQ-lite + LeakyReLU(0.5)^2 + QAT@0.15

This folder is based on the public `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`
family, with the MLP activation changed from `relu^2` to `LeakyReLU(0.5)^2`.

## Stack

- 11 transformer layers
- model dim 512
- 8 attention heads
- 4 KV heads
- 3x MLP expansion
- XSA on late layers
- partial RoPE
- LN scaling
- EMA
- late QAT at LR scale `< 0.15`
- warmdown 3500
- GPTQ-lite style int6 export
- LeakyReLU(0.5)^2

## Implementation Notes

- falls back to PyTorch SDPA when `flash_attn_interface` is unavailable
- uses `torch.no_grad()` in eval paths
- clones rotary cache tensors before reuse
- supports `USE_TORCH_COMPILE=0`

## 8xH100 Run

This folder produced the following preliminary result on `8xH100`:

- `step:4260/20000 val_bpb:0.8705`
- `DIAGNOSTIC post_ema val_bpb:0.8705`
- `final_int6_roundtrip_exact val_bpb:0.87762377`
- `Total submission size int6+zstd: 15825448 bytes`

## Run Command

```bash
cd /workspace/ParamGoldOpenAI/records/track_10min_16mb/2026-03-27_nandh_11L_ema_gptqlite_leakyrelu
RUN_ID=nandh_11l_gptqlite_leakyrelu \
DATA_PATH=/workspace/ParamGoldOpenAI/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/workspace/ParamGoldOpenAI/data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
python -m torch.distributed.run --standalone --nproc_per_node=8 train_gpt.py
```
