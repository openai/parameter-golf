# 8L Depth Recurrence + LeakyReLU² (1.2066 BPB)

**Author:** [Điền Tên/Github Handle của bạn]
**Score:** 1.2066
**Model Size:** 15.73 MB (Int8 + Zlib)
**Hardware:** 8xH100

## Architecture Innovations
- **Depth Recurrence:** 8 physical layers looped 2 times (16 logical layers).
- **LeakyReLU(0.5)²:** Solved dead-neuron gradients in the MLP, enabling deeper gradient flow.
- **Partial RoPE:** RoPE applied to only 25% of head dimensions (16/64).
- **Untied Embeddings:** Decoupled input token embeddings and LM head.
- **Depth Embeddings:** Added learned vectors for each logical layer to provide depth awareness.
- **Cross-Layer State Aggregation (XSA):** Skip connections across all previous logical layers initialized at zero.
- **Hardware Optimized:** `MLP_MULT = 2.75` (hidden dim 1408, highly aligned for Tensor Cores).

## Run Command
```bash
RUN_ID=depth_recurrence_8xh100_overclocked \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 NUM_LAYERS=8 NUM_KV_HEADS=4 MLP_MULT=2.75 \
TIE_EMBEDDINGS=0 EMBED_LR=0.4 HEAD_LR=0.003 MATRIX_LR=0.025 \
WARMUP_STEPS=20 WARMDOWN_ITERS=3500 MUON_MOMENTUM_WARMUP_START=0.9 \
MUON_MOMENTUM_WARMUP_STEPS=1500 MUON_MOMENTUM=0.95 LOGIT_SOFTCAP=15.0 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Logs
```text
step:7400/20000 train_loss:2.0282 train_time:595325ms step_avg:80.45ms
step:7458/20000 val_loss:2.0343 val_bpb:1.2048 train_time:600057ms step_avg:80.46ms
stopping_early: wallclock_cap train_time:600057ms step:7458/20000
peak memory allocated: 19139 MiB reserved: 19202 MiB
Serialized model: 74592783 bytes
Code size: 48738 bytes
Total submission size: 74641521 bytes
Serialized model int8+zlib: 15686654 bytes (payload:18987464 raw_torch:19026099 payload_ratio:3.93x)
Total submission size int8+zlib: 15735392 bytes
final_int8_zlib_roundtrip val_loss:2.0374 val_bpb:1.2066 eval_time:2658ms
final_int8_zlib_roundtrip_exact val_loss:2.03735762 val_bpb:1.20663794