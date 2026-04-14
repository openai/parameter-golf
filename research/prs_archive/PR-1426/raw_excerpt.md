# PR 1426 — 13L Int4-Packed MLP + Depth Recurrence + Pre-Quant TTT + Full Stack

**Author:** aravhawk
**Claimed BPB:** TBD / null (pending evaluation per README and submission.json)
**Artifact size:** not stated
**Seeds:** not stated

## Files retrieved
- `records__track_10min_16mb__2026-04-06_14L_Int4Packed_GPTQ_XSA__README.md`
- `records__track_10min_16mb__2026-04-06_14L_Int4Packed_GPTQ_XSA__train_gpt.py`
- `records__track_10min_16mb__2026-04-06_14L_Int4Packed_GPTQ_XSA__submission.json`

## Environment variables (from run command)
Default: torchrun --nproc_per_node=8 train_gpt.py
SP8192 mode: DATA_PATH=./data/datasets/fineweb10B_sp8192, TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe.model, VOCAB_SIZE=8192
14L stretch: NUM_LAYERS=14, XSA_LAST_N=14, VE_LAYERS=12,13, RECUR_LAYERS=5,6
Ablation knobs: NUM_LAYERS=13, RECUR_LAYERS=4,5, RECUR_EXTRA_LOOPS=1, TTT_EPOCHS=6, TTT_LR=0.0005, TTT_FREEZE_BLOCKS=2, QK_GAIN_INIT=5.0, TRIGRAM=1, BIGRAM_VOCAB_SIZE=4096, BIGRAM_DIM=112, VE_LAYERS=10,11,12

## Claimed changes (from README, verbatim)
> Novel Techniques:
>
> True Int4 Bit-Packing (first in this competition). Standard int4 quantization stores values in [-7,7] as full int8 bytes, wasting 4 bits per value. Our pack_int4 function stores two int4 values in a single byte, cutting raw MLP storage in half before LZMA. Combined with Full Hessian GPTQ error compensation, int4 MLP achieves high reconstruction quality.
>
> 13 Layers (first submission beyond 11). Int4 GPTQ + bit-packing saves ~3MB vs uniform int6, funding 2 extra transformer layers within 16MB. With depth recurrence on layers 4,5, effective depth is 15 virtual layers.
>
> Pre-Quant TTT: After EMA weights are loaded and before GPTQ quantization, fine-tune with AdamW for 6 epochs (lr=0.0005, cosine decay, freeze first 2 blocks). Expected gain: -0.020 to -0.034 bpb.
>
> Full Stack: 13L (15 virtual), 512d, 8H, 4KV GQA, U-Net encoder 6/decoder 7, MLP 3x LeakyReLU(0.5)^2, XSA all 13 layers, SmearGate, BigramHash(4096,112), Trigram, VE at layers 10,11,12, Partial RoPE 16/64, LN Scale, tied embeddings, logit softcap 30.0, QK-Gain 5.0. Int4 for MLP, int6 for attention.
