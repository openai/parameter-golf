## Model architecture (two scripts)

**`train_gpt.py` (naive / full-attention baseline)**  
9-layer U-Net style transformer (encoder half with skip writes, decoder half with skip adds), width **512**, **8** query heads and **4** KV heads (GQA), **2×** MLP expansion, vocab 1024, tied embeddings, RoPE, RMSNorm, residual scaling + skip weights. Every layer uses **causal scaled dot-product attention** (FlashAttention-style path). About **17.06M** parameters (`model_params` in logs).

**`train_gpt_gdn.py` (hybrid)**  
Same depth (9) and head layout (**8** / **4** GQA where attention is used), width **448**, same MLP multiplier and global structure. **Seven** blocks use a **Gated Delta Net** mixer (FLA `chunk_gated_delta_rule`, causal depthwise conv on Q/K/V, per-head gates/decay), and **two** blocks (layers **3** and **7**) keep **full causal attention**. Default GDN shape: `gdn_expand_v=2.0`, `gdn_head_dim_ratio=0.75`, `gdn_conv_size=4`. About **17.42M** parameters. The width and GDN recipe are chosen so capacity stays close to the baseline despite the cheaper mixers.

---

## Log-derived step time and 600s step budget


All experiments are run on 1 H100 GPU setup
Source: `logs/{4096,8192,16384,32768}_{baseline,gdn}_params.txt`.  
**Average step time** = mean **marginal** training time per step over logged steps **6–10** (difference in cumulative `train_time` between consecutive steps), i.e. steady throughput after the first few training steps.

| `TRAIN_SEQ_LEN` | Baseline avg step (ms) | Hybrid avg step (ms) | Est. steps in **600s** wall (baseline) | Est. steps in **600s** wall (hybrid) |
|-----------------|------------------------|----------------------|----------------------------------------|--------------------------------------|
| 4096            | 546                    | 790                  | **~1100**                              | **~760**                             |
| 8192            | 757                    | 899                  | **~793**                               | **~667**                             |
| 16384           | 1220                   | 1037                 | **~492**                               | **~579**                             |
| 32768           | 2160                   | 1363                 | **~278**                               | **~440**                             |

Formula used: `steps_600s ≈ 600_000 / avg_step_ms` (rounded to nearest integer). This ignores validation and other non-training overhead; real runs with `VAL_LOSS_EVERY` cadence will complete **slightly fewer** optimizer steps in 600s.

**Note:** At **4096–8192** context the hybrid is **slower** per step (extra conv + recurrent kernel vs heavily optimized FlashAttention at moderate lengths). At **16k–32k** the hybrid is **faster** per step because most layers avoid quadratic attention cost while two layers still provide full attention.



# Compute experiment: full attention vs GDN hybrid

## Run commands

RUN_ID=4096_baseline_params \
TRAIN_SEQ_LEN=4096 \
MAX_WALLCLOCK_SECONDS=100 \
torchrun --standalone --nproc_per_node=1 train_gpt.py


RUN_ID=8192_baseline_params \
TRAIN_SEQ_LEN=8192 \
MAX_WALLCLOCK_SECONDS=100 \
torchrun --standalone --nproc_per_node=1 train_gpt.py


RUN_ID=16384_baseline_params \
TRAIN_SEQ_LEN=16384 \
MAX_WALLCLOCK_SECONDS=100 \
torchrun --standalone --nproc_per_node=1 train_gpt.py


RUN_ID=32768_baseline_params \
TRAIN_SEQ_LEN=32768 \
MAX_WALLCLOCK_SECONDS=100 \
torchrun --standalone --nproc_per_node=1 train_gpt.py



RUN_ID=4096_gdn_params \
TRAIN_SEQ_LEN=4096 \
MAX_WALLCLOCK_SECONDS=100 \
torchrun --standalone --nproc_per_node=1 train_gpt_gdn.py


RUN_ID=8192_gdn_params \
TRAIN_SEQ_LEN=8192 \
MAX_WALLCLOCK_SECONDS=100 \
torchrun --standalone --nproc_per_node=1 train_gpt_gdn.py


RUN_ID=16384_gdn_params \
TRAIN_SEQ_LEN=16384 \
MAX_WALLCLOCK_SECONDS=100 \
torchrun --standalone --nproc_per_node=1 train_gpt_gdn.py


RUN_ID=32768_gdn_params \
TRAIN_SEQ_LEN=32768 \
MAX_WALLCLOCK_SECONDS=100 \
torchrun --standalone --nproc_per_node=1 train_gpt_gdn.py

---