# Mini-Hymba: Hybrid Attention + SSM Heads on Recurrent Layers

**Author:** Aparna Sivanandam ([@aparna-1407](https://github.com/aparna-1407))  
**Track:** Non-record (unlimited compute)  
**val_bpb:** 3.0721 (post int8+zlib roundtrip, 134 steps, 1× NVIDIA RTX PRO 6000 Blackwell)  
**Artifact size:** 6,909,879 bytes (6.9 MB — well under 16MB limit)  
**Status:** Full 4550-step run on 8×H100 pending compute grant  

---

## What this submission does

Replaces `CausalSelfAttention` in the recurrent layers (3, 4, 5) with a **HymbaLayer** — a
hybrid block that runs attention heads and Mamba-lite SSM heads **in parallel** on the same
input and concatenates their outputs before projecting back to `d_model`.

The core insight from the Hymba paper: attention heads provide high-resolution recall of
specific tokens, while SSM heads provide efficient context summarization via a recurrent
state. Running them in parallel lets each pathway specialize — the SSM heads accumulate a
compressed summary of the entire prior context, while attention heads handle precise token
retrieval. This complementarity is especially valuable in the recurrent layers (3, 4, 5),
which already loop 2-3× in the SOTA depth-recurrence stack — the SSM hidden state persists
and compounds across those loops in a way that pure attention cannot.

---

## Architecture changes over baseline

### HymbaLayer (drop-in for CausalSelfAttention)

```
Input x (B, T, d_model)
  ├── Attention pathway
  │     n_attn = num_heads // 2 = 4 heads
  │     + learnable meta tokens (4) prepended
  │     + RMS norm + RoPE (identical to host script)
  │     + QK-Gain per head
  │     + GQA (n_attn=4, n_kv=4, rep=1)
  │     → (B, T, n_attn × head_dim)
  │
  └── SSM pathway (Mamba-lite)
        n_ssm = num_heads // 2 = 4 heads
        operates on original x (no meta tokens)
        diagonal A, input-dependent B/C/dt
        stable fp32 sequential scan
        → (B, T, n_ssm × head_dim)

cat([attn_out, ssm_out]) → out_proj → (B, T, d_model)
```

**Head split rationale:** `n_attn = n_head // 2 = 4`. With `n_kv_head = 4`, this gives
`kv_rep = n_attn // n_kv_head = 1` — exact GQA, no remainder issues.

### Learnable meta tokens

4 learnable tokens prepended to each sequence before the attention computation in hybrid
layers. From the Hymba paper: these store critical global information and reduce the
"forced-to-attend" burden — attention heads no longer need to route through positional
tokens to access global context. Cost: `4 × 512 × 3 layers = 6,144` extra parameters.

### Cross-layer KV sharing (actual parameter reduction)

Adjacent Hymba layers share K and V projections: layer 4 reuses layer 3's `c_k`/`c_v`,
layer 5 reuses layer 3's (via chain walk). Sharing layers do not register `c_k`/`c_v` at
all — they are absent from `state_dict`, so no dead weights enter the artifact. This is
a genuine parameter reduction, not just a forward-pass reuse.

### Everything else unchanged

TTT, GPTQ int8+zlib quantization, BPB evaluation, artifact compression, Muon optimizer,
and the depth-recurrence loop on layers 3-5 all run exactly as in the host script.
The implementation is a clean drop-in: `hymba_layer.py` registers host classes
(`CastedLinear`, `Rotary`, `apply_rotary_emb`) at import time and delegates all
normalization and RoPE to the host script's implementations.

---

## Results

| Metric | Value |
|--------|-------|
| val_bpb (unquantized, step 134) | 3.0435 |
| val_bpb (post int8+zlib roundtrip) | **3.0721** |
| val_loss (post roundtrip) | 5.1871 |
| Artifact size int8+zlib | 6,909,879 bytes |
| Training steps completed | 134 / 200 (wallclock cap) |
| Hardware | 1× NVIDIA RTX PRO 6000 Blackwell (96GB) |
| Step time | ~6.7s/step (sequential SSM scan, torch.compile disabled) |

**Loss trajectory (val_bpb):**
```
step   0:  4.1078  (untrained baseline)
step  25:  3.2532
step  50:  3.1918
step  75:  3.1630
step 100:  3.0936
step 125:  3.0454
step 134:  3.0435  (wallclock cap hit)
```

Clean monotonic descent, no spikes or instability. Quantization roundtrip cost: +0.029 BPB.

---

## Why the BPB is high (and what would close the gap)

This submission is limited by two factors unrelated to the architecture:

1. **Only 134 steps** — the sequential Python SSM scan runs at ~6.7s/step on a single GPU,
   vs ~0.13s/step for the SOTA on 8×H100. A full 4550-step run requires 8×H100 with a
   parallel SSM scan. Pending compute grant.

2. **sp1024 tokenizer** — SOTA uses sp8192. The larger vocabulary directly improves BPB
   and the tokenizer data was the only variant available on the cached HuggingFace dataset
   at submission time.

A full 4550-step run on 8×H100 with sp8192, parallel SSM scan, and tuned LRs is expected
to produce results competitive with the 1.1-1.2 BPB range. This submission documents the
architecture and demonstrates stable training.

---

## Run command

```bash
HYMBA_ENABLED=1 \
  WARMUP_STEPS=100 \
  ITERATIONS=200 \
  MAX_WALLCLOCK_SECONDS=900 \
  MATRIX_LR=0.004 \
  SCALAR_LR=0.004 \
  EMBED_LR=0.005 \
  TIED_EMBED_LR=0.005 \
  VAL_LOSS_EVERY=50 \
  DATA_PATH=./data/datasets/fineweb10B_sp1024 \
  TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
  python train_gpt.py 2>&1 | tee train_log.txt
```

**Dependencies:** `hymba_layer.py` must be in the same directory as `train_gpt.py`.
No additional packages required beyond the repo's `requirements.txt`.

---

## Implementation notes

The SSM scan in `MambaLiteHead` uses a stable fp32 sequential recurrence (explicit for-loop
over T) rather than a parallel log-cumsum approximation. The parallel cumsum approach
produced `inf × 0` NaN instability under `torch.compile` + bf16 autocast. The sequential
scan is numerically exact and stable but disables `torch.compile` (logged as
`torch_compile:disabled reason=hymba_ssm_scan_inductor`). A fused parallel Triton kernel
would restore compile compatibility and reduce step time by ~50×.

---

## References

1. **Hymba: A Hybrid-head Architecture for Small Language Models**  
   Xin Dong, Yonggan Fu, ..., **Yingyan (Celine) Lin**, Jan Kautz, Pavlo Molchanov  
   ICLR 2025 — [arXiv:2411.13676](https://arxiv.org/abs/2411.13676)  
   *Primary architectural inspiration. The parallel hybrid-head design, meta tokens,
   and KV sharing are all from this paper.*

2. **CPT: Efficient Deep Neural Network Training via Cyclic Precision**  
   **Yingyan Lin** et al., Georgia Tech EIC Lab  
   ICLR 2021 Spotlight — [arXiv:2101.09868](https://arxiv.org/abs/2101.09868)  
   *Motivates quantization-aware training direction for future work.*

3. **Mamba: Linear-Time Sequence Modeling with Selective State Spaces**  
   Albert Gu, Tri Dao  
   [arXiv:2312.00752](https://arxiv.org/abs/2312.00752)  
   *The MambaLiteHead SSM design is a simplified version of Mamba's selective scan.*

4. **PR #1493** (bigbag) — SP8192 + 3-layer recurrence + parallel residuals + QK-Gain + legal TTT  
   *Base SOTA stack that this submission patches into.*
