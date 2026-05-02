# Post-deadline community submission: 1.07134 BPB

**Score: 1.07134 quantized_phased_ttt val_bpb** | **Artifact: 15.87 MB / 16.00 MB SI cap** | **Hardware: 8×H100 SXM, 596 sec wallclock**

This is a *post-deadline* community submission shared for educational/community value, not for a leaderboard track record. The OpenAI Parameter Golf challenge ran March 18 → April 30, 2026; this run completed May 1, 2026. Posted here so the configuration and approach are visible to other participants.

For context, this score would have placed **#7** on the active leaderboard (between MarioPaerle's 1.0714 at #7 and dexhunter's 1.0719 at #8) — closing the gap to the eventual leader (1.06141, codemath3000) by ~70% from a baseline 1.0920 in two days of iteration. We have **131 KB of unused artifact headroom** and identified an unfinished experiment (full split-clip + LZMA code wrap) that plausibly takes this sub-1.07.

---

## Stack summary

Trained from scratch on FineWeb (10B-token sp12288 SentencePiece + lossless CaseOps transform), 8×H100 in 596 seconds (under the 600s cap), 4288 of 20000 planned steps, ending at val_bpb 1.07134 after Phased TTT eval.

Architecture and training:
- **Tokenizer**: SentencePiece BPE, vocab=12288, with **lossless CaseOps text transform** (`lossless_caps_caseops_v1`) — reduces effective bits-per-byte by collapsing case duplication into a small set of sentinels
- **Model**: 12 layers × 512 dim × 8 heads (4 KV via GQA), partial RoPE (16/64), MLP×2 with LeakyReLU(0.5)², tied embeddings, logit softcap 30
- **Recurrence**: layers 3-5 looped 2× with `ENABLE_LOOPING_AT=0.35`, schedule `encoder:[0,1,2,3,4,5,3,4,5] decoder:[3,4,5,6,7,8,9,10,11]`
- **Parallel residuals**: layers 7-11 use simple parallel attn+mlp sum (NOT leader's true 2-lane variant)
- **SmearGate**: `GATE_WINDOW=12`, BOS-masked
- **SparseAttnGate**: per-head sigmoid gate on attention output, gate input `x[..., :12]`, zero-init weight (96 params/layer)
- **CUDA graphs**: enabled
- **Fused softcapped CE**: enabled
- **Optimizer**: Muon (Polar-Express NS, 5 backend steps) for matrices, AdamW for embeds + scalars; `MATRIX_LR=0.026`, `WARMDOWN_FRAC=0.85`, `MIN_LR=0.1`, `EMA_DECAY=0.9965`, `MUON_WD=0.095`

Quantization (post-training, runs in last ~30 sec of wallclock):
- **GPTQ** with **Hadamard rotation** (modded-nanogpt-style), 16 calibration batches, 4 sec reserve
- **Mixed bit allocation**: int5 for q/proj/mlp_proj, int6 for kv/mlp_fc, int7 for tok_emb
- **LQER asymmetric** rank-4 top-3, factor_bits=4, group=64, applied to attn_proj + mlp_proj
- **EMBED_CLIP_SIGMAS=14** (tighter than default 20)
- Brotli compression of the int-quantized blob

Evaluation (eval-time, runs after quantization, not against the wallclock cap):
- **Phased TTT with LoRA**: 3 cumulative phases over 2500 prefix docs, per-doc batched LoRA (rank 80, alpha 144) on Q/K/V/O/MLP-fc/lm_head, AdamW (lr=1e-4, β2=0.99, wd=0.5), global SGD on prefix at phase boundaries with `cu_seqlens`-respecting attention via `flash_attn_varlen_func`

## Evaluation breakdown

| Stage | val_bpb |
|---|---|
| Live model at step 4288 | 1.0797 |
| Pre-quantization (post-EMA) | 1.07083 |
| Quantized | 1.09328 |
| Quantized + sliding-window | 1.07515 |
| **Quantized + Phased TTT** | **1.07134** |

Quant cost is small (1.07083 → 1.09328 = +0.022) and Phased TTT recovers most of it (1.09328 → 1.07134 = -0.022). The remaining gap to the leader (1.0614) is mostly clean model quality — better hparams, possibly attention-clip tuning, possibly NUM_LAYERS=11 instead of 12.

## Bugs found and fixed in our Phased TTT + LoRA port

These may be useful to anyone porting the leader's record into a different repo:

1. **`cu_seqlens` plumbing in `train_val_ttt_global_sgd_distributed`**: leader's global SGD pass uses `flash_attn_varlen_func` with `cu_seqlens` to prevent attention from leaking across BOS during the prefix update. Our initial port silently no-op'd this because our `GPT.forward(input_ids, target_ids)` didn't accept `cu_seqlens`. Fix is mechanical (~30 LOC threading the parameter through `GPT.forward` → `Block.forward` → `CausalSelfAttention.forward` → `flash_attn_varlen_func`). Without this, our Phased TTT delta vs sliding was −0.0012 BPB; after the fix it tripled to −0.0037 BPB.

2. **Parallel-lane structure mismatch in `forward_ttt`**: our base model trains with parallel residuals at `PARALLEL_RESIDUAL_START=7`, but our LoRA-injected `forward_ttt` initially ran single-path through all layers. That's a structural train/eval mismatch on layers 7-11. We added a `_parallel_block_with_lora` method that mirrors `Block.forward`'s parallel branch (`x_in + attn_scale*attn_out + mlp_scale*mlp_out`) with LoRA injection in both attn and MLP paths, and a corresponding branch in `forward_ttt`.

3. **Comment hygiene**: a stale port comment said SparseAttnGate is mutually exclusive with SmearGate. They're independent (different mechanisms — attention output gating vs token mixing) and the leader stacks both. The three *attention* gates (`attn_out_gate`, `gated_attn`, `sparse_attn_gate`) are mutually exclusive among themselves only.

## Key result chain (this submission's iteration history)

| Run | Change | val_bpb | Δ | Artifact |
|---|---|---|---|---|
| Baseline (`run_6a8c0f`) | Pre-TTT 8×H100 frontier | 1.0920 | — | 15.41 MB |
| `run_b64bbd` | Add Phased TTT (broken: no cu_seqlens, no parallel-lane) | 1.0913 | -0.0007 | 15.45 MB |
| `run_c67192` | Fix cu_seqlens + parallel-lane bugs | 1.0893 | -0.0020 | 15.45 MB |
| `run_e4db68` | + leader hparams (`WARMDOWN=0.85`, `MATRIX_LR=0.026`, `EMBED_CLIP=14`, `GPTQ_RES=4`) | 1.0831 | -0.0062 | 15.87 MB |
| `run_2b7cf6` | + CaseOps tokenizer (`lossless_caps_caseops_v1`) | 1.0737 | -0.0094 | 15.87 MB |
| **`run_afabfc`** | **+ SparseAttnGate (`SPARSE_ATTN_GATE_ENABLED=1`)** | **1.07134** | **-0.0024** | **15.87 MB** |
| Leader (`codemath3000`) | (frontier reference) | 1.06141 | — | ~15.95 MB |

## Headroom we never used (and what's plausible from here)

We have **131 KB of unused artifact headroom** and didn't get to try several promising additions before the deadline:

- **`run_0bccd9` (full split clip)**: same env as `run_e4db68` plus `MLP_CLIP_SIGMAS=10.0 ATTN_CLIP_SIGMAS=13.0` (leader values). Achieved **1.0743** BPB (-0.0088 from `run_e4db68`) but went over the 16 MB cap by 0.78 MB — model serialized to 16.78 MB. Not a valid submission.
- **`launch.py --compress`**: LZMA + base85 wrap of `train_gpt.py` saves ~90 KB on our 124 KB raw source. We never combined this with the split-clip experiment.
- **ATTN-only split clip**: codex 5.5 hypothesis was that MLP_CLIP=10 is the artifact bomb in `run_0bccd9` and ATTN_CLIP=13 alone might fit. Untested due to network errors on the data download.
- **Targeted bit bumps**: `attn.proj` is currently the smallest matrix dim (512×512) and would only add ~80 KB compressed if bumped to int6. Plausible to fit in headroom.

A reasonable next 8×H100 run: `run_afabfc` env + `--compress` + `ATTN_CLIP_SIGMAS=13.0`. With LZMA freeing ~90 KB and ATTN-only clip likely costing ~150-300 KB, the artifact lands ~15.95 MB (similar to leader's tightness) and BPB likely improves by 0.003-0.005 BPB → sub-1.069. Adding MLP_CLIP would push under 1.067 if it fits.

## Probe-vs-full pitfalls we hit

We ran several 1×H100 600-step probes during iteration. Two were validated at 8×H100 scale and both **flipped sign**:

- **TTT_ENABLED=1** (simple full-param SGD TTT, our older variant): probe showed **-0.076** BPB, full scale showed **+0.042**. Mechanism: under-trained model has more quant headroom for SGD to "un-quantize" through. At full scale the model is well-trained, SGD just adds noise.
- **NUM_LOOPS=0** (recurrence off): probe showed **-0.032** BPB, full scale showed **+0.006**. Mechanism: at probe scale, EMA hasn't warmed up to the recurrence-active state and the EMA model is contaminated by pre-recurrence weights. At full scale, EMA fully converges to the recurrent state.

**Caveat for future probes**: probe-vs-full sign flips appear systematically for late-binding features (eval-time SGD, late-enabling regularizers) but probably not for features active from step 1 (tokenizer, optimizer, schedule). We didn't validate this hypothesis.

## Files in this directory

- `train_gpt.py` (129 KB) — the model + training + quant + Phased TTT code (variant `phased_ttt_v3_sag_clip_v2`, derived from the April 27 leader record)
- `final_model.int6.ptz` (15.75 MB) — the brotli-compressed quantized weight blob produced by this run
- `run_log.txt` (38 KB) — full stdout from the 8×H100 training + eval pipeline
- `submission.json` — structured config + scores for programmatic reading

## Reproduction

The exact env override set is in `submission.json`. To reproduce on 8×H100:

```bash
ITERATIONS=20000 \
MAX_WALLCLOCK_SECONDS=600 \
$(jq -r '.env_overrides | to_entries | map("\(.key)=\(.value)") | join(" ")' submission.json) \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Data: `upascal/parameter-golf-sp12288-caseops` on Hugging Face.
