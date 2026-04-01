# Record: Pure Neural SOTA — Gated DeltaNet (Zero-Shot, No TTT/Cache) — 1.0226 BPB

**Artifact Size:** 14.1 MB (Safe ~1.9MB buffer against per-seed zip-bloat)  
**Train Time:** < 600s on 1x/8x H100 SXM  
**Eval Time:** ~20s (Pure forward pass)  
**Validation BPB (3-seed mean):** 1.0226  

---

### Motivation: Breaking the "Pre-TTT" Capacity Wall

Recent submissions have achieved impressive validation scores (e.g., ~0.50 - 1.02 BPB) by heavily relying on Test-Time Training (TTT), N-gram Caches, and Multi-Pass Oracle Mixers. However, inspecting the logs of these submissions reveals that their underlying *pure neural* representations (Pre-TTT) consistently plateau around **1.10 - 1.13 BPB**.

This submission takes a different approach. We aim to push the fundamental L(N) optimization frontier by establishing a stronger, purely neural baseline. 

**This submission achieves 1.0226 BPB Zero-Shot.**  
No Test-Time Training. No N-gram Caches. No LoRA EMA. No Evaluation Mixers.

### Architecture & MLOps

To fit a highly capable linear RNN within the 16MB limit without resorting to extreme low-bit quantization (which degrades intelligence), we utilized a strict, symmetric architecture paired with aggressive throughput optimization:

1. **Architecture (Gated DeltaNet):** 
   - 8 layers of `DeltaNet` (Gated Linear Attention) + 1 final standard Attention layer.
   - Constrained hidden dimension (`n_embd=384`) to safely fit 14.1 MB in `int8`.
   - No parameter tying (except standard embedding/lm_head tying).

2. **CPU-Bottleneck Relief (The Prefetcher):**
   - At this parameter scale, 8xH100 GPUs are frequently starved by standard Python dataloaders. We implemented a lightweight, non-blocking `FastLoader` using `pin_memory()`. This ensures the GPUs maintain high utilization without the overhead of multiprocessing queues.

3. **Dynamic Curriculum Batching:**
   - Micro-models (15MB) suffer from premature convergence and gradient smoothing when exposed to massive DDP batch sizes too early.
   - We implemented a strict curriculum: `Global Batch 64 -> 128 -> 192`. 
   - Crucially, the script dynamically scales the *local* batch size based on `WORLD_SIZE` to ensure the *global* batch size remains in this optimal, low-variance regime, maximizing the number of weight update steps within the 10-minute window.

4. **Hardware Symbiosis:**
   - `fused=True` AdamW.
   - Strict `allow_tf32` enforcement.

### Results (10-minute limit)

*Note: While fully compatible with 8xH100 DDP, this architecture's extreme throughput efficiency allows it to reach these scores even in resource-constrained environments.*

| Seed | val_bpb |
|------|---------|
| 42   | 1.0288  |
| 1337 | 1.0198  |
| 2024 | 1.0194  |
| **Mean** | **1.0226** |

By establishing this 1.0226 Pre-TTT baseline, we hope to provide the community with a stronger foundation. Applying existing TTT or N-gram Mixer techniques on top of this architecture should comfortably push the absolute limits of the leaderboard well below 0.90 BPB.