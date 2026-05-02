Your current ~**1.118 BPB** (sliding window, I assume) is already very strong — it's clearly beating the public top of **1.1428** (thwu1's 10L int5-MLP + BigramHash(10240) + Muon WD=0.04 + SWA) and even the next few entries (~1.145–1.150 range from March 20). You're in a great spot, but since you're artifact-size maxed out (weights + code ~16 MB post-zstd), the realistic moves left are:

- tiny efficiency/quality-of-fit improvements that cost almost no extra parameters/bytes
- better post-training recovery / adaptation that fits in the eval budget
- compression / quantization tweaks that free up a few KB to allow a small widening or extra trick
- activation / init / regularizer micro-tweaks that interact well with your leaky_relu(0.5)^2 + int5 MLP + TTT + sliding eval stack

Here are the highest-EV small-ish changes I would try in priority order (ordered roughly by expected gain per effort, assuming you can afford 1–3 more quick experiments before submitting).

### Top-priority low-risk / high-signal tweaks (try these first)

1. **Increase TTT epochs to 4–5 but lower lr a bit (0.0015–0.0018 instead of 0.002)**  
   Your current TTT setup (3 epochs, lr=0.002, momentum=0.9, freeze first 2 layers) is already good, but many people report that 4–5 epochs at slightly lower lr recovers a bit more quantization loss without diverging.  
   → Cost: negligible runtime (still << 30 s total TTT)  
   → Expected: +0.003 to +0.008 BPB recovery in some configurations  
   → Try: `TTT_EPOCHS=4` or `5`, `TTT_LR=0.0016`

2. **Try TTT freezing only the first **1** layer instead of 2** (or none at all if stable)  
   Freezing fewer early layers often lets TTT repair more of the damage from int5 MLP quantization.  
   The risk is slight instability — watch the per-epoch loss in logs.  
   → Cost: zero  
   → Try: `TTT_FREEZE_LAYERS=1` (or `0` if loss doesn't explode)

3. **Add a very small amount of extra **dropout-like noise during TTT** (not full dropout)**  
   People sometimes add small Gaussian noise to the input embeddings or after smear gate during TTT only (std ~0.01–0.025) — helps generalization / fights overfit on val during adaptation.  
   Example addition in `run_ttt()` after `x = chunk[:-1].reshape(...)`:

   ```python
   if args.ttt_enabled:
       x = x + torch.randn_like(x, device=x.device) * 0.015   # or 0.020
   ```

   → Cost: ~nothing  
   → Often gives +0.002–0.006 in quantization-heavy models

4. **Switch prune_frac from magnitude → **structured** (remove weakest output channels in MLP.fc)**  
   Your current magnitude pruning is unstructured → zstd doesn't compress it as well.  
   Try zeroing entire output channels (columns) of `mlp.fc.weight` with the lowest average magnitude instead.  
   This is almost free in bytes after zstd and lets the surviving channels be a bit stronger.

   Quick change in `magnitude_prune()`:

   ```python
   if "mlp.fc.weight" in name and t.ndim == 2:
       # structured: weakest output channels
       channel_mags = t.abs().mean(dim=1)
       k = int(t.shape[0] * frac)
       _, idx = torch.topk(channel_mags, k, largest=False)
       t[idx] = 0
   ```

   → Expected: small compression win + tiny quality bump

### Medium-effort / potentially bigger moves

5. **Try a different activation in MLP (keep square, swap leaky)**  
   Your `leaky_relu(0.5)^2` is already aggressive — try these variants (change `MLP._activation` and/or forward):

   - `leaky_relu(negative_slope=0.3)^2` → slightly less aggressive negative part
   - `silu(x) * x` → SwiGLU-like without extra param (very popular in small LMs)
   - `relu(x + 0.1 * x**2)` or `relu(x) + 0.05 * x**2` → adds a bit of quadratic without squaring everything

   → Often one of these edges out leaky2 by 0.003–0.010 depending on quant level

6. **Increase BigramHash vocab to 12288 or 16384 (if you have ~5–10 KB spare)**  
   Current top uses 10240. Going higher helps a surprising amount on byte-level-ish data like FineWeb if the embedding table still fits after zstd.  
   → Cost: ~ few KB → try if your artifact has headroom

7. **Add very light **label smoothing** only during the last 30–40% of training** (0.05–0.08)  
   Helps fight overconfidence, especially useful with logit_softcap + TTT.  
   In forward:

   ```python
   if self.training and step > args.iterations * 0.6:
       main_loss = F.cross_entropy(logits.float(), targets, label_smoothing=0.06, reduction="mean")
   ```

   → Often +0.002–0.007 when quant is aggressive

### Smaller micro-tweaks worth ablating quickly

- `MUON_WD=0.035` → 0.045 range (instead of 0.02–0.04); many top entries converged around there
- `qk_gain_init=1.6` or `1.4` (instead of 1.5) — tiny but sometimes moves the needle
- `logit_softcap=28` or `32` (instead of 30)
- Try `warmdown_iters=1500` instead of 1200 — softer LR decay can help final quality
- In TTT: use AdamW instead of SGD+momentum for the last 1–2 epochs (very small code change)

### Quick reality check on possible floor

Given public top ~1.143 and your 1.118 (likely sliding), you're already ~0.025 ahead. The next 0.01–0.03 is going to come from stacking 2–4 of the small things above rather than one magic bullet. The biggest single unlock left is probably better TTT recovery + slightly better activation/quant interplay.

If you want to share which direction feels most promising given your logs (e.g. is TTT still dropping loss a lot in epoch 3? Is quant penalty still >0.01?), I can narrow it down further.

Good luck — you're in striking distance of something very hard to beat. 🚀