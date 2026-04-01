# 12L rANS + LeakyReLU(0.95)² + Soft XSA

val_bpb: 1.1601 (sliding window stride=64, post int5/int6+rANS quantization roundtrip)
artifact: 15,912,601 bytes (87 KB under 16,000,000 limit)
hardware: 1×H100 SXM, extrapolating 8×H100 submission
n-gram eval: forthcoming (expected to add -0.005 to -0.01 BPB at eval time)
reproduce: `script-log-evalcode-etc/eval/reproduce_results.sh` (full training + eval from scratch)

## About me

I'm not a language model guy at all. I'm a deep learning practitioner who tends to focus on small models with tabular data - typically generated from appliances of various sorts. I've done some kernel work to offload stats operations using tiling patterns. That background is what drew me to check out this competition. Some of my knowledge transferred — the compression instincts, the kernel work, the comfort with squeezing models into tight budgets.  The methods I started to see here spawned a huge research and testing project on my side. It's been a lot of fun and very informative on transformer architectures for language. It's also made me think deeply about compression itself and how far down you can actually get in terms of source compression, both for this domain and others that might be closer to my own wheelhouse.  I enjoyed making this submission a lot, and plan to improve it throughout the duration of the competition.  


## Methodology note

All ablation experiments were 500-step runs on 1×H100 with torch.compile. I was compute-constrained and couldn't afford full-length runs for every idea, so I used 500-step ablations to check for early signal. This means many of the techniques listed as "didn't work" may actually perform differently at full training length — 500 steps is enough to detect large effects but not subtle ones that compound over time. The ablation table should be read with that caveat.


## Writeup note

If I didn't mention a PR that had an item before today - I apologize, I may have missed you, and if that's the case I want to apologize for not providing prior refrence. 
As of a hour or two ago I believe my accredidations for inspiration and where I started are all accurate.  This is a late night submission, I don't want to wake up and find out someone else got the rANS thing and did it better than me - so there will be more information coming, and maybe better phrasing in some parts.


## The story

Started from the SOTA stack at competition launch (PR #287 era, 11L baseline). Began experimenting with compression early, eventually landing on per-tensor adaptive rANS entropy coding as a replacement for zstd-22. Each weight tensor gets its own frequency table tuned to its actual value distribution after int5/int6 quantization. We test both rANS and zstd per tensor and pick the winner. In practice rANS wins on nearly everything - only 1 tensor won on zstd this run.  However the margin varies significantly depending on what the underlying weight distributions look like. Some training methods that improve BPB turn out to be surprisingly expensive in artifact bytes because they change weight distributions in ways rANS handles less well. I seem to get a better ratio on the current arch at full train then at 500/2k.  XSA was a notable example of variation changing the expected compression. 
To validate, we ran tensors though and confirmed bit-identical eval BPB - matching to 10 decimal places.  rANS works.


### Fitting under 16MB

The rANS savings opened up enough headroom to move from 11L to 12L. From there we found several more ways to reclaim artifact bytes:

- **Bigram removal (bigram_vocab_size=0).** At 12L, BigramHash contributes effectively zero BPB. Removing it saved roughly 500KB of artifact space. Never attempted this at lower layer counts so unclear if this is a depth thing or if bigram was always marginal.

- **10% magnitude pruning + safety valve.** A post-training pruning step zeros out the smallest 10% of weights by magnitude (2D tensors > 65536 params only), improving compressibility. A safety valve loop auto-bumps prune_pct by 1% increments until the artifact fits under 15,950,000 bytes (50KB buffer), max 25%. In this run the safety valve didn't trigger — 10% was enough with 87KB to spare.

- **Code size reduction.** XSA Triton kernel inlined into train_gpt.py (counts toward code_bytes). Eval-only functions moved to a separate import. Comments and docstrings trimmed. Net savings ~16KB.

- **MLP width interaction with XSA.** While trying to fit the artifact under 16MB, we tested several MLP widths. MLP 2.8125x with XSA produced better BPB than MLP 3.0x with XSA or MLP 3.0x without. I assume this is due to the narrower MLP operates a bit like dropout and stopped its capacity from beating out using attention(?). Not planned, just fell out of the  search for something that would fit.

### Training quality

- **LeakyReLU(0.95)² activation.** Inspired by PR #885 which used slope=0.9. Ran a sweep across relu², leaky 0.01², 0.05², 0.9², 0.95², 1.0², and silu². 0.95 was the peak. A win at -0.014 BPB.

- **Soft XSA with learned per-head alpha on all 12 layers.** Inspired by XSA4 from PR #549 and PR #374. My favorite thing to do with all magic numbers - param them up and make the tell me.  And since we're at it, do it differenly per layer (or similar).  However running this and the initial naive PyTorch implementation exploded twice.  First it became apparent the memory shape was wrong (more on this below) creating a massive iteration issue and even after that was resovled through first-pass kerenel the step time was still too high. from ~485ms to ~856ms.

The first Triton kernel (v2) using per-position programsbut lost gradients flow through q/k (bad implementation). Next a position-tiled v3 kernel ala-flash-attention-style.  This brought the forward pass down significantly. BPB still was a loss though.  Then I finally saw the implementation issue, added backward to preserve q/k.  This scopecrept on me but I left it in, I'm curious what other people will say about the work.  

Regardless this recovered most of the quality and landed at 512ms. There's likely more performance to be found — the tiling and online softmax patterns in the forward kernel could be further optimized I think, possibly there's an adjacent but much faster method I'm not aware of that could replace this.  Either way, step time recovered I called it done for now. Hopefully someone will pick up on tweaks here, I am sure there's something we've left on the table.

### Eval-time augmentation

- **CMS n-gram eval cache with bilateral confidence gating.** First - not included in score right now.  Eval assist in this competition is interesting, and not something I expected to see work so well.  I tried to do something similar with at twist so it's not a direct copy.  I spent more time here than I would have anticipated as well - this is a fascinating set of solutions to use.  

What I was playing with:  Count-min sketch (64M counters, 4 hash functions, orders 3-7) built from backward-looking scored tokens during sliding window eval. The mixing gates on both sides: neural uncertainty (PR #727 style) AND n-gram confidence (my twist).  This took roughly 30 iterations to get from catastrophic (+0.125 BPB) to helpful (-0.008 BPB), and launched me off into a few topics I was not familar with.  This was fun and if anyone see's the direction I was heading instead with it I hope it opens up more ideas for those who are more familar with n-grams and similar methods.  My earl failures here: using fingerprint verification / cache and top-K truncation.  These turned out to be the wrong approach. 

## What didn't work at 12L

500-step ablations on 1×H100 with torch.compile (see methodology caveat above). Lots of methods on the 11L leaderboard at the time were neutral or negative at 12L in SHORT runs:

| Technique | Delta vs baseline | Verdict |
|---|---|---|
| XSA4 (fixed alpha, last 4 layers) | -0.0005 | Noise |
| EMA decay=0.997 | +0.294 roundtrip | Catastrophic |
| Partial RoPE 16/64 + LN Scale | +0.012 | Hurt |
| Label smoothing 0.05 | +0.034 | Hurt |
| Label smoothing 0.10 | +0.068 | Hurt |
| Focal loss gamma=0.5 | +0.007 | Hurt |
| Focal loss gamma=1.0 | +0.019 | Hurt |
| Loss cap 99th percentile | +0.015 | Hurt |
| Loss cap 95th percentile | +0.123 | Catastrophic |
| Entropy-hump weighting (c=0.4) | +0.039 | Hurt |
| Entropy-hump weighting (c=0.3) | +0.122 | Catastrophic |
| Per-layer slope ramp (0.85 to 0.99) | -0.001 | Noise |
| Hyperbolic embedding reg (λ=0.01) | +0.107 | Catastrophic |
| Hyperbolic embedding reg (λ=0.001) | +0.006 | Hurt |
| STE QAT | NaN | Diverged |
| SiLU² | +0.000 | Noise |
| OHEM (hard example replay) | +0.108 | Catastrophic |

Our read: 12L isn't capacity-starved, it's data-hungry. Regularization and capacity-focusing techniques are solving a problem 12L doesn't have. The things that really helped were gradient flow (activation) and attention efficiency (learned XSA). More data and longer warmdown are where the remaining BPB lives. My search for step-time gains will continue.

## Training curve

```
Step 1000:  1.4456
Step 2000:  1.3784
Step 3000:  1.3447
Step 4000:  1.3215
Step 5000:  1.3046
Step 6000:  1.2965
Step 7000:  1.2913
Step 8000:  1.2862
Step 9000:  1.2836
Step 10000: 1.2804 (plateau begins)
Step 12000: 1.2790
Step 14000: 1.2742
Step 16000: 1.2707
Step 18000: 1.2712 (warmdown starts at 18600)
Step 19000: 1.2626
Step 20000: 1.2319
Step 21000: 1.1928 (SWA applied at 20450)
Step 21600: 1.1707
Roundtrip:  1.1601 (post int5/int6 + rANS, sliding stride=64)
```

The warmdown dropped 0.1005 BPB in 3000 steps after the model sat on a plateau for ~8000 steps. The quant roundtrip cost 0.0106 BPB.
I immediately think, if I could just run this longer....  But it's more nuanced than that of course.  Regardless, the warmdown period has one heck of a slope - obviously a place to concentrate more investigation.  This also falls into situations that happen outside of language models during training - and I have a series of things to start messing with here next from my usual tricks bag.  I hope I can get some extra compute time even on 1xH100 pods to keep going with longer runs as I'm not sure anything at 12L is gonna show up on cheap 500 step ablations like I was doing - this arch just wants to run further before stopping.

## What's next

- Longer WARMDOWN (the plateau-to-drop pattern)
- 8×H100 3-seed validation for record submission  (anyone with runpod credits still reading?  my previous PR was the onne from the form, I'll link it again shortly)
- Several eval-time augmentation methods we haven't had compute to test yet
- Some training-time ideas that specifically target what 12L seems to respond to
- Better understanding of the rANS compression variance across training methods - what if I can find what sort of training makes weights that are the most compressable to it.  And a long shot: Enforce that during training (weight penalization, something) and still not kill BPB?  Perhaps we can buy another few hundred K for more bells and whistles, broader MLP layers (find the next value UP the size curve that interacts well with XSA), etc...
- The n-gram CMS stack on this checkpoint (expect improvements of -0.005 to -0.01 at eval time)

Should the RunPod credit granters see this, hi! I could use some more time. I ran everything on 1×H100 extrapolating the 8×H100 token budget but can't really swing more testing on 8 right now and there are several angles I'd like to investigate. I'll get a more structured writeup of the full ablation history as we progress. Happy to be exploring the higher layer count and odd compression algo's even if the first results aren't going to top the leaderboard today.

## Reproduction

```bash
bash prepare.sh

DATA_PATH=./data/datasets/fineweb10B_sp1024 \
  NUM_LAYERS=12 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2.8125 \
  TIE_EMBEDDINGS=1 USE_FUSED_QKV=1 BIGRAM_VOCAB_SIZE=0 \
  LEAKY_RELU2=1 LEAKY_SLOPE=0.95 \
  XSA_LAST_N=12 SOFT_XSA=1 \
  PRUNE_PCT=0.10 \
  ITERATIONS=21600 WARMDOWN_ITERS=3000 WARMUP_STEPS=20 \
  TRAIN_BATCH_TOKENS=262144 TRAIN_SEQ_LEN=2048 \
  SWA_ENABLED=1 SWA_START_FRAC=0.4 SWA_EVERY=50 \
  MUON_BACKEND_STEPS=5 MUON_MOMENTUM=0.99 \
  TORCH_COMPILE=1 SEED=42 USE_RANS=1 \
  MAX_WALLCLOCK_SECONDS=0 \
  python train_gpt.py
```
## contributions

1. Per-tensor adaptive rANS replacing zstd-22 (consistently better, varies by weight distribution)
2. LeakyReLU(0.95)² (new sweep peak, beats PR #885's 0.9)
3. Soft XSA with learned per-head alpha on all layers + position-tiled Triton diagonal kernel with approximate q/k gradients to regain step time from same
4. MLP width × XSA capacity interaction (neat)
5. Bilateral confidence-gated CMS n-gram cache with product mixing
6. Comprehensive 12L ablation results (19 techniques tested, 2 helped)
7. Kernels!  Int8matmul failed to be better, but the work I did for XSA the only way to make the learned kind work without step time explosion
9. Odds and ends (fused QKV, other small items in submission)