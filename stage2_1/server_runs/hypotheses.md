# Parameter Golf Hypotheses

This revision updates `stage2_1` using the 2026-03-23 frontier read in [stage3/attack_surfaces.md](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/stage3/attack_surfaces.md).

The important change is strategic, not cosmetic: the old `stage2_1` set overweighted older public winners such as NorMuon, OrthoInit, and solo SmearGate. The current frontier says the dominant no-TTT stack is now:

- 11 layers
- stronger deployment quantization, especially GPTQ
- LeakyReLU(0.5)^2
- EMA
- XSA4
- Muon weight decay
- longer warmdown and stronger base schedule

That means `stage2_1` should no longer ask "which old missing trick beats our root baseline?" It should ask "which missing frontier mechanisms still explain the 1.1631 -> 1.12 gap once the base recipe is moved closer to the frontier?"

## 2026-03-24 Addendum

After the latest upstream PR scan, three adjustments matter:

- `XSA4` is still the right runnable proxy, but `XSA-all` is now the higher-priority end-state hypothesis.
- curriculum is now a real lane because shard ordering has measured leaderboard impact under the 600-second cap.
- K-LoRA plus Min-NLL epoch selection should be treated as a separate eval-time adaptation protocol, not as evidence for the no-TTT base model.

## Objective

Optimize final `val_bpb` under:

- `600s` wallclock
- `16MB` artifact cap
- real post-export deployment score
- small budget for full 8xH100 confirmations

Updated public anchors:

- our current: `1.1631`
- merged SOTA: `1.1233`
- open no-TTT frontier: `1.1163` 3-seed mean, `1.1158` best seed
- open TTT frontier: `0.6864` claimed, but from a closed non-merged PR and therefore a separate provisional signal

## Revised Priors

Strong positive priors:

- full GPTQ or GPTQ-lite export is now a first-order bottleneck, not a late polish step
- 11L/MLP3x/seq2048/warmdown3500 style env profile should be the control, not the child branch
- LeakyReLU(0.5)^2 is a must-test free lunch
- EMA has displaced SWA as the more standard smoothing mechanism
- XSA is now a central architecture/context mechanism, not an optional curiosity
- curriculum via shard ordering is now a real training-budget allocation hypothesis

Weaker priors than before:

- NorMuon
- OrthoInit + muP
- solo SmearGate
- solo BigramHash
- FA3 as the default throughput answer
- label smoothing and MTP as mainline first-wave screens

## Frontier Families

### H301: Frontier-Aligned Env Base

- Mechanism: move the control itself closer to the frontier before testing code patches.
- Core settings:
  - `NUM_LAYERS=11`
  - `MLP_MULT=3`
  - `TRAIN_SEQ_LEN=2048`
  - `TRAIN_BATCH_TOKENS=786432`
  - `MATRIX_LR=0.025`
  - `SCALAR_LR=0.025`
  - `TIED_EMBED_LR=0.035`
  - `MUON_MOMENTUM=0.99`
  - `MUON_MOMENTUM_WARMUP_START=0.92`
  - `MUON_MOMENTUM_WARMUP_STEPS=1500`
  - `GRAD_CLIP_NORM=0.3`
  - `WARMDOWN_ITERS=3500`
  - `VAL_LOSS_EVERY=4000`
- Why it matters: if the base recipe is too weak, patch ranking is misleading.

### H302: Full GPTQ Export

- Mechanism: replace simple clip-and-round quantization with Hessian-aware post-training quantization.
- Primary lane: deployment/export
- Expected signal: large improvement in post-export score on the same checkpoint
- Why promoted: biggest single gap-closing mechanism in the new leaderboard evidence

### H303: GPTQ-Lite Clip Search

- Mechanism: lightweight per-row clip-percentile search as a cheaper export fallback.
- Primary lane: deployment/export
- Expected signal: smaller but real post-export gain without full GPTQ complexity

### H304: LeakyReLU(0.5)^2

- Mechanism: replace `relu(x).square()` with `leaky_relu(x, 0.5).square()`.
- Primary lane: training dynamics
- Expected signal: modest but consistent pre-quant gain with almost zero implementation risk

### H305: EMA(0.997)

- Mechanism: maintain a shadow exponential moving average and export EMA weights instead of raw final weights.
- Primary lane: training-to-deployment bridge
- Expected signal: small but consistent post-export gain, better than relying on SWA alone

### H306: XSA4

- Mechanism: add cross-sequence KV context on a limited number of layers, starting with the last four.
- Primary lane: architecture/context
- Expected signal: clear pre-quant gain that survives deployment

### H306b: XSA-All

- Mechanism: push XSA beyond the last four layers and treat it as the actual architecture destination if deployment and runtime headroom exist.
- Primary lane: architecture/context
- Expected signal: larger no-TTT gain than `XSA4`, but with higher interaction and artifact risk
- Status: deferred child of `H306`, not the first runnable screen

### H307: Muon Weight Decay 0.04

- Mechanism: keep matrix weights tighter and more deployment-friendly during training.
- Primary lane: training plus export robustness
- Expected signal: better deployment score or smaller quant gap
- Status: still live, but now it is a helper in a stronger stack rather than the lead story

### H308: VRL

- Mechanism: carry a learned value residual from early layers into later ones.
- Primary lane: architecture
- Expected signal: real quality gain without needing the old BigramHash-heavy path
- Status: deferred target, not in the current runnable Pack B

### H309: Partial RoPE 16/64 Plus LN Scale

- Mechanism: improve deep-context geometry with low-cost refinements that show up repeatedly in the merged SOTA.
- Primary lane: architecture refinement
- Expected signal: smaller but stackable gain on top of XSA and 11L
- Note: this is intentionally treated as a refinement bundle, not a mechanism-level attribution claim

### H310: Sliding Eval And Doc-Isolated Eval

- Mechanism: late-stage scoring lift on the exact same checkpoint.
- Primary lane: evaluation policy
- Expected signal: lower final score without retraining
- Status: still strong, but now should be paired with the stronger deployment lane instead of treated as the whole story

### H311: Funded Architecture Children

- Mechanism: once deployment saves bytes or improves robustness, spend that budget on VE128, BigramHash, or similar helpers.
- Primary lane: architecture under budget reallocation
- Expected signal: only fair after GPTQ/zstd/fp16 rules are real

### H312: TTT Lane

- Mechanism: score-first or multi-pass legal test-time training.
- Primary lane: evaluation-time compute
- Status: separate game
- Rule: do not let TTT hide whether the no-TTT model is actually strong

### H313: Curriculum Ordering

- Mechanism: reorder training shards so the fixed 600-second budget sees more leverage-per-step data earlier.
- Primary lane: data/training
- Expected signal: modest but real pre-quant and post-quant gain under equal wallclock
- Why promoted: open `#650` makes this a measured lane instead of a vague systems story
- Rule: compare on equal wallclock, not equal nominal step count

### H314: K-LoRA And Min-NLL TTT

- Mechanism: adapt low-rank deltas on K as well as Q/V and select the best epoch per document rather than blindly taking the last one.
- Primary lane: evaluation-time adaptation
- Expected signal: potentially huge final-score lift on the TTT metric, with little implication for no-TTT base quality
- Status: separate game from `stage2_1` no-TTT chasing
- Implementation note: current merged root does not expose a built-in TTT path, so this is a fresh port target rather than a dormant env-gated mutation
- Rule: do not mix this lane into Pack B or treat it as evidence that the trained model itself improved

## Demoted Families

These are not deleted forever. They are no longer first-wave `stage2_1` priorities.

- NorMuon
- OrthoInit + muP
- solo SmearGate
- solo BigramHash
- FA3-first throughput bets
- label smoothing
- MTP auxiliary loss

## Pack Implications

The revised pack structure should be:

- deployment lane centered on GPTQ, sliding eval, and deployment rules
- training lane centered on frontier base control plus LeakyReLU^2, EMA, XSA4, MuonWD, a GPTQ-lite export proxy, and refinement helpers
- deferred lane for `XSA-all` and curriculum once the first training winners are known
- TTT protocol lane kept separate from no-TTT ranking
- child stacks only after one deployment winner and one training winner are both real

The main design rule is:

- the control must already look like a plausible 2026 frontier base
- patches should explain the remaining gap from that base, not the gap from an outdated baseline
