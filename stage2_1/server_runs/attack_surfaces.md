# Parameter Golf Attack Surfaces

This is a patch-first map of the editable surfaces in `parameter-golf`.

The point is not just to list ideas from the survey. The point is to identify:

- where the code can be changed,
- which metric channel it should move,
- and what kind of `val_bpb` win it can plausibly produce.

Primary code targets:

- [train_gpt.py](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/train_gpt.py)
- record reference: [train_gpt.py](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/records/track_10min_16mb/2026-03-19_WarmdownQuantization/train_gpt.py)
- public survey: [community_pr_survey.md](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/stage2/community_pr_survey.md)

## 2026-03-24 Reprioritization

The original `stage2_1` surface map is still useful, but the priority order has changed after the new `stage3` leaderboard intelligence in [stage3/attack_surfaces.md](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/stage3/attack_surfaces.md).

The main shift is:

- full GPTQ or GPTQ-lite is now a first-order deployment bottleneck
- `11L + 2048 seq + 3x MLP + longer warmdown` should be treated as the control profile, not as a speculative child
- LeakyReLU(0.5)^2, EMA, and XSA4 now outrank older first-wave scouts such as NorMuon, OrthoInit, and solo SmearGate
- `XSA-all` is now the more plausible architecture destination, with `XSA4` acting as the current runnable proxy
- curriculum via shard ordering is now a measured training surface, not just a speculative systems idea
- K-LoRA plus Min-NLL is a real eval-time protocol surface, but it should remain separate from no-TTT training attribution
- Partial RoPE and LN Scale are live modern refinement candidates, while VRL remains a deferred structural target

So this document should now be read with the revised `stage2_1` slate in:

- [hypotheses.md](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/stage2_1/hypotheses.md)
- [portfolio.md](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/stage2_1/portfolio.md)
- [patch_specs.md](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/stage2_1/patch_specs.md)

## Metric Channels

There is not one `val_bpb`.

For this repo, there are four distinct score channels:

1. Pre-quant fixed-chunk `val_bpb`
- Produced by the standard validation path in [train_gpt.py#L226](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/train_gpt.py#L226).
- This is the cleanest readout of pure training quality.

2. Post-quant fixed-chunk `val_bpb`
- Produced after round-trip quantization in [train_gpt.py#L1308](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/train_gpt.py#L1308).
- This is where export quality shows up.

3. Post-quant sliding-window `val_bpb`
- Not in the root script today, but implemented in the record reference at [record train_gpt.py#L779](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/records/track_10min_16mb/2026-03-19_WarmdownQuantization/train_gpt.py#L779).
- This is a major eval-policy lane.

4. Post-quant TTT `val_bpb`
- Not present in the current merged root script.
- This now belongs to the "fresh port from records/open PRs" category rather than the "already exposed env surface" category.

Different patch families should be judged on different channels:

- training patches should improve channel `1`, then survive into `2`
- export patches may do little to `1` but should improve `2`
- eval-policy patches can leave `1` and `2` unchanged but improve `3` or `4`
- TTT protocol patches can dominate channel `4` without saying much about the no-TTT model

## Surface A: Evaluation Policy

### A1. Fixed-chunk validation geometry

Target:

- [train_gpt.py#L226](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/train_gpt.py#L226)

Mechanism:

- This path scores each token once with one causal context window.
- It under-uses available long context compared with sliding-window evaluation.

Expected effect on `val_bpb`:

- no training change
- no artifact change
- can materially lower final eval `val_bpb` by giving more usable context at scoring time

Patch families:

- add `EVAL_SEQ_LEN`
- add `EVAL_STRIDE`
- add sliding-window evaluation port from the record script
- add doc-isolated evaluation

Expected strongest effect:

- direct improvement in final benchmark score, not pre-quant learning

Retire if:

- the improvement is tiny after exact replay, or the rule interpretation is questionable

### A2. Sliding-window eval

Reference:

- [record train_gpt.py#L59](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/records/track_10min_16mb/2026-03-19_WarmdownQuantization/train_gpt.py#L59)
- [record train_gpt.py#L779](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/records/track_10min_16mb/2026-03-19_WarmdownQuantization/train_gpt.py#L779)

Mechanism:

- overlap validation windows
- score only the suffix of each window
- let each token see much more left context

Expected effect on `val_bpb`:

- strong positive on final eval `val_bpb`
- no effect on training loss
- no effect on artifact bytes

Strong priors:

- `stride=64` and `stride=256` are both live in the survey
- this is one of the largest known score levers

### A3. TTT-LoRA evaluation

Mechanism:

- adapt rank-`r` low-rank deltas per document at eval time
- convert base-model competence into per-document specialization

Expected effect on `val_bpb`:

- mostly on final competition score
- little or no change to pre-quant fixed-chunk score
- requires a fresh patch port on top of the current merged root, because the root script no longer contains a built-in LoRA TTT path

Patch families:

- tune rank
- tune chunk size
- tune learning rate
- change adapted modules
- reset policy by document or chunk
- K-LoRA
- Min-NLL epoch selection

Retire if:

- eval cost rises a lot with no score lift

### A4. Doc-isolated eval

Targets:

- [train_gpt.py#L805](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/train_gpt.py#L805) for existing doc boundary logic
- [train_gpt.py#L226](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/train_gpt.py#L226) for standard eval path

Mechanism:

- do not let context spill across document boundaries
- align scoring context with natural document structure

Expected effect on `val_bpb`:

- can improve final eval score without touching training
- especially relevant when combined with sliding eval or TTT

## Surface B: Export And Quantization

### B1. Quantizer bitwidth and per-row scaling

Target:

- [train_gpt.py#L328](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/train_gpt.py#L328)
- [train_gpt.py#L349](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/train_gpt.py#L349)

Mechanism:

- quantization changes the deployed weights, not the trained weights
- smaller bitwidth improves artifact budget but increases round-trip error

Expected effect on `val_bpb`:

- often no change in pre-quant score
- large effect on post-quant `val_bpb`
- indirect architecture effect if saved bytes fund more parameters

Patch families:

- int6 export
- mixed int5/int6 export
- per-group bitwidth
- int6 stored in int8 containers
- scale dtype and clip percentile changes

Reference implementation:

- int6 export in [record train_gpt.py#L326](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/records/track_10min_16mb/2026-03-19_WarmdownQuantization/train_gpt.py#L326)

### B2. Passthrough rules

Target:

- [train_gpt.py#L320](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/train_gpt.py#L320)

Mechanism:

- selectively keep quant-sensitive tensors in higher precision
- trade bytes for much lower quantization error on the most fragile tensors

Expected effect on `val_bpb`:

- positive effect on post-quant `val_bpb`
- zero or negligible effect on pre-quant score

Patch families:

- fp16 tied embedding passthrough
- late-K fp16 passthrough
- explicit keep-float policy for `q_gain`, `resid_mix`, `skip_weights`
- per-name or per-size passthrough

Reference:

- fp16 embedding and late-K in [record train_gpt.py#L374](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/records/track_10min_16mb/2026-03-19_WarmdownQuantization/train_gpt.py#L374)

### B3. Compression backend

Target:

- [train_gpt.py#L1312](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/train_gpt.py#L1312)

Mechanism:

- better compression buys more artifact headroom at the same weights

Expected effect on `val_bpb`:

- no direct effect on model quality
- indirect positive effect because it permits better export rules or larger models under the cap

Patch families:

- zstd
- grouped LZMA
- layout-aware serialization

## Surface C: Training Dynamics

### C1. Muon optimizer law

Target:

- [train_gpt.py#L119](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/train_gpt.py#L119)

Mechanism:

- changes the geometry of matrix updates
- affects both early convergence and final compressibility of weights

Expected effect on `val_bpb`:

- can improve pre-quant score
- can also improve post-quant score if weights become more quant-friendly

Patch families:

- NorMuon
- Muon weight decay
- Muon backend step count
- adaptive momentum
- cautious or decoupled decay

Survey tie-in:

- this is one of the five most repeated missing techniques

### C2. Warmup and warmdown schedule

Targets:

- [train_gpt.py#L1156](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/train_gpt.py#L1156)
- [train_gpt.py#L1253](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/train_gpt.py#L1253)

Mechanism:

- in a 10-minute budget, bootstrap behavior matters a lot
- flatter late weights can improve quantized score even when pre-quant score is flat

Expected effect on `val_bpb`:

- can improve both pre-quant and post-quant score
- often larger effect on post-quant via smoother weight geometry

Patch families:

- longer warmdown
- different LR decay shapes
- SWA during warmdown
- momentum warmup retunes

### C3. Gradient clipping

Target:

- [train_gpt.py#L1262](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/train_gpt.py#L1262)

Mechanism:

- prevents unstable spikes
- can help longer context or deeper stacks survive

Expected effect on `val_bpb`:

- usually small positive or neutral
- mostly a stability helper for other branches

### C4. STE QAT

Best insertion point:

- around linear weight use in [train_gpt.py#L516](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/train_gpt.py#L516) or module-level forward paths

Mechanism:

- train the model to tolerate the deployed quantizer rather than applying quantization only after training

Expected effect on `val_bpb`:

- small effect on pre-quant
- potentially large reduction in post-quant gap

Survey tie-in:

- this is another one of the five most repeated missing techniques

## Surface D: Architecture And Parameter Allocation

### D1. Depth, width, MLP budget

Targets:

- [train_gpt.py#L63](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/train_gpt.py#L63)
- [train_gpt.py#L616](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/train_gpt.py#L616)

Mechanism:

- reallocate fixed parameter budget across representation depth and feedforward capacity

Expected effect on `val_bpb`:

- direct effect on pre-quant score
- indirect effect on post-quant score depending on export viability

Patch families:

- `NUM_LAYERS=10/11`
- explicit `MLP_HIDDEN`
- width-depth rebalance
- MLP 3x-like branches

Important constraint:

- this is only meaningful together with export/compression changes if bytes are already tight

### D2. SmearGate

Best insertion point:

- embedding path around [train_gpt.py#L714](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/train_gpt.py#L714)

Mechanism:

- add a cheap inductive bias for adjacent-token dependence before the transformer layers spend capacity reconstructing it

Expected effect on `val_bpb`:

- likely direct pre-quant improvement
- small byte cost
- should survive post-quant well because the parameter count is tiny

Survey tie-in:

- one of the strongest repeated missing architecture features

### D3. BigramHash

Best insertion points:

- token embedding path
- tokenizer/data metadata path, since the data tooling already exposes recommended bigram vocab sizing

Mechanism:

- inject cheap token-pair memory into the model
- offload easy local compositional patterns from the main transformer

Expected effect on `val_bpb`:

- direct pre-quant gain
- some byte cost, so compare against a byte-matched control

### D4. Initialization

Targets:

- [train_gpt.py#L706](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/train_gpt.py#L706)
- [train_gpt.py#L587](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/train_gpt.py#L587)
- [train_gpt.py#L647](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/train_gpt.py#L647)

Mechanism:

- in a short-horizon run, better initialization can matter as much as a mild optimizer change

Expected effect on `val_bpb`:

- mostly pre-quant training-quality improvement
- may also help post-quant score if it finds smoother weights

Patch families:

- OrthoInit
- muP-style output scaling
- overtone spectral init
- phase-transition `resid_mix` init

Survey tie-in:

- OrthoInit is one of the five repeated missing techniques

### D5. Existing skip/reuse structure

Target:

- [train_gpt.py#L683](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/train_gpt.py#L683)

Mechanism:

- the baseline already has encoder/decoder skip reuse, so "U-Net" is not a greenfield feature here

Expected effect on `val_bpb`:

- improvements here likely come from skip weighting or initialization, not from adding the whole family from scratch

## Surface E: Tokenizer And Data Representation

### E1. Vocabulary size

Targets:

- [train_gpt.py#L63](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/train_gpt.py#L63)
- [train_gpt.py#L1037](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/train_gpt.py#L1037)

Mechanism:

- changes embedding parameter budget
- changes tokenization efficiency in bytes-per-token
- changes the shape of the validation metric itself, because BPB depends on tokenizer behavior

Expected effect on `val_bpb`:

- can move score a lot either way
- much harder to attribute than export/eval changes

Best use:

- separate branch after more local surfaces are settled

### E2. Tokenizer family

Targets:

- [train_gpt.py#L44](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/train_gpt.py#L44)
- [data/download_hf_docs_and_tokenize.py](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/data/download_hf_docs_and_tokenize.py)

Mechanism:

- changes both token-count efficiency and parameter allocation

Expected effect on `val_bpb`:

- potentially large, but high implementation and attribution cost

## Surface F: Throughput And Wallclock Conversion

### F1. Step-time reduction

Targets:

- compilation and warmup path in [train_gpt.py#L1167](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/train_gpt.py#L1167)
- attention path in [train_gpt.py#L604](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/train_gpt.py#L604)
- validation cadence in [train_gpt.py#L1208](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/train_gpt.py#L1208)

Mechanism:

- more steps inside the same 600 seconds can beat a slightly better per-step recipe

Expected effect on `val_bpb`:

- indirect but real
- improves pre-quant score by buying more optimization steps

Patch families:

- FA3 if it really beats current SDPA
- lower validation overhead
- adaptive Muon if it reduces `ms/step`

### F2. Wallclock-aware stopping and schedule shape

Target:

- [train_gpt.py#L1156](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/train_gpt.py#L1156)

Mechanism:

- maps elapsed time into LR decay
- under a hard budget, schedule shape is a score surface, not just a training detail

Expected effect on `val_bpb`:

- can improve pre-quant and post-quant score by spending the limited horizon better

## Highest-Leverage Patch Families By Gain Channel

### Direct final-score levers

- sliding-window eval
- doc-isolated eval
- TTT-LoRA tuning

These can cut `val_bpb` without changing training at all.

### Post-quant gap levers

- int6 or mixed-bit export
- fp16 embedding passthrough
- late-K passthrough
- STE QAT
- SWA
- Muon weight decay

These are the strongest candidates when pre-quant quality is already respectable but export loses too much.

### Pre-quant training-quality levers

- NorMuon
- OrthoInit
- SmearGate
- BigramHash
- depth/MLP reallocation

These should first prove they move channel `1`.

### Throughput-to-quality levers

- FA3 if it really lowers `ms/step`
- lighter validation cadence
- low-overhead optimizer simplifications

These matter only if more steps can be bought within the cap.

## Practical Reading Of The Updated Survey

The survey's "every submission beating us uses at least 3 of these 5" finding is useful because those five techniques span three different gain channels:

- training quality: `OrthoInit`, `SmearGate/BigramHash`, `NorMuon`
- export robustness: `STE QAT`, `MuonWD`, `SWA`
- stackability: they are mostly orthogonal enough to compose

That means the correct patch portfolio is not "pick any three."

It is:

1. one eval-policy move
2. one export-gap move
3. one training-quality move
4. optionally one structural move

That gives a real chance to move the final score, instead of only retuning one neighborhood.
