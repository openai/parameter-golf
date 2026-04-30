# JEPArdy! Non-Record Submission - JEPA + Leader-Stack - val_bpb 1.1230

This record captures our JEPA-focused non-record submission and the slightly improbable chain of decisions that produced it.

## Results (`8x H100 SXM`)

| Seed | Raw over-cap bpb | Under-cap exact bpb | Packaged bytes total |
|------|------------------|---------------------|----------------------|
| `1337` | `1.12128254` | `1.12271348` | `15,922,511` |
| `42` | `1.12176502` | `1.12337480` | `15,910,983` |
| `2026` | `1.12157046` | `1.12300861` | `15,809,375` |
| **Mean** | — | **1.12303230** | — |
| **Std** | — | **0.00033130** | — |

It beats the [March 17 Naive Baseline](https://github.com/openai/parameter-golf/tree/main/records/track_10min_16mb/2026-03-17_NaiveBaseline) (`val_bpb: 1.22436570`) by a wide margin, and it does so with an ablation saga that still lets us say, with a reasonably straight face, that JEPA itself is doing useful work here rather than merely riding shotgun inside a stronger stack.

## Why We Believe JEPA Is Doing Real Work
This submission reflects the way we ended up thinking about JEPA after a week of arguing with it: not as a total replacement for the rest of the stack, but as a component that has to earn its place inside a coherent recipe.

I had not built with JEPA before and was skeptical that it would contribute anything useful to this model family. So the method was deliberately antagonistic. JEPA had to win in a cleaner setting before it was allowed into the stronger stack, and then it had to win again there under longer-horizon validation.

The first proof came in a byte-level isolation lane built against matched controls on `8xH100`. There, JEPA gave a real if modest improvement:
- control: `1.36692884`
- `JEPA_LOSS_WEIGHT=0.10`: `1.36182961`

After adding full-model EMA, JEPA still won:
- control: `1.36487466`
- `JEPA_LOSS_WEIGHT=0.10`: `1.36169919`

Only after that did we move into the stronger leader-family stack with `TTT_ENABLED=0`, where the question changed from “does JEPA help at all?” to “how much JEPA can this recipe absorb before it becomes self-defeating?” The short screens made several settings look plausible. The longer `600s` runs were less easily charmed:
- control: `2.22466503`
- `weight:0.05`: `2.22707166`
- `weight:0.15`: `2.29532844`
- `weight:0.10`: `2.19728869`

That back-and-forth was the point. A weaker weight looked harmless without helping much. A stronger weight looked briefly promising and then turned into a liability. `0.10` was the Goldilocks setting that survived contact with the actual budget, so that is the one we promoted to the final `8xH100` candidate.

This hybrid "symphony" of multiple instruments tuned and retuned to each other is what we should have expected to see, based on [Monsieur LeCun's original JEPA paper](https://openreview.net/pdf?id=BZ5a1r-kVsf) -- though of course LeCun put it in terms of the modular structure of the mammalian brain. More recent JEPA work makes the same point from the engineering side: the [2026 LeWorldModel paper](https://arxiv.org/abs/2603.19312v2) says existing JEPA methods often depend on multi-term losses, EMA, pre-trained encoders, or auxiliary supervision to stay stable; that was the inspiration for our initial decision to add EMA to the isolation lane and the leader-stack translation lane. 

All this to say: modern JEPA stacks are usually hybrids of one sort or another. The interesting question is not whether JEPA does all the work by itself, but whether it changes the behavior of the whole system in a fruitful way that survives ablation and scale-up, and that's our claim here.

## What We Tried And What Failed

This JEPA result did not arrive as a single clean idea. It came out of several dead ends, some misleading short screens, and a wise decision to keep explicit logs. In the hope that our failed experiments might prevent fellow travelers from tripping over the same rocks, we're including further details here, and I'd be happy to expand on any of them if asked.

**Key failed or inconclusive directions:**
- exploratory byte-level JEPA hybrid: early positive screens, but not competitive at longer horizon
- minimal-diff isolation lane before the stronger MLP transplant: JEPA did not help
- tiny learned predictor head alone on the weak isolation lane: did not produce a robust win
- leader-stack short-screen weight `0.15`: looked best at `180s`, then lost badly at `600s`
- simple stronger-LZMA recompression: did not solve the artifact overage

**Key things that helped:**
- `LeakyReLU(0.5)^2`
- full-model EMA for export/eval
- choosing `JEPA_LOSS_WEIGHT=0.10` by longer-horizon validation, not by the cheapest screens alone.* 
- storage-only export change from the saved checkpoint

\*Actually, really interestingly, we homed in on the 0.10 after a pretty simple "0.05 too little JEPA, 0.15 too much JEPA" screen, where Claude and I concluded "ah, clearly 0.10 is the sweet spot -- the *ideal* amount of JEPA -- right in the middle." But then I said to Claude, "do we have any strong antecedent theoretical evidence that the distribution of JEPA losses would just flatten out beyond 0.15? Why not suppose that it changes shape multiple times in the interval up to 1?" After giving me some side-eye, Claude relented and we did a sweep at every 0.05 interval, results below, indicating that the distribution is indeed not monotonic, and I was "Absolutely Right™".
|jepa_weight|val_loss  |val_bpb   |
|-----------|----------|----------|
|0.65       |1.29404186|1.86690778|
|0.15       |1.29404957|1.86691890|
|0.55       |1.29478845|1.86798488|
|0.05       |1.29605058|1.86980574|
|1.00       |1.29631343|1.87018496|
|0.45       |1.29670615|1.87075153|
|0.95       |1.29781500|1.87235127|
|0.90       |1.29793175|1.87251969|
|0.50       |1.29838871|1.87317895|
|0.10       |1.29852931|1.87338179|
|0.80       |1.29902096|1.87409110|
|0.25       |1.29983709|1.87526852|
|0.40       |1.29986623|1.87531057|
|0.30       |1.30032384|1.87597075|
|0.70       |1.30298896|1.87981571|
|0.75       |1.30324671|1.88018757|
|0.85       |1.30369632|1.88083622|
|0.20       |1.30434502|1.88177209|
|0.35       |1.30475990|1.88237064|
|0.60       |1.30593691|1.88406871|
|0.00       |1.31746854|1.90070533|


## Final Candidate Snapshot

Training model:
- Base stack: March 23 leader-family stack descended from the [March 23 leader submission](https://github.com/openai/parameter-golf/tree/main/records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon)
- Implementation: the record-local `train_gpt.py` included in this folder
- JEPA setting: `JEPA_LOSS_WEIGHT=0.10`
- Tokenizer/data: `fineweb10B_sp1024` + `fineweb_1024_bpe.model`
- Eval mode for this candidate: `TTT_ENABLED=0`

Storage-only export pass:
- Implemented directly in the record-local `train_gpt.py`
- Changes relative to the original March 23 export:
  - remove duplicate top-level JEPA alias weights (`jepa_in.weight`, `jepa_out.weight`) from the exported state
  - quantize `attn`, `mlp`, `embed`, and `other` floating tensors with the int6 path
- This pass starts from the trained checkpoint only. It does not retrain, does not use validation tokens to change weights, and does not introduce any new data source. It is compression, not absolution.

## Legality and Compliance
- [x] `train_gpt.py` in this folder is the self-contained submission script.
- [x] All three packaged artifacts are under `16,000,000` bytes.
- [x] Training fits within the `10` minute budget on `8xH100`.
- [x] Evaluation fits within the additional `10` minute budget.
- [x] Validation uses the full validation shard set.
- [x] BPB accounting uses the official SentencePiece LUT path.
- [x] Final scoring is fixed-model, single-pass sliding-window evaluation with `TTT_ENABLED=0`.
- [x] JEPA is a training-time auxiliary loss only; no JEPA target is computed from validation tokens during scoring.
- [x] The storage-only export pass starts from the trained checkpoint only and does not consume training or validation data.
- [x] No network calls or external side information are used during training or evaluation.
