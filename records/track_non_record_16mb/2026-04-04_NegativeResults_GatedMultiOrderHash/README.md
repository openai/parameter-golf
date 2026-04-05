# Non-Record Submission: Negative Results from Gated Multi-Order Hash N-grams

This non-record submission documents a small line of experiments that tried to replace the simple `BigramHashEmbedding` prior with a more expressive **multi-order hash n-gram** module using scalar gates and, in earlier runs, per-order hard top-1 routing across hash heads.

The main result is negative but, we think, informative:

- the more complex gated hash-n-gram line did **not** beat the March 25 BigramHash SOTA stack in this 10 minute / 16MB regime
- the main gap showed up **before compression**, not only after GPTQ/pruning
- once systems regressions were fixed, the best simplified variant got very close in pre-quant quality, but still lost after post-training compression

Our best legal run in this line reached:

- `post_ema val_bpb = 1.1347`
- `final_int6_sliding_window_exact val_bpb = 1.11959957`
- `bytes_total = 15,937,956`

This is competitive for a non-record submission, but still behind the March 25 BigramHash reference:

- `post_ema val_bpb = 1.1340`
- `final_int6_sliding_window_exact val_bpb = 1.11437394`

## High-Level Takeaways

1. More expressive candidate-level gating did not help in this regime.
   The original `2,3,4-gram` + `2,2,2` hash design underperformed the simpler baseline even before compression.

2. Hard top-1 over multiple hash heads appears risky.
   Our working hypothesis is that near-zero initialization plus top-1 routing caused early winner-take-all behavior: once one hash slot got a little training signal, it kept winning even when semantics were weak.

3. Tail-loss emphasis did not improve sliding-window relative gain.
   The `roundtrip -> sliding` improvement stayed almost unchanged, while the absolute numbers got slightly worse. In particular, the legal multi-hash run with tail emphasis had almost the same `roundtrip_exact - sliding_exact` gap as the earlier no-tail run, suggesting that suffix-weighted training did not actually buy extra stride-64 specialization in this setup.

4. System fixes mattered, but did not change the final conclusion.
   This experiment originally paid hidden runtime taxes from training-time calibration caching, synchronous shard reloads, and synchronous logging. After fixing these, throughput matched the baseline regime much better, but the model family still did not surpass BigramHash.

5. Collision "noise" may not be purely harmful.
   One interpretation of these results is that stable hash collisions can act as cheap identity features. Aggressive semantic gating may destroy useful fixed pseudo-identifiers instead of removing only harmful noise.

6. For this model family, the largest remaining gap was post-training.
   The best legal run in this line got very close to the March 25 BigramHash SOTA before compression, but still lost a few additional points after GPTQ + selective pruning. We do **not** think this means GPTQ is bad for the challenge in general, since the March 25 SOTA itself uses GPTQ successfully. Rather, our evidence suggests that this particular gated hash-n-gram family was not a great fit for our post-training stack.

## Representative Runs

| Run | Key idea | post_ema bpb | roundtrip exact | sliding exact | total bytes | Notes |
|-----|----------|-------------:|----------------:|--------------:|------------:|-------|
| `run_overcap_multi_hash.log` | `2,3,4-gram`, `2,2,2` hash, no tail emphasis | 1.1376 | 1.14104006 | 1.11738440 | 16,197,853 | Best raw quality in this line, but over the true 16,000,000-byte cap |
| `run_submit_multi_hash_tail64.log` | legal submit-style run, `2,2,2` hash + tail loss (`64`, `2.0`) | 1.1393 | 1.14628762 | 1.12254768 | 15,971,636 | Tail emphasis did not help; fully legal but worse |
| `run_submit_single_hash.log` | systems fixes + `1,1,1` hash + no tail loss | **1.1347** | **1.14309182** | **1.11959957** | **15,937,956** | Best legal run in this line; still behind BigramHash SOTA after compression |

For reference, the March 25 BigramHash SOTA seed-42 run reported:

- `post_ema val_bpb = 1.1340`
- `final_int6_roundtrip_exact val_bpb = 1.13808963`
- `final_int6_sliding_window_exact val_bpb = 1.11437394`
- `bytes_total = 15,984,850`

## What Was Actually Negative?

The original hypothesis was that a richer lexical prior should beat a simple bigram additive prior:

- use `2/3/4-gram` instead of only bigrams
- use multiple hashes per order to provide collision insurance
- use query-conditioned scalar gates to select the semantically right candidate

This did not happen.

The strongest evidence is that even after the line was simplified and the runtime path was cleaned up, the best legal result still came from removing the extra same-order competition (`NGRAM_NUM_HASHES=1,1,1`), and even then it did not beat the simpler BigramHash baseline after compression.

## Negative Result: Tail-Loss Emphasis

We also tried explicitly emphasizing the final tokens during training, motivated by the fact that sliding-window evaluation only scores the newest positions in each window.

In practice this did not help.

- no-tail multi-hash run: `1.14104006 -> 1.11738440` (`delta = 0.02366`)
- tail-emphasized multi-hash run: `1.14628762 -> 1.12254768` (`delta = 0.02374`)

The absolute metrics got worse, while the relative gain from `roundtrip` to `sliding` stayed effectively unchanged.

Our interpretation is that this tail-weighted objective changed the training target, but did not change the model's actual evaluation-time advantage under longer-context sliding evaluation in a useful way.

## Open Question: What Does Hash Embedding Encode?

One of the more interesting lessons from this line is that our original prior may have been backwards.

We started from the assumption that hash collisions primarily inject harmful semantic noise, and that the right thing to do was to route, filter, or gate that noise away as early as possible.

But these results suggest a different possibility:

- a collided hash feature is not just "noise"
- if it is deterministic, it may act as a stable pseudo-identifier
- the model may be using that pseudo-identifier as a cheap feature, even when it is not semantically clean

This raises a useful open question for future work:

**Is hash embedding mainly encoding local knowledge, or is it mainly providing stable identity-like tags that the backbone can exploit?**

If the latter is true, then aggressive candidate-level semantic gating may be removing the very thing that makes hash features useful in the first place. That would help explain why a simpler additive BigramHash prior can outperform a more "semantic" routed n-gram design in this regime.

## Relation to Engram-Style Gating vs LongCat-Style Additive Fusion

This line also made us rethink a broader design question that shows up in public n-gram memory systems.

From public materials, DeepSeek's Engram is explicitly presented as a **retrieval + gated fusion** memory module: it retrieves static n-gram memory and then fuses it back into the hidden state through a context-aware gate.[^engram-readme] In contrast, Meituan LongCat's public model description presents its n-gram layer more simply: the current token and its recent context are mapped to an n-gram embedding, which is then fused with the base embedding / hidden representation as a direct added feature.[^longcat-model]

We do **not** claim a direct historical lineage from Engram to LongCat. But the contrast in public descriptions is striking, and our results line up more with the simpler additive philosophy:

- the more we asked the n-gram branch to make candidate-level semantic decisions, the more fragile training became
- removing same-order hash competition helped
- the remaining gap suggests that a simple stable feature injection may be more useful than an early routing mechanism in this short-budget setting

One plausible reading is that in a compressed, short-training regime, the backbone may be better at deciding how to use a noisy but stable local feature than the n-gram branch is at deciding, on its own, whether that feature is semantically "clean" enough to keep.

## Practical Contributions From This Line

Even though the model family itself did not win, this work still produced a few practical contributions that made experimentation cheaper and easier to debug:

1. Better AR calibration control.
   We added a prompt-bank AR mode, mild repeat penalty, and optional printed generations. This made it much easier to see when autoregressive GPTQ calibration was degenerating, instead of treating AR as a black box.

2. Higher-throughput post-training calibration defaults.
   We raised AR/Hessian batch defaults and made calibration outputs more inspectable, which reduced iteration cost on expensive 8xH100 runs without introducing an obvious quality regression on this line.

3. Faster and more transparent pruning.
   We improved selective-prune search behavior, added progress/debug logging, and allowed bounded early acceptance near the artifact target. This reduced the amount of time wasted in the very tail of post-processing.

4. Lower hidden systems tax during training and post-processing.
   We fixed unnecessary training-time calibration caching when not using `train_cache`, added next-shard prefetch, and made logging optional/asynchronous. These changes materially improved the fairness of throughput comparisons and reduced wasted wallclock on multi-GPU runs.

We think these system and calibration changes are more transferable than the gated n-gram architecture itself.

## Behind the Scenes: How the Hypothesis Evolved

This section is less about claiming a polished final theory, and more about recording the reasoning path that led us to the final interpretation.

1. We started from a "noise reduction" prior.
   The original intuition was that multi-hash and query-conditioned gates should help clean up collision noise: let the model select the semantically right n-gram feature and suppress the wrong ones.

2. Very quickly, the optimization problem looked more important than the static expressivity.
   We spent time thinking through toy cases where several true n-grams shared one hash slot but differed on another. In those thought experiments, pure addition made the shared slot dirty, but still let every private slot keep receiving gradients. Gating looked cleaner in principle, but also created a way for low-frequency true features to be starved.

3. That shifted the question from "how do we denoise collisions?" to "where should selection happen?"
   We considered several intermediate designs:

   - per-hash candidate gates
   - degree-level gates (`2g/3g/4g`) with within-degree addition
   - hard top-1 per order
   - stronger separation between shared vs non-shared embedding regions

   In hindsight, most of these ideas were different ways of managing the same fear: that collision noise written into the residual stream would be irreversible.

4. We then discovered an implementation mismatch that clarified the real issue.
   The gate had briefly been implemented as a per-channel gate rather than a scalar-per-hash gate. Fixing that bug improved conceptual alignment, but did not rescue the line. That was useful in itself, because it suggested the main problem was not one accidental implementation detail.

5. Hard top-1 eventually became the most suspicious part of the design.
   Once we thought through the interaction between very small initialization, same-order multiple hashes, and top-1 routing, a different failure mode became plausible: the model may be selecting early winners before semantics are actually formed, then reinforcing those winners simply because they were selected first.

6. From there, the prior almost inverted.
   Instead of seeing collisions as something that should obviously be removed, we started considering the possibility that stable collisions are useful. A collided hash feature may be semantically messy, but still function as a reproducible pseudo-identifier. In that framing, the problem with our gated design is not merely that it fails to clean noise, but that it may destroy exactly the stable "dirty" features the backbone would have learned to use.

7. That final shift made the negative result easier to interpret.
   The story stopped being "we need a smarter gate" and became closer to "maybe the simple additive prior was right for this regime." That perspective also made the public contrast between Engram-style gated memory and LongCat-style additive fusion feel more relevant to our own experiments.

## Future Experiments Suggested by This Negative Result

This negative result does not make us pessimistic about hash features in general. It mostly makes us more skeptical of **early semantic routing**. If we were to continue this line, the next experiments we would prioritize are simpler and more additive:

1. Smaller or cheaper BigramHash-style priors.
   If hash features are helping primarily as stable pseudo-identifiers, then they may not need as much semantic capacity as we originally assumed. That suggests revisiting whether bigram/hash embeddings can work with lower dimensions, smaller tables, or more aggressive parameter sharing.

2. Random-map or lightly learned hash projections.
   Instead of learning a fully semantic hash branch, we would test whether a random or semi-random projection is already enough to provide useful stable identity features. If performance holds up, that would support the "identity feature" interpretation over the "knowledge store" interpretation.

3. Multi-projection without candidate gating.
   Another direction would be to keep multiple views of the same local context, but remove the routing machinery entirely. For example:

   - multiple projected hash views summed together
   - multiple projected hash views averaged together
   - multiple projected hash views concatenated and linearly mixed once

   The goal would be to let the backbone see several stable local fingerprints without forcing the n-gram module itself to decide which one is semantically correct.

4. Order-level control rather than candidate-level control.
   If some gating is still useful, we would move it upward in the hierarchy: first combine within-order hash views additively, then decide how much `2g`, `3g`, or `4g` should contribute as a whole. This is much less aggressive than asking same-order candidates to compete from the start.

5. Directly testing the "knowledge vs identity" hypothesis.
   A very simple diagnostic program would be:

   - compare learned additive bigram features vs random additive bigram features
   - compare single-view additive features vs multi-view additive features
   - compare high-dimensional semantic projections vs low-dimensional pseudo-identifier projections

   If cheap, noisy, stable features perform surprisingly well, that would be strong evidence that the main value of hash embedding in this regime is not clean local knowledge retrieval, but providing extra reproducible identity structure.

## Implementation Notes

The included `train_gpt.py` is a snapshot from the best legal run in this line (`run_submit_single_hash.log`), not from the earlier over-cap multi-hash run.

Important aspects of the final snapshot:

- `11L / 512d / 8 heads / 4 KV heads`
- `2,3,4-gram` orders with `3072,1536,768` vocab sizes
- `NGRAM_NUM_HASHES=1,1,1`
- `GPTQ_CALIB_SOURCE=ar_prompt_bank`
- printed AR calibration sequences for inspection
- mild AR repeat penalty
- loader next-shard prefetch
- async logging
- selective prune with bounded undershoot acceptance

## Included Files

- `train_gpt.py` — code snapshot for the best legal run in this line
- `run_overcap_multi_hash.log` — strongest raw multi-hash result, but over artifact cap
- `run_submit_multi_hash_tail64.log` — legal multi-hash run with tail emphasis
- `run_submit_single_hash.log` — best legal non-record run from this line
- `results.tsv` — compact metrics table
- `submission.json` — metadata for this non-record submission

[^engram-readme]: DeepSeek Engram README: https://github.com/deepseek-ai/Engram
[^longcat-model]: LongCat-Flash-Lite model page: https://www.longcatai.org/models/flash-lite
