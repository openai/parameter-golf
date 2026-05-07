# Mercury in Retrograde*

**This is a non-record submission for the Parameter Golf request for text diffusion.** Instead of taking a leading model and then sprinkling a little text diffusion on top for fun, we aimed to make text diffusion unmistakably present in the training objective and in the evaluation story, then to report, with as much honesty as possible, what happened when that idea was squeezed into a 16MB artifact and a 10-minute training budget. The result was a failure for `val_bpb` but I'm calling it a win for educational golfing. The model acquired a real diffusion-like interface. It learned to revise whole spans in parallel. It learned to infill. It learned a speed-quality knob. It also learned, with a kind of stubborn little diligence, how to be wrong at extremely high throughput.

> *Mercury retrograde is an optical illusion in which the planet appears to move backward; it is superstitiously associated with communication breakdowns, travel delays, and technological glitches--the perfect metaphor for a model that seems to be going backward on the one metric the challenge actually uses to keep score.

## Why Text Diffusion Was Worth Trying

Autoregressive language models do one thing with almost tyrannical consistency: they predict the next token, then the next, then the next, each one contingent on the whole left context and inaccessible from the right. This is a fantastically strong way to model text distributionally, but it is also sequential by construction. Diffusion models tempt us with another picture. Instead of writing the sentence one token at a time, they begin from corruption and iteratively denoise, revising many positions at once, sometimes all of them. In images this has already redrawn the map.

In text it has been harder, but not for lack of serious attempts: we took some inspiration from e.g. [Diffusion-LM](https://arxiv.org/abs/2205.14217) which made the case that diffusion objectives could be useful for controllable text generation. Some [later papers](https://arxiv.org/abs/2410.18514) [suggested](https://arxiv.org/abs/2602.15014) that text diffusion is not merely a curiosity, but something that may become more competitive with the right scaling laws, formulations, and systems.

A diffusion language model should, in principle generate multiple tokens in parallel, revise its own mistakes, handle infill and other any-order generation tasks naturally, expose an explicit speed-vs-quality tradeoff through the number of denoising rounds, and perhaps, if the stars are properly aligned, do all this much faster than ordinary left-to-right decoding.

That is also roughly how [Inception’s Mercury announcement](https://www.inceptionlabs.ai/blog/introducing-mercury) presents the idea: a coarse-to-fine model that modifies multiple tokens in parallel, can serve as a drop-in replacement for Transformer-based LLM infrastructure, and gets much of its practical force not only from modeling but from the surrounding systems stack. (That last clause matters a lot.)

## Why It Probably Does Not Work Here

The short version is that Parameter Golf is a peculiarly hostile habitat for text diffusion.

This challenge rewards compression quality on FineWeb through exact `val_bpb`, under an extremely tight artifact budget and a brutal training wall clock. This environment is almost offensively well-suited to small autoregressive models. AR spends nearly all of its capacity on one job: estimating the next-token distribution as faithfully as possible. A compact diffusion model, by contrast, has to learn something more baroque. It must model text well enough to know what should go in a corrupted position, while also learning how to denoise under a corruption process, while also staying stable under iterative refinement, while also paying for the fact that its natural interface is no longer the exact objective by which the challenge ranks submissions.

This becomes worse, not better, when the model is tiny. A large diffusion model may have enough slack capacity to learn both language statistics and denoising dynamics and maybe even some graceful error correction. A small model, trained briefly, tends instead to discover a grimly economical compromise: guess common tokens, repeat punctuation, repair locally if possible, and otherwise retreat into the high-frequency regions of the distribution. Which is, to be fair, a distributional strategy of a sort. It just is not the sort you want.

There is also the systems issue. Inception's public framing is not merely "diffusion is better." It is "diffusion plus an optimized inference engine plus batching plus kernels plus scale is fast enough to matter." We did not reproduce that stack here. We reproduced, on purpose, the modeling gesture: parallel coarse-to-fine text denoising inside the challenge’s compact Transformer setup. That makes this a useful scientific negative result rather than a failed product launch. It isolates the part we could test.

## What We Actually Tried

### Phase 1: naive hybrid text diffusion

We began with a modest premise: keep the ordinary causal LM machinery, then add text-diffusion-style corruption objectives and see whether some measured amount of denoising helps without wrecking BPB.

**Early 180-second ladder:**

| Variant | Description | val_bpb |
| --- | --- | ---: |
| `td_control` | plain causal LM | 2.0027 |
| `td_span15` | 15% span corruption | 2.1725 |
| `td_span30` | 30% span corruption | 2.1831 |
| `td_prog15` | progressive corruption to 15% | 2.1752 |
| `td_prog30` | progressive corruption to 30% | 2.1893 |

The immediate lesson was not subtle. Even gentle diffusion-style corruption moved BPB in the wrong direction. That did not kill the project, but it did force a conceptual fork. If we optimized only for BPB, the best "text diffusion" model would simply be the one with the least text diffusion in it, which is not what we wanted to chase here.

### Phase 2: late auxiliary losses and corruption sweeps

We then tried to make diffusion less destructive by deferring or demoting it. The denoising objective was pushed later in training, or weakened, or made more selective.

**Late auxiliary ladder:**

| Variant | Description | val_bpb |
| --- | --- | ---: |
| `td_control` | plain causal LM | 1.9997 |
| `td_late02` | 2% late diffusion weight | 2.0437 |
| `td_late05` | 5% late diffusion weight | 2.0485 |
| `td_late05p` | 5% late diffusion, progressive | 2.0489 |
| `td_late10` | 10% late diffusion weight | 2.0489 |

**Corruption sweep:**

| Variant | Description | val_bpb |
| --- | --- | ---: |
| `td_control` | plain causal LM | 2.0035 |
| `td_cons05` | consistency-style late objective | 2.0477 |
| `td_hyb05` | 5% hybrid corruption | 2.0515 |
| `td_hyb05m` | 5% hybrid, mask-heavier | 2.0503 |
| `td_unif05` | 5% uniform corruption | 2.0601 |
| `td_unif10` | 10% uniform corruption | 2.0690 |

These runs were important precisely because they sort of worked, in the narrow sense that they did not completely explode. But that turned out to be the problem. They were not diffusion-native enough to be interesting. They preserved more of the AR model because they were, in spirit, still AR models with a denoising side hustle.

### Phase 3: make diffusion explicit, then see what survives

At that point the project changed from "can we smuggle in a little diffusion without hurting BPB" to "what is the most informative compact text-diffusion submission we can make, even if the BPB is worse."

That led to the Mercury-style branch: a direct denoising mode, mixed continuation/infill masking, self-conditioning, progressive hybrid corruption, and parallel refinement metrics logged on purpose. If the model was going to lose on the main challenge metric, it should at least lose in a way that teaches something specific.

**Initial Mercury screens, 180 seconds on 1xH100:**

| Variant | Description | val_bpb |
| --- | --- | ---: |
| `mercury_mask25` | pure masked denoising | 12.9566 |
| `mercury_hybrid25` | hybrid corruption, 25% | 3.5492 |
| `mercury_uniform50` | uniform corruption, 50% | 3.5424 |
| `mercury_hybrid50p` | progressive hybrid to 50% | 3.5412 |
| `mercury_auxlate` | AR-heavy late auxiliary fallback | 2.0131 |

This is where the shape of the problem finally became clear. The AR-heavy fallback was numerically healthier but aesthetically boring. The diffusion-native Mercury variants were much worse on BPB, but at least they behaved like diffusion models rather than as causal models wearing a fake mustache.

We then tried to improve the Mercury path without betraying it:

- align training masks with continuation and infill tasks,
- add self-conditioning,
- keep a small clean-language prior,
- reduce the corruption ceiling,
- bias somewhat more toward continuation,
- and, because one always hopes the obvious next thing will be the thing, try explicit 2-step unrolled training.

Those produced the following sequence.

**Self-conditioning and task-alignment screens, 180 seconds:**

| Variant | Description | val_bpb |
| --- | --- | ---: |
| `mercury_uniform50` | mixed tasks, no strongest self-conditioning | 3.5056 |
| `mercury_hybrid50p` | mixed tasks | 3.4987 |
| `mercury_uniform50_suffixsc` | continuation-only, stronger self-conditioning | 3.5043 |
| `mercury_hybrid50p_mixsc` | mixed tasks, stronger self-conditioning | 3.4980 |
| `mercury_auxlate` | AR-heavy fallback | 2.0150 |

**Longer 600-second Mercury screen:**

| Variant | Description | val_bpb |
| --- | --- | ---: |
| `mercury_hybrid50p_mixsc` | progressive hybrid 25% to 50% | 2.2261 |
| `mercury_hybrid37p5_mixsc` | progressive hybrid 25% to 37.5% | 2.2168 |
| `mercury_hybrid50p_cont90` | stronger continuation bias | 2.2199 |

**Fine local search around the best recipe, 180 seconds:**
| Variant | Description | val_bpb |
| --- | --- | ---: |
| `mercury_hybrid35_mixsc` | progressive hybrid 25% to 35%, mixed tasks | 3.4973 |
| `mercury_hybrid37p5_cont85` | 37.5% ceiling, continuation 0.85 | 3.4980 |
| `mercury_hybrid3125_mixsc` | 31.25% ceiling, mixed tasks | 3.4982 |
| `mercury_hybrid35_cont85` | 35% ceiling, continuation 0.85 | 3.4983 |
| `mercury_hybrid37p5_mixsc` | 37.5% ceiling, mixed tasks | 3.4984 |

And the explicit **two-step formulation**, which is worth recording because it failed so cleanly:

| Variant | val_bpb |
| --- | ---: |
| `mercury_hybrid37p5_2step` | 6.8082 |
| `mercury_hybrid37p5_2step_cont90` | 6.7054 |
| retry `mercury_hybrid37p5_2step` | 7.0219 |
| retry `mercury_hybrid37p5_2step_cont90` | 6.9992 |

This was not merely slow. It was unstable. The runs reached only about 25 steps in 180 seconds and produced catastrophically bad BPB almost immediately. In other words, the obvious "make it more diffusion-like by literally unrolling more denoising" move was, in this setting, precisely the wrong move.

For completeness, we also explored anchored/block diffusion.

**Anchored/block ladder, 180 seconds:**

| Variant | Description | val_bpb |
| --- | --- | ---: |
| `ab_control` | plain causal LM | 1.9970 |
| `ab_anchor32late` | late-onset anchored block | 2.3869 |
| `ab_anchor32` | anchored block | 12.2302 |
| `ab_anchor64` | larger anchored block | 12.7226 |
| `ab_block32` | raw block diffusion | 12.9688 |

Anchors helped relative to naked block diffusion. Delaying the objective helped much more. None of it was healthy enough to become the submission.

## How We Landed On The Final Recipe

The final submission recipe is `mercury_hybrid35_mixsc`, and it is here because it satisfied the one criterion that became more important than vanity: it made diffusion visible without making the whole model nonsensical.

In concrete terms, it keeps:

- direct Mercury-style denoising as the main auxiliary interface,
- progressive hybrid corruption from 25% to 35%,
- mixed continuation and infill tasks,
- 75% continuation bias,
- self-conditioning with 75% commit fraction,
- and a small clean-language prior to keep the denoiser from drifting completely into local noise repair.

We did not choose it because it had the absolute best number from every exploratory run. We chose it because it was the best compromise between two competing obligations:

- it had to be diffusion-native enough that calling it "text diffusion" was not a euphemism,
- and it had to preserve enough language modeling competence that the model remained legibly a language model rather than an expensive punctuation fountain.

This is, in a way, the entire retrograde story. The more diffusion we added in the naive sense, the worse BPB became. The more we retreated back toward AR, the less interesting the submission became. The final recipe is the point on that curve where the model is still recognizably about diffusion, yet not so broken that the whole experiment collapses into pure parody.

## Final 8xH100 Result

**Final 8xH100 SXM runs used seeds `1337`, `42`, and `2026`.**
*The submitted `train_gpt.py` defaults to this recipe*
| Seed | Final val_loss | Final val_bpb | Artifact bytes | Stop step |
| --- | ---: | ---: | ---: | ---: |
| 1337 | 2.45349105 | 1.45309560 | 15,677,283 | 4628 |
| 42 | 2.46577237 | 1.46036929 | 15,531,183 | 4912 |
| 2026 | 2.45803749 | 1.45578825 | 15,500,938 | 4926 |
| Mean | 2.45910030 | 1.45641771 | - | - |
| Std | 0.00620926 | 0.00367747 | - | - |

This is much worse than the official naive AR baseline. That sentence should not be hidden in a footnote or apologetically coughed into a sleeve. It is the primary result. In this challenge setting, compact text diffusion simply does not beat compact autoregression on `val_bpb`.

All three artifacts fit under the decimal `16,000,000` byte cap. The largest logged total was `15,677,283` bytes.

## What The Model Is Good At, If "Good" Is Used Carefully

Our experiment exposes diffusion-native behavior clearly enough to inspect.

**Three-seed mean `parallel_eval` results from the 8x logs:**

| Task | Refinement steps | Token accuracy | Tokens/sec |
| --- | ---: | ---: | ---: |
| continuation | 1 | 0.0348 | 25,123.81 |
| continuation | 2 | 0.0345 | 36,784.34 |
| continuation | 4 | 0.0348 | 25,990.89 |
| continuation | 8 | 0.0355 | 13,005.42 |
| infill | 1 | 0.0377 | 104,971.51 |
| infill | 2 | 0.0384 | 52,357.51 |
| infill | 4 | 0.0410 | 26,075.30 |
| infill | 8 | 0.0404 | 12,922.12 |

**These accuracies are awful.** They are also informative. They tell us that the model did learn the interface. It can denoise in parallel. It can infill. It can trade additional refinement steps for different speed and slightly different accuracy. What it did not learn was how to make that interface semantically robust under this budget. But it did reveal where some of the tradeoff was happening.


## Matched AR vs Mercury Decode Benchmark

To make the speed-quality tradeoff legible, I also ran a matched 1xH100 decode benchmark using the actual seed `2026` 8x checkpoint and the official naive baseline checkpoint.

Setup:

- 32 validation examples
- 128-token prefix
- 64-token continuation target
- 64-token infill span, with a visible 64-token suffix
- same tokenizer, same validation source, same hardware class for the matched comparison

Highlights from `decode_benchmark.md`:

- AR continuation throughput: `1518.79` tokens/sec
- Mercury continuation, 1 step: `0.0400` token accuracy at `33315.36` tokens/sec, `21.94x` AR throughput
- Mercury continuation, 2 steps: `0.0400` token accuracy at `52423.93` tokens/sec, `34.52x` AR throughput
- Mercury infill, 1 step: `0.0400` token accuracy at `147729.24` tokens/sec, `97.27x` AR continuation throughput

The model is able to be breathtakingly fast, spectacularly parallel, but very, very incorrect. The diffusion-native interface is real, but (at this scale) the diffusion-native model is not very good at all.

> The raw examples in `decode_benchmark.md` are included because aggregate metrics flatter degeneration. In several samples the model collapses into `the`, punctuation, or repeated fragments. The token accuracy is therefore not evidence of meaningful long-span understanding in the ordinary human sense. It is evidence that the model can sometimes land near the high-frequency skeleton of the target while moving very quickly.

## Reproduction

The submitted `train_gpt.py` defaults to the final `mercury_hybrid35_mixsc` recipe. On the official RunPod image with cached SP1024 FineWeb data:

```bash
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Repeat with `SEED=42` and `SEED=2026` to reproduce the three-seed set. Included logs:

- `train_seed1337.log`
- `train_seed42.log`
- `train_seed2026.log`

The script writes:

- `final_model.pt`
- `final_model.int8.ptz`
- final `val_loss` and `val_bpb`
- post-roundtrip exact BPB
- continuation and infill `parallel_eval` metrics

## Compliance

- This is a non-record submission under `records/track_non_record_16mb`.
- `train_gpt.py` is self-contained and runnable from this folder.
- No network calls or external side information are used during training or validation.
- Validation uses the full FineWeb validation split through the standard Parameter Golf BPB path.
- The compressed artifact is `int8+zlib`, and all logged artifacts are under `16,000,000` bytes.
- Training fits in the 10-minute 8xH100 SXM budget used for the challenge.
- Evaluation fits in the separate evaluation budget.
