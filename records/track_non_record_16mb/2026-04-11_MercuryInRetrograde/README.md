# Mercury in Retrograde

This is a non-record text-diffusion submission for the Parameter Golf request for weird and creative approaches. It is intentionally not presented as a leaderboard contender. It is a compact, Mercury-style discrete denoising experiment that shows a useful negative result: under this parameter and time budget, making the model more diffusion-native gives a real parallel decode interface, but it moves validation BPB in the wrong direction.

The title is literal enough to be useful. Mercury retrograde is an optical illusion where Mercury appears to move backward in the sky for roughly three weeks, 3-4 times a year. In astrology, it is associated with disrupted communication and a period for reflection, reviewing, and redoing. That is a fair description of what happened here: text diffusion gave us an interesting communication mechanism, but it did not move this small language model forward on compression quality.

## What This Implements

This submission keeps the Parameter Golf validation and artifact accounting intact, but changes the training objective and reporting to make text diffusion visible rather than hiding it as a tiny auxiliary loss.

Key ingredients:

- A Transformer backbone with the ordinary Parameter Golf full-validation BPB path retained.
- A Mercury-style denoising training mode, enabled by default in `train_gpt.py`.
- Progressive hybrid token corruption, ramping from 25% to 35%.
- Mixed continuation and infill denoising tasks, with a 75% continuation bias.
- Self-conditioning on denoised tokens, with 75% of corrupted positions committed into the second pass.
- A small clean-language prior (`MERCURY_CLEAN_LOSS_WEIGHT=0.10`) so the denoiser does not become pure noise reconstruction.
- Extra `parallel_eval` logs for continuation and infill at 1, 2, 4, and 8 refinement steps.

This is not an implementation of Mercury's proprietary inference engine or systems stack. It is a compact, challenge-local translation of the modeling idea: predict many corrupted text tokens in parallel, then optionally refine.

## Result Summary

Final 8xH100 SXM runs used seeds `1337`, `42`, and `2026`, with the submitted recipe now set as the default in `train_gpt.py`.

| Seed | Final val_loss | Final val_bpb | Artifact bytes | Stop step |
| --- | ---: | ---: | ---: | ---: |
| 1337 | 2.45349105 | 1.45309560 | 15,677,283 | 4628 |
| 42 | 2.46577237 | 1.46036929 | 15,531,183 | 4912 |
| 2026 | 2.45803749 | 1.45578825 | 15,500,938 | 4926 |
| Mean | 2.45910030 | 1.45641771 | - | - |
| Std | 0.00620926 | 0.00367747 | - | - |

This is much worse than the official naive AR baseline. That is the main result: for this model size and training budget, direct text denoising is not competitive as a compression model.

All three artifacts fit under the decimal 16,000,000 byte cap. The largest logged total was `15,677,283` bytes.

## Diffusion-Native Metrics

The interesting part is not the BPB. The interesting part is that the model exposes a parallel refinement interface for both suffix continuation and middle-span infill.

Three-seed mean `parallel_eval` results from the training logs:

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

These token accuracies are dreadful. They are still useful because they make the failure mode measurable: the model learns a fast, parallel denoising interface, but the denoising distribution is dominated by high-frequency-token collapse rather than coherent text recovery.

## Matched Decode Benchmark

I also ran a matched 1xH100 decode benchmark using the actual seed `2026` 8x checkpoint and the official naive baseline checkpoint. The goal was not to claim good text generation. It was to ask what the diffusion interface buys when quality is allowed to be bad.

Setup:

- 32 validation examples.
- Prefix length 128 tokens.
- Continuation target length 64 tokens.
- Infill span length 64 tokens, with a 64-token visible suffix.
- Baseline AR uses greedy left-to-right decode.
- Mercury predicts/refines the whole target block in parallel.

Highlights from `decode_benchmark.md`:

- AR continuation throughput: `1518.79` tokens/sec.
- Mercury continuation at 1 step: `0.0400` token accuracy, `33315.36` tokens/sec, `21.94x` AR throughput.
- Mercury continuation at 2 steps: `0.0400` token accuracy, `52423.93` tokens/sec, `34.52x` AR throughput.
- Mercury infill at 1 step: `0.0400` token accuracy, `147729.24` tokens/sec, `97.27x` AR continuation throughput.

That is the tradeoff in one sentence: this tiny diffusion model is very fast at being very wrong.

The raw examples in `decode_benchmark.md` are included because they are more honest than the aggregate metrics. They show collapse into tokens like `the`, punctuation, and repeated suffix fragments. That qualitative failure matters for interpreting the token accuracy.

## What We Learned

The negative result is fairly consistent across our exploratory ladder:

- Small auxiliary text-diffusion losses were less destructive, but also less interesting as text diffusion.
- More diffusion-native training made the submission more distinctive, but worsened BPB.
- Uniform, masked, hybrid, anchored-block, self-conditioned, and 2-step variants did not reveal a competitive BPB path in short or 10-minute screens.
- The best "interesting" recipe was not the least-diffusive one. It was the one that preserved a real denoising interface while staying under the artifact cap and avoiding total validation collapse.
- The model's 3-4% token accuracy may be partly explained by high-frequency-token behavior. In sampled examples, correct-looking behavior often reflects guessing common words or punctuation rather than understanding the continuation.

This suggests a constraint-specific lesson: text diffusion may need substantially more capacity, data budget, denoising curriculum, or systems work before it becomes competitive for tiny fixed-artifact compression. Mercury-style parallelism is attractive, but in this setting the distribution modeling problem dominates the systems advantage.

## Reproduction

The submitted `train_gpt.py` defaults to the final `mercury_hybrid35_mixsc` recipe. On the official RunPod image with cached SP1024 FineWeb data:

```bash
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Repeat with `SEED=42` and `SEED=2026` to reproduce the submitted three-seed set. The included logs are:

- `train_seed1337.log`
- `train_seed42.log`
- `train_seed2026.log`

The script also writes:

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
- The compressed artifact is `int8+zlib`, and all logged artifacts are under 16,000,000 bytes.
- Training fits in the 10-minute 8xH100 SXM budget used for the challenge.
- Evaluation fits in the separate evaluation budget.
- This submission is not intended to beat SOTA or the naive baseline. It is intended to document a text-diffusion-native attempt, including the parts that clearly failed.
