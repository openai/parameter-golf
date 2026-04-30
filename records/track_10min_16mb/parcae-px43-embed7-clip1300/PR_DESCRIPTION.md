# Draft: parcae-px43-embed7-clip1300

This is a draft/non-record research submission based on the Parcae loop-injection direction from @mikeapedia's PR #1674: [Non-record: Parcae Loop Injection + Gemma-style Attention + Gram NS](https://github.com/openai/parameter-golf/pull/1674).

## What This Architecture Is

The main idea I wanted to test was whether the Parcae-style loop boundary can improve a small recurrent-depth transformer under the 8xH100 / 16MB setting. PR #1674 describes Parcae constrained loop injection as an SSM-inspired boundary condition at loop re-entry points: instead of passing the recurrent hidden state through unchanged, the loop boundary learns a stable decay term and a residual re-injection term from the original stream. In my run, this is combined with the px43/embed7/clip1300 compression setup and evaluated with the legal sliding-window path.

The submitted package uses:

- recurrent-depth transformer loop structure over the middle blocks
- QK-gain attention initialization
- skip gates and tied embedding/head path
- EMA post-training weights
- Hessian-aware mixed GPTQ
- 6-bit matrix quantization and 7-bit embedding quantization
- Brotli compression
- final sliding-window evaluation

## Tokenizer / Data

This run uses the Mikeapedia SP8192 tokenizer and pretokenized data from:

- Hugging Face dataset: [Mikeapedia/parameter-golf-sp8192](https://huggingface.co/datasets/Mikeapedia/parameter-golf-sp8192/tree/main/datasets)
- Tokenizer file: `datasets/tokenizers/fineweb_8192_bpe.model`

The tokenizer SHA256 used by the runner is:

```text
a24fd9326f81c9456e24484aae2a05b209898738a0082f37b085ef2fe873cec7
```

## Results

Three completed 8xH100 seeds are included:

| Seed | Sliding BPB | Train Time | Eval Time | Artifact Bytes |
|------|-------------|------------|-----------|----------------|
| 42 | 1.08802944 | 600.024s | 89.275s | 15,633,824 |
| 1337 | 1.08783878 | 600.117s | 89.174s | 15,630,505 |
| 2024 | 1.08760994 | 600.093s | 89.318s | 15,630,862 |
| Mean | 1.08782605 | 600.078s | 89.256s | 15,631,730 |

The run is not being represented as a valid record. The local gate report is included because the logs exceed the strict 600s training budget by 24-117 ms and the score does not beat the current record threshold.

## Credits

Thanks to @mikeapedia for PR #1674 and the Parcae loop-injection research direction, plus the public Mikeapedia SP8192 tokenizer/data bundle used here. PR #1674 also points to its upstream inspirations, including xIELU/per-layer QK-gain work and the Parcae paper lineage; this experiment is an attempt to test that family of ideas under a 3-seed 8xH100 run.

Thanks also to the Parameter Golf community for the prior work on depth recurrence, QK gain, GPTQ, SP8192 tokenization, and compression/eval tooling that this run builds on.
