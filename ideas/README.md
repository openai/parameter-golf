# ASK_FOR_IDEAS_v2 Notes

This folder saves the response to `ASK_FOR_IDEAS_v2.md` as one Markdown file per idea or hypothesis.

Files:

- `01-train-shorter-than-evaluate.md`
- `02-use-byte-slack-for-quantization-outliers.md`
- `03-late-qat-for-the-submitted-model.md`
- `04-tie-blocks-and-reinvest-in-width.md`
- `05-cut-mlp-before-cutting-attention.md`
- `06-retune-for-a-600-second-trainer.md`
- `07-exploit-eval-time-slack.md`
- `trap-01-dont-untie-embeddings-or-grow-vocab-first.md`
- `trap-02-dont-jump-straight-to-int4.md`
- `trap-03-dont-spend-compute-on-longer-train-context-first.md`
- `ablation-01-train-vs-eval-seq-len.md`
- `ablation-02-batch-vs-warmdown.md`
- `ablation-03-export-only-quantization-sweep.md`
- `single-best-bet.md`

The ranking is grounded in these repo facts:

- The 10-minute baseline is still improving when it hits the wallclock stop.
- The same 9x512 model keeps improving over 4 hours, so the current setup is still optimization-limited.
- The baseline artifact is already close to the 16,000,000-byte cap.
- The current int8 export path gives back a meaningful amount of the gain, especially on stronger checkpoints.
