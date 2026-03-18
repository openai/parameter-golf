# GPT Pro Ideas

This folder saves `gpt-pro.md` as one Markdown file per idea, plus a short summary of the highest-conviction branch and a note on submission-risky directions.

Files:

- `01-streaming-long-context-evaluation-with-decoupled-eval-seq-len.md`
- `02-strictly-causal-cache-model-at-evaluation.md`
- `03-untie-the-output-head.md`
- `04-maintain-an-ema-and-export-the-averaged-weights.md`
- `05-drop-to-1-2-kv-heads-and-spend-the-savings-elsewhere.md`
- `06-train-shorter-than-you-evaluate.md`
- `07-do-a-compression-aware-tail-fake-quant-or-qat.md`
- `08-make-the-model-modestly-larger-and-train-it-on-fewer-tokens.md`
- `09-pairwise-shared-block-recurrence-with-per-layer-gates-left-untied.md`
- `10-replace-deterministic-sequential-streaming-with-randomized-block-sampling.md`
- `11-replace-some-or-all-relu2-mlps-with-a-parameter-matched-swiglu-or-geglu.md`
- `12-add-a-tiny-causal-depthwise-conv-or-token-mixing-branch.md`
- `13-replace-torch-save-plus-pickle-metadata-with-a-custom-flat-packer.md`
- `14-sweep-tokenizer-size-upward-a-bit.md`
- `15-tiny-online-adaptation-of-output-biases-during-evaluation.md`
- `submission-risky.md`
- `single-best-bet.md`

Summary of the source document:

- The baseline is a 9-layer, 512-dim GPT with 8 attention heads, 4 KV heads, tied embeddings by default, `TRAIN_SEQ_LEN=1024`, and a final per-row int8 plus zlib export.
- The challenge score is `val_bpb`, the counted artifact must stay under `16,000,000` bytes, and evaluation must finish under 10 minutes on `8xH100`.
- The strongest claimed upside is in exploiting evaluation freedom, optimizing for the quantized artifact instead of the bf16 checkpoint, and reallocating parameters away from low-value places.

The first five concrete experiments proposed in `gpt-pro.md` were:

1. streamed 4k-8k evaluation with causal KV cache and RoPE scaling
2. a strictly causal cache-LM / n-gram mixture at eval
3. untie the output head and cut KV heads to 1
4. EMA plus a short fake-quant tail before export
5. 512-train / long-eval or a 512->1024 curriculum, then sweep a slightly larger model
