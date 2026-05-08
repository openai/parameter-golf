Adds a non-record 16MB submission for a compact autoregressive GPT trained on the FineWeb SP1024 setup.

Implementation details:
- candidate_id: `gpuq_ttt_disabled_runtime_control`
- architecture: autoregressive GPT using the SP1024 tokenizer and a 17.1M parameter model before packaging
- runtime policy: TTT is disabled (`TTT_ENABLED=0`), so training uses the score-first autoregressive path without test-time-training overhead
- attention/initialization: QK gain initialization is set to `5.25`
- training batch: `2,097,152` train tokens per step on one H100
- evaluation: the final int8+zlib artifact is round-tripped and evaluated after packaging
- packaging: self-contained `train_gpt.py` plus compressed int8-zlib model artifact under the 16MB cap

Result:
- val_bpb: `1.44421409`
- val_loss: `2.438495`
- hardware: `1x NVIDIA H100 80GB HBM3`
- seed: `1337`
- train time: `668.949s`
- model bytes: `10,095,619`
- code bytes: `104,891`
- total package bytes: `10,200,510`

This is submitted as non-record because I could not get access to the required 8xH100 SXM setup before the deadline. The packaged run was done on a single H100, includes one seed, and exceeded the 600 second record-track limit.

The purpose of the submission is to provide a reproducible non-record comparison point for the cost/benefit of leaving TTT disabled under the artifact cap. In this configuration, avoiding the TTT runtime path preserves throughput and stays within the compressed artifact budget.

Validation:
- `python3 -m json.tool records/track_non_record_16mb/2026-04-30_gpuq-ttt-disabled-runtime-control_1p44421/submission.json >/dev/null`
- `python3 -m py_compile records/track_non_record_16mb/2026-04-30_gpuq-ttt-disabled-runtime-control_1p44421/train_gpt.py`
- `git diff --check`
