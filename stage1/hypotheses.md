# Stage 1 Mutation Families

The previous Stage 1 pass over-counted hyperparameter sweeps as separate surfaces.
That is too narrow for Enigma.

The correct unit for Stage 1 is a mutation neighborhood:

- one main causal mechanism,
- one main edit region,
- one main bottleneck attacked,
- one family-specific question that teaches us something if it wins or loses.

## Recut Rule

- at most `2` families may be mostly config-only
- at least `8` families must require nontrivial code or data changes
- cheap knob changes are seeds inside a family, not separate families by themselves
- if two ideas hit the same region for the same bottleneck, merge them

## Consolidations From The Earlier List

- old `H01/H03/H04` collapse into one allocation family
- old `H08/H09` collapse into one token/update geometry family
- old `H12/H13` collapse into one optimizer partition family
- old `H10` is absorbed into a broader wallclock schedule family
- old `H17/H18` become one export-aware quantization family
- old `H19/H20` become one tokenizer/data frontier family

## Evidence Update From Public Records

The public leaderboard materially changes the prior.

What is now clearly supported:

- export-aware quantization is first-order, not cleanup
- longer training context can beat shorter-context step farming
- optimizer partition and decay/schedule matter under the 10-minute cap
- evaluation-time context is a real frontier, not a footnote

What is currently weaker than expected:

- pure reuse / recurrence for the first 10-minute pass
- aggressive control-tensor simplification as an early priority

What this means:

- Stage 1 should build on the public record frontier instead of re-deriving it from scratch
- A100 runs should be used to rank these families cheaply, then promoted to `8xH100`

## Final Stage 1 Eleven

The public records exposed one missing family:
evaluation-time context and scoring policy.

These are the families we should actually treat as distinct.

| ID | Type | Surface | Target region | Main mechanism | Why it survives |
| --- | --- | --- | --- | --- | --- |
| M01 | config | architecture allocation | `Hyperparameters`, `GPT.__init__` | matched-budget depth/width/KV/MLP reallocation | one necessary architecture baseline family, but not more than one |
| M02 | config | token/update geometry | `Hyperparameters`, loader usage in main loop | change sequence length and batch-token geometry under the 600s cap | validated by public long-context records, but still one family not many |
| M03 | code | parameter reuse | `GPT.blocks`, `GPT.block_map`, skip path | alternate-layer sharing or light recurrence | direct structural reuse test imported from small-model design |
| M04 | code | nonuniform stack roles | `GPT.__init__`, `Block`, `CausalSelfAttention`, `MLP` | make encoder and decoder halves intentionally different instead of a uniform stack | distinct from sharing because it changes specialization, not reuse |
| M05 | code | control-tensor simplification | `q_gain`, `attn_scale`, `mlp_scale`, `resid_mix`, `skip_weights`, quantizer keep-float sets | replace per-dimension control vectors with cheaper grouped or scalar controls | direct attack on hidden float-preserved byte surface |
| M06 | code | optimizer partitioning | optimizer param split in main, `Muon` assignment logic | assign different tensor families to different update laws | now supported by public AdamW-for-embed/scalar plus Muon-for-matrix records |
| M07 | code | adaptive Muon compute | `Muon.step`, `get_muon_backend_steps`, `apply_muon_schedule` | change Muon work over time or by matrix family | still strong, but now subordinate to the broader optimizer family evidence |
| M08 | code | wallclock-aware schedule | `lr_mul`, warmup/reset flow, validation cadence in main loop | spend more of the 600s budget on useful training and less on schedule tax | strongly supported by warmdown-heavy public records |
| M09 | code | export-aware quantization | `keep_float_tensor`, `quantize_float_tensor`, `quantize_state_dict_int8`, `dequantize_state_dict_int8` | redesign what stays float and how large tensors quantize | one of the strongest public win surfaces |
| M10 | code/data | tokenizer/data frontier | `data/tokenizer_specs.json`, `data/download_hf_docs_and_tokenize.py`, validation LUT path in `train_gpt.py` | change vocab family or tokenizer family on the fixed docs cache | high-upside frontier outside the optimizer local minimum |
| M11 | code/eval | evaluation-time context | `eval_val`, sliding-window eval path, eval sequence-length path, `forward_logits` | score tokens with richer context at evaluation time | clearly validated by public record submissions and missing from the earlier ontology |

## Family-Specific Cheapest Scouts

| ID | Cheapest scout | Expected teaching signal |
| --- | --- | --- |
| M01 | one matched-budget architecture fork | whether parameter allocation dominates local training tricks |
| M02 | one longer-context fork plus matched LR/batch retune | whether context per token beats total step count in the 600s regime |
| M03 | tie every other block | whether reuse beats local width/depth retuning |
| M04 | asymmetric encoder/decoder halves | whether specialization beats uniformity in tiny models |
| M05 | scalarize or group one control family first | whether control vectors are over-engineered for the byte budget |
| M06 | AdamW for embeddings/scalars, Muon for matrices | whether tensor-family optimizer assignment is the real win |
| M07 | lower early Muon work, restore later | whether Muon cost is mistimed rather than unnecessary |
| M08 | longer always-on decay and wallclock-aware schedule | whether post-quant quality is schedule-limited |
| M09 | fp16 embedding passthrough or late-K passthrough first | whether keep-float exceptions beat global quantizer tweaks |
| M10 | one alternate vocab or tokenizer-family export | whether the current token inventory is locally good but globally wrong |
| M11 | sliding-window eval or moderate longer eval context | how much score headroom exists in evaluation policy alone |

## Current Posterior Read

Highest-priority families after the public evidence:

- `M09` export-aware quantization
- `M11` evaluation-time context
- `M02` long-context training geometry
- `M06` optimizer partitioning
- `M08` wallclock-aware schedule

Demoted but not killed:

- `M03` parameter reuse
- `M05` control-tensor simplification

## What This Recut Means

- Stage 1 is no longer mostly a hyperparameter sweep.
- The family set now explicitly covers evaluation as well as training and export.
- The earlier level-one hooks in `train_gpt.py` are still useful, but they only cover part of the real Stage 1 frontier.
