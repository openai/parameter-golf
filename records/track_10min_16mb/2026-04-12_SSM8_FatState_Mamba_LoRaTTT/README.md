# SSM8 "Fat State" Mamba — Non-Record Submission

## Summary

This submission presents **SSM8**, a Mamba-based State Space Model (SSM) entry for
the Parameter Golf challenge. It is submitted as a non-record submission in response
to the challenge organisers' request for SSM implementations. The primary
goal is to establish a reproducible SSM baseline within the 16 MB / 10-minute constraint and to document the engineering challenges specific to this architecture.

**Final result:** BPB = **1.3587** on the FineWeb validation set (8×H100, 10 min).

---

## Architecture

SSM8 is a pure Mamba recurrent language model. I deliberately avoided mixing
attention and SSM layers in order to produce a clean SSM baseline.

| Hyperparameter         | Value   |
|------------------------|---------|
| Architecture           | Mamba SSM (recurrent) |
| d_model                | 640     |
| d_inner (expand=2.0)   | 1280    |
| d_state ("Fat State")  | 34      |
| d_conv                 | 4       |
| num_layers             | 8       |
| head_adapter_rank      | 16      |
| vocab_size             | 1056 (SP1024 + special tokens) |
| Total parameters       | 22,305,952 |

### Key architectural choices

**Fat State (d_state=34).** Standard Mamba implementations use d_state=8 or 16.
I increased this to 34, giving the SSM a 2–4× richer hidden state at no additional
parameter cost compared to deepening the stack. The motivation is that FineWeb text
has long-range dependencies (paragraph structure, coreference chains) that benefit
from wider recurrent memory.

**No positional encoding.** Mamba encodes sequence order implicitly through the
recurrent state dynamics, which frees the parameter budget from RoPE frequency tables.

**State hand-off interface in evaluation.** The evaluation loop threads
per-layer `ssm_states` across windows (`return_states=True`), but the current
`_run_selective_scan` invocation does not consume an explicit external initial
state. Accordingly, this implementation should be interpreted as an attempted
state-carry path with inactive external-state injection.

**Low-rank logit adapter.** A rank-16 factored head adapter
(`head_adapter_A × head_adapter_B`) is applied on top of tied embedding logits,
adding exactly 27,136 parameters
($16\times640 + 1056\times16$) at negligible compressed footprint.

**Online entropy-based data heuristic.** Each training batch is filtered at load
time using a zlib compression ratio test. A chunk is accepted only if its
compression ratio (raw bytes / zlib-compressed bytes) falls below a progressive
threshold: 4.0 in the first 20% of training, tightening to 2.5, 2.0, and finally
1.8 in the last 20%. The intuition is that high-compressibility text (HTML
boilerplate, repeated patterns, low-diversity sequences) carries little information
per token and wastes compute on trivially predictable examples.

---

## Compression Pipeline

Weights are compressed in two stages:

1. **GPTQ-lite int6** — per-row quantisation with 5 candidate clip percentiles
   (0.999, 0.9995, 0.9999, 0.99999, 1.0), selecting the one with minimum MSE.
   Embeddings use int8 per-row due to their heavier-tailed distribution.

2. **zstd level 22** — applied to the PyTorch serialised quantised tensors.

Final artifact: **15.93 MB** (code 0.04 MB + model 15.89 MB), within the 16.0 MB limit.

---

## Training

**Optimizer.** Muon for 2D weight matrices, AdamW (fused) for embeddings and
scalar parameters.

**Schedule.** Warmup–Stable–Decay: 10% warmup / 70% stable / 20% cosine decay.

**QAT.** Weights are snapped to the int6 grid every 20 steps when
$\text{lr\_scale}<0.15$. Under the stated 10/70/20 warmup-stable-decay schedule,
this threshold is reached only in the late cosine tail (approximately the final
5% of wall-clock training), before GPTQ-lite export.

**EMA.** A shadow EMA model (decay=0.997) is maintained on GPU throughout training.
The EMA weights are saved to the artifact rather than the live weights, as they
produce smoother weight distributions that compress better under GPTQ-lite.

**Gradient checkpointing** is enabled throughout training, trading memory for
backward-pass recomputation.

---

## Evaluation

**Score-First Test-Time Training (TTT).** LoRA adapters (r=8) are injected into
`in_proj` and `out_proj` of each SSM block. During evaluation, adapters are updated
on the previous window's tokens before scoring the current window.

**Online prior.** A token-level unigram prior is accumulated from all scored tokens
and blended into the logits with a small weight (PRIOR_STRENGTH=0.03). This gives
a small systematic improvement on text with domain-specific vocabulary distributions.

**Temperature scaling.** A learnable temperature τ is updated via SGD alongside the
LoRA adapters, calibrating the sharpness of the output distribution per-document.

---

## Training Log (8×H100, 10 min)

```
Found 80 training shards.
[TIMING] Budget: 600s | Safe buffer: 5s | Effective training: 595s
[MODEL] Params: 22,305,952 | BF16: 44.6MB | d_state=34 head_r=16
[OPT] ws_scale=2.071 BASE_LR=0.02071 EMBED_LR=0.06000 GRAD_CLIP=0.15 MU_END=0.950
Warmup + compile: 16.7s
[STEP 00079] t=30s/595s loss=4.7655 lr_scale=0.505 mu=0.930 qat=False yield=100%
[STEP 00241] t=91s/595s loss=3.0230 lr_scale=1.000 mu=0.950 qat=False yield=100%
[STEP 00403] t=151s/595s loss=2.8291 lr_scale=1.000 mu=0.950 qat=False yield=100%
[STEP 00564] t=212s/595s loss=2.6611 lr_scale=1.000 mu=0.950 qat=False yield=100%
[STEP 00726] t=272s/595s loss=2.4971 lr_scale=1.000 mu=0.950 qat=False yield=100%
[STEP 00888] t=333s/595s loss=2.4217 lr_scale=1.000 mu=0.950 qat=False yield=100%
[STEP 01050] t=393s/595s loss=2.4268 lr_scale=1.000 mu=0.950 qat=False yield=99%
[STEP 01212] t=454s/595s loss=2.3872 lr_scale=1.000 mu=0.950 qat=False yield=99%
[STEP 01374] t=515s/595s loss=2.3410 lr_scale=0.764 mu=0.950 qat=False yield=78%
[STEP 01536] t=575s/595s loss=2.2313 lr_scale=0.076 mu=0.950 qat=True yield=58%
[TIMING] Time limit reached, saving artifact.

[EVAL RESULT] BPB: 1.3587 | tokens_processed=6487339
```

---

## Future Work

The current BPB of 1.3587 is above the Transformer baseline (~1.22). The gap is
expected given that pure SSM architectures are generally weaker at token-level
prediction on short-range patterns. Planned directions:

- Reduce d_state and increase num_layers to shift the parameter budget toward
  more representational depth.
- Hybrid architecture: interleave Mamba layers with a small number of sliding-window
  attention layers to recover short-range granularity.
- Longer training (non-record unlimited compute track) to establish the true
  parameter-limited BPB floor for SSM architectures at this scale.
- Explore selective state initialisation strategies beyond log-HiPPO.
