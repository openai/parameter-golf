# Pure Raw-Byte JEPA With Detached Exact Probe: Negative Result

This folder documents the cleanest pure-JEPA attempt we ran for Parameter Golf. The setup is strict: raw `byte260`, no tokenizer, no exact byte-NLL into the backbone, and exact byte prediction only through a later detached Transformer probe on frozen features. The best result from this path was **`2.3839 bpb`** with `transformer_rope_gqa_localglobal + slot_ema_teacher`, which is a real improvement over our earlier pure-JEPA runs but still about **`+1.16 bpb`** above the repo simple baseline `1.22436570`.

## Protocol

The backbone is trained with JEPA-style latent prediction plus collapse regularization only. Each `8`-byte patch is encoded into one summary latent and four ordered `2`-byte slot latents. A Transformer backbone predicts the next summary and slot bank. Later, a detached Transformer exact decoder probe is trained on frozen features consisting of:

- the causal context state
- the predicted next summary
- the predicted next slot bank

This is intentionally different from hybrid JEPA paths that let exact next-token or next-byte gradients shape the backbone during training.

## Results

### Pure-JEPA milestones

| Stage | Best `bpb` | Notes |
|------|-----------:|------|
| `BytePatchJEPA_PurityFirst` | `2.8583` | earlier raw-byte JEPA with a coupled exact decoder term |
| `BytePatchJEPA_PureProbeScaling` | `3.0774` | first clean frozen-probe pipeline |
| Transformer-only backbone screen | `2.3889800525604903` | `transformer_rope_gqa_localglobal` with `slot_l2` |
| Transformer-only objective screen | `2.3839` | `transformer_rope_gqa_localglobal + slot_ema_teacher` |

### Backbone screen

| Backbone | `bpb` |
|------|------:|
| `transformer_rope_gqa_localglobal` | `2.3889800525604903` |
| `transformer_rope_gqa_base` | `2.389990501438125` |
| `transformer_rope_gqa_convstem` | `2.5803010001832605` |

### Objective screen

These values were recovered from copied-back final strong-probe logs because `results/objective_screen/summary.json` never synced back.

| Objective | `bpb` |
|------|------:|
| `slot_ema_teacher` | `2.3839` |
| `slot_cosine` | `2.3885` |
| `slot_l2` | `2.3888` |
| `slot_vicreg` | `2.3918` |
| `masked_slot_jepa` | `2.5098` |

### Encoder screen

All encoder variants used the winning backbone and objective above under the same short equal-budget rerun.

| Patch encoder | `bpb` |
|------|------:|
| `conv_patch` | `2.746384624395377` |
| `mlp_baseline` | `2.7525905146099565` |
| `patch_transformer` | `2.8835849452702482` |
| `latent_queries` | `2.899715507869489` |

## Comparison to Other JEPA PRs

These are useful comparison points, but they are not the same experiment.

| PR | Training path | Tokenization | Reported result | Why it differs |
|------|------|------|------:|------|
| This folder | pure detached-probe JEPA | raw bytes | `2.3839` | no exact-loss gradients into backbone |
| [PR #708](https://github.com/openai/parameter-golf/pull/708) | hybrid JEPA + exact next-byte scorer | raw bytes | about `2.1252` | exact next-byte compression objective is in the main training path and predicted chunk latents are fused back into the scorer |
| [PR #896](https://github.com/openai/parameter-golf/pull/896) | JEPA self-distillation auxiliary loss on top of autoregressive LM | tokenized | PR author reports vanilla CE beats JEPA by `0.005 BPB` and is `40%` faster | CE remains the main path and the comparison is token-level, not raw-byte pure JEPA |
| [PR #903](https://github.com/openai/parameter-golf/pull/903) | LeWorldModel-style JEPA + SIGReg + CE head, plus a detached diagnostic probe | BPE and byte | reported `1.2064` sliding / `1.2235` standard for best long BPE, `1.2566` 10-minute BPE, `1.3348` standard 10-minute byte | includes a detached probe diagnostic, but the main reported model is still CE-trained, CE is described as dominant by mid-training, and the JEPA-only contribution remains open |

PRs #708 and #896 are hybrid or auxiliary-loss approaches. PR #903 is closer to this line of work because it also includes a detached diagnostic probe, but its main reported model is still a CE-trained JEPA-augmented system rather than a backbone trained in a pure detached-probe regime. So none of them are apples-to-apples comparisons with this setup. The clean question here is narrower: can a pure raw-byte JEPA backbone, trained without exact-loss gradients, carry enough information that a later detached exact decoder can recover good `bpb`?

## What We Learned

- Stronger Transformer backbone plus slot-based targets improved pure JEPA substantially over earlier attempts.
- Objective changes within the slot-target family only moved the result a little, except `masked_slot_jepa`, which was clearly worse.
- Richer within-patch encoders mostly did not help; `conv_patch` only barely beat the baseline MLP encoder.
- Lower JEPA loss did not reliably translate into lower exact-byte `bpb`.
- Pure JEPA remained far above the repo baseline even after moving to the stronger Transformer-only setup.

## What Seems Wrong

- The temporal path still looks too summary-dominant: the backbone mostly reasons over patch summaries, not the full slot history.
- The future-latent predictor is still effectively too deterministic for byte compression, so it likely averages over plausible futures.
- The detached exact decoder probe can learn, but the frozen JEPA features still appear too lossy for exact byte prediction.

## Supporting Artifacts

- [Historical notes](JEPA_SUMMARY.md)
- [Top-line result table](results/topline_results.tsv)
- [Objective screen recovered from logs](results/objective_screen_from_logs.md)
- [Objective screen TSV](results/objective_screen_summary.tsv)
- [Backbone screen summary](results/backbone_screen/summary.json)
- [Encoder screen summary: `mlp_baseline`](results/encoder_screen_mlp_baseline/summary.json)
- [Encoder screen summary: `conv_patch`](results/encoder_screen_conv_patch/summary.json)
- [Encoder screen summary: `patch_transformer`](results/encoder_screen_patch_transformer/summary.json)
- [Encoder screen summary: `latent_queries`](results/encoder_screen_latent_queries/summary.json)
- [Objective JEPA loss plot](results/live_pull/plots/objective_jepa_loss_vs_bytes.png)
- [Objective final probe plot](results/live_pull/plots/objective_final_probe_bpb_vs_time.png)

## Reproduction

Smoke:

```bash
cd records/track_non_record_16mb/2026-03-26_BytePatchJEPA_TransformerOnly
env SELF_TEST=1 python3 train_gpt.py
python3 summarize_sweep.py --self-test
python3 launch_runpod_probe.py --phase smoke --gpu-count 1
```

Backbone screen:

```bash
cd records/track_non_record_16mb/2026-03-26_BytePatchJEPA_TransformerOnly
python3 launch_runpod_probe.py --phase backbone_screen --gpu-count 4
```

Objective screen:

```bash
cd records/track_non_record_16mb/2026-03-26_BytePatchJEPA_TransformerOnly
python3 launch_runpod_probe.py --phase objective_screen --gpu-count 4
```

This folder is a research non-record writeup. It does **not** claim a validated 16MB artifact submission.
