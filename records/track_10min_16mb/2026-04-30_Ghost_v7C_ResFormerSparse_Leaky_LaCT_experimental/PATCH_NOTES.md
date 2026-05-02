# GPT patch notes for Ghost v7

This patch keeps the v7 branch intact, but fixes blockers that made it unsafe to spend H100 credits on directly.

## Fixed

1. **Dequantization scale bug** — reconstruction is now `q * scale`, matching `scale = clip_abs / max_val`.
2. **Dirty TTT/export path** — the clean trained/EMA model is serialized; legal TTT now runs after quantized roundtrip and reports `final_legal_ttt_roundtrip`.
3. **EMA TTT crash risk** — TTT explicitly re-enables gradients before applying the freeze/no-QV mask.
4. **Distributed final eval mismatch** — every rank loads the same quantized/dequantized `final_model.ptz` before final scoring.
5. **Sliding-window loss bug** — added `return_per_token=True` so sliding eval sums only the last-stride losses instead of using whole-window mean loss.
6. **zstd dependency declared** — added `requirements.txt`.

## Still not proven

- LaCT is implemented as a legal chunked TTT branch, but it needs real RunPod timing and BPB logs.
- This is not merged onto the 1.061 public SOTA base.
- No run logs are included yet; `submission.json` remains placeholder until real runs populate it.

## Score line to cite if TTT is enabled

```text
final_legal_ttt_roundtrip_exact val_loss:... val_bpb:...
```

Do not cite any score from a model that has adapted on validation tokens and then rescored them.

## GPT v7C patch — Kimi report integration

Added two low-cost candidates from the Kimi competitive-techniques report:

1. `LeakyReLU(0.5)^2` activation in MLPs, controlled by `LEAKY_RELU_SLOPE`.
2. ResFormer-style Value Residual Learning, controlled by:
   - `RESFORMER_ENABLED=1`
   - `RESFORMER_MODE=sparse|all|off`
   - `RESFORMER_LEARNED=1|0`
   - `RESFORMER_DETACH_V0=1|0`

Default mode is sparse + learned: cache layer-0 V, then blend it into the last third of layers with two learned softmax logits per active layer. This is intentionally narrower than a full architecture rewrite.

Micro-sim note: this branch includes `MICRO_SIM_REPORT_V7C.md`. The sim patches optimizer behavior locally only because this sandbox's AdamW initialization path hung. The submitted training script still uses real AdamW.
