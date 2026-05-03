# Vocab1792 + FlashMuon + Linear Scale Init + XSA5 Last-Gated + RReLU2 + Int6 AWQ Mixed Bits

## Summary

This submission keeps the same main 10-layer branch as the previous FlashMuon/XSA/RReLU2 recipe, but makes two practical changes:

- move to a larger vocabulary (`1792`)
- use mixed-bit AWQ quantization so a small set of sensitive tensors can stay at `int8` while most tensors remain `int6`

The goal is simple: spend bytes where they matter most, while keeping the rest of the model on the cheaper `int6` path.

## Final Construction

- `10` transformer layers
- large vocabulary branch (`1792`)
- `XSA` enabled on the last `5` layers
- only the final XSA layer is gated
- `RReLU2` MLP activation
- Flash Muon optimizer path
- `Muon` weight decay `0.01`
- late EMA plus post-train best-choice selection
- linear phase init for `resid_mix`
- linear-by-depth initialization for `attn_scale` and `mlp_scale`
- mixed-bit `int6_awq + lzma` export
- validation-tail calibration for quantization

## Main Ideas

The best gains in this branch came from four places:

1. Stronger late-layer routing with XSA only near the top of the stack.
2. Better quantization, especially for a few tensors that are unusually sensitive.
3. Better late-training selection through EMA and post-train checkpoint comparison.
4. A larger vocabulary that improved quality without meaningfully hurting step time.

## Quantization

### Mixed-Bit AWQ

The export path is now mixed-bit AWQ:

- default stays `int6` for most tensors
- selected sensitive tensors can be exported as `int8`
- dequantization stays compatible because bit width is stored per tensor in `qmeta`

This is controlled by:

- `QUANT_INT8_NAME_PATTERNS`

Default sensitive-tensor list:

- `tok_emb.weight`
- `lm_head.weight`

These tensors are expensive to quantize too aggressively, so giving them `int8` while keeping most of the model at `int6` gave a better final tradeoff.

### Calibration

Chosen calibration source:

- `val_tail`

Reason:

- it remained the most reliable calibration source for the final AWQ export path

### Compression

Chosen backend:

- `lzma`

Reason:

- best final size for the chosen mixed-bit AWQ object format

## Vocabulary Change

This branch also uses a larger vocabulary (`1792`).

Important observation:

- the step-time difference between `1024` and `1792` vocab was only about `1 ms` for this model

So for this architecture, vocabulary growth was almost free from a speed perspective, while still improving the quality/size tradeoff enough to be worth keeping.

## Architecture Notes

- best current branch still uses `10` layers
- `MLP_MULT=3` remained the best practical choice in this family
- deeper models improved raw quality but pushed compressed size too high for the `16 MB` target
- the final branch still benefits from putting more modeling power into the last few layers instead of spreading complexity everywhere

## Initialization

Chosen initialization:

- `resid_mix`: simple linear phase initialization
- `attn_scale`: linear by depth
- `mlp_scale`: linear by depth

Takeaway:

- stronger late-layer scales helped more than donor-style random initialization
- simple structured initialization was more stable than more aggressive learned-statistic initialization

## Post-Train Selection

The final branch uses a small best-choice module instead of exporting the final step blindly.

Candidate set:

- raw final checkpoint
- `EMA` checkpoint
- selected late checkpoints
- average of the selected late checkpoints

Conclusion:

- late training is noisy under the short wallclock budget
- explicit post-train comparison gave a more reliable exported float model before quantization

## Final Result

Final metrics are recorded in `submission.json`.

This submission should be read as:

- a large-vocab continuation of the FlashMuon/XSA/RReLU2 branch
- with mixed-bit AWQ replacing uniform `int6` export
- and with bytes intentionally concentrated in the most quantization-sensitive tensors
