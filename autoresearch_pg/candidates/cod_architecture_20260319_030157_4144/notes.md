# cod_architecture_20260319_030157_4144

1. Hypothesis

Scheduler cycle 36: action=propose family=architecture, parent=port_architecture_20260319_023023_5357, source=train_gpt_mlx.py. Reparameterize each block linear as a unit-RMS row matrix plus a learned per-row `out_scale` vector, and renormalize after each optimizer step so most magnitude information moves into small float control tensors instead of the int8-quantized matrices.

2. Expected upside

Lower `final_int8_zlib_roundtrip_exact val_bpb` from the architecture parent by shrinking the post-quant gap without changing the inherited 6x576, 9/3-head, MLP=1 trunk. The new `out_scale` vectors are tiny relative to the matrices they protect, so byte growth should stay modest while export fidelity improves.

3. Expected risk

The extra weight/scale factorization introduces another symmetry into optimization and could slightly hurt pre-quant learning if Muon + Adam do not like the per-step rebalance. Zero-init projection layers are also now learning through a separate scale path, so convergence could get a bit slower or noisier.

4. Exact knobs changed

No env override changes. Code changes only:
- `CastedLinear` now stores normalized rows plus learned `out_scale`.
- `SplitOptimizers.step` now rebalances row norms back into `out_scale` after optimizer updates.
- `out_scale` / `out_scales` were added to the control-tensor export patterns so they stay in float on serialization.

5. Promotion bar

Promote if `smoke_mlx_local` stays valid and either beats `3.24020662` on `final_int8_zlib_roundtrip_exact val_bpb` or shows a clearly smaller quant gap at comparable pre-quant loss and bytes.
