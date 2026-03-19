# cod_architecture_20260319_025235_3818

1. Hypothesis

Scheduler cycle 35: action=propose family=architecture, parent=port_architecture_20260319_023023_5357, source=train_gpt_mlx.py

Keep the parent 6x576x9/3 architecture and skip layout, but share only the MLP cores across mirrored depth positions while leaving attention and the small per-layer control tensors untied. The recent same-tier evidence says extra untied depth and extra MLP bytes were basically flat or worse, so the next architecture move should test whether the feed-forward stack is over-parameterized relative to the attention path and the export metric.

2. Expected upside

Lower `final_int8_zlib_roundtrip_exact val_bpb` by removing a chunk of the large quantized MLP matrix budget without collapsing the whole stack into a full recurrent block. Mirrored MLP sharing keeps layer-specific attention, skip connections, and per-layer `attn_scale` / `mlp_scale` / `resid_mix` controls, so the model can still specialize by depth while paying for fewer heavy feed-forward tensors at export.

3. Expected risk

The current best architecture already uses a narrow `MLP_MULT=1`, so further reducing feed-forward freedom may bottleneck representation and lose more than it gains. There is also some risk that the mirrored sharing pattern is too symmetric for the encoder/decoder halves even with untied controls.

4. Exact knobs changed

Added one new default-on architecture knob in `train_gpt_mlx.py`:
- `SHARE_MIRROR_MLPS=1`

Refactored the model so:
- attention modules remain unique per layer
- MLP modules can be reused across mirrored layer indices, e.g. `0,1,2,2,1,0` for the inherited 6-layer setup
- per-layer control tensors (`attn_scale`, `mlp_scale`, `resid_mix`) remain untied
- optimizer grouping now treats the new shared core matrices as Muon params and the control tensors as scalar Adam params

5. Promotion bar

`smoke_mlx_local`: no runtime breakage, logs show `mlp_sharing:mirror_pairs`, and int8+zlib bytes drop materially versus the untied parent.
`proxy_1gpu_fast`: beat the parent on `final_int8_zlib_roundtrip_exact val_bpb`, or at minimum show a smaller post-quant gap at meaningfully lower bytes without a large raw-loss regression.
