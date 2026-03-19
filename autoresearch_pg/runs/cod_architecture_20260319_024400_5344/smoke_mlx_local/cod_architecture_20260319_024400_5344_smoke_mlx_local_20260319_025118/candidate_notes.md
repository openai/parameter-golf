# cod_architecture_20260319_024400_5344

1. Hypothesis

Keep the parent 6x576 architecture intact, but borrow the strongest low-risk train-only MTP habit from the better same-tier recipe family: apply a small `t+2` auxiliary next-token objective before warmdown, then fall back to pure next-token loss for the late export-facing phase. The width/depth sweeps were basically flat, so improving sample efficiency without adding deploy bytes is a cleaner next move than another parameter-count tweak.

2. Expected upside

Lower `final_int8_zlib_roundtrip_exact val_bpb` by extracting a bit more learning signal from each training sequence while leaving validation, serialization, and int8 roundtrip evaluation on the original deployed objective. This is especially attractive here because the architecture family still has byte headroom and a nontrivial post-quant gap.

3. Expected risk

If even a small future-token target is too noisy for this compact architecture, it can dilute the main next-token objective or cost enough extra logits work to erase the gain. Disabling it during warmdown reduces late-phase risk, but it may also make the effect too weak to matter.

4. Exact knobs changed

Added two default-on knobs in `train_gpt_mlx.py`:
- `MTP_LOSS_WEIGHT_2=0.03`
- `MTP_DISABLE_ON_WARMDOWN=1`

Wired `train_gpt_mlx.py` so:
- validation, export, and final roundtrip evaluation still use the original next-token loss
- training can switch between pure next-token loss and a weighted next-token + `t+2` objective
- the common no-chunk path reuses one logits projection for both targets to keep the wallclock tax modest
- warmup compiles the MTP path when enabled, and warmdown automatically drops back to the deploy objective

5. Promotion bar

`smoke_mlx_local`: no runtime breakage, logs show the MTP objective phase switch, and submission bytes stay flat.
`proxy_1gpu_fast`: beat the parent on `final_int8_zlib_roundtrip_exact val_bpb`, or at minimum improve early optimization without worsening the raw-vs-roundtrip gap.
