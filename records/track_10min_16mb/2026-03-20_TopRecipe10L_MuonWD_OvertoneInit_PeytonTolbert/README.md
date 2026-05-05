# Top Recipe 10L + Muon WD + Overtone Init

Single-seed 8xH100 record-track run using the upgraded trainer configured to match the strongest known recipe family as closely as possible:

1. Sliding-window final evaluation with `stride=64`
2. FP16 tied embedding export
3. 10 transformer layers
4. Muon weight decay `0.02`
5. Overtone spectral embedding initialization with power `0.5`
6. Phase-transition residual-mix initialization

## Result

- Pre-quant eval at stop: `val_loss:2.0602`, `val_bpb:1.2202`
- Post-quant eval: `val_loss:2.0053`, `val_bpb:1.1876`
- Exact printed metric: `final_int8_zlib_roundtrip_exact val_bpb:1.18762449`

## Run Details

- Hardware: `8x H100 80GB`
- Train wallclock: `599846ms`
- Steps completed: `8260`
- Final eval time: `118402ms`
- Total submission size int8+zlib: `15842628` bytes
- Code size: `128619` bytes

## Included Files

- `train_gpt.py`: exact trainer used for the run, copied from the upgraded trainer
- `train.log`: full training and evaluation log
- `submission.json`: leaderboard metadata
