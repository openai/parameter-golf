# MergedTop3_v3 clean H100 rerun

This folder contains a clean single-seed `track_10min_16mb` submission based on a merged top-stack recipe:

- 11 layers
- XSA on the last 4 layers
- EMA
- 3x MLP
- SmearGate
- BigramHash with 2048 buckets
- mixed int6 quantization with zstd
- sequence length 2048
- Muon/AdamW weight decay 0.04
- sliding-window eval with stride 64
- Partial RoPE with `ROPE_DIMS=16`
- layerwise LN scaling
- GPTQ-lite clip search
- `WARMDOWN_ITERS=3500`

## Clean run result

Fresh uninterrupted `8x H100` run completed on 2026-03-25 with:

- `step_stop=5347`
- `train_time=580.213s`
- `final_int6_roundtrip_exact val_loss=1.96565872`
- `final_int6_roundtrip_exact val_bpb=1.16417381`
- `eval_time=44.398s`
- `bytes_model_int6_zstd=15,562,277`
- `bytes_code=72,924`
- `bytes_total=15,635,201`

This run stayed under both required caps:

- training time `< 600s`
- evaluation time `< 600s`
- artifact size `< 16,000,000`

## Files

- `train_gpt.py`
- `README.md`
- `submission.json`
- `train_seed1337.log`
- `requirements.txt`

## Notes

- This is a clean single-seed run, not a multi-seed statistical record claim.
- `train_seed1337.log` is the original remote run log recovered after the run.
