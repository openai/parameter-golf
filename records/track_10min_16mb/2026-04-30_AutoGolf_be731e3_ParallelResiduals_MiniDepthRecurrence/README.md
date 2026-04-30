# Non-Record Submission: AutoGolf be731e3 Parallel Residuals + Mini Depth Recurrence

**val_bpb: 1.10931401** from one complete seed-2024 run. This is packaged as a non-record submission candidate rather than a SOTA claim: the archive contains one complete run and one incomplete retry, so it does not demonstrate the multi-run statistical win required for new records.

This submission packages the internal `be731e3` experiment into the standard Parameter Golf record layout. It is based on the March 31 `ParallelResiduals_MiniDepthRecurrence` family and keeps the same core recipe: delayed mini depth recurrence on layers 4 and 5, untied repeated MLPs, parallel residual routing from layer 7, XSA on the last 11 layers, AR self-generated GPTQ calibration, mixed int6/int5 quantization, and Brotli compression with byte shuffling.

## Results

Run environment: 8xH100 80GB SXM, `fineweb10B_sp1024`, SentencePiece 1024-token BPE, no TTT, 600 second training budget.

| Run | Seed | Steps | Train time | Step avg | Post-EMA BPB | Roundtrip BPB | Sliding BPB | Artifact bytes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `train_seed2024.log` | 2024 | 5520 | 600.049s | 108.70ms | 1.1267 | 1.13373293 | **1.10931401** | 15,849,959 |

Additional diagnostics from the complete run:

- Sliding-window validation loss: `1.87302528` nats
- Serialized quantized model: `15,755,929` bytes
- Code size reported by the script: `94,030` bytes
- Peak allocated memory: `26,997` MiB
- GPTQ calibration: AR self-generated, 64 sequences x 2048 tokens, temperature 0.80
- Mixed quantization: 32 int6 layers and 34 int5 layers, with recurrent layers forced int6

`train_seed2024_retry_incomplete.log` is included for transparency. It reached post-EMA evaluation but stopped before GPTQ/final exact evaluation, so it is not counted in the reported score.

## Reproduction

The complete archived run used the local Slurm launcher on an 8xH100 node. The equivalent direct command is:

```bash
DATA_PATH=/public/home/h202308375/jyg/hpc/sync/auto-golf/parameter-golf/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/public/home/h202308375/jyg/hpc/sync/auto-golf/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
SEED=2024 \
MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

If running from the Parameter Golf repository root with the standard data layout, the explicit `DATA_PATH` and `TOKENIZER_PATH` values can be replaced by the script defaults under `./data`.

## Files

- `train_gpt.py`: self-contained training, quantization, compression, and validation script from internal commit `be731e3`
- `train_seed2024.log`: complete run used for the reported result
- `train_seed2024_retry_incomplete.log`: incomplete retry log, retained as supporting context
- `submission.json`: metadata for leaderboard ingestion
- `requirements.txt`: Python package dependencies inherited from the source record family
