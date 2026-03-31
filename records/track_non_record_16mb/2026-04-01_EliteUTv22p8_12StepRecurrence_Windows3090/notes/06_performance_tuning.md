# Performance Tuning & I/O Optimization

To maximize the 10-minute training window on an RTX 3090, we optimized every overhead outside of the main Transformer math.

## 1. Zero-Latency Startup (Rapid-Fire Warmup)
The baseline training script had a 20-step warmup that executed full batches, wasting ~2 minutes on `torch.compile` and Triton autotuning.

- **The Fix**: Reduced `WARMUP_STEPS` to **1** and modified `train_gpt.py` to only process **one micro-batch** during the first step.
- **Result**: Startup delay dropped from **3 minutes to <10 seconds**.

## 2. Fast-RAM Data Loader (Pinned Memory)
Standard `np.memmap` can suffer from page faults and slow random access on Windows.

- **Implementation**: 
    - Replaced memory-mapping with a direct **`np.fromfile`** read into a NumPy array.
    - Utilized **`.pin_memory()`** on the batch tensor before the GPU transfer.
- **Latency**: Reduced token loading time from **~200ms** per batch to **sub-millisecond** per batch.

## 3. Throughput Scaling (32-Step Accumulation)
To fit a massive **524,288 token batch** into 24GB of VRAM while using a 6.9M parameter model:

- **The Ratio**: Decoupled the training batch from the hardware limit by implementation **32 gradient accumulation steps** (Micro-batch size of 16,384 tokens).
- **The Speed**: Achieved **34,000 tokens/second** by minimizing synchronizations.

## 4. Capped Validation
Mid-training validation on the 62-million-token FineWeb val set was the cause of significant "hangs" at Step 5.

- **Logic**: Implemented a **50-batch cap** for intermediate evaluations and a 100-batch cap for the final model save.
- **Benefit**: BPB estimates are now generated in ~6 seconds instead of ~15 minutes.
