# Parameter Golf Marathon Submission: 128-Cluster Specialist Model

This repository contains the final standalone submission for the **16MB Parameter Golf Challenge**.

## Model Overview: Specialist 128-Cluster Architecture
The model is a highly specialized Transformer with the following configuration:
- **Architecture**: 800D, 6 Layers, 10 Heads per Layer.
- **Specialist MoE**: 128 Bigram-Gating Specialists (FastClusterGating).
- **Optimization**: 10,000 Step Marathon training using **Muon** for internal representations and **Adam** for clusters.
- **Weight Averaging**: Stochastic Weight Averaging (SWA) for enhanced generalization.

### Performance Records (Local RTX 2080 Ti)
| Metric | Value |
| :--- | :--- |
| **SWA BPB (Pre-Quant)** | **1.3366** |
| **Quantized BPB (Final)** | **1.3540** |
| **Submission Size** | **15.15 MB** |
| **Est. Server BPB** | **1.20 - 1.25** (Projected for 8H100) |

---

## 🛠️ Rule Compliance & Architecture
- **Hard 10-Minute Wallclock**: This script includes a hard **600-second (MAX_WALLCLOCK_SECONDS)** break to ensure strict compliance with the competition's training time limit.
- **Strict 16MB Guard**: Automatically calculates and enforces the **16,000,000 byte** submission limit (Compressed weights + Python code size).
- **H100 Performance Target**: Designed to target a **~10-minute runtime** on 8x H100 GPUs using `torch.compile` and adaptive batch scaling.
- **Specialist MoE**: The 128-cluster specialist architecture captures high-frequency linguistic patterns while keeping the global parameter count extremely low.
- **Fair-Start**: No pre-trained weights or external datasets. Initialized from scratch.

---

## Submission Package Contents
- `train_gpt.py`: Consolidated script containing architecture, training, and quantization.
- `submission.json`: Leaderboard metadata.
- `server_setup_guide.md`: Detailed H100 deployment guide.

---

## Instructions: Running on Server (e.g., H100 GPU)

For a final validation run on a clean server environment:

1.  **Set Environment Variables**:
    ```bash
    export DATA_PATH="/path/to/fineweb10B_sp1024"
    export TOKENIZER_PATH="/path/to/fineweb_1024_bpe.model"
    ```
2.  **Option A: Immediate Validation & Quantization**:
    Run the script with 0 iterations to trigger immediate quantization and validation of existing weights (if present):
    ```bash
    ITERATIONS=0 python train_gpt.py
    ```
    
3.  **Option B: Full Server Record Run (Recommended)**:
    Use the maximum power of the H100/A100 (80GB VRAM):
    ```bash
    export TRAIN_BATCH_TOKENS=524288
    export USE_COMPILE=1
    python train_gpt.py
    ```
    This mode uses automated kernel compilation and massive batches for maximum throughput.

---

## Quantization Strategy
The model uses a hybrid **5-Bit MLP / 6-Bit Rest** quantization scheme. By shifting values to specific sparse bitmasks (uint8 shift trick), we maximize `zlib` compression efficiency to stay comfortably under the **16,000,000 byte** limit.

Contact the developer for detailed ablation studies on the 128-specialist gating mechanism.
