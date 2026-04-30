# SOTA Submission: 1.1565 BPB @ 5.64MB (10min/16mb Track)

This PR submits a new State-of-the-Art (SOTA) entry for the **10min/16mb** track, achieving **1.1565 BPB** with an artifact size of **5.64MB**.

### 🚀 Key Improvements & Technical Details

1.  **Architecture: Depth Recurrence + Parallel Residuals**
    *   Implements a looped layer structure (layers 4-5 repeated twice) to increase effective depth without increasing parameter count.
    *   Utilizes **Parallel Residuals** (GPT-J style) from layer 0-10, allowing attention and MLP to be computed in parallel for better gradient flow.
    *   Includes **Untied Loop MLPs**: Attention weights are shared across loops, but MLPs are untied to capture loop-specific state.

2.  **Quantization: Hessian-aware SDClip + GPTQ**
    *   Uses **GPTQ** for all matrix weights (int6) and embedding weights (int8).
    *   Implements **Hessian-aware SDClip**: Clipping ranges are modulated by the diagonal of the Hessian, prioritizing preservation of high-importance features.
    *   All dequantization operations utilize `bfloat16` to ensure precision alignment with the training regime.

3.  **Serialization: ByteShuffle + LZMA**
    *   Implements a custom **ByteShuffle** algorithm prior to compression to improve LZMA efficiency on quantized integer streams.
    *   The final artifact `final_model.ternary.ptz` is a standard XZ-compatible stream (lzma) containing the shuffled state dict.

### 📊 Performance Summary

*   **Track**: 10min/16mb
*   **Validation Loss**: 2.9869
*   **Validation BPB**: 1.1565
*   **Artifact Size**: 5,645,856 bytes (5.38 MiB)
*   **Training Time**: ~9.8 minutes on a single T4 GPU.

### 🛠️ Reproduction Instructions

1.  Open the provided notebook: `notebooks/Parameter_golf.ipynb`.
2.  Install dependencies: `pip install -r records/track_10min_16mb/hardik-sota-final/requirements.txt`.
3.  Set environment variables:
    ```bash
    export DATA_DIR="./data/"
    export MAX_WALLCLOCK_SECONDS="600"
    export TERNARY_TARGET_BYTES="5645856"
    ```
4.  Run the script: `python records/track_10min_16mb/hardik-sota-final/train_gpt.py`.

---
*Note: This submission addresses all previous feedback regarding environment variable typos, precision casting, and script-artifact synchronization.*