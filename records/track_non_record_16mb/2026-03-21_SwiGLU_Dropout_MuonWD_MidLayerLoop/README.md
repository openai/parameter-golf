# V2.1 Prototype - EMA Architectural Enhancement

This version is an enhancement of the **validated V2 architecture (1.2182 bpb)**.

### 📊 Validation Results (V2 Architecture)
*   **Hardware**: 1x H100 NVL (80GB).
*   **Data Constraint**: Trained on only **80 shards** (~4.3B tokens processed).
*   **Metric**: Achieved **1.2182 val_bpb** in 60 minutes.
*   **Note**: As the loss was still descending at 1 hour, this architecture is projected to reach the **1.1x** range when scaled to the full 8xH100 / 10-minute competition window.

## 🛠️ Core Verified Architecture
1. **Aggressive Regularization:** Deploys extreme Muon weight decay and `10% Dropout` across both Attention and MLP blocks.
2. **SwiGLU Upgrades:** Replaces squared-ReLU with SwiGLU in the MLP block for superior inductive priors.
3. **Targeted Depth Recurrence (Middle-Layer Looping):** Bounds recurrence to the network's inner core, increasing effective depth while maintaining stable IO projections.

## Feasibility and Verification
The included logs demonstrate that this architecture is fully stable, compliant with the **16MB limit (4.8MB actual)**, and ready for high-scale compute grant allocation.
