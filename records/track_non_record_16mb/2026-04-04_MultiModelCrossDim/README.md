# Multi-Model Cross-Attention with Dimensional Asymmetry

### Parameter Golf -- Non-Record Submission (16MB Track)

## Summary

This submission explores a **multi-model architecture with shared
representation and cross-attention across asymmetric dimensions**,
trained on a single NVIDIA 5090 GPU.

The core finding is that:

-   **Loss curve shape is largely invariant across model types**
-   **Model dimension controls the shape of the curve**
-   **Model type controls how efficiently the model traverses that
    curve**

By combining different model types at different dimensions into a shared
representation, we achieve improved performance with fewer parameters.

------------------------------------------------------------------------

## Final Results

-   **Validation Loss:** 2.1021
-   **Validation BPB:** 1.2450
-   **Int8 Roundtrip Loss:** 2.1086
-   **Total Submission Size (int8+zlib):** \~13.6 MB

------------------------------------------------------------------------

## Model Configuration

-   **Model Type:** multi_model_single_representation

-   **Number of Models:** 3

-   **Model Types:** transformer, mlp, causal_depthwise

-   **Model Dims:** 468, 198, 186

-   **Shared Representation Dim:** 852

-   **Cross Attention Dim:** 480

-   **Layers:** 4

-   **Recurrence:** 1

-   **Heads:** 6

-   **KV Heads:** 2

-   **Conv Kernel:** 4

------------------------------------------------------------------------

## Key Findings

### 1. Loss Curve Shape is Invariant

Across all experiments:

-   Different model types produced **nearly identical loss curve
    shapes**
-   Specific features (e.g., spikes at step \~4000) appeared
    **consistently across runs**

This suggests the dataset and optimization regime define the difficulty
landscape, not the model architecture.

------------------------------------------------------------------------

### 2. Model Type Affects Ease of Learning

Different model types traverse the curve more or less efficiently:

-   vertical shifts (lower/higher loss)
-   smoother vs sharper transitions

Complementary pairs outperform identical pairings.

------------------------------------------------------------------------

### 3. Dimension Controls Curve Shape

Model dimension directly alters the shape of the loss curve.

------------------------------------------------------------------------

### 4. Cross-Dimensional Attention Improves Efficiency

Asymmetric dimensions + shared representation + cross-attention improved
performance and stability.

------------------------------------------------------------------------

### 5. Model Combinations Matter

Some combinations produce smoother learning curves, others sharper
spikes.

------------------------------------------------------------------------

### 6. Recurrence Amplifies Behavior

Recurrence reinforces existing model behavior.

------------------------------------------------------------------------

### 7. Tradeoffs

-   speed vs quality
-   capacity vs stability

------------------------------------------------------------------------

## Conclusion

-   Dimension dominates training dynamic
s
-   Model type influences efficiency
-   Cross-attention enables useful interaction


## Notes

This submission is not intended as a record attempt, but as an exploration of ideas building on [previous work](https://github.com/alientony/Split-brain). The goal was to test alternative architectural directions and contribute additional observations to ongoing research.

The results suggest that cross-attention across asymmetric model dimensions may offer a useful direction for further study.

These ideas are open for extension and modification. Others are encouraged to explore, adapt, or build upon this approach.

![Dim Test 1](https://github.com/user-attachments/assets/eeead890-bb10-4b10-8f97-fd210e693870)
![Dim Test 2](https://github.com/user-attachments/assets/b4a51f3f-77ab-4436-b9a2-fb16b33e8791)
![Dim Test 3](https://github.com/user-attachments/assets/beb20316-08cf-4129-aa8e-1764fd2ccd69)
