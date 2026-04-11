# Muon "Polar Express" (Degree-5 Minimax)

To break the "Step 2 Explosion" at high model widths (768-1024), we transitioned from the standard Degree-7 Newton-Schulz polynomial to a custom **Quintic Minimax** (Degree-5) polynomial.

## 1. The Coefficients
The "Polar Express" polynomial $P(x)$ for matrix orthogonalization ($X_{n+1} = X_n \cdot P(X_n^T X_n)$) uses the following degree-5 minimax coefficients:

- **$c_0$**: $3.4445$
- **$c_1$**: $-4.7750$
- **$c_2$**: $2.0315$

**Polynomial**: $3.4445I - 4.7750(X^TX) + 2.0315(X^TX)^2$

## 2. Advantages over Degree-7

| Feature | Degree-7 (Standard) | Degree-5 (Polar Express) |
| :--- | :--- | :--- |
| **Derivative at 0** | ~4.375 | **~3.4445** |
| **Spectral Radius** | ~1.65 | **~2.10+** |
| **Convergence Rate** | 7th order (Fast) | 5th order (Stable) |
| **Step 2 Stability** | High risk of explosion | **Extremely Robust** |

## 3. Implementation Details
- **FP32 Accumulation**: Even if the model is training in `bf16`, the Polar Express iterations are executed in **FP32** to prevent cumulative rounding errors in the $X^2$ terms.
- **Backend Steps**: Set `MUON_BACKEND_STEPS=5`. This ensures the matrix is near-perfectly orthogonal ($X^TX \approx I$) after each update.
- **Momentum Warmup**: The Muon momentum starts at `0.85` and ramps to `0.95` over 100 steps to allow the orthogonalization to settle.
