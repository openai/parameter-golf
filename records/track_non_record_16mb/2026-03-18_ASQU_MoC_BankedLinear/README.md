This work explores whether architectural changes can improve intrinsic capacity per parameter under fixed-step training budgets.

We introduce three components — ASQU, MoC, and BankedLinear — and evaluate their effects through controlled ablations.

---

## Results

Ablations are run from the base train_gpt.py script for a fixed 10k steps. MLP expansion is adjusted as needed to remain within the 16MB limit.

| Model Variant | Pre-quant BPB | Post-quant BPB | Size (bytes) | MLP Mult |
|--------------|--------------:|---------------:|-------------:|---------:|
| Baseline | 1.2262 | 1.2328 | 15861272 | 2.00 |
| + ASQU | 1.2232 | 1.2301 | 15898146 | 2.00 |
| + Short Conv (k=1) | 1.2157 | 1.2217 | 15973462 | 1.99 |
| + MoC (k=8) | 1.2121 | 1.2182 | 15911167 | 1.93 |
| + BankedLinear | 1.2098 | 1.2164 | 15852659 | 2.6 |

Across these changes, we observe an improvement of ~0.016 bpb over baseline under identical training conditions.

---

## ASQU (Asymmetric Squared Unit)

ASQU extends ReLU^2 by learning a per-channel scaling for the negative branch.

Per-channel parameterization provides ~0.001 bpb over learning a global scalar.

---

## MoC (Mixture of Convolutions)

Short convolutions are highly effective in this regime, providing consistent improvements despite their low parameter cost. However, in their standard form, all tokens share the same convolutional weights regardless of identity.

MoC lifts this restriction by allowing each token to use its own dynamically generated convolution, constructed as a mixture over a small shared set of basis kernels.

Basis interpolation offers enough expressivity to be useful without making optimization unstable.

The gate/router input is the same hidden state used to generate QKV.


## BankedLinear

BankedLinear replaces standard projections with a shared weight basis across layers, where each layer constructs its weights as a learned mixture of:
- a small set of learned weight matrices
- a larger set of fixed random projections

Fixed random projections provide a cheap, high-dimensional basis from which weight matrices can be constructed. They enable the model to synthesize structured weight matrices as mixtures over many fixed components.

In practice, removing the random projections significantly degrades performance, suggesting that combining many simple fixed components is an effective way to approximate more expressive transformations under a fixed parameter budget.

When focusing on the learned weights, BankedLinear can be viewed as a form of continuous, learned weight reuse across depth, allowing layers to share structure without being identical, and in a way that is guided by the model’s learned mixing coefficients.

A depth-aware initialization of the mixing coefficients on learned layers is important for best performance. 

This approach enables parameter reuse across layers, allowing saved capacity to be reinvested elsewhere.

We use 9 total layers, and with 3 learned projections total, and 512 fixed random projections.

MoC and BankedLinear can be viewed as related forms of mixture-based parameterization: MoC constructs dynamic mixtures over layer-local operators on a per-token basis, while BankedLinear constructs mixtures over a global weight basis shared across layers.

This suggests a broader design space combining dynamic and shared mixtures (e.g. globally banked operators with token-dependent routing).


## Practical Considerations
All experiments were done on fixed steps, so it is unclear which components remain competitive under strict wallclock-constrained settings.

some things to keep in mind if you do want to try to integrate something:
- ASQU does not currently have a fused implementation, and does not offer a massive gain over Leaky ReLU^2, so unclear if beneficial given the overhead of the current naive implementation, though if cost is the same, ASQU should perform modestly better.
- MoC is significantly more expensive and unlikely to be competitive under strict wallclock constraints.
- Basic short conv is much more efficient than MoC due to available efficient implementations. Short conv likely makes smear + bigram redundant, and short conv likely has better performance but is still moderately less efficient (though also less parameters), unclear which one is better.
- BankLinear has a manageable overhead but needs significant steps to show advantage due to more complex optimization, unclear if there is enough time for advantage to emerge, or if advantage is greater than overhead induced.
