This submission explores architectural changes aimed at improving capacity per parameter separate from throughput considerations.

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


BankedLinear replaces standard linear projections with a shared weight bank across layers. Each layer constructs its weights as a learned mixture over:

- a small set of learned weight matrices
- a larger set of fixed random projections

This can be viewed as a generalization of prior work on weight sharing / recursive transformers, where layers are no longer forced to reuse identical weights, but learn a mixture from a shared pool of weights.

A depth-aware initialization of the mixing coefficients on learned layers is important for best performance. 

This approach enables parameter reuse across layers, allowing saved capacity to be reinvested elsewhere.

We use 9 total layers, and with 3 learned projections total, and 512 fixed random projections.


## General Notes
All experiments were done on fixed steps, so not currently known which are competitive on main leaderboard.

some things to keep in mind if you do want to try to integrate something:
- ASQU does not currently have a fused implementation, and does not offer a massive gain over Leaky ReLU^2, so unclear if beneficial given the overhead of the current naive implementation, though if cost is the same, ASQU will almost certainly perform modestly better.
- MoC is prohibitively slow and would not be competitive.
- Basic short conv is much more efficient than MoC due to available efficient implementations. Short conv likely makes smear + bigram redundant, and short conv likely has better performance but is still moderately less efficient (though also less parameters), unclear which one is better.
- BankLinear has a manageable overhead but needs significant steps to show advantage due to more complex optimization, unclear if there is enough time for advantage to emerge, or if advantage is greater than overhead induced.
