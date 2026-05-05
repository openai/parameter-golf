---
title: "PPM-D Byte-Level Scoring Report Draft"
---

# Report: PPM-D byte-level scoring is not a valid probability distribution, and why it appears to gain

This is an investigation of the recent byte-level PPM-D mixture submissions
(https://github.com/openai/parameter-golf/pull/1835,
https://github.com/openai/parameter-golf/pull/1850, https://github.com/openai/parameter-golf/pull/1854,
https://github.com/openai/parameter-golf/pull/1858,
https://github.com/openai/parameter-golf/pull/1862, https://github.com/openai/parameter-golf/pull/1833,
https://github.com/openai/parameter-golf/pull/1871, https://github.com/openai/parameter-golf/pull/1865,
https://github.com/openai/parameter-golf/pull/1885 (mine)
-- the cluster flagged in https://github.com/openai/parameter-golf/issues/1872). The cluster reports
val_bpb ranging from 0.90 to 1.014, all using the same scoring construction: standard token-level NN
log-probability "bit-conservingly spread" across each token's bytes, mixed in probability space with a
classical byte-level PPM-D order-5 model. We focus on
https://github.com/openai/parameter-golf/pull/1850 as a clean reference implementation -- it isolates
the mechanism without additional architectural changes, and the same scoring formula appears verbatim
in the others. This report is not about the `Σ_tok` vs `Σ_byte` dispute in
https://github.com/openai/parameter-golf/issues/1872. The problem is more fundamental: the
uniform-spread NN side is not a valid probability distribution over 256 bytes, and therefore does not
satisfy C2.

All errors in the analysis below are mine and mine alone.

## Contents

This report has two parts.

Part 1 -- The PPM-D byte-level mixture is not a valid probability distribution.

Part 2 -- We investigate how this scoring system erroneously produces the reported gain.

## Part 1 -- The PPM-D byte-level mixture is not a valid probability distribution

This part is due to @sharpobject in https://github.com/openai/parameter-golf/issues/1872. I am
restating that argument here because it is the premise for Part 2.

Under the byte-level reading of C2, the scorer must define a probability distribution over the 256
possible next bytes at each scored position.

The submission class scores bytes via

$$
p_{\mathrm{mix}}(b)=\lambda p_{\mathrm{uniform}}(b)+(1-\lambda)p_{\mathrm{PPM}}(b),
$$

where $p_{\mathrm{PPM}}$ is the PPM-D byte distribution and $p_{\mathrm{uniform}}$ is obtained by
uniformly spreading token log-probability across bytes.

Since $p_{\mathrm{PPM}}$ already sums to 1, $p_{\mathrm{mix}}$ can sum to 1 only if
$p_{\mathrm{uniform}}$ does. So the question reduces to whether the NN-side byte object is itself a
probability distribution.

For the first byte of a token, the natural extension is

$$
p_{\mathrm{uniform}}(b)=\sum_{t=b\ldots} p_{\mathrm{NN}}(t)^{1/n(t)}.
$$

For later bytes one conditions on the already-realized within-token prefix first; the same normalization
problem remains, so I ignore that extra notation here.

But this object does not sum to 1. The reason is simple: for any multi-byte token with $0<p<1$, one
has $p^{1/n}>p$. So the uniform-spread construction systematically inflates small token probabilities
before summing them by byte.

A toy example already breaks normalization. Suppose

$$
p_{\mathrm{NN}}(t_1)=0.25,\qquad p_{\mathrm{NN}}(t_2)=0.25,\qquad p_{\mathrm{NN}}(t_3)=0.5,
$$

where $t_1,t_2$ are two-byte tokens starting with `a`, and $t_3$ is a one-byte token starting with
`b`. Then

$$
p_{\mathrm{uniform}}(\texttt{a})=0.25^{1/2}+0.25^{1/2}=1,\qquad
p_{\mathrm{uniform}}(\texttt{b})=0.5,
$$

so

$$
p_{\mathrm{uniform}}(\texttt{a})+p_{\mathrm{uniform}}(\texttt{b})=1.5>1.
$$

Therefore $p_{\mathrm{uniform}}$ is not a probability distribution over bytes, and neither is
$p_{\mathrm{mix}}$. This is exactly the C2 failure pointed out in Issue #1872.

The natural byte-level object induced by the same token softmax is instead the conditional distribution

$$
p_{\mathrm{cond}}(b\mid \pi)=
\frac{\sum_{t:\pi b \preceq \mathrm{bytes}(t)} p_{\mathrm{NN}}(t)}
     {\sum_{t:\pi \preceq \mathrm{bytes}(t)} p_{\mathrm{NN}}(t)},
$$

where $\pi$ is the within-token byte prefix already realized. Unlike the uniform-spread construction,
this is a genuine distribution over next bytes. Part 2 uses it as the correct reference point.

## Part 2 -- Why the apparent gain comes from the scoring system, not from PPM itself

The uniform-spread construction was chosen for a reason: if PPM is turned off, summing byte losses
reproduces the original token-level score exactly. But that same bookkeeping choice is what makes PPM
appear much stronger than it really is.

What the uniform-spread distribution tends to do is move uncertainty from the later parts of a token
toward the front and flatten it out across all of the token's bytes. In other words, it takes token
loss that in a natural conditional view would be concentrated on a few genuinely uncertain later bytes,
and redistributes that loss onto earlier bytes that may already be almost certain. The clearest
examples are tokens whose first byte is a space: the model may be very sure that the next byte is ` `,
while still being unsure which full token follows after that. Uniform spread erases that distinction and
charges the early space byte as if it carried an equal share of the token's uncertainty.

All numbers below use a post-quantized model, together with the same PPM configuration used in #1850.

The key comparison is this:

| Method | val_bpb |
|---|---:|
| Token baseline | 1.08335 |
| Uniform-spread + PPM | 1.03242 |
| Conditional distribution + PPM | 1.12144 |

So under the submitted uniform-spread scoring rule, PPM appears to gain about 0.051 val_bpb. But under
the conditional distribution, the very same PPM scorer is not better than the baseline at all: it is
worse by about 0.038 val_bpb.

That is the central empirical fact of this report. The apparent gain is not coming from PPM
outperforming the model on a valid next-byte scoring problem. It is coming from replacing the
conditional distribution with the uniform-spread one before mixing.

A concrete token-level example makes the mechanism clear.

Consider the real token `" today"` in the context
`...half the total starters) and today it is 17 of the 24...`.

For this token, the two scoring systems assign the same total baseline loss,
but distribute it very differently across bytes:

| byte | gate_hi | uniform | conditional | PPM | mix(uniform) | mix(conditional) | gain vs uniform | gain vs conditional |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ` ` | 1 | 1.47564 | 0.00841 | 0.01511 | 0.05426 | 0.01478 | +1.42138 | -0.00637 |
| `t` | 0 | 1.47564 | 1.51869 | 1.94185 | 1.51362 | 1.55381 | -0.03797 | -0.03511 |
| `o` | 0 | 1.47564 | 2.38802 | 1.93680 | 1.51329 | 2.33256 | -0.03764 | +0.05546 |
| `d` | 0 | 1.47564 | 4.93874 | 5.98645 | 1.57978 | 5.00587 | -0.10414 | -0.06713 |
| `a` | 1 | 1.47564 | 0.00000 | 0.06454 | 0.10308 | 0.06121 | +1.37257 | -0.06121 |
| `y` | 1 | 1.47564 | 0.00000 | 0.05716 | 0.09579 | 0.05422 | +1.37985 | -0.05422 |

Token totals:

| token | uniform baseline | conditional baseline | uniform+PPM mix | conditional+PPM mix | PPM gain vs uniform | PPM gain vs conditional |
|---|---:|---:|---:|---:|---:|---:|
| `" today"` | 8.85387 | 8.85387 | 4.85983 | 9.02245 | +3.99404 | -0.16858 |

This is the key point. For the very same realized token, the submitted scoring rule makes PPM look
helpful by `+3.99` nats, while the conditional distribution shows that the same PPM configuration is
actually slightly harmful (`-0.17` nats).

The reason is visible byte-by-byte. Uniform spread assigns `1.47564` nats to every byte, including the
easy bytes ` `, `a`, and `y`, where the conditional distribution is already essentially zero and where
`gate_hi=1` gives PPM high weight. PPM then gets credit for “improving” those bytes only because the
scoring rule first assigned them artificial cost.

So the sign of the apparent token-level gain flips depending on how the same token loss is allocated
across bytes. That is exactly the claim of Part 2: the reported gain is not a stable property of PPM
itself, but of the scoring rule used to mix it with the model.

This shows that a large part of the reported gain is created by the scoring construction itself. Under
the conditional distribution, the same PPM configuration does not improve the score; it makes it worse.
So the headline improvement is not evidence that PPM is winning on a valid next-byte prediction problem.

## Reproducibility

- `testing/inspect_with_ppm.py` runs the uniform-spread versus conditional-distribution comparison with
  the same PPM configuration.
- `testing/ppm_scorer.c` is the byte-level PPM scorer used in both comparisons.
- `testing/show_5_artifact_examples.py` regenerates the worked artifact examples from a saved per-byte
  dump.
