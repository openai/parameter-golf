AnchorPrefixCodec Submission
1. Overview

This submission implements a structure-first compression codec with a train-only constrained fallback mechanism.

The system focuses on reducing uncertainty before applying probabilistic modeling.

Pipeline:

Deterministic structure recovery
Prefix-retained fallback encoding
Train-only candidate constraint on first suffix piece
2. Result (Proxy Only)

codec_proxy_bpb_official_byte_lut: 0.8044310761489103

This is NOT an official eval_val result.

This value uses:

Official byte LUT denominator
Codec structural bits (not model cross-entropy)
3. Important Clarification

This is a codec-space estimator, not a model-space estimator.

It does NOT optimize model P(x).
It reduces entropy through structure and constrained decoding.

4. Files
train_gpt.py
codec_base.py
codec_prefix.py
prepare_canonical_units.py
submission.json
records/
5. Pipeline Description

Input tokens
-> decode into canonical units

-> structure recovery
-> anchor
-> pattern
-> compound

-> residual recovery (causal only)

-> fallback decoding
-> prefix_keep = 3
-> suffix language model
-> online scoring (score-first)
-> first-piece constraint (train-only)

-> bit estimation

6. Core Idea

Compression is achieved by:

Removing entropy via structure
Reducing uncertainty via prefix
Shrinking candidate space via constraints
Leaving minimal uncertainty to the language model
7. First-Piece Constraint

Mechanism:

P(piece | context)
-> P(piece | context restricted to candidate set)

Candidate set:

Built only from training data
Maps retained prefix to next-piece distribution

Usage:

Applied only to the first predicted suffix piece
Used only if it reduces bit cost
No future context is used
8. Legality Notes

This submission is designed to respect Parameter Golf constraints:

Train-only statistics
No validation data leakage
No right-context usage
Online scoring without future tokens
Illegal actions are repriced as fallback

This has not yet been verified using official eval_val.

9. Reproducibility

Environment:

$env:PYTHONPATH="."
$env:PREFIX_KEEP="3"
$env:ONLINE_WEIGHT="1.0"
$env:CONSTRAINT_ENABLED="1"
$env:MAX_CONSTRAINT_PIECES="2"

Run:

python ".\train_gpt.py" --seed 0
10. Limitations
Result is proxy only, not official evaluation
Constraint is applied only to first suffix piece
No modification to model logits
Evaluation is performed in codec space
11. Key Findings
Structure-based recovery provides the largest gain
Prefix fallback is efficient
Candidate space constraint improves compression

Rejected directions:

Right-context optimization
Direct model probability modification
Reducing fallback count as primary objective
12. Insight

The key idea is:

Do not improve probability directly
Instead, reduce the search space of valid continuations

13. Status

Structure compression: complete
Fallback system: complete
Constraint system: active
Official evaluation: pending

