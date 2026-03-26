# Short Version

This PR documents the cleanest pure raw-byte JEPA attempt we ran for Parameter Golf. The best result was **`2.3839 bpb`** with `transformer_rope_gqa_localglobal + slot_ema_teacher`, which is a real improvement over our earlier pure-JEPA runs but still far from the simple baseline `1.2244`.

# What Makes This “Pure JEPA”

- raw `byte260`
- no tokenizer
- no exact byte-NLL into the backbone
- backbone trained only with JEPA-style latent prediction plus regularization
- exact byte prediction only through a later detached Transformer probe on frozen features

So the clean question here is: can a pure raw-byte JEPA backbone, trained without exact-loss gradients, carry enough information that a later detached exact decoder can recover good `bpb`?

# Top-Line Results

| Result | `bpb` | Notes |
|------|------:|------|
| Best pure detached-probe result | `2.3839` | `transformer_rope_gqa_localglobal + slot_ema_teacher` |
| Earlier purity-first milestone | `2.8583` | raw-byte JEPA with coupled exact decoder term |
| First clean frozen-probe milestone | `3.0774` | earlier pure-probe campaign |
| Best backbone screen result | `2.3889800525604903` | `transformer_rope_gqa_localglobal` |
| Best encoder-screen result | `2.746384624395377` | `conv_patch`, still worse than the best objective-screen result |

# How This Compares to Other JEPA Attempts

| PR | Setup | Reported result | Interpretation |
|------|------|------:|------|
| This PR | pure raw-byte detached-probe JEPA | `2.3839` | no exact-loss gradients into the backbone |
| [#708](https://github.com/openai/parameter-golf/pull/708) | byte-level hybrid JEPA + exact next-byte scorer | about `2.1252` | stronger numerically, but exact next-byte loss is in the main training path |
| [#896](https://github.com/openai/parameter-golf/pull/896) | tokenized JEPA self-distillation on top of autoregressive LM | PR author reports vanilla CE wins by `0.005 BPB` and is `40%` faster | useful negative result, but not raw-byte pure JEPA |
| [#903](https://github.com/openai/parameter-golf/pull/903) | LeWorldModel-style JEPA + SIGReg + Mamba-2 SSM + CE head, plus a detached diagnostic probe | reported `1.2064` sliding / `1.2235` standard for best long BPE; `1.3348` standard for 10-minute byte | closest comparison, but the main reported model is still CE-trained and the JEPA-only contribution remains open |

PRs #708 and #896 are hybrid or auxiliary-loss approaches. PR #903 is closer to this work because it also includes a detached probe diagnostic, but its main reported model is still a CE-trained JEPA-augmented system rather than a pure backbone-only JEPA path. So none of them are apples-to-apples comparisons to this PR.

# Main Negative Findings

- Stronger Transformer + slot targets helped a lot, but pure JEPA still remained far above baseline.
- Objective changes were small once the slot-target family was in place.
- Richer patch encoders mostly did not help.
- Lower JEPA loss did not reliably translate into lower exact-byte `bpb`.
- The main bottleneck now looks like latent/interface design, not just backbone size or JEPA loss choice.

# Why This Still Matters

This PR isolates the “pure JEPA” question more cleanly than the hybrid JEPA-related PRs in the repo. That makes it a useful lower bound and negative control for future JEPA claims: the best-performing JEPA-adjacent results still rely on a strong main CE path, which strengthens rather than weakens the negative result from the pure setup.
