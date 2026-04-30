# Run 2 Summary: Yahya's byte-token handling — BUG_PRESENT

## Verdict
**BUG_PRESENT.** Yahya's `train_gdn_7k.py` (lines 206-219) has no `sp.is_byte` branch in `build_sentencepiece_luts`. Byte tokens fall through to `base_bytes[i] = len(piece.encode("utf-8"))`. For every byte piece (`<0x00>`, `<0x01>`, ..., `<0xFF>`), this gives 6 bytes. Canonical PR #1727 code assigns 1.

## Numerical evidence

| | yahya | canonical | delta |
|---|---|---|---|
| Per-byte-token byte count | 6 | 1 | +5 |
| Total over 256 byte tokens in vocab | 1,536 | 256 | +1,280 |
| Byte tokens in val (40.5M tokens) | 269,220 occurrences | same | — |
| Contribution to byte sum in val | 1,615,320 | 269,220 | +1,346,100 |

## Code (yahya's lines 206-219)

```python
def build_sentencepiece_luts(sp, vocab_size, device):
    base_bytes = torch.zeros(vocab_size, dtype=torch.float32, device=device)
    has_space = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    is_boundary = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    for i in range(vocab_size):
        piece = sp.id_to_piece(i)
        raw = piece.encode("utf-8")
        base_bytes[i] = len(raw)            # NO sp.is_byte branch
        if piece.startswith("\u2581"):
            has_space[i] = True
            base_bytes[i] = len(piece[1:].encode("utf-8")) + 1   # +1 bug
        if sp.is_control(i) or sp.is_unknown(i):
            is_boundary[i] = True            # missing sp.is_unused
    return base_bytes, has_space, is_boundary
```

Three deviations from canonical PR #1727 are visible directly:
1. **Byte-token bug** (no `sp.is_byte` branch). Confirmed PRESENT this run.
2. **Leading-space `+1` bug** (line 216). Yahya's own self-disclosure.
3. **Missing `sp.is_unused`** (line 217 boundary predicate). Confirmed PRESENT.

## Implication for the audit tool

The current `canonical_rescore.py` returns INDETERMINATE for the byte-token detector when a script has no `sp.is_byte` branch. This is a false negative when the default branch produces a non-canonical value. The detector should either:

- (a) Tighten: when no `sp.is_byte` branch is present AND the default branch is `len(piece.encode("utf-8"))`, classify as DEVIATES, OR
- (b) Stay conservative but document the false-negative case explicitly in methodology.md

We propose (b). The static detector cannot in general know what the default branch evaluates to for byte pieces without execution. INDETERMINATE remains a valid conservative call. But the methodology should note that "INDETERMINATE for byte-token does NOT mean the byte-token handling is correct; it only means the detector cannot statically verify."

## Implication for the residual ratio gap

The byte-token bug increases yahya's canonical denominator by 1,346,100 bytes, which decreases his `buggy/canonical` ratio relative to ours. This works in the *opposite* direction from what's needed to explain his 1.1746 vs our 1.1671 gap. So the byte-token bug is real but cannot, by itself, explain the residual gap. Run 3 will reconstruct yahya's full LUT and compute the actual ratio it produces.

## Files
- run2_yahya_byte_token_check.py / .json / .log
