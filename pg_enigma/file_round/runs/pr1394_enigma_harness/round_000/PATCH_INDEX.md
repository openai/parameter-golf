# Patch index

## Intended tournament reading order

| Order | File | Role | One-line rationale |
| --- | --- | --- | --- |
| 1 | `controls/C0.md` | Control | Published baseline anchor for every downstream comparison. |
| 2 | `controls/C1.md` | Control | Exact replay anchor used to measure control spread before promotion. |
| 3 | `candidates/H0.md` | F0 | Same-trace late EMA checkpoint selector that tests whether final is really best. |
| 4 | `candidates/H1.md` | F1 | Fixed alternate export profile that reallocates entropy budget with int7 plus `lzma`. |
| 5 | `candidates/H2.md` | F1 | Same-checkpoint export bakeoff that selects between two existing-menu recipes. |
| 6 | `candidates/H4.md` | F6 | Class-specific bit allocation rewrite for attention versus MLP tensors. |
| 7 | `candidates/H5.md` | F6 | Class-specific clip-law rewrite that breaks the one-matrix-rule invariant. |
| 8 | `candidates/H3.md` | F2 | Transition-band recurrence handoff, kept last because it changes training rather than post-train export only. |

## File checklist

| File | Rationale |
| --- | --- |
| `controls/C0.md` | Canonical baseline anchor with no novel mechanism. |
| `controls/C1.md` | Replay anchor for control-spread estimation. |
| `candidates/H0.md` | Uses the sole `F0` budget on a downstream checkpoint selector. |
| `candidates/H1.md` | First `F1` slot: one coordinated alternate export profile. |
| `candidates/H2.md` | Second `F1` slot: same-checkpoint export tournament instead of one fixed recipe. |
| `candidates/H3.md` | Uses the lone `F2` slot on an explicit handoff program, not a naked threshold nudge. |
| `candidates/H4.md` | First `F6` slot: bit-allocation rewrite by tensor class. |
| `candidates/H5.md` | Second `F6` slot: clip-law rewrite by tensor class. |

## Family-budget check

- `F0`: `H0`
- `F1`: `H1`, `H2`
- `F2`: `H3`
- `F6`: `H4`, `H5`

This uses only surviving families with positive budget, respects the `1/2/1/2` allocation from `FAMILY_INDEX.md`, and keeps four distinct source families alive across the six candidates.
