# Stage 1 Frontier Family R2

This package replaces the old config-heavy family as the active runner scaffold.

Execution rule:

- use A100 runs to rank family strength
- use the same family ids as the Stage 1 doctrine
- promote only clear survivors to `8xH100`

## Slots

| Slot | Family | Role | Why it exists now |
| --- | --- | --- | --- |
| P0 | baseline | comparator | control run for all public-frontier probes |
| P1 | `M02` long-context training geometry | scout | public records say longer context is real |
| P2 | `M09` export-aware quantization | exploit | fp16 embedding passthrough is one of the cleanest public wins |
| P3 | `M06` optimizer partitioning | exploit | public top run validates AdamW-for-some and Muon-for-others |
| P4 | `M11` evaluation-time context | exploit | sliding-window eval is a first-order public frontier |
| P5 | `M08` wallclock-aware schedule | exploit | always-on decay and long warmdown look real |
| P6 | `M01` architecture reallocation | alternate | keeps one non-public-following architecture anchor alive |
| P7 | `M07` adaptive Muon compute | alternate | keeps our best optimizer-systems family alive |

## Notes

- `P1-P5` are the active public-frontier set.
- `P6-P7` are retained as disciplined alternates so the search does not collapse entirely into public imitation.
- This runner package is substrate only. It does not launch anything by itself.
