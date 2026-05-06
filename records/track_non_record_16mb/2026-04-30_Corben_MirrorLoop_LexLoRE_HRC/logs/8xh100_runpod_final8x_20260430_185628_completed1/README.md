# 8xH100 Completed First-Row Snapshot

This directory preserves the first completed 8xH100 row from the initial final8x matrix, plus the partial second row that was stopped after the first result showed the same-shape e832 configuration exceeded the 16MB artifact cap.

Completed row:

| Candidate | Final export BPB | Train-time val BPB | Steps | Step avg | Artifact bytes | Headroom |
|---|---:|---:|---:|---:|---:|---:|
| `final8x_196k_r2_d704e832_w2200_wd02_lqer8t16_vocabmoe_qk55` | `1.35704747` | `1.3174` | `6628` | `90.54ms` | `16,413,081` | `-413,081` |

Interpretation: the 8x run used the GPUs well and improved train-time validation, but export/compression pushed this e832 row over the decimal cap. The follow-up queue switched to e768 legalizer candidates rather than spending more paid time on larger LQER variants with the same over-cap shape.
