# 8xH100 Legal Fallback Completed Two Rows

This directory preserves the first two completed legal-size fallback rows from the one-hour 8xH100 RunPod window.

Completed rows:

| Candidate | Final export BPB | Train-time val BPB | Steps | Step avg | Artifact bytes | Headroom |
|---|---:|---:|---:|---:|---:|---:|
| `final8x_legal_196k_r2_d704e768_w2200_wd02_lqer6t12_vocabmoe_qk55` | `1.35496419` | `1.3191` | `6658` | `90.13ms` | `15,989,749` | `10,251` |
| `final8x_legal_196k_r2_d704e768_w2200_wd02_lqer8t16_vocabmoe_qk55` | `1.35536174` | `1.3158` | `6655` | `90.17ms` | `15,803,789` | `196,211` |

The `lqer6t12` row is the best completed under-cap 8xH100 evidence currently preserved for the MirrorLoop HRC + LexLoRE submission. It nearly fills the decimal 16MB cap while staying legal. This was produced during a self-funded, approximately one-hour 8xH100 RunPod window after no compute-grant response had been received.
