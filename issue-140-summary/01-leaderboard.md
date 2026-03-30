# Leaderboard Summary

## Official Leaderboard (Top 5)

| Rank | BPB | Author | Key Techniques | PR |
|------|-----|--------|---------------|-----|
| 1 | **1.1428** | @thwu1 | 10L Int5-MLP + BigramHash(10240) + SWA(0.4) + WD 0.04 | #180 |
| 2 | 1.1458 | @raahilshah | Int6 MLP3x + SmearGate + BigramHash + OrthoInit + MuonWD + SWA | #162 |
| 3 | 1.1502 | @aruniyer | 11L + Int6 QAT + MLP3x + WD 0.04 + zstd-22 | #86 |
| 4 | 1.1556 | @aquariouseworkman | SmearGate + OrthoInit + Int6 STE QAT + MLP3x + Sliding Window | #65 |
| 5 | 1.1586 | @yahya010 | 10L Int6 QAT + Zstd MLP2.6x + Muon 0.99 + Sliding Window | #63 |

## Best Pending Validated (Top 6)

| BPB | Author | Delta nats | Seeds | Techniques | PR |
|-----|--------|-----------|-------|-----------|-----|
| **1.1250** | @jfprincz | 0.030 | 3 | 11L + Partial RoPE (16/64) + LN Scale + Late QAT + XSA4 + EMA (0.997) + FA3 | #315 |
| **1.1280** | @jfprincz | 0.025 | 3 | 11L + XSA4 + EMA (0.997) + SmearGate + BigramHash + WD 0.04 + FA3 | #287 |
| **1.1313** | @timowhite88 | 0.019 | 3 | 11L Int6 MLP3x + SmearGate + TTT + SWA + FA3 (pre-eval TTT ruled invalid) | #254 |
| **1.1320** | @saml212 | 0.018 | 3 | 12L + Gradient-Guided Quant (int7/6/5) + Partial RoPE + LN Scale + XSA4 + EMA + 524K batch | #332 |
| **1.1326** | @jfprincz | 0.017 | 3 | 11L + Int6 MLP3x + SmearGate + BigramHash + WD 0.04 + SWA + FA3 | #198 |
| **1.1400** | @saml212 | 0.005 | 3 | 11L Int6 + SmearGate + BigramHash + 524K batch + SWA + WD 0.04 | #236 |

## Unvalidated Below SOTA

| BPB | Author | Techniques | PR |
|-----|--------|-----------|-----|
| **1.1307** | @unnir | 11L + SmearGate + Partial XSA (last 3) + SWA + WD 0.04 + FA3 | #265 |
| **1.1354** | @ibarrajo | 11L + Partial XSA (last 3) + TTT + 524K batch (pre-eval TTT) | #290 |
| **1.1357** | @dennisimoo | 11L + XSA4 + EMA + 524K batch + WD 0.04 (no FA3) | #307 |
| **1.1381** | @charmquark1984 | 11L + SmearGate + TTT + 524K batch + WD 0.042 (pre-eval TTT) | #281 |
| **1.1419** | @chris-buckley | 11L + XSA4 + EMA + TTT (no FA3, SDPA fallback) | #317 |
