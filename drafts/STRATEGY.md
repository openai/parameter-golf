# Draft Submissions Strategy

## Draft A: Current (NorMuon + full-weight TTT + batch 524K)
- Base: PR #198 (1.1318)
- Additions: NorMuon, full-weight SGD TTT (3 epochs), batch 524K, temp scaling
- Expected: ~1.12 bpb
- Risk: NorMuon might not help or hurt
- Location: records/track_10min_16mb/2026-03-20_QAT_TTT_ValueEmbed/

## Draft B: Ablation plan for H100 runs
When you get compute, run these experiments in order:
1. PR #198 base reproduced verbatim → confirm 1.1318
2. PR #198 + TTT only → isolate TTT impact
3. PR #198 + NorMuon only → isolate NorMuon impact
4. PR #198 + batch 524K only → isolate batch impact
5. PR #198 + all three → our submission
6. Sweep TTT lr [0.001, 0.002, 0.005, 0.01]
7. Sweep TTT epochs [1, 2, 3, 5]
8. Sweep TTT freeze_blocks [0, 1, 2, 3]

## Draft C: Nuclear option — distillation
- Train a huge model (no size constraint) for the full 10 min
- Use it to generate soft targets for the 16MB model
- Re-train the 16MB model with KD loss
- Requires 2 training runs but the rules allow "any training data"

## Draft D: Error Correction Table (PR #232 inspired)
- After training, run eval pass to find worst predictions
- Encode corrections into the 16MB artifact (~2-3 MB)
- Remaining ~13 MB for the model
- Could give massive bpb improvement on fixed val set

## Draft E: Paid prefix (PR #168 inspired, controversial)
- Use ~8 MB of the artifact for a compressed context prefix
- ~8 MB for the model
- Every eval window gets massive pre-computed context
- Rules unclear on whether this is allowed

## Priority order: A → B (ablations) → C → D → E
