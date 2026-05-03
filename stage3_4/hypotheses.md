# Stage 3.4 Hypotheses

These are not controller hypotheses.
They are branch-design hypotheses.

## H401 Tri-Branch Default

- Mechanism:
  - one shared trunk
  - three late finishers:
    - EMA-heavy
    - deploy-aligned QAT
    - family-split warmdown
- Why:
  - the late phase may have multiple plausible winners and we should not choose too early

## H402 Earlier Tri-Branch

- Mechanism:
  - same finisher set as H401
  - branch earlier to give each finisher more budget
- Why:
  - if branching is real but weak, the likely failure is not enough finisher time

## H403 Late Dual-Branch

- Mechanism:
  - branch later
  - only two finishers
- Why:
  - depth per branch may matter more than branch count

## H404 Deploy-vs-Family Duel

- Mechanism:
  - branch into only two nontrivial finishers
  - remove the trivial EMA fallback
- Why:
  - tests whether the branch value is truly in differentiated late mechanisms

## H405 Raw-EMA-Deploy Triple

- Mechanism:
  - separate state-style uncertainty from policy-style uncertainty
- Why:
  - the late win may come from export-state style, not just late loss shaping

## H406 Aggressive Tri-Branch

- Mechanism:
  - same branching shape as H401
  - more aggressive finisher policies
- Why:
  - branching should let us try harder late swings because bad branches can be discarded
