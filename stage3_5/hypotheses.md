# Stage 3.5 Hypotheses

These are not controller hypotheses and not fixed-time branch hypotheses.

They are event-triggered branch tournament hypotheses.

## H501 Adaptive Tri Portfolio

- Mechanism:
  - branch on late training state
  - three finisher programs
  - raw vs EMA export portfolio inside each branch
- Why:
  - late uncertainty is about both branch timing and export state

## H502 Scale-Gated Dual Deep

- Mechanism:
  - branch only when LR scale gets low enough
  - two deeper finishers instead of three shallower finishers
- Why:
  - branch depth may matter more than branch breadth

## H503 Plateau-Gated Aggressive

- Mechanism:
  - branch when training plateaus or hits a failsafe time
  - aggressive QAT and matrix-push finishers compete against EMA
- Why:
  - branching should let us try harder late swings because bad branches can be discarded

## H504 State-Style Tournament

- Mechanism:
  - raw-finish, EMA-heavy, and deploy-QAT compete as export-state styles
- Why:
  - the dominant late uncertainty may be the export object, not the optimizer path

## H505 Family-vs-Deploy Event Duel

- Mechanism:
  - no trivial EMA branch
  - deploy-QAT competes directly against family/matrix late specialization
- Why:
  - if branching is real, nontrivial late mechanisms should be able to duel directly

## H506 Failsafe Event Tri

- Mechanism:
  - adaptive trigger with a hard max-frac failsafe
  - three non-identical finisher programs
- Why:
  - robust branching may need both adaptation and a guaranteed late branch point
