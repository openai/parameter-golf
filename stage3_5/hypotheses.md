# Stage 3.5 Hypotheses

These are event-triggered late-program tournament hypotheses on the active executable
strong local base. They are not controller policies and not fixed-time branch
hypotheses.

## H501 Pre-Quant TTT Tri Portfolio

- Mechanism:
  - branch on late training state
  - run three different pre-quant TTT finishers
  - choose the best export mode inside each branch
- Why:
  - late uncertainty is mainly in the TTT law and export object, not just branch timing

## H502 TTT Breadth-vs-Depth Duel

- Mechanism:
  - trigger only when the late scale condition is met
  - compare two deeper TTT programs instead of a broader tri-set
- Why:
  - branch depth may matter more than branch breadth once the branch family is TTT-focused

## H503 Plateau-Gated Aggressive Swing

- Mechanism:
  - branch on plateau or a hard late failsafe
  - let aggressive dTTT-style and recurrent-deploy finishers compete against a conservative TTT fallback
- Why:
  - branching should buy permission to swing harder late because failed branches can be discarded

## H504 Export-State Style Tournament

- Mechanism:
  - keep the late TTT program simple
  - compete over raw-state, EMA-state, and broader freeze-2 state styles
- Why:
  - the dominant late uncertainty may be the starting/export object rather than the finisher law itself

## H505 TTT-vs-Recurrent Deploy Duel

- Mechanism:
  - no trivial EMA fallback
  - dTTT-style tail adaptation competes directly against recurrent deploy shaping
- Why:
  - if branching is real, two nontrivial late mechanisms should be able to duel directly

## H506 Failsafe Event Tri

- Mechanism:
  - adaptive trigger with a hard max-frac failsafe
  - three TTT/deploy programs always get a guaranteed late tournament window
- Why:
  - robust branching may need both adaptivity and a guaranteed branch point
