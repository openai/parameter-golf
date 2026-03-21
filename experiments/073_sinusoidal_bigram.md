# Experiment 073: Sinusoidal BigramHash — DIDN'T HELP

## Results
- Sliding: 1.1495 (worse than 071's 1.1480 without any bigram)
- Artifact: 15.45MB ✅

## Finding
Random fixed table adds noise, not signal. The LEARNED embedding was the valuable part.
Sinusoidal bigram is worse than no bigram at all.
**RULED OUT: sinusoidal/fixed bigram tables don't help.**
