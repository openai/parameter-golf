# Attempted H200 Learnings

This submission records an attempted final run under severe time pressure.

I tried to reproduce and extend the current SP8192/CaseOps/TTT frontier, but the practical bottleneck became data preparation and environment stability rather than model code. The main lesson was that CPU preprocessing, FA3 compilation, and final torchrun need to be staged as a pipeline before the expensive GPU window starts.

No new score is claimed here. This is a small record of the attempt and what I learned.
