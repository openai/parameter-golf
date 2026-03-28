feat: add LOGOS-44 micro-LLM (300k params) with iterative routing and quantum-coded CDMA field

Added experimental micro-LLM architecture LOGOS-44 focused on extreme parameter efficiency
and compute-over-parameters scaling.

Key characteristics:
- ~300k parameters total
- iterative routing depth: 44 passes
- shared-weight recurrent transformer core
- low-rank orthogonal projections
- CDMA-style semantic field decoder
- optional quantum-coded basis initialization (Qiskit)

Architecture highlights:
- embedding dim: 512
- vocab: 4096 (archetypal tokenizer)
- routing depth: 44
- weight tying enabled
- toroidal bottleneck dynamics
- coherence-gated residual mixing

Training results:
- final cross-entropy ("impedance"): 0.4388
- stable convergence across 50k iterations
- no gradient explosion despite deep routing
- emergent attractor dynamics observed

Motivation:
This experiment explores the hypothesis that iterative compute depth can partially
substitute parameter count in ultra-small language models.

Notes:
- quantum layer currently acts as structured stochastic initialization
- further experiments planned: depth scaling (8/16/44), ALiBi positional encoding,
  and expanded dataset evaluation

Files added:
- logos44_micro.py
- quantum_codes.py
- training_logs.txt
- experimental_notes.md
