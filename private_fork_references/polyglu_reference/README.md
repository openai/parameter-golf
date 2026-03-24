# PolyGLU Reference for Parameter Golf

> **Purpose**: This directory gives Claude Code everything it needs to integrate the PolyGLU activation routing mechanism into the Parameter Golf challenge. Read all files before writing any code.

## Directory Structure

```
polyglu_reference/
├── README.md                 ← You are here. Start here.
├── STRATEGY.md               ← High-level strategy: WHY PolyGLU helps in Parameter Golf and HOW to apply it
├── PAPER.md                  ← Full PolyGLU paper content (arXiv:2603.13347v1) — theory, method, results
├── POLYGLU_IMPLEMENTATION.md ← The reference PolyGLU PyTorch code with detailed annotations
├── PARAMETER_GOLF_CONTEXT.md ← Analysis of the Parameter Golf challenge: rules, baseline, SOTA tricks
└── INTEGRATION_PLAN.md       ← Concrete step-by-step plan for modifying train_gpt.py
```

## Reading Order

1. **STRATEGY.md** — Understand the "why" and the hypothesis
2. **PAPER.md** — Deep understanding of PolyGLU's mechanism and findings
3. **PARAMETER_GOLF_CONTEXT.md** — Understand what you're modifying and the constraints
4. **POLYGLU_IMPLEMENTATION.md** — The exact code you'll be adapting
5. **INTEGRATION_PLAN.md** — Execute this plan

## Key Constraint

The Parameter Golf challenge has a **16MB artifact size limit** and a **10-minute training time on 8xH100**. PolyGLU must be adapted to work within these constraints — it cannot be copied verbatim from the 600M-parameter model. The routing overhead must be minimal relative to the ~4M parameter budget.
