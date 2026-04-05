# Multi-Cube Face Letter Assignment

This project explores a structured reasoning problem in 3D space: assigning letters to visible faces of multiple cubes under strict symbolic constraints.

---

## Task Definition

Each sample consists of:
- 6 cubes  
- each cube exposes exactly 3 visible faces  
- each face contains exactly one letter  

The system must assign **18 unique letters (A–R)** across all visible faces, following a fixed mapping:

| Cube | Letters |
|------|--------|
| 1    | A, B, C |
| 2    | D, E, F |
| 3    | G, H, I |
| 4    | J, K, L |
| 5    | M, N, O |
| 6    | P, Q, R |

---

## Constraints

The task enforces strict global and local constraints:

- exactly 3 letters per cube  
- no duplicated letters across cubes  
- no missing letters  
- one letter per visible face  
- consistent spatial assignment across all 6 cubes  

Any violation (duplication, omission, misplacement) is considered a failure.

---

## Motivation

Modern models often fail on tasks requiring:

- global consistency across multiple objects  
- structured symbolic reasoning  
- strict constraint satisfaction  
- spatial coherence  

Although visually simple, this task exposes a key limitation:  
models tend to make **locally correct but globally inconsistent predictions**.

---

## Dataset

This project uses:

👉 https://huggingface.co/datasets/8Planetterraforming/cube_text_constraints

The dataset is designed as a **constraint-focused benchmark**, where:

- correctness is deterministic  
- ambiguity is minimal  
- failures are clearly measurable  
- reasoning errors are easy to detect  

---

## Method

This project uses a lightweight neural model designed for structured reasoning.

Key elements:
- explicit cube-to-letter grouping (A–R per cube)
- constraint-aware training
- compact architecture optimized for small model size

The focus is on maintaining **global consistency across all cubes**, rather than independent local predictions.

---

## Results

Evaluation on structured cube assignment task:

### Training Performance (TinyModel)
- Accuracy: **100%** (on training dataset of 6 samples)  
- All cube-face mappings correctly learned  

### Baseline Behavior (Naive / General Models)
Manual testing shows common failure patterns:
- inconsistent assignments across cubes  
- duplicated or missing letters  
- mixing labels between cube groups  

These failures highlight the difficulty of maintaining global constraints.

---

## Key Insight

The task appears simple locally, but requires strict global coordination.

Standard models fail because they:
- treat each face independently  
- lack constraint enforcement  
- do not maintain cross-object consistency  

Even a very small structured model can outperform larger general models when constraints are explicitly modeled.

---

## Limitations

- evaluation currently limited to small dataset  
- results reflect memorization rather than full generalization  
- no testing yet on unseen cube configurations  

---

## Future Work

- generalization to unseen cube layouts  
- larger and more diverse datasets  
- constraint-aware decoding methods  
- evaluation under limited parameter budgets (Parameter Golf setting)

---

## Status

Research prototype — prepared for submission to the OpenAI Parameter Golf challenge.

---

## Goal

To develop a compact model capable of maintaining strict symbolic and spatial consistency in a multi-object 3D environment under strong constraints.
