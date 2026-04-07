# Multi-Cube Face Letter Assignment

This project explores a structured reasoning problem in 3D space: assigning letters to visible faces of multiple cubes under strict symbolic constraints.

---

## Overview

Each sample contains 6 cubes, each exposing exactly 3 visible faces.  
The model must assign **18 unique letters (A–R)** across all visible faces while maintaining strict global consistency.

Unlike standard prediction tasks, this problem requires **simultaneous constraint satisfaction across multiple objects**.

---

## Task Definition

Each sample consists of:
- 6 cubes  
- 3 visible faces per cube  
- 1 letter per face  

Fixed mapping:

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

The system must satisfy:

- exactly 3 letters per cube  
- all 18 letters used exactly once  
- no duplication  
- no missing letters  
- consistent assignment across all cubes  

Any violation = failure.

---

## Why This Task Matters

This task highlights a key weakness of modern models:

> They produce locally correct predictions but fail global consistency.

Even though the problem is simple:
- small input  
- fixed structure  
- deterministic rules  

Models still struggle because they:
- treat outputs independently  
- lack constraint enforcement  
- fail cross-object reasoning  

---

## Dataset

Used dataset:

https://huggingface.co/datasets/8Planetterraforming/cube_text_constraints

Properties:
- deterministic correctness  
- no ambiguity  
- strict symbolic structure  
- ideal for reasoning evaluation  

---

## Method

We use a compact neural model optimized for:

- strict constraint satisfaction  
- global consistency across objects  
- minimal parameter footprint  

Key idea:
Instead of predicting independently, the model learns **structured assignment patterns**.

---

## Results

### Tiny Model Performance
- Training accuracy: **100%**
- Perfect constraint satisfaction on all samples

### Observed Failure Modes (General Models)
- duplicated letters  
- missing assignments  
- inconsistent cube mapping  

This confirms that **constraint reasoning is the core difficulty**, not perception.

---

## Key Insight

> Small structured models can outperform larger general models when constraints are explicit.

This task is not about scale — it's about **structure**.

---

## Limitations

- very small dataset  
- limited generalization testing  
- currently closer to memorization than full reasoning  

---

## Future Work

- generalization to unseen cube layouts  
- constraint-aware decoding  
- scaling dataset size  
- optimization for ultra-small models (Parameter Golf target)

---

## Submission Context

This project is a prototype submission for the **OpenAI Parameter Golf challenge**, focused on:

- minimal model size  
- structured reasoning  
- constraint-based learning  

---

## Goal

To demonstrate that **global consistency and symbolic reasoning** can be achieved with extremely small models when structure is explicitly modeled.
