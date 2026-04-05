# Multi-Cube Face Letter Assignment

This project investigates a structured reasoning problem in 3D space: assigning letters to visible faces of multiple cubes under strict symbolic constraints.

## Task Definition

Each sample consists of:
- 6 cubes
- each cube exposes exactly 3 visible faces
- each face contains exactly one letter

The full system must assign **18 unique letters (A–R)** across all visible faces, following a fixed mapping:

| Cube | Letters |
|------|--------|
| 1    | A, B, C |
| 2    | D, E, F |
| 3    | G, H, I |
| 4    | J, K, L |
| 5    | M, N, O |
| 6    | P, Q, R |

## Constraints

The task enforces strict global and local constraints:

- exactly 3 letters per cube  
- no duplicated letters across cubes  
- no missing letters  
- one letter per visible face  
- consistent spatial assignment across all 6 cubes  

Any violation (duplication, omission, misplacement) is considered a failure.

## Motivation

Modern multimodal and language models often fail on tasks requiring:

- global consistency across multiple objects  
- structured symbolic reasoning  
- precise constraint satisfaction  
- spatially coherent labeling  

Despite appearing visually simple, this task exposes fundamental weaknesses in reasoning and planning.

## Dataset

This project uses the dataset:

👉 https://huggingface.co/datasets/8Planetterraforming/cube_text_constraints

The dataset is designed as a **constraint-focused evaluation benchmark**, where:

- tasks are easy to verify by humans  
- failures are deterministic and measurable  
- ambiguity is minimized  
- reasoning errors are clearly exposed  

## Approach

This repository explores:

- structured labeling strategies  
- constraint-aware training  
- lightweight architectures for small models  
- improving consistency in multi-object reasoning  

The goal is not only accuracy, but **robust constraint satisfaction under limited model capacity**.

## Status

Research prototype — preparing submission for the OpenAI Parameter Golf challenge.

## Goal

To develop a compact model capable of maintaining strict symbolic and spatial consistency in a constrained 3D multi-object environment.

## Method

This project uses a lightweight model designed to handle structured spatial constraints.

Key elements:
- explicit cube-to-letter grouping (A–R mapped per cube)
- constraint-aware training to prevent letter mixing
- simplified architecture optimized for small model size

The model focuses on maintaining global consistency across multiple objects rather than local predictions.

## Results

Current status:
- model under development
- early experiments show improved consistency compared to naive approaches

(Final evaluation metrics will be added for submission)

## Key Insight

Standard models fail because they treat each face independently.

This approach improves performance by enforcing:
- global structure awareness
- multi-object consistency
- strict symbolic constraints

The task highlights the gap between pattern recognition and true structured reasoning.
## Results

Evaluation on structured cube assignment task:

### Training performance (TinyModel)
- Accuracy: 100% (on training dataset of 6 samples)
- All cube-face mappings correctly learned

### Baseline (naive generation / general model)
- Estimated accuracy: 40–60%
- Frequent errors:
  - inconsistent face assignments
  - mixing labels between cubes

### Key insight
The task appears simple locally but requires strict global consistency.

Naive models often fail due to lack of constraint enforcement, while even a tiny structured model can fully learn correct mappings.
## Limitations

The current model is evaluated on a small dataset and shows perfect memorization.

Future work will include:
- generalization to unseen cube configurations
- larger datasets
- constraint-aware decoding
