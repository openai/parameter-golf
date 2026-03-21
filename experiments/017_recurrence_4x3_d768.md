# Experiment 017: Recurrence 4x3 d768

## Status: FAILED (instance 6 killed)

## Hypothesis
4 unique blocks × 3 loops = 12 effective layers, dim=768, GQA 12:6. Maximum width with fewer unique blocks. Tests if width > unique block diversity.

## Configuration
- **Architecture**: 4 unique × 3 loops = 12 eff, dim=768, 12 heads, 6 KV

## Results
Never ran — instance 6 GPU was stuck/killed.
