# Multi-Cube Face Letter Assignment

## Summary

This submission explores a structured reasoning problem: assigning letters to visible faces of multiple 3D cubes under strict constraints.

Each sample consists of:
- 6 cubes
- each cube has 3 visible faces
- each face contains exactly one letter

A total of 18 unique letters (A–R) must be assigned consistently across all cubes.

## Key Challenge

Standard models fail due to:
- lack of global consistency
- mixing letters between cubes
- treating each face independently

## Approach

This work focuses on:
- constraint-aware assignment
- fixed cube-to-letter grouping
- enforcing global consistency across all cubes

The task is treated as a single structured system rather than independent predictions.

## Status

Prototype submission. The focus is on exploring constraint-based reasoning under limited model capacity.

## Notes

This work highlights the gap between pattern recognition and structured reasoning in compact models.
