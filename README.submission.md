# Submission Readme — Multi Cube Face Letter Assignment

## Status
This entry is currently positioned as a **non-record submission path** until official Parameter Golf benchmark numbers are measured on the OpenAI pipeline.

## Competition alignment
This repository keeps two layers separate:

1. **Official submission path**
   - target metric: official validation BPB on the OpenAI Parameter Golf task
   - target constraints: under 16,000,000 total artifact bytes and official training budget
   - target workflow: official OpenAI repo-compatible training and evaluation path

2. **Auxiliary evaluation path**
   - internal diagnostic datasets for multi-object consistency, duplicate avoidance, omission avoidance, and structured assignment
   - used only to rank candidate ideas before paying for official BPB runs
   - not used as a replacement for the official benchmark

## Why this project exists
The project hypothesis is that better internal organization of structured assignments may reduce duplication, omission, and global inconsistency. Those properties are measured on auxiliary datasets, but only changes that survive the official BPB benchmark are considered valid for the final competition path.

## Current artifact facts
- current local compressed artifact observed: `final_model.int8.ptz`
- current observed compressed size: `9,741,199` bytes
- this only confirms artifact size, not official benchmark validity by itself

## Required final reporting before claiming a real submission result
The following fields must be filled with real measured values from an official run:
- exact artifact path used by the official pipeline
- exact artifact size in bytes
- exact code bytes counted by the competition rules if applicable
- exact train time
- exact validation BPB
- exact seed(s)
- exact commit SHA

## Official run checklist
- one baseline run on the official pipeline
- one saved log per run
- one artifact per run
- one row in the experiment tracker per run
- no external downloads during final evaluation path unless allowed by the official repo workflow
- no hidden dependencies outside the repo

## Auxiliary datasets
These datasets are treated as internal diagnostics only:
- Cube-Multi-Object-Consistency-Dataset
- planet-alphabet-mapping-1-26

They are useful if they help predict which architectural or training changes improve the official metric. They are not the final metric.
