# Parameter Golf Compliance Kit

This kit is a strict cleanup package for turning the current project into a competition-aligned workflow.

## What this kit does
- separates the official submission path from auxiliary dataset evaluation
- gives you a submission README that talks in competition terms
- gives you a submission JSON template that avoids fake numbers
- gives you an audit script to check artifact size and basic file presence
- gives you an experiment tracker for baseline vs candidate runs

## What this kit does not do
- it does not fabricate BPB, train time, or official benchmark results
- it does not claim the current cube model is already a valid Parameter Golf final submission
- it does not replace the official OpenAI pipeline

## Immediate use
1. Copy the `records/non-record/multi-cube-face-letter-assignment/` files into your repo.
2. Put your current artifact path into `submission.template.json` only after you know the exact final file used by the official run.
3. Run `audit_submission.py` from the repo root.
4. Run one official baseline with the OpenAI pipeline.
5. Fill the experiment tracker with real values only.

## Local artifact facts already confirmed
- `/mnt/data/final_model.int8.ptz` size: 9,741,199 bytes
- decompressed size: 17,223,564 bytes
- `/mnt/data/best_model.pth` size: 2,532,149 bytes
- `best_model.pth` appears to be a small feed-forward classifier with 631,938 parameters, not a confirmed official PG submission model
