# Hardik Final SOTA Submission

This folder contains the submission bundle for the 10min / 16MB track.

## Included files
- `submission.json`: submission metadata and reported metrics.
- `train.log`: training output captured from the run.
- `final_model.ternary.ptz`: included model artifact.
- `train_gpt.py`: training/export script used for the submission.
- `requirements.txt`: pinned runtime dependencies.

## Reproduction
1. Install dependencies from `requirements.txt`.
2. Run `train_gpt.py` from this folder or from the notebook in `notebooks/Parameter_golf.ipynb`.
3. Use the exported artifact and log files in this directory to verify the submission metrics.
