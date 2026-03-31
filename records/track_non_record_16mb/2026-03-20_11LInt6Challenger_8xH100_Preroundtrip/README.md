This record captures an in-progress non-record submission built from the 2026-03-20 challenger workbench.

The goal of this run was to stay inside the normal `10 minute / 8xH100 / 16MB` competition shape while copying the simpler public `11L int6 + zstd` recipe first, before layering back in `SmearGate`, `BigramHash`, or `SWA`.

This folder is being submitted as a non-record attempt because the strongest `8xH100` run only completed the training-phase validation before the provider terminated the pod during export, so there is no final post-roundtrip score to claim yet.

Configuration:
- Track: `non-record`, main-track-style `600s` attempt on `8xH100`
- Layout: `vocab=1024 layers=11 dim=512 heads=8 kv=4 mlp_hidden=1536`
- Training: `batch_tokens=786432 train_seq_len=2048 eval_seq_len=2048 eval_stride=64`
- Optimizer: `muon matrix_lr=0.025 scalar_lr=0.025 tied_embed_lr=0.03 muon_wd=0.038`
- Quantization target: `compressor=zstd weight_bits=6 embed_bits=16`
- Keep-float tensors: `tok_emb.weight`, `blocks.9.attn.c_k.weight`, `blocks.10.attn.c_k.weight`
- Context features enabled in this run: `bigram=0 smeargate=0 swa=0`

Command:
```bash
./run_challenger.sh
```

Key observed metrics (from `train.log`):
- Timed training stopped at `5193/20000` steps due to the `600s` wallclock cap.
- Pre-roundtrip eval at stop: `val_loss:1.9627`, `val_bpb:1.1624`
- Train time at stop: `600022ms`
- Printed code size before export work: `35761` bytes
- The strongest public result I was comparing against at the time was PR `#180` at `1.14526`, so this run was `+0.01714 bpb` away before roundtrip/export.

Why this is non-record:
- The `8xH100` run did not finish the export + roundtrip-eval phase.
- RunPod terminated the pod during export on the strongest run, and later terminated a fresh replacement pod during dataset preparation before training even began.
- Because of that, this folder logs the strongest measured training-phase result and the exact code used, but does not claim a valid leaderboard score.

Notes on the code snapshot:
- `train_gpt.py` is the records entry point used for this run family.
- Helper files are vendored locally so the folder can compile and run without reaching back into repo root.

Included files:
- `train_gpt.py` (records entry point used for the attempt family)
- `root_train_gpt_vendor.py` plus local helper modules vendored for portability
- `run_challenger.sh` (track-relevant entrypoint used for the logged run family)
- `train.log` (exact remote log from the strongest `8xH100` attempt that reached `1.1624` pre-roundtrip)
- `submission.json` (metadata for this non-record attempt)
