# Retained Local Artifacts

Large model artifacts are not committed in this non-record folder. They are
currently retained locally at:

| Artifact | Path | Bytes | SHA256 |
|---|---|---:|---|
| final full-precision model | `/home/simon/castorv2/runs/castor_l7grow_v4_12h_seed1337/final_model.pt` | 135431355 | `02959aa988dd1668ca696ce1a0058309ea4fe52d3505f2a560f5240d74f6bac9` |
| final full-precision snapshot | `/home/simon/castorv2/logs/castor_l7grow_v4_12h_seed1337.final_model_snapshot.pt` | 135431355 | `02959aa988dd1668ca696ce1a0058309ea4fe52d3505f2a560f5240d74f6bac9` |
| latest training checkpoint | `/home/simon/castorv2/runs/castor_l7grow_v4_12h_seed1337/checkpoints/latest.pt` | 286390027 | `734a0f69d377a96439ae3ba8a4814741c5b270f6ef921e912f10ed48f93e4466` |

The run used `SKIP_FINAL_PACKAGING=1`, so no final compressed int6 `.ptz`
artifact was produced for this archive.
