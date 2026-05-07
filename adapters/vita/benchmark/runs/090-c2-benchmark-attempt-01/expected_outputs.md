# Expected outputs and acceptance checks

Required outputs (from real benchmark execution):
- evidence/train.log
- evidence/eval.log
- evidence/submission.json
- evidence/artifact_sizes.json
- evidence/environment.json

Acceptance checks:
- [ ] Benchmark pipeline actually executed (not scaffold-only)
- [ ] Exact benchmark metric line extracted from logs
- [ ] Artifact-size accounting verified from produced artifacts
- [ ] Hardware/protocol comparability documented
- [ ] claims_guard updated from non-claiming only when all checks pass
