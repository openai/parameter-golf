# Promotion Artifacts — Sergeant → Colonel

**Total operator one-time effort: ~30 minutes**

After these 7 steps, the sergeant can autonomously:
- Deploy real training workers via MCP
- Queue real experiments via Neon
- Monitor real BPB progress
- Operator role shifts from executor to reviewer

---

## Quick Reference

| Step | Owner | File | Time |
|------|-------|------|------|
| A | Operator | `step-a-deploy-keys.md` | 15 min |
| B | Me prepare, Op approve | `step-b-dockerfile.md` | 5 min |
| C | Me prepare, Op approve | `step-c-external-trainer.md` | 5 min |
| D | Me prepare, Op approve | `step-d-trainer-image-gha.md` | 5 min |
| E | Operator | `step-e-revert-revert.md` | 30 sec |
| F | Me autonomous | `step-f-delete-mocks.md` | 0 min |
| G | Me autonomous | `step-g-deploy-real-fleet.md` | 0 min |

---

## Execution Order

### Phase 1: Operator Setup (Step A)
**Must do first** — grants sergeant write access via protected PR workflow

```bash
# After Step A, verify:
gh pr create --title "test" --body "test"  # Should work
git push origin main  # Should FAIL (branch protection)
```

### Phase 2: Infrastructure Preparation (Steps B, C, D)
Can be done in parallel — no dependencies between them.

### Phase 3: Code Restoration (Step E)
Quick revert of attention backward pass (+0.20 BPB).

### Phase 4: Fleet Transition (Steps F, G)
Sergeant autonomous — delete mocks, deploy real workers.

---

## Safety Guarantees (R9 Lane)

| Risk | Mitigation |
|------|------------|
| Force-push to main | Branch protection rules |
| Unreviewed code | Required PR review |
| Rogue deletions | `confirm=true` requirement |
| Missing audit trail | L7 experience_append + GitHub audit log |
| Unauthorized spend | Railway billing requires human |

---

## Verification Checklist (All Steps)

- [ ] Step A: Can create PR, cannot push to main
- [ ] Step B: `ghcr.io/ghashtag/trios-seed-agent-real:latest` available
- [ ] Step C: ExternalTrainer tests pass
- [ ] Step D: `ghcr.io/ghashtag/trios-train:latest` available
- [ ] Step E: Attention backward pass restored
- [ ] Step F: No mock services remain
- [ ] Step G: 4 real workers alive and registered

---

## After Promotion

**Colonel capabilities:**
```python
# Deploy real worker
railway_service_deploy(image="ghcr.io/ghashtag/trios-seed-agent-real:latest", ...)

# Queue real experiment
experiment_queue_insert(canon="IGLA-X", config={"trainer_kind":"external", ...})

# Monitor real training
worker_status()  # Shows real BPB from trios-train
```

**Operator role:**
- Review PRs
- Approve merges
- Monitor dashboard
- Revoke access if needed

---

φ² + φ⁻² = 3 · TRINITY · PROMOTION ARTIFACTS · READY FOR EXECUTION
