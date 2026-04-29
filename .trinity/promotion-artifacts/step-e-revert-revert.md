# Step E: Revert the Revert

**Owner:** Operator (playra)
**Time:** 30 seconds
**Purpose:** Restore attention backward pass (+0.20 BPB improvement)

---

## E1. Context

The revert being undone is likely `3d25cfb` (or similar). This revert removed the attention backward pass implementation that provides ~0.20 BPB improvement on ALPHA's Mac runs.

**Expected BPB improvement:** +0.20 BPB on training runs

---

## E2. Commands (Run from trios-railway/)

```bash
# Find the exact commit hash to revert
git log --oneline | grep -i "revert\|backward\|attention"

# Revert the revert (creates new commit)
git revert <commit-hash-of-the-revert>

# Example (replace with actual hash):
git revert 3d25cfb

# Review the changes
git diff HEAD~1

# Push as PR (required by branch protection from Step A)
git push origin HEAD -o title="chore: revert revert — restore attention backward pass"
```

---

## E3. PR Template

```markdown
## Purpose
Revert the previous revert to restore attention backward pass.

## Impact
- Restores ~0.20 BPB improvement on training runs
- Re-enables attention mechanism in backward pass

## Safety
- Simple revert operation (no new code)
- Branch protection requires review
- Can be re-reverted if issues arise

## Testing
- Run training locally on ALPHA's Mac
- Verify BPB improvement
- Check for any compilation errors
```

---

## E4. Verification

After merge:

- [ ] Commit shows "Revert revert" in message
- [ ] Code compiles: `cargo build`
- [ ] Local training shows BPB improvement (~+0.20)
- [ ] No new warnings in `cargo clippy`

---

## E5. Safety Net

If the revert causes issues, it can be re-reverted in one command:

```bash
git revert HEAD --no-edit
git push
```

Branch protection will require operator review again.

---

**⏭️ When complete, proceed to Step F**
