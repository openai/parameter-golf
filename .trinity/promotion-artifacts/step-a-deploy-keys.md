# Step A: One-time Grant — Deploy Keys & Branch Protection

**Owner:** Operator (playra)
**Time:** 15 minutes (one-time setup)
**Purpose:** Grant sergeant write access via protected PR workflow

---

## A1. GitHub Deploy Keys Creation

For each repository, create a deploy key with scoped permissions:

| Repository | Permissions | Purpose |
|------------|-------------|---------|
| gHashTag/trios-railway | Contents:Write, Pull requests:Write | Main project access |
| gHashTag/trios-trainer-igla | Contents:Write, Pull requests:Write | Trainer image builds |
| gHashTag/zig-golden-float | Contents:Write, Pull requests:Write | Zig component |
| gHashTag/t27 | Contents:Write, Pull requests:Write | T27 component |

### How to create deploy keys:

1. Generate SSH key pair:
```bash
ssh-keygen -t ed25519 -C "perplexity-mcp-grandmaster@trios" -f ~/.ssh/trios_deploy_key
```

2. For each repo, go to: Settings → Deploy Keys → Add deploy key
   - Title: `perplexity-mcp-grandmaster`
   - Key: paste contents of `~/.ssh/trios_deploy_key.pub`
   - ✅ Allow write access
   - ✅ Allow read access

3. Store the private key securely and pass it to the MCP configuration

---

## A2. Branch Protection Rules

For EACH repository (trios-railway, trios-trainer-igla, zig-golden-float, t27):

1. Go to: Settings → Branches → Add rule
2. Rule name: `main-branch-protection`
3. Apply to: `main` branch
4. Settings:
   - ✅ **Require pull request reviews before merging**
     - Required reviewers: 1
     - Approve PRs with at least: 1
   - ✅ **Require status checks to pass before merging**
     - Require branches to be up to date before merging
   - ✅ **Require branches to be up to date before merging**
   - ✅ **Restrict who can push to matching branches**
     - Add: `perplexity-mcp-grandmaster[bot]` (if using GitHub App)
     - Do NOT add operator directly - force PR workflow
   - ✅ **Do not allow bypassing the above settings**

### Why this matters:

| Risk | Mitigation |
|------|------------|
| Force-push to main | Blocked by branch protection |
| Unreviewed code | Requires 1 approval |
| Rogue deletions | Require PR, can't force-delete |
| Audit trail | GitHub shows all PR activity |

---

## A3. Key Distribution to MCP

After creating deploy keys, they need to be accessible to the MCP connector.

Option 1: Environment variable (for testing)
```bash
export TRIOS_DEPLOY_KEY_PATH="$HOME/.ssh/trios_deploy_key"
```

Option 2: GitHub App (recommended for production)
1. Create GitHub App: https://github.com/settings/apps/new
   - App name: `perplexity-mcp-grandmaster`
   - Description: `Trinity R7-R9 automation bot`
2. Permissions:
   - Contents: Read & Write
   - Pull requests: Read & Write
   - Issues: Read & Write
3. Install on all 4 repos
4. Pass App ID and private key to MCP config

---

## A4. Verification

After setup, verify:

```bash
# From sergeant's perspective, test access
gh auth status
gh repo view gHashTag/trios-railway --json visibility
gh pr list --repo gHashTag/trios-railway
```

Expected: Can create PRs, but CANNOT push directly to main.

---

## Safety Checklist Before Moving to Step B

- [ ] Deploy keys created for all 4 repos
- [ ] Branch protection enabled on all 4 `main` branches
- [ ] Can create PR via git push to fork
- [ ] CANNOT push directly to main (test this: `git push origin main` should fail)
- [ ] MCP connector configured with deploy key or GitHub App credentials

---

**⏭️ When complete, proceed to Step B**
