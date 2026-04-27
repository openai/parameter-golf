# Cross-repo PAT for the audit watchdog

Anchor: `phi^2 + phi^-2 = 3`. Refs [#16].

The hourly `audit-watchdog.yml` workflow lives in `gHashTag/trios-railway`
but posts its digest to `gHashTag/trios#143`. The default `GITHUB_TOKEN`
is scoped to the workflow's own repo — cross-repo writes fail with:

```
GraphQL: Resource not accessible by integration (addComment)
```

To enable comment + auto-close, add a **fine-grained Personal Access Token**.

## One-time setup (5 min)

1. Open <https://github.com/settings/personal-access-tokens/new>.
2. Settings:
   - **Token name:** `trios-railway audit watchdog`
   - **Resource owner:** `gHashTag` (the user/org that owns `trios`)
   - **Expiration:** 90 days (recommended) or longer
   - **Repository access:** `Only select repositories` → pick `gHashTag/trios`
   - **Repository permissions** (only these — least privilege):
     - `Issues` → `Read and write`
     - everything else → leave at `No access`
3. Click **Generate token**, copy the value (starts with `github_pat_…`).
4. Save it to `gHashTag/trios-railway` Actions secrets:

```bash
gh secret set TRIOS_REPO_PAT --repo gHashTag/trios-railway
# (paste token, hit enter)
```

5. Verify:

```bash
gh secret list --repo gHashTag/trios-railway | grep TRIOS_REPO_PAT
gh workflow run audit-watchdog.yml --repo gHashTag/trios-railway --ref main \
    -f target=1.85 -f skip_comment=false
```

The next watchdog run posts a digest to `trios#143` and (on combined exit 0)
auto-closes it.

## How the workflow uses it

```yaml
env:
  GH_TOKEN: ${{ secrets.TRIOS_REPO_PAT || secrets.GITHUB_TOKEN }}
```

If `TRIOS_REPO_PAT` is missing, the comment and close steps short-circuit
on `if:` guards and the workflow logs a `::warning::` instead of failing.
This means the same workflow file works in three modes:

| `TRIOS_REPO_PAT` | `inputs.skip_comment` | Behaviour |
|---|---|---|
| set    | false | Posts digest, closes on PASS |
| unset  | false | Logs warning, prints draft comment, exits green |
| any    | true  | Dry run — runs audit, no side effects on `trios#143` |

## Rotation

Fine-grained PATs expire. Set a calendar reminder one week before
expiration; rotate by repeating step 1 with a fresh token, then
`gh secret set TRIOS_REPO_PAT --repo gHashTag/trios-railway` again.

If the watchdog suddenly starts logging the warning above, the PAT
likely expired.
