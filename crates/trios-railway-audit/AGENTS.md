# AGENTS.md — trios-railway-audit

Local invariants for this ring. See repo-root [AGENTS.md](../../AGENTS.md)
for the constitution.

## Scope

trios-railway-audit is one ring in the trios-railway workspace. It MUST NOT depend on
any sibling crate other than `trios-railway-core` (transport + ids).

## Do not

- Hand-write GraphQL response JSON; let the upstream Railway API drive.
- Spawn shells, write `.sh` files, or call out to Python.
