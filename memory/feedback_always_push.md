---
name: Always push after every commit
description: User requires git push after every commit — never leave commits local
type: feedback
---

Always run `git push fork research` immediately after every `git commit`. Never leave commits sitting locally.

**Why:** Execution sessions pull from the remote. Unpushed commits mean execution runs stale code or specs, which caused a preflight failure when sanity greps referred to an unpushed spec fix.

**How to apply:** Every single commit in this repo must be followed immediately by a push. No exceptions.
