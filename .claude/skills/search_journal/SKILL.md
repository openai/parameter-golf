---
name: search_journal
description: Quick reference for searching past journals and summaries. Three modes — browse all headings, search for a specific topic, or drill into a known heading. Invoke when you forget the exact command, or when you're about to ask "have we tried X before?"
---

# Search Journal

Three modes for finding past work. Pick the one that matches what you know:

- **Browse** — TOC across active sessions:
  ```
  grep "^## " journal.md journals/[!_]*.md summaries/[!_]*.md
  ```
  The `[!_]` glob skips files starting with underscore — i.e., excludes `_archive_transformer/` content. Default browse focuses on the current research arc.

- **Search** — topic across active sessions:
  ```
  grep -i "<topic>" journal.md journals/[!_]*.md summaries/[!_]*.md
  ```
  Full content. Use this when the topic might be discussed inside an entry whose heading doesn't name it.

- **Drill** — full content of a known heading:
  ```
  mdq '# "<keyword>"' journal.md journals/[!_]*.md
  ```
  `#` matches any heading at any depth (single hash regardless of `##` / `###`). Quoted substring match.

If a query returns nothing or too much, switch modes — usually browse/search → drill, narrowing each time.

## Searching archives explicitly (opt-in)

Archived transformer-session journals/summaries/walks live in `journals/_archive_transformer/`, `summaries/_archive_transformer/`, `walks/_archive_transformer/`. They contain ~65 experiments of transformer-axis history. Default search excludes them. To search archives:

```
# Browse archive headings:
grep "^## " journals/_archive_transformer/*.md summaries/_archive_transformer/*.md

# Search archive contents:
grep -i "<topic>" journals/_archive_transformer/*.md summaries/_archive_transformer/*.md
```

When in doubt about whether an archive search is warranted: usually it isn't. The transferable lessons most relevant to a hybrid (TIED_EMBED_INIT_STD=0.05, MUON_BACKEND_STEPS=15, batch=24k+matrix_lr=0.045, etc.) live in `summaries/_archive_transformer/2026-04-25_overnight_session.md`.
