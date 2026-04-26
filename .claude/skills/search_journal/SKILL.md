---
name: search_journal
description: Quick reference for searching past journals and summaries. Three modes — browse all headings, search for a specific topic, or drill into a known heading. Invoke when you forget the exact command, or when you're about to ask "have we tried X before?"
---

# Search Journal

Three modes for finding past work. Pick the one that matches what you know:

- **Browse** — you don't have a specific target, want to scan what exists:
  ```
  grep "^## " journal.md journals/*.md summaries/*.md
  ```
  Headings only, TOC view across all sessions.

- **Search** — you know the topic (e.g. "sliding-window", "32k batch", "qk_gain"):
  ```
  grep -i "<topic>" journal.md journals/*.md summaries/*.md
  ```
  Full content. Use this when the topic might be discussed inside an entry whose heading doesn't name it.

- **Drill** — you have a heading and want the section's full content:
  ```
  mdq '# "<keyword>"' journal.md journals/*.md
  ```
  `#` matches any heading at any depth (single hash regardless of `##` / `###`). Quoted substring match.

  That's it. If a query returns nothing or too much, switch modes — usually browse/search → drill, narrowing each time.
