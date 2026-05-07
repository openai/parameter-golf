"""Analyze docs_selected.jsonl for high-frequency web patterns to inform user_defined_symbols."""
import json
import sys
from collections import Counter
from pathlib import Path

SAMPLE_SIZE = 100_000

# Patterns to count — grouped by category
PATTERNS = {
    # Already in user's list (for comparison)
    "http://": "url",
    "https://": "url",
    "www.": "url",
    ".com": "tld",
    ".org": "tld",
    ".net": "tld",
    ".io": "tld",
    "</": "html",
    "/>": "html",
    "href=": "html-attr",
    "src=": "html-attr",
    "class=": "html-attr",
    "id=": "html-attr",
    # Candidate additions — HTML close tags
    "</div>": "html-close",
    "</p>": "html-close",
    "</span>": "html-close",
    "</a>": "html-close",
    "</li>": "html-close",
    "</ul>": "html-close",
    "</ol>": "html-close",
    "</td>": "html-close",
    "</tr>": "html-close",
    "</th>": "html-close",
    "</table>": "html-close",
    "</h1>": "html-close",
    "</h2>": "html-close",
    "</h3>": "html-close",
    "</h4>": "html-close",
    "</strong>": "html-close",
    "</em>": "html-close",
    "</b>": "html-close",
    "</i>": "html-close",
    "</script>": "html-close",
    "</style>": "html-close",
    # Candidate additions — HTML open/self-closing
    "<div": "html-open",
    "<span": "html-open",
    "<img": "html-open",
    "<br>": "html-open",
    "<br/>": "html-open",
    "<br />": "html-open",
    "<hr>": "html-open",
    "<input": "html-open",
    "<meta": "html-open",
    "<link": "html-open",
    # Candidate additions — HTML attributes
    "style=": "html-attr",
    "type=": "html-attr",
    "name=": "html-attr",
    "value=": "html-attr",
    "alt=": "html-attr",
    "title=": "html-attr",
    "width=": "html-attr",
    "height=": "html-attr",
    "rel=": "html-attr",
    "target=": "html-attr",
    "data-": "html-attr",
    "aria-": "html-attr",
    "onclick=": "html-attr",
    "action=": "html-attr",
    "method=": "html-attr",
    "content=": "html-attr",
    "charset=": "html-attr",
    "placeholder=": "html-attr",
    # Candidate additions — HTML entities
    "&amp;": "entity",
    "&nbsp;": "entity",
    "&lt;": "entity",
    "&gt;": "entity",
    "&quot;": "entity",
    "&#": "entity",
    "&copy;": "entity",
    "&mdash;": "entity",
    "&ndash;": "entity",
    "&rsquo;": "entity",
    "&lsquo;": "entity",
    "&rdquo;": "entity",
    "&ldquo;": "entity",
    "&hellip;": "entity",
    # Candidate additions — file extensions
    ".html": "ext",
    ".htm": "ext",
    ".php": "ext",
    ".asp": "ext",
    ".jsp": "ext",
    ".css": "ext",
    ".js": "ext",
    ".json": "ext",
    ".xml": "ext",
    ".jpg": "ext",
    ".jpeg": "ext",
    ".png": "ext",
    ".gif": "ext",
    ".svg": "ext",
    ".pdf": "ext",
    ".mp4": "ext",
    ".mp3": "ext",
    # Candidate additions — protocols/misc
    "mailto:": "proto",
    "ftp://": "proto",
    "://": "proto",
    "javascript:": "proto",
    # Candidate additions — TLDs
    ".edu": "tld",
    ".gov": "tld",
    ".co.uk": "tld",
    ".de": "tld",
    ".fr": "tld",
    ".info": "tld",
    ".biz": "tld",
    # Common multi-char sequences in web text
    "the ": "common",
    " the": "common",
    "ing ": "common",
    "tion": "common",
    " of ": "common",
    " and": "common",
    "and ": "common",
    " in ": "common",
    "Copyright": "legal",
    "Privacy": "legal",
    "Terms": "legal",
    "All rights reserved": "legal",
    # Punctuation patterns
    '="': "punct",
    "='": "punct",
    '">': "punct",
    "'>": "punct",
    "<!--": "comment",
    "-->": "comment",
}

def main():
    jsonl_path = Path(__file__).parent / "docs_selected.jsonl"
    if not jsonl_path.exists():
        print(f"ERROR: {jsonl_path} not found", file=sys.stderr)
        sys.exit(1)

    counts = Counter()
    doc_counts = Counter()  # how many docs contain each pattern (for coverage)
    total_bytes = 0

    print(f"Sampling {SAMPLE_SIZE:,} docs from {jsonl_path.name}...")
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= SAMPLE_SIZE:
                break
            try:
                doc = json.loads(line)
                text = doc.get("text", "")
            except json.JSONDecodeError:
                continue

            total_bytes += len(text.encode("utf-8"))
            seen = set()
            for pattern in PATTERNS:
                c = text.count(pattern)
                if c > 0:
                    counts[pattern] += c
                    if pattern not in seen:
                        doc_counts[pattern] += 1
                        seen.add(pattern)

            if (i + 1) % 10_000 == 0:
                print(f"  processed {i+1:,} docs...")

    print(f"\nAnalyzed {min(i+1, SAMPLE_SIZE):,} docs, {total_bytes:,} bytes total")
    print(f"Average doc size: {total_bytes / min(i+1, SAMPLE_SIZE):,.0f} bytes\n")

    # Sort by total occurrences descending
    print(f"{'Pattern':<25} {'Category':<12} {'Total Hits':>12} {'Docs w/ Pattern':>16} {'Hits/Doc':>10}")
    print("-" * 80)

    already_in_list = {
        "http://", "https://", "www.", ".com", ".org", ".net", ".io",
        "</", "/>", "href=", "src=", "class=", "id="
    }

    # Print patterns already in the user's list
    print("\n=== ALREADY IN YOUR LIST ===")
    for pattern, total in counts.most_common():
        if pattern in already_in_list:
            cat = PATTERNS[pattern]
            docs = doc_counts[pattern]
            ratio = total / min(i+1, SAMPLE_SIZE)
            print(f"  {pattern:<23} {cat:<12} {total:>12,} {docs:>16,} {ratio:>10.1f}")

    # Print candidate additions sorted by frequency
    print("\n=== CANDIDATE ADDITIONS (sorted by total hits) ===")
    candidates = [(p, c) for p, c in counts.most_common() if p not in already_in_list]
    for pattern, total in candidates:
        cat = PATTERNS[pattern]
        docs = doc_counts[pattern]
        ratio = total / min(i+1, SAMPLE_SIZE)
        marker = " ***" if total >= 10_000 else ""
        print(f"  {pattern:<23} {cat:<12} {total:>12,} {docs:>16,} {ratio:>10.1f}{marker}")

    # Summary: top candidates above threshold
    print("\n=== TOP CANDIDATES (>= 10K total hits in 100K docs) ===")
    top = [(p, c) for p, c in candidates if c >= 10_000]
    for pattern, total in top:
        cat = PATTERNS[pattern]
        bytes_saved = len(pattern.encode("utf-8")) - 1  # bytes saved per occurrence (1 token instead of N bytes)
        print(f"  {pattern:<23} {total:>12,} hits  ({bytes_saved} bytes saved/hit)")

    print(f"\n  Total top candidates: {len(top)}")
    print(f"  Current symbols: {len(already_in_list)}")
    print(f"  Proposed total: {len(already_in_list) + len(top)}")
    print(f"  BPE merges remaining: {1024 - 3 - 256 - len(already_in_list) - len(top)}")


if __name__ == "__main__":
    main()
