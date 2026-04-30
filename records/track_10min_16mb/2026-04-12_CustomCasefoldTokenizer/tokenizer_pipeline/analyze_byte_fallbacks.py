"""Analyze byte-fallback usage in a case-folded SentencePiece tokenizer."""
import sys, os, io, json, random, unicodedata
from collections import Counter, defaultdict

# Fix Windows encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import sentencepiece as spm

MODEL_PATH = "data/tokenizers/fineweb_8192_bpe_casefold_refined.model"
DOCS_PATH = "data/docs_selected.jsonl"
NUM_DOCS = 10_000
SEED = 1337
TOP_N = 100

def classify_char(ch):
    """Classify a character into a broad category."""
    cp = ord(ch)
    cat = unicodedata.category(ch)
    if cp < 128:
        if ch.isupper():
            return "ASCII uppercase"
        if ch.isdigit():
            return "ASCII digit"
        if ch.isalpha():
            return "ASCII lowercase"
        if ch in ' \t\n\r':
            return "ASCII whitespace"
        return "ASCII punctuation/symbol"
    if cat.startswith('L'):
        return "Non-ASCII letter"
    if cat.startswith('N'):
        return "Non-ASCII digit"
    if cat.startswith('P'):
        return "Non-ASCII punctuation"
    if cat.startswith('S'):
        return "Non-ASCII symbol"
    if cat.startswith('Z'):
        return "Non-ASCII whitespace"
    if cat.startswith('M'):
        return "Combining mark"
    if cat.startswith('C'):
        return "Control/format char"
    return f"Other ({cat})"

def classify_string(s):
    cats = Counter()
    for ch in s:
        cats[classify_char(ch)] += 1
    return cats

def byte_fb_id(piece_str):
    """Check if piece string is a byte fallback like <0xAB>."""
    return piece_str.startswith('<0x') and piece_str.endswith('>') and len(piece_str) == 6

def decode_byte_fb(piece_str):
    """Decode <0xAB> -> byte value."""
    return int(piece_str[3:5], 16)

def main():
    sp = spm.SentencePieceProcessor()
    sp.Load(MODEL_PATH)

    # Build set of byte fallback token IDs and mapping
    byte_fb_ids = set()
    id_to_byte = {}
    for i in range(sp.GetPieceSize()):
        piece = sp.IdToPiece(i)
        if byte_fb_id(piece):
            byte_fb_ids.add(i)
            id_to_byte[i] = decode_byte_fb(piece)
    print(f"Byte fallback token count: {len(byte_fb_ids)}", flush=True)

    # Load first NUM_DOCS from file (streaming, no full load needed)
    print(f"Loading first {NUM_DOCS} docs from {DOCS_PATH}...", flush=True)
    docs = []
    with open(DOCS_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            text = json.loads(line).get('text', '')
            if text:
                docs.append(text)
                if len(docs) >= NUM_DOCS:
                    break
    print(f"Loaded {len(docs)} docs", flush=True)

    # Analyze
    fallback_substr_counter = Counter()
    fallback_seq_lengths = []
    char_category_counts = Counter()
    total_tokens = 0
    total_fb_tokens = 0

    for doc_idx, doc in enumerate(docs):
        if doc_idx % 2000 == 0:
            print(f"  Processing doc {doc_idx}/{len(docs)}...", flush=True)

        text = doc.lower()
        ids = sp.Encode(text)
        total_tokens += len(ids)

        # Find consecutive runs of byte fallback tokens
        i = 0
        while i < len(ids):
            if ids[i] in byte_fb_ids:
                # Collect consecutive byte fallback tokens
                fb_bytes = bytearray()
                j = i
                while j < len(ids) and ids[j] in byte_fb_ids:
                    fb_bytes.append(id_to_byte[ids[j]])
                    j += 1
                fb_len = j - i
                total_fb_tokens += fb_len

                # Decode bytes to string
                try:
                    orig_substr = fb_bytes.decode('utf-8')
                except UnicodeDecodeError:
                    orig_substr = fb_bytes.decode('utf-8', errors='replace')

                # Remove leading SentencePiece space marker if present
                # The SP marker is U+2581 which is 3 bytes: E2 96 81
                orig_substr = orig_substr.replace('\u2581', ' ').strip()
                if not orig_substr:
                    orig_substr = fb_bytes.hex()

                fallback_substr_counter[orig_substr] += 1
                fallback_seq_lengths.append(fb_len)

                for ch in orig_substr:
                    char_category_counts[classify_char(ch)] += 1

                i = j
            else:
                i += 1

    # Report
    print("\n" + "="*80)
    print("BYTE FALLBACK ANALYSIS")
    print("="*80)
    print(f"\nTotal tokens:          {total_tokens:,}")
    print(f"Total fallback tokens: {total_fb_tokens:,}")
    print(f"Fallback rate:         {total_fb_tokens/total_tokens*100:.3f}%")
    print(f"Unique fallback substrings: {len(fallback_substr_counter):,}")
    print(f"Total fallback sequences:   {len(fallback_seq_lengths):,}")
    if fallback_seq_lengths:
        avg_len = sum(fallback_seq_lengths) / len(fallback_seq_lengths)
        print(f"Avg fallback seq length:    {avg_len:.2f} tokens")
        print(f"Max fallback seq length:    {max(fallback_seq_lengths)} tokens")

        len_dist = Counter(fallback_seq_lengths)
        print(f"\nFallback sequence length distribution:")
        for length in sorted(len_dist.keys())[:20]:
            pct = len_dist[length] / len(fallback_seq_lengths) * 100
            print(f"  {length:3d} tokens: {len_dist[length]:8,} ({pct:5.1f}%)")
        if max(len_dist.keys()) > 20:
            remaining = sum(v for k, v in len_dist.items() if k > 20)
            print(f"  >20 tokens: {remaining:8,}")

    print(f"\n{'='*80}")
    print("CHARACTER CATEGORY BREAKDOWN (by chars in fallback substrings)")
    print("="*80)
    total_chars = sum(char_category_counts.values())
    for cat, cnt in char_category_counts.most_common():
        pct = cnt / total_chars * 100
        print(f"  {cat:30s}: {cnt:8,} ({pct:5.1f}%)")

    print(f"\n{'='*80}")
    print(f"TOP {TOP_N} MOST COMMON FALLBACK SUBSTRINGS")
    print("="*80)
    print(f"{'Rank':>4s}  {'Count':>7s}  {'Len':>3s}  {'Bytes':>5s}  {'Repr':<50s}  Category")
    print("-"*110)
    for rank, (substr, count) in enumerate(fallback_substr_counter.most_common(TOP_N), 1):
        byte_len = len(substr.encode('utf-8', errors='replace'))
        cats = classify_string(substr)
        dominant_cat = cats.most_common(1)[0][0] if cats else "Unknown"
        r = repr(substr)
        if len(r) > 50:
            r = r[:47] + '...'
        print(f"{rank:4d}  {count:7,}  {len(substr):3d}  {byte_len:5d}  {r:<50s}  {dominant_cat}")

    # Group by dominant category
    print(f"\n{'='*80}")
    print("FALLBACK SUBSTRINGS GROUPED BY DOMINANT CATEGORY")
    print("="*80)
    cat_groups = defaultdict(lambda: {"count": 0, "unique": 0, "examples": []})
    for substr, count in fallback_substr_counter.items():
        cats = classify_string(substr)
        dominant_cat = cats.most_common(1)[0][0] if cats else "Unknown"
        cat_groups[dominant_cat]["count"] += count
        cat_groups[dominant_cat]["unique"] += 1
        if len(cat_groups[dominant_cat]["examples"]) < 5:
            cat_groups[dominant_cat]["examples"].append((substr, count))

    sorted_cats = sorted(cat_groups.items(), key=lambda x: -x[1]["count"])
    for cat, info in sorted_cats:
        print(f"\n  {cat}: {info['count']:,} occurrences, {info['unique']:,} unique substrings")
        for substr, cnt in sorted(info["examples"], key=lambda x: -x[1])[:5]:
            r = repr(substr)
            if len(r) > 60:
                r = r[:57] + '...'
            print(f"    {cnt:6,}x  {r}")

if __name__ == "__main__":
    main()
