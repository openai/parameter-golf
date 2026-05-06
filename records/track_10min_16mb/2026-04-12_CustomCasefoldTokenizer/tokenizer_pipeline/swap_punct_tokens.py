"""Swap low-value tokens for bare ASCII punctuation in the casefold refined vocab.

Loads the refined hex vocab, identifies the N lowest-use learned tokens on
lowercased val text, drops them, inserts bare punctuation characters, and
rebuilds the SentencePiece model.
"""
import sys, io, json
from pathlib import Path
from collections import Counter

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import sentencepiece as spm

# Adjust import path
sys.path.insert(0, str(Path(__file__).parent))
from hybrid_tokenizer import create_sp_model

# --- Config ---
REFINED_MODEL = "data/tokenizers/fineweb_8192_bpe_casefold_refined.model"
REFINED_HEX   = "data/tokenizers/vocab_8k_casefold_refined.hex"
BASELINE_MODEL = "data/tokenizers/fineweb_8192_bpe.model"  # template for normalizer
DOCS_PATH     = "data/docs_selected.jsonl"
NUM_DOCS      = 10_000
OUTPUT_MODEL  = "data/tokenizers/fineweb_8192_bpe_casefold_refined_v2.model"
OUTPUT_HEX    = "data/tokenizers/vocab_8k_casefold_refined_v2.hex"

# Bare ASCII punctuation to add (ordered by fallback frequency from analysis)
PUNCT_TO_ADD = [
    ',', '.', ':', '-', ')', '?', '!', ';', '"', '/',
    "'", '|', ']', '%', '(', '*', '+', '>', '=', '_',
    '<', '~', '@', '#', '^',
]

def load_hex_vocab(path: str) -> list[bytes]:
    tokens = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                tokens.append(bytes.fromhex(line))
    return tokens

def save_hex_vocab(tokens: list[bytes], path: str):
    with open(path, 'w') as f:
        for t in tokens:
            f.write(t.hex() + '\n')

def main():
    # 1. Load refined model and hex vocab
    sp = spm.SentencePieceProcessor()
    sp.Load(REFINED_MODEL)
    hex_tokens = load_hex_vocab(REFINED_HEX)
    print(f"Loaded {len(hex_tokens)} learned tokens from hex vocab")

    # 2. Figure out which bare punctuation tokens need adding
    unk = sp.unk_id()
    need = []
    for ch in PUNCT_TO_ADD:
        bare_id = sp.PieceToId(ch)
        if bare_id == unk:
            need.append(ch)
        else:
            print(f"  Already have bare {ch!r} at ID {bare_id}")
    print(f"\nNeed to add {len(need)} bare punctuation tokens: {need}")

    # 3. Count token usage on lowercased val docs
    print(f"\nCounting token usage on {NUM_DOCS} lowercased docs...")
    usage = Counter()
    n = 0
    with open(DOCS_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            text = json.loads(line).get('text', '')
            if text:
                ids = sp.Encode(text.lower())
                usage.update(ids)
                n += 1
                if n >= NUM_DOCS:
                    break
    print(f"Encoded {n} docs, {sum(usage.values()):,} tokens")

    # 4. Rank learned tokens by usage (lowest first)
    learned_usage = []
    for idx, tok_bytes in enumerate(hex_tokens):
        piece_str = tok_bytes.decode('utf-8')
        piece_id = sp.PieceToId(piece_str)
        count = usage.get(piece_id, 0)
        learned_usage.append((idx, piece_str, count, tok_bytes))

    learned_usage.sort(key=lambda x: x[2])

    print(f"\nTokens to DROP (lowest {len(need)} by usage):")
    drop_indices = set()
    for i in range(len(need)):
        idx, piece, count, _ = learned_usage[i]
        print(f"  Drop [{idx:4d}] {piece!r:30s} ({count} uses)")
        drop_indices.add(idx)

    # 5. Build new token list: remove dropped, add punctuation
    new_tokens = [t for i, t in enumerate(hex_tokens) if i not in drop_indices]
    for ch in need:
        new_tokens.append(ch.encode('utf-8'))

    print(f"\nNew vocab: {len(new_tokens)} learned tokens (was {len(hex_tokens)})")

    # 6. Save hex vocab
    save_hex_vocab(new_tokens, OUTPUT_HEX)
    print(f"Saved hex vocab to {OUTPUT_HEX}")

    # 7. Build SP model
    print(f"Building SP model...")
    create_sp_model(new_tokens, BASELINE_MODEL, OUTPUT_MODEL)
    print(f"Saved model to {OUTPUT_MODEL}")

    # 8. Verify bare punctuation exists in new model
    sp2 = spm.SentencePieceProcessor()
    sp2.Load(OUTPUT_MODEL)
    print(f"\nVerification — bare punctuation in new model:")
    unk2 = sp2.unk_id()
    ok = 0
    for ch in need:
        pid = sp2.PieceToId(ch)
        status = "OK" if pid != unk2 else "MISSING"
        if pid != unk2:
            ok += 1
        print(f"  {ch!r:5s} -> ID {pid:5d}  {status}")
    print(f"\n{ok}/{len(need)} bare punctuation tokens added successfully")
    print(f"Total vocab size: {sp2.GetPieceSize()}")

    # 9. Quick byte fallback comparison
    print(f"\n{'='*60}")
    print("BYTE FALLBACK COMPARISON (first 1000 docs)")
    print(f"{'='*60}")
    for label, model in [("v1 (refined)", sp), ("v2 (+punct)", sp2)]:
        fb_ids = set()
        for i in range(model.GetPieceSize()):
            p = model.IdToPiece(i)
            if p.startswith('<0x') and p.endswith('>') and len(p) == 6:
                fb_ids.add(i)

        total_tok = 0
        total_fb = 0
        m = 0
        with open(DOCS_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                text = json.loads(line).get('text', '')
                if text:
                    ids = model.Encode(text.lower())
                    total_tok += len(ids)
                    total_fb += sum(1 for x in ids if x in fb_ids)
                    m += 1
                    if m >= 1000:
                        break
        pct = total_fb / total_tok * 100 if total_tok else 0
        print(f"  {label:20s}: {total_fb:,}/{total_tok:,} fallback tokens ({pct:.2f}%)")

if __name__ == "__main__":
    main()
