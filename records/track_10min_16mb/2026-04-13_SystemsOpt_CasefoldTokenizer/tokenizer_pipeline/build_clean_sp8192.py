"""Build a cleaned SP8192 model by removing:
1. L=1 learned tokens (redundant with byte fallback)
2. Undertrained tokens (<5000 exposure in 10-min training budget)

Switches from BPE to Unigram model type since removing L=1 tokens
breaks BPE merge chains. Unigram uses Viterbi decoding which finds
optimal segmentation without merge hierarchies.

Usage:
    uv run data/build_clean_sp8192.py
    uv run data/build_clean_sp8192.py --exposure-threshold 5000
    uv run data/build_clean_sp8192.py --dry-run  # just report, don't build
"""
import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2 as sp_pb2

DATA_DIR = Path(__file__).parent
DOCS_PATH = DATA_DIR / "docs_selected.jsonl"
TOKENIZERS_DIR = DATA_DIR / "tokenizers"

# Training budget constants (from PR #1394)
TRAIN_BATCH_TOKENS = 2048 * 48 * 8  # 786,432
TRAIN_STEPS_10MIN = 5000
TOKENS_SEEN = TRAIN_BATCH_TOKENS * TRAIN_STEPS_10MIN  # ~3.9B
FULL_TRAIN_TOKENS = 12_748_339_616
FRACTION_SEEN = TOKENS_SEEN / FULL_TRAIN_TOKENS  # ~0.308

TRAIN_DOCS_TOTAL = 15_318_808


def load_texts(path: Path, limit: int, offset: int = 0) -> list[str]:
    texts = []
    skipped = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if skipped < offset:
                skipped += 1
                continue
            if len(texts) >= limit:
                break
            try:
                doc = json.loads(line)
                text = doc.get("text", "")
                if text:
                    texts.append(text)
            except json.JSONDecodeError:
                continue
    return texts


def estimate_training_exposure(sample_freq: int, sample_docs: int) -> int:
    """Estimate how many times a token is seen during 10-min training."""
    extrapolation = TRAIN_DOCS_TOTAL / sample_docs
    full_freq = sample_freq * extrapolation
    return int(full_freq * FRACTION_SEEN)


def main():
    parser = argparse.ArgumentParser(description="Build cleaned SP8192")
    parser.add_argument("--model", type=str,
                        default=str(TOKENIZERS_DIR / "fineweb_8192_bpe.model"))
    parser.add_argument("--val-docs", type=int, default=50000)
    parser.add_argument("--train-docs", type=int, default=500000)
    parser.add_argument("--exposure-threshold", type=int, default=5000,
                        help="Remove tokens with fewer than this many exposures in 10-min training")
    parser.add_argument("--bpb-threshold-pct", type=float, default=5.0)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    sp = spm.SentencePieceProcessor(model_file=args.model)
    vs = sp.vocab_size()
    print(f"Source model: {args.model} (vocab_size={vs})")

    # ── Step 1: Identify L=1 learned tokens ─────────────────────────
    print(f"\n--- L=1 learned tokens (redundant with byte fallback) ---")
    l1_remove = set()
    for tid in range(vs):
        if sp.is_byte(tid) or sp.is_control(tid) or sp.is_unknown(tid):
            continue
        piece = sp.id_to_piece(tid)
        if len(piece.encode("utf-8")) == 1:
            l1_remove.add(tid)

    print(f"  Found {len(l1_remove)} L=1 learned tokens to remove")

    # ── Step 2: Identify undertrained tokens via BPB + training freq ─
    print(f"\n--- Undertrained token analysis ---")
    print(f"  Loading {args.val_docs:,} val docs...")
    val_texts = load_texts(DOCS_PATH, args.val_docs)

    # Get BPE frequencies on val
    val_freq = Counter()
    total_tokens = 0
    total_bytes = 0
    all_ids = sp.encode(val_texts, out_type=int, num_threads=8)
    for text, ids in zip(val_texts, all_ids):
        val_freq.update(ids)
        total_tokens += len(ids)
        total_bytes += len(text.encode("utf-8"))

    # BPB contribution
    N = total_tokens
    learned_bpb = {}
    for tid in range(vs):
        if tid < 4 or sp.is_byte(tid):
            continue
        uses = val_freq.get(tid, 0)
        if uses > 0:
            surprisal = -math.log2(uses / N)
            bpb_contrib = uses * surprisal / total_bytes
        else:
            bpb_contrib = 0.0
        learned_bpb[tid] = {"uses": uses, "bpb_contrib": bpb_contrib}

    active = [v for v in learned_bpb.values() if v["uses"] > 0]
    mean_bpb = sum(v["bpb_contrib"] for v in active) / max(len(active), 1)
    bpb_threshold = mean_bpb * (args.bpb_threshold_pct / 100.0)

    # Get training frequencies
    print(f"  Loading {args.train_docs:,} training docs (offset={args.val_docs})...")
    train_texts = load_texts(DOCS_PATH, args.train_docs, offset=args.val_docs)
    print(f"  Encoding training docs...")
    train_freq = Counter()
    train_ids = sp.encode(train_texts, out_type=int, num_threads=8)
    for ids in train_ids:
        train_freq.update(ids)

    # Identify undertrained tokens: flagged by BPB AND low training exposure
    undertrained_remove = set()
    for tid, info in learned_bpb.items():
        if tid in l1_remove:
            continue  # already removing
        if info["bpb_contrib"] >= bpb_threshold:
            continue  # BPB contribution is fine

        tf = train_freq.get(tid, 0)
        exposure = estimate_training_exposure(tf, args.train_docs)
        if exposure < args.exposure_threshold:
            undertrained_remove.add(tid)

    print(f"  BPB threshold: {bpb_threshold:.6f} ({args.bpb_threshold_pct}% of mean {mean_bpb:.6f})")
    print(f"  Exposure threshold: <{args.exposure_threshold:,} in 10-min training")
    print(f"  Undertrained tokens to remove: {len(undertrained_remove)}")

    # ── Summary ──────────────────────────────────────────────────────
    all_remove = l1_remove | undertrained_remove
    print(f"\n{'='*70}")
    print(f"REMOVAL SUMMARY")
    print(f"{'='*70}")
    print(f"  L=1 learned (redundant with byte fallback): {len(l1_remove)}")
    print(f"  Undertrained (<{args.exposure_threshold:,} exposure):      {len(undertrained_remove)}")
    print(f"  Total to remove:                            {len(all_remove)}")
    print(f"  Remaining learned tokens:                   {vs - 260 - len(all_remove)}")
    print(f"  New vocab size:                             {260 + (vs - 260 - len(all_remove))}")

    # Print all removals sorted by category
    print(f"\n  Undertrained tokens being removed:")
    print(f"  {'TID':>5} {'Val':>7} {'Train':>9} {'Exp10m':>8} {'BPB%':>7} {'L':>2}  Piece")
    rows = []
    for tid in sorted(undertrained_remove):
        piece = sp.id_to_piece(tid)
        vf = val_freq.get(tid, 0)
        tf = train_freq.get(tid, 0)
        exp = estimate_training_exposure(tf, args.train_docs)
        bpb_pct = learned_bpb[tid]["bpb_contrib"] / mean_bpb * 100
        bl = len(piece.encode("utf-8"))
        rows.append((tid, vf, tf, exp, bpb_pct, bl, piece))

    rows.sort(key=lambda r: r[3])  # by exposure
    for tid, vf, tf, exp, bpb_pct, bl, piece in rows:
        safe = piece.encode("ascii", errors="backslashreplace").decode("ascii")
        print(f"  {tid:>5} {vf:>7,} {tf:>9,} {exp:>8,} {bpb_pct:>6.2f}% {bl:>2}  {safe}")

    if args.dry_run:
        print(f"\n  --dry-run: not building model")
        return

    # ── Build new model ──────────────────────────────────────────────
    print(f"\n--- Building cleaned model ---")

    original = sp_pb2.ModelProto()
    with open(args.model, "rb") as f:
        original.ParseFromString(f.read())

    # Collect pieces to keep (preserving original scores for ranking)
    kept_pieces = []
    for piece_proto in original.pieces:
        # Find this piece's token ID
        tid = None
        for i in range(vs):
            if sp.id_to_piece(i) == piece_proto.piece:
                tid = i
                break
        if tid is None:
            continue

        # Skip special, byte, and removed tokens
        if sp.is_control(tid) or sp.is_unknown(tid) or sp.is_byte(tid):
            continue
        if tid in all_remove:
            continue

        kept_pieces.append((piece_proto.piece, piece_proto.score))

    # Build new model
    model = sp_pb2.ModelProto()
    model.normalizer_spec.CopyFrom(original.normalizer_spec)
    if original.HasField("denormalizer_spec"):
        model.denormalizer_spec.CopyFrom(original.denormalizer_spec)
    model.trainer_spec.CopyFrom(original.trainer_spec)
    # Switch to Unigram for Viterbi decoding (no merge chain dependency)
    model.trainer_spec.model_type = 1  # UNIGRAM
    model.trainer_spec.byte_fallback = True

    TYPE_CONTROL = 3
    TYPE_UNKNOWN = 2
    TYPE_BYTE = 6
    TYPE_NORMAL = 1

    def add(piece_str, score, ptype):
        p = model.pieces.add()
        p.piece = piece_str
        p.score = score
        p.type = ptype

    # Special tokens
    add("<pad>", 0.0, TYPE_CONTROL)
    add("<s>", 0.0, TYPE_CONTROL)
    add("</s>", 0.0, TYPE_CONTROL)
    add("<unk>", 0.0, TYPE_UNKNOWN)

    # All 256 byte fallback tokens
    for i in range(256):
        add(f"<0x{i:02X}>", 0.0, TYPE_BYTE)

    # Learned tokens — use uniform score so Viterbi minimizes token count
    UNIFORM_SCORE = -1.0
    for piece_str, _original_score in kept_pieces:
        add(piece_str, UNIFORM_SCORE, TYPE_NORMAL)

    new_vocab = 260 + len(kept_pieces)
    model.trainer_spec.vocab_size = new_vocab

    # Write
    if args.output:
        output_path = args.output
    else:
        output_path = str(TOKENIZERS_DIR / "fineweb_8192_bpe_clean.model")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(model.SerializeToString())

    print(f"  Written: {output_path}")
    print(f"  Vocab: {new_vocab} (4 special + 256 byte + {len(kept_pieces)} learned)")
    print(f"  Model type: Unigram (Viterbi optimal encoding)")
    print(f"  File size: {Path(output_path).stat().st_size:,} bytes")

    # ── Verify ───────────────────────────────────────────────────────
    print(f"\n--- Verification ---")
    sp_clean = spm.SentencePieceProcessor(model_file=output_path)
    cv = sp_clean.vocab_size()
    print(f"  Loaded cleaned model: vocab_size={cv}")

    # Quick compression comparison on first 1000 val docs
    test_docs = val_texts[:1000]
    test_bytes = sum(len(t.encode("utf-8")) for t in test_docs)

    orig_tokens = sum(len(ids) for ids in sp.encode(test_docs, out_type=int, num_threads=4))
    clean_tokens = sum(len(ids) for ids in sp_clean.encode(test_docs, out_type=int, num_threads=4))

    print(f"\n  Compression on 1000 val docs ({test_bytes:,} bytes):")
    print(f"    Original (BPE):  {orig_tokens:>10,} tokens  ({orig_tokens/test_bytes:.4f} tok/byte)")
    print(f"    Cleaned (Uni):   {clean_tokens:>10,} tokens  ({clean_tokens/test_bytes:.4f} tok/byte)")
    delta_pct = (clean_tokens - orig_tokens) / orig_tokens * 100
    print(f"    Delta:           {delta_pct:>+.2f}%")

    # Verify no UNK on a sample
    test_hard = "Hello $100 & 50% off!! q=42 Z-scores (x, y, z) | pipe *asterisk*"
    ids_clean = sp_clean.encode(test_hard)
    has_unk = any(sp_clean.is_unknown(tid) for tid in ids_clean)
    print(f"\n  UNK test: {'PASS (no UNKs)' if not has_unk else 'FAIL (UNKs found)'}")
    print(f"    Input:  {test_hard}")
    print(f"    Tokens: {len(ids_clean)}")


if __name__ == "__main__":
    main()
