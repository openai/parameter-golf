import argparse
import importlib.util
import json
import math
import os
import sys
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path


AUTHOR = "xu robert"
NAME = "AnchorPrefixCodec_first_piece_constraint_codec"
PREVIOUS_BEST_BPB = 1.2464

HERE = Path(__file__).resolve().parent
CODEC_PREFIX_PATH = HERE / "codec_prefix.py"
OUT_PATH = HERE / "records" / "train_gpt_result_first_piece_constraint_codec.json"

def ensure_canonical_units():
    required = [
        HERE / "records" / "canonical_units" / "per_file" / f"train_{i:06d}_canonical_units_v1.jsonl"
        for i in range(8)
    ]
    required.append(
        HERE / "records" / "canonical_units" / "per_file" / "val_000000_canonical_units_v1.jsonl"
    )

    if all(p.exists() for p in required) and os.environ.get("FORCE_PREPARE_CANONICAL_UNITS", "0") != "1":
        print("[prepare] canonical units found")
        return

    print("[prepare] building canonical units from official data")
    subprocess.check_call([sys.executable, str(HERE / "prepare_canonical_units.py")])

MAX_TRAIN_UNITS_DEFAULT = 4_000_000
MAX_VAL_UNITS_DEFAULT = 200_000

MIN_CTX_COUNT = 3
MAX_CANDIDATES = 64


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def env_int(name, default):
    raw = os.environ.get(name)
    return default if raw is None or raw == "" else int(raw)


def env_float(name, default):
    raw = os.environ.get(name)
    return default if raw is None or raw == "" else float(raw)


def count_code_bytes():
    return sum(p.stat().st_size for p in HERE.glob("*.py"))


def action_legality(action):
    if action is None:
        return "FALLBACK"

    name = action.get("action")
    if name in {"fragment", "subword_split", "subword_single", "left_phrase"}:
        return "LEGAL_CAUSAL"
    if name in {"right_phrase"}:
        return "ILLEGAL_FUTURE_CONTEXT"
    return "UNCLEAR"


def official_like_byte_count(sp, units):
    total = 0
    prev_is_boundary = True

    for u in units:
        ids = sp.encode(u["s"], out_type=int)
        for tid in ids:
            piece = sp.id_to_piece(int(tid))
            if piece.startswith("▁"):
                if not prev_is_boundary:
                    total += 1
                piece = piece[1:]

            total += len(piece.encode("utf-8"))
            prev_is_boundary = True

    return total

def build_official_byte_luts(sp):
    vocab_size = int(sp.vocab_size())
    base_bytes = [0] * vocab_size
    has_leading_space = [False] * vocab_size
    is_boundary_token = [True] * vocab_size

    for token_id in range(vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue

        is_boundary_token[token_id] = False

        if sp.is_byte(token_id):
            base_bytes[token_id] = 1
            continue

        piece = sp.id_to_piece(token_id)
        if piece.startswith("▁"):
            has_leading_space[token_id] = True
            piece = piece[1:]

        base_bytes[token_id] = len(piece.encode("utf-8"))

    return base_bytes, has_leading_space, is_boundary_token


def official_byte_count_from_units(sp, val_units):
    base_bytes, has_leading_space, is_boundary_token = build_official_byte_luts(sp)

    token_ids = []
    for u in val_units:
        token_ids.extend(int(t) for t in u.get("raw_token_ids", []))

    if len(token_ids) < 2:
        return 0

    total = 0
    for prev_id, tgt_id in zip(token_ids[:-1], token_ids[1:]):
        total += base_bytes[tgt_id]
        if has_leading_space[tgt_id] and not is_boundary_token[prev_id]:
            total += 1

    return total

def empirical_piece_counter(v4, sp, train_units):
    counts = Counter()
    for u in train_units:
        for piece in v4.word_piece_sequence(sp, u["s"]):
            counts[piece] += 1
    return counts


def piece_bits_from_counter(piece_counts, piece, alpha):
    vocab = max(1, len(piece_counts))
    total = sum(piece_counts.values())
    return -math.log2((piece_counts.get(piece, 0) + alpha) / (total + alpha * vocab))


def build_first_piece_index(v4, sp, train_units, prefix_keep):
    idx = defaultdict(Counter)

    for u in train_units:
        seq = v4.word_piece_sequence(sp, u["s"])
        pieces = [p for p in seq if p != v4.EOW]
        keep = min(prefix_keep, len(pieces))
        retained = pieces[:keep]
        predicted = pieces[keep:] + [v4.EOW]

        if not predicted:
            continue

        first = predicted[0]

        for k in range(1, min(3, len(retained)) + 1):
            ctx = tuple(retained[-k:])
            idx[ctx][first] += 1

    return idx


def constrained_first_piece_bits(index, retained_prefix, target):
    best = None
    best_meta = None

    for k in range(1, min(3, len(retained_prefix)) + 1):
        ctx = tuple(retained_prefix[-k:])
        counter = index.get(ctx)

        if not counter:
            continue

        total = sum(counter.values())
        if total < MIN_CTX_COUNT:
            continue

        if len(counter) > MAX_CANDIDATES:
            continue

        if target not in counter:
            continue

        p = counter[target] / total
        bits = -math.log2(max(p, 1e-12))

        if best is None or bits < best:
            best = bits
            best_meta = {
                "ctx_len": k,
                "candidate_size": len(counter),
                "ctx_total": total,
                "target_count": counter[target],
            }

    return best, best_meta

def load_val_token_ids_from_bin(data_dir):
    import numpy as np
    import os

    path = os.path.join(data_dir, "fineweb_val_000000.bin")

    if not os.path.exists(path):
        raise FileNotFoundError(f"val bin not found: {path}")

    header_bytes = 256 * np.dtype("<i4").itemsize
    header = np.fromfile(path, dtype="<i4", count=256)

    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {path}")

    num_tokens = int(header[2])
    max_tokens = int(os.environ.get("MAX_VAL_TOKENS", "500000"))

    if max_tokens <= 0:
        take = num_tokens
    else:
        take = min(max_tokens, num_tokens)

    arr = np.fromfile(path, dtype="<u2", count=take, offset=header_bytes)

    if arr.size == 0:
        raise RuntimeError("val bin loaded but empty")

    return arr.astype(np.int64).tolist()

def official_byte_count_from_token_ids(sp, token_ids):
    base_bytes, has_leading_space, is_boundary_token = build_official_byte_luts(sp)

    total = 0
    for prev_id, tgt_id in zip(token_ids[:-1], token_ids[1:]):
        total += base_bytes[tgt_id]
        if has_leading_space[tgt_id] and not is_boundary_token[prev_id]:
            total += 1

    return total


class OnlineSuffixLM:
    def __init__(self, vocab_size, alpha=0.05):
        self.vocab_size = max(1, vocab_size)
        self.alpha = alpha
        self.uni = Counter()
        self.bi = defaultdict(Counter)
        self.tri = defaultdict(Counter)

    def _prob_from_counter(self, counter, piece):
        total = sum(counter.values())
        return (counter.get(piece, 0.0) + self.alpha) / (total + self.alpha * self.vocab_size)

    def prob(self, ctx, piece):
        if len(ctx) >= 2:
            key = tuple(ctx[-2:])
            if key in self.tri and sum(self.tri[key].values()) > 0:
                return self._prob_from_counter(self.tri[key], piece), "online_trigram"

        if len(ctx) >= 1:
            key = tuple(ctx[-1:])
            if key in self.bi and sum(self.bi[key].values()) > 0:
                return self._prob_from_counter(self.bi[key], piece), "online_bigram"

        if sum(self.uni.values()) > 0:
            return self._prob_from_counter(self.uni, piece), "online_unigram"

        return None, None

    def update(self, ctx, piece):
        ctx_snapshot = tuple(ctx[-2:])
        self.uni[piece] += 1

        if len(ctx_snapshot) >= 1:
            self.bi[tuple(ctx_snapshot[-1:])][piece] += 1

        if len(ctx_snapshot) >= 2:
            self.tri[tuple(ctx_snapshot[-2:])][piece] += 1


def score_suffix_v11(
    v4,
    sp,
    base_lm,
    online_lm,
    first_piece_index,
    text,
    prefix_keep,
    online_weight,
    constraint_enabled,
):
    if online_weight <= 0.0 and not constraint_enabled:
        scored = v4.score_suffix_with_prefix(
            sp=sp,
            lm=base_lm,
            text=text,
            prefix_keep=prefix_keep,
        )
        seq = v4.word_piece_sequence(sp, text)
        pieces_only = [p for p in seq if p != v4.EOW]
        keep = min(prefix_keep, len(pieces_only))
        scored["retained_prefix"] = pieces_only[:keep]
        scored["constraint_used"] = False
        scored["constraint_saved_bits"] = 0.0
        scored["constraint_meta"] = None
        return scored

    base_scored = v4.score_suffix_with_prefix(
        sp=sp,
        lm=base_lm,
        text=text,
        prefix_keep=prefix_keep,
    )

    seq = v4.word_piece_sequence(sp, text)
    pieces_only = [p for p in seq if p != v4.EOW]
    keep = min(prefix_keep, len(pieces_only))

    retained_prefix = pieces_only[:keep]
    predicted_suffix = pieces_only[keep:] + [v4.EOW]

    ctx = list(retained_prefix[-2:])
    bits = 0.0
    source = Counter()

    avg_base_bits = base_scored["bits"] / max(1, base_scored["predicted_piece_count"])

    constraint_used = False
    constraint_saved_bits = 0.0
    constraint_meta = None

    for pos, piece in enumerate(predicted_suffix):
        op, osrc = online_lm.prob(ctx, piece)

        if op is None or online_weight <= 0.0:
            online_or_base_bits = avg_base_bits
            src = "base_avg"
        else:
            online_bits = -math.log2(max(op, 1e-12))
            mixed_bits = (1.0 - online_weight) * avg_base_bits + online_weight * online_bits
            online_or_base_bits = min(avg_base_bits, mixed_bits)
            src = f"safe_{osrc}" if online_or_base_bits < avg_base_bits else "base_avg"

        final_bits = online_or_base_bits

        MAX_CONSTRAINT_PIECES = int(os.environ.get("MAX_CONSTRAINT_PIECES", "2"))

        if constraint_enabled and pos < MAX_CONSTRAINT_PIECES:
            cb, meta = constrained_first_piece_bits(first_piece_index, retained_prefix, piece)

            if cb is not None and cb < final_bits:
                final_bits = cb
                src = f"constraint_pos{pos}"

                constraint_used = True
                constraint_saved_bits += online_or_base_bits - cb

                # 记录统计（区分pos）
                if constraint_meta is None:
                    constraint_meta = {"positions": {}}
                constraint_meta["positions"][pos] = meta

        bits += final_bits
        source[src] += 1

        # score-first update after current piece is scored
        online_lm.update(ctx, piece)

        ctx.append(piece)
        if len(ctx) > 2:
            ctx = ctx[-2:]

    return {
        "bits": bits,
        "predicted_piece_count": len(predicted_suffix),
        "source": source,
        "retained_prefix": retained_prefix,
        "constraint_used": constraint_used,
        "constraint_saved_bits": constraint_saved_bits,
        "constraint_meta": constraint_meta,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    t0 = time.time()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    prefix_keep = env_int("PREFIX_KEEP", 3)
    max_train_units = env_int("MAX_TRAIN_UNITS", MAX_TRAIN_UNITS_DEFAULT)
    max_val_units = env_int("MAX_VAL_UNITS", MAX_VAL_UNITS_DEFAULT)
    online_weight = env_float("ONLINE_WEIGHT", 1.0)
    constraint_enabled = env_int("CONSTRAINT_ENABLED", 1) == 1

    print("[start] AnchorPrefixCodec_first_piece_constraint_codec")
    print(f"[config] PREFIX_KEEP={prefix_keep}")
    print(f"[config] ONLINE_WEIGHT={online_weight}")
    print(f"[config] CONSTRAINT_ENABLED={int(constraint_enabled)}")
    print(f"[config] MAX_TRAIN_UNITS={max_train_units}")
    print(f"[config] MAX_VAL_UNITS={max_val_units}")

    v4 = load_module(CODEC_PREFIX_PATH, "codec_prefix")
    m = v4.load_base_module()
    sp = v4.load_sp()
    DATA_DIR = HERE.parents[2] / "data" / "datasets" / "fineweb10B_sp1024"

    m.MAX_TRAIN_UNITS = max_train_units
    m.MAX_VAL_UNITS = max_val_units
    ensure_canonical_units()


    print("[load] train canonical units")
    train_units = m.load_units(m.TRAIN_PATHS, m.MAX_TRAIN_UNITS)

    print("[load] val canonical units")
    val_units = m.load_units(m.VAL_PATHS, m.MAX_VAL_UNITS)

    print("[build] train-only structure knowledge")
    knowledge = m.build_train_knowledge(train_units)

    print("[build] train-only suffix LM")
    base_lm = v4.build_piece_ngram_lm(train_units, sp)

    print("[build] train-only first-piece index")
    first_piece_index = build_first_piece_index(v4, sp, train_units, prefix_keep)

    piece_counts = empirical_piece_counter(v4, sp, train_units)
    online_lm = OnlineSuffixLM(vocab_size=len(piece_counts), alpha=v4.LM_ALPHA)

    print("[eval] first-piece constraint codec")
    eval_t0 = time.time()
    val_sentences = m.split_sentences(val_units)

    totals = Counter()
    legality_counts = Counter()
    fallback_source = Counter()

    retained_prefix_bits_empirical = 0.0
    fallback_bits = 0.0
    fallback_predicted_piece_count = 0

    legal_recoverable_bits = 0.0
    illegal_or_unclear_repriced_fallback_bits = 0.0
    illegal_or_unclear_repriced_prefix_bits = 0.0

    constraint_count = 0
    constraint_saved_bits = 0.0
    constraint_candidate_size_dist = Counter()
    constraint_ctx_len_dist = Counter()
    constraint_examples = []
    illegal_examples = []

    for sent in val_sentences:
        rec = v4.recover_sentence(sent, knowledge, m)

        totals["unit_count"] += rec["unit_count"]
        totals["anchor_count"] += rec["anchor_count"]
        totals["recovered_non_anchor"] += rec["recovered_non_anchor"]
        totals["recoverable_residual_count"] += rec["recoverable_residual_count"]
        totals["fallback_count"] += rec["fallback_count"]

        for k, val in rec["action_counts"].items():
            totals[k] += val

        for item in rec.get("recoverable_items", []):
            u = item["unit"]
            action = item["action"]
            status = action_legality(action)
            legality_counts[status] += 1

            if status == "LEGAL_CAUSAL":
                legal_recoverable_bits += v4.BITS_ACTION_TAG + v4.AVG_CHOICE_BITS_CANDIDATE_LE_3
            else:
                scored = score_suffix_v11(
                    v4=v4,
                    sp=sp,
                    base_lm=base_lm,
                    online_lm=online_lm,
                    first_piece_index=first_piece_index,
                    text=u["s"],
                    prefix_keep=prefix_keep,
                    online_weight=online_weight,
                    constraint_enabled=constraint_enabled,
                )

                illegal_or_unclear_repriced_fallback_bits += scored["bits"]
                fallback_source.update(scored["source"])

                for piece in scored["retained_prefix"]:
                    illegal_or_unclear_repriced_prefix_bits += piece_bits_from_counter(
                        piece_counts, piece, alpha=v4.LM_ALPHA
                    )

                if len(illegal_examples) < 20:
                    illegal_examples.append({"unit": u["s"], "action": action, "status": status})

        for item in rec["fallback_items"]:
            u = item["unit"]

            legality_counts["FALLBACK"] += 1

            scored = score_suffix_v11(
                v4=v4,
                sp=sp,
                base_lm=base_lm,
                online_lm=online_lm,
                first_piece_index=first_piece_index,
                text=u["s"],
                prefix_keep=prefix_keep,
                online_weight=online_weight,
                constraint_enabled=constraint_enabled,
            )

            fallback_bits += scored["bits"]
            fallback_predicted_piece_count += scored["predicted_piece_count"]
            fallback_source.update(scored["source"])

            for piece in scored["retained_prefix"]:
                retained_prefix_bits_empirical += piece_bits_from_counter(
                    piece_counts, piece, alpha=v4.LM_ALPHA
                )

            if scored["constraint_used"]:
                constraint_count += 1
                constraint_saved_bits += scored["constraint_saved_bits"]

                meta = scored["constraint_meta"] or {}
                constraint_candidate_size_dist[meta.get("candidate_size", -1)] += 1
                constraint_ctx_len_dist[meta.get("ctx_len", -1)] += 1

                if len(constraint_examples) < 80:
                    constraint_examples.append({
                        "unit": u.get("s", ""),
                        "type": u.get("type", "UNK"),
                        "saved_bits": scored["constraint_saved_bits"],
                        "meta": meta,
                    })

    eval_time = time.time() - eval_t0

    unit_count = max(1, totals["unit_count"])
    fallback_count = max(1, totals["fallback_count"])

    anchor_bits = totals["anchor_count"] * v4.BITS_ANCHOR
    structure_bits = totals["recovered_non_anchor"] * v4.BITS_STRUCTURE_RECOVERED
    original_recoverable_bits = totals["recoverable_residual_count"] * (
        v4.BITS_ACTION_TAG + v4.AVG_CHOICE_BITS_CANDIDATE_LE_3
    )

    causal_total_with_empirical_prefix = (
        anchor_bits
        + structure_bits
        + legal_recoverable_bits
        + fallback_bits
        + retained_prefix_bits_empirical
        + illegal_or_unclear_repriced_fallback_bits
        + illegal_or_unclear_repriced_prefix_bits
    )

    canonical_space_bytes = len(" ".join(u["s"] for u in val_units).encode("utf-8"))
    val_token_ids = load_val_token_ids_from_bin(DATA_DIR)
    print(f"[debug] DATA_DIR={DATA_DIR}")
    print(f"[debug] val_token_ids={len(val_token_ids)}")
    official_like_bytes = official_byte_count_from_token_ids(sp, val_token_ids)

    result = {
        "author": AUTHOR,
        "github_id": "",
        "name": NAME,
        "seed": args.seed,
        "seeds": [args.seed],
        "status": "OK",
        "mode": "first_piece_constraint_codec",
        "prefix_keep": prefix_keep,
        "online_weight": online_weight,
        "constraint_enabled": constraint_enabled,
        "train_unit_count": len(train_units),
        "eval_unit_count": len(val_units),
        "counts": {
            "unit_count": totals["unit_count"],
            "anchor_count": totals["anchor_count"],
            "recovered_non_anchor": totals["recovered_non_anchor"],
            "recoverable_residual_count": totals["recoverable_residual_count"],
            "fallback_count": totals["fallback_count"],
            "effective_fallback_rate": totals["fallback_count"] / unit_count,
            "action_fragment": totals["action_fragment"],
            "action_subword_split": totals["action_subword_split"],
            "action_subword_single": totals["action_subword_single"],
            "action_left_phrase": totals["action_left_phrase"],
            "action_right_phrase": totals["action_right_phrase"],
        },
        "causal_legality": {
            "counts": dict(legality_counts),
            "illegal_or_unclear_count": (
                legality_counts.get("ILLEGAL_FUTURE_CONTEXT", 0)
                + legality_counts.get("UNCLEAR", 0)
            ),
            "illegal_examples": illegal_examples,
        },
        "constraint": {
            "count": constraint_count,
            "saved_bits": constraint_saved_bits,
            "avg_saved_bits": constraint_saved_bits / max(1, constraint_count),
            "candidate_size_dist": dict(sorted(constraint_candidate_size_dist.items())),
            "ctx_len_dist": dict(sorted(constraint_ctx_len_dist.items())),
            "examples": constraint_examples,
            "params": {
                "MIN_CTX_COUNT": MIN_CTX_COUNT,
                "MAX_CANDIDATES": MAX_CANDIDATES,
            },
        },
        "bits": {
            "anchor_bits": anchor_bits,
            "structure_bits": structure_bits,
            "original_recoverable_bits": original_recoverable_bits,
            "legal_recoverable_bits": legal_recoverable_bits,
            "fallback_bits": fallback_bits,
            "retained_prefix_bits_empirical": retained_prefix_bits_empirical,
            "illegal_or_unclear_repriced_fallback_bits": illegal_or_unclear_repriced_fallback_bits,
            "illegal_or_unclear_repriced_prefix_bits": illegal_or_unclear_repriced_prefix_bits,
            "causal_total_with_empirical_prefix": causal_total_with_empirical_prefix,
        },
        "fallback": {
            "fallback_avg_bits_per_word": fallback_bits / fallback_count,
            "fallback_avg_bits_per_predicted_piece": fallback_bits / max(1, fallback_predicted_piece_count),
            "predicted_piece_count": fallback_predicted_piece_count,
            "source": dict(fallback_source),
        },
        "denominators": {
            "canonical_space_joined_utf8_bytes": canonical_space_bytes,
            "official_byte_count_lut": official_like_bytes,
        },
        "scores": {
            "previous_best_bpb": PREVIOUS_BEST_BPB,
            "causal_proxy_bpb_space_bytes": causal_total_with_empirical_prefix / max(1, canonical_space_bytes),
            "causal_proxy_bpb_official_byte_lut": causal_total_with_empirical_prefix / max(1, official_like_bytes),
            "delta_vs_previous_best_official_like": (
                causal_total_with_empirical_prefix / max(1, official_like_bytes)
            ) - PREVIOUS_BEST_BPB,
        },
        "bytes_total": count_code_bytes(),
        "bytes_code": count_code_bytes(),
        "bytes_model": 0,
        "wallclock_seconds": time.time() - t0,
        "eval_time_seconds": eval_time,
        "notes": [
            "V11 applies train-only first-piece candidate constraints from retained prefix pieces.",
            "Constraint is used only when it lowers score for the first predicted suffix piece.",
            "Online LM remains score-first: update occurs after scoring each piece.",
            "No right context is used.",
            "Still proxy alignment before official eval_val adapter.",
        ],
    }

    OUT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[result]")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"[saved] {OUT_PATH}")


if __name__ == "__main__":
    main()