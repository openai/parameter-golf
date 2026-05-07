import json
from collections import Counter, defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
CANONICAL_UNITS_DIR = HERE / "records" / "canonical_units" / "per_file"

TRAIN_PATHS = [
    CANONICAL_UNITS_DIR / f"train_{i:06d}_canonical_units_v1.jsonl"
    for i in range(8)
]

VAL_PATHS = [
    CANONICAL_UNITS_DIR / "val_000000_canonical_units_v1.jsonl",
]

OUT_DIR = HERE / "records"

MAX_TRAIN_UNITS = 4_000_000
MAX_VAL_UNITS = 200_000

ANCHOR_FUNCTION_WORDS = {
    "of", "to", "in", "by", "for", "with", "as", "from", "and",
    "is", "was", "are", "were", "be", "been", "on", "at", "into",
    "than", "that", "which", "who", "what", "when", "where", "why",
}

DERIV_SUFFIXES = (
    "ization", "isation", "ational", "fulness", "ousness",
    "ically", "ingly", "edly",
    "tion", "sion", "ment", "ness", "able", "ible", "ally",
    "ing", "ed", "er", "est", "ly", "al", "ic", "ive",
    "ous", "ful", "less", "ity", "ism", "ist", "ize", "ise",
    "ate", "ian", "an", "es", "s",
)

LEFT_RADIUS = 2
RIGHT_RADIUS = 2
MAX_CANDIDATES = 3

SUBWORD_MIN_LEN = 3
SUBWORD_MAX_LEN = 8
SUBWORD_MIN_FREQ = 20


def load_units(paths, max_units):
    units = []
    count = 0

    for path in paths:
        p = Path(path)
        if not p.exists():
            print(f"[warn] missing train/val path: {p}")
            continue

        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                j = json.loads(line)

                if j.get("is_dropped"):
                    continue

                typ = j.get("canonical_type")
                s = j.get("canonical_string")
                key = j.get("canonical_key")
                file_id = j.get("file_id")
                unit_id = j.get("unit_id")

                if not typ or s is None or key is None:
                    continue

                if typ == "SPECIAL":
                    continue

                units.append({
                    "file_id": file_id,
                    "unit_id": unit_id,
                    "s": s,
                    "lower": s.lower(),
                    "key": key,
                    "type": typ,
                    "len": len(s),
                    "token_count": int(j.get("token_count", 1)),
                    "has_suffix_punct": bool(j.get("has_suffix_punct", False)),
                    "suffix_piece_preview": j.get("suffix_piece_preview", ""),
                    "suffix_punct_pieces": j.get("suffix_punct_pieces", []),
                    "raw_piece_preview": j.get("raw_piece_preview", ""),
                })

                count += 1
                if count >= max_units:
                    return units

    return units


def is_sentence_boundary(u):
    s = u.get("s", "")
    suffix_preview = u.get("suffix_piece_preview", "") or ""
    suffix_pieces = u.get("suffix_punct_pieces", []) or []
    raw_preview = u.get("raw_piece_preview", "") or ""

    boundary_marks = {".", "?", "!"}

    if any(mark in s for mark in boundary_marks):
        return True

    if any(mark in suffix_preview for mark in boundary_marks):
        return True

    if any(any(mark in str(piece) for mark in boundary_marks) for piece in suffix_pieces):
        return True

    if any(mark in raw_preview for mark in boundary_marks):
        return True

    return False


def split_sentences(units):
    sentences = []
    cur = []

    for u in units:
        cur.append(u)
        if is_sentence_boundary(u):
            sentences.append(cur)
            cur = []

    if cur:
        sentences.append(cur)

    return sentences


def is_name_like(s):
    if not s or len(s) <= 1:
        return False
    return s[0].isupper() and any(ch.islower() for ch in s[1:])


def stem_variants(w):
    out = {w}

    for suf in DERIV_SUFFIXES:
        if len(w) <= len(suf) + 2:
            continue
        if not w.endswith(suf):
            continue

        base = w[:-len(suf)]
        out.add(base)

        if suf in {"ing", "ingly", "ed"}:
            out.add(base + "e")

        if suf in {"es", "s"} and base.endswith("i"):
            out.add(base[:-1] + "y")

        if suf in {"ian", "an"}:
            out.add(base)
            if base.endswith("d") or base.endswith("c"):
                out.add(base + "a")

    return out


def unit_signature(u):
    lower = u["lower"]

    if is_name_like(u["s"]):
        return ("NAME", u["len"])

    stems = stem_variants(lower)
    shortest = min(stems, key=len)

    return (u["type"], u["len"], shortest)


def residual_lm_key(u):
    return unit_signature(u)


def word_fragment(u):
    lower = u["lower"]

    if len(lower) <= 4:
        return lower

    stems = stem_variants(lower)
    shortest = min(stems, key=len)

    if len(shortest) >= 4:
        return shortest[:4]

    return lower[:4]


def fragment_key(u):
    return (
        word_fragment(u),
        u["len"],
        u["type"],
    )


def phrase_left_key(prev_u, cur_u):
    return (
        prev_u["lower"],
        word_fragment(cur_u),
        cur_u["len"],
        cur_u["type"],
    )


def phrase_right_key(cur_u, next_u):
    return (
        word_fragment(cur_u),
        cur_u["len"],
        cur_u["type"],
        next_u["lower"],
    )


def clean_subword_source(w):
    return "".join(ch for ch in w.lower() if ch.isalpha())


def build_subword_vocab(train_units, min_freq=SUBWORD_MIN_FREQ):
    counts = Counter()

    for u in train_units:
        w = clean_subword_source(u["lower"])
        if len(w) < SUBWORD_MIN_LEN:
            continue

        max_len = min(SUBWORD_MAX_LEN, len(w))
        for size in range(SUBWORD_MIN_LEN, max_len + 1):
            for i in range(0, len(w) - size + 1):
                sub = w[i:i + size]
                counts[sub] += 1

    vocab = {
        sub for sub, c in counts.items()
        if c >= min_freq
    }

    return vocab, counts


def build_subword_signature_index(train_units, subword_vocab):
    """
    Maps subword-sequence signatures to possible full unit signatures.

    Example:
      television -> possible split ("tele", "vision")
      index[("tele", "vision", len=10, type=WORD)] -> television signature
    """
    index = defaultdict(Counter)
    single_index = defaultdict(Counter)

    for u in train_units:
        raw = clean_subword_source(u["lower"])
        if len(raw) < SUBWORD_MIN_LEN:
            continue

        target = residual_lm_key(u)

        # single subword key
        for size in range(SUBWORD_MIN_LEN, min(SUBWORD_MAX_LEN, len(raw)) + 1):
            for i in range(0, len(raw) - size + 1):
                sub = raw[i:i + size]
                if sub in subword_vocab:
                    single_key = (
                        sub,
                        u["len"],
                        u["type"],
                    )
                    single_index[single_key][target] += 1

        # exact two-piece split key
        for cut in range(SUBWORD_MIN_LEN, len(raw) - SUBWORD_MIN_LEN + 1):
            left = raw[:cut]
            right = raw[cut:]

            if left not in subword_vocab or right not in subword_vocab:
                continue

            key = (
                left,
                right,
                u["len"],
                u["type"],
            )
            index[key][target] += 1

    return index, single_index


def best_subword_action(u, knowledge, max_candidates=MAX_CANDIDATES):
    raw = clean_subword_source(u["lower"])
    if len(raw) < 2 * SUBWORD_MIN_LEN:
        return None

    target = residual_lm_key(u)
    vocab = knowledge["subword_vocab"]
    split_index = knowledge["subword_split_index"]
    single_index = knowledge["subword_single_index"]

    best = None

    # Try exact two-piece full cover.
    for cut in range(SUBWORD_MIN_LEN, len(raw) - SUBWORD_MIN_LEN + 1):
        left = raw[:cut]
        right = raw[cut:]

        if left not in vocab or right not in vocab:
            continue

        key = (
            left,
            right,
            u["len"],
            u["type"],
        )
        r = candidate_rank_and_size(split_index.get(key), target)

        if r is not None and r["rank"] is not None and r["candidate_size"] <= max_candidates:
            cand = {
                "action": "subword_split",
                "rank": r["rank"],
                "candidate_size": r["candidate_size"],
                "prob": r["prob"],
                "count": r["count"],
                "split": [left, right],
            }

            if best is None or cand["candidate_size"] < best["candidate_size"]:
                best = cand

    # Fallback: single strong subword cue, not full-cover.
    # This can catch words where one data-driven piece is highly diagnostic.
    for size in range(SUBWORD_MAX_LEN, SUBWORD_MIN_LEN - 1, -1):
        if size > len(raw):
            continue

        for i in range(0, len(raw) - size + 1):
            sub = raw[i:i + size]
            if sub not in vocab:
                continue

            key = (
                sub,
                u["len"],
                u["type"],
            )
            r = candidate_rank_and_size(single_index.get(key), target)

            if r is not None and r["rank"] is not None and r["candidate_size"] <= max_candidates:
                cand = {
                    "action": "subword_single",
                    "rank": r["rank"],
                    "candidate_size": r["candidate_size"],
                    "prob": r["prob"],
                    "count": r["count"],
                    "subword": sub,
                }

                if best is None or cand["candidate_size"] < best["candidate_size"]:
                    best = cand

    return best


def is_anchor(u, short_high_freq):
    lower = u["lower"]
    typ = u["type"]

    if lower in ANCHOR_FUNCTION_WORDS:
        return True

    if lower in short_high_freq:
        return True

    if typ in {"REL", "FUNC"}:
        return True

    if typ == "WORD" and len(lower) <= 3 and lower in short_high_freq:
        return True

    return False


def anchor_signature(u):
    lower = u["lower"]

    if lower in ANCHOR_FUNCTION_WORDS:
        return ("FUNC_WORD", lower)

    return ("TYPE_LEN", u["type"], u["len"])


def candidate_rank_and_size(counter, target_key):
    if not counter:
        return None

    ranked = counter.most_common()
    size = len(ranked)
    total = sum(counter.values())

    for rank, (key, count) in enumerate(ranked, start=1):
        if key == target_key:
            return {
                "rank": rank,
                "candidate_size": size,
                "count": count,
                "prob": count / max(1, total),
            }

    return {
        "rank": None,
        "candidate_size": size,
        "count": 0,
        "prob": 0.0,
    }


def build_train_knowledge(train_units):
    freq = Counter(u["lower"] for u in train_units)

    short_high_freq = {
        w for w, c in freq.items()
        if len(w) <= 3 and c >= 10
    }

    print("building subword vocab...")
    subword_vocab, subword_counts = build_subword_vocab(train_units)

    print("building subword indexes...")
    subword_split_index, subword_single_index = build_subword_signature_index(
        train_units,
        subword_vocab,
    )

    local_patterns = Counter()
    compound2 = Counter()
    compound3 = Counter()

    fragment_candidates = defaultdict(Counter)
    left_phrase_candidates = defaultdict(Counter)
    right_phrase_candidates = defaultdict(Counter)

    train_sentences = split_sentences(train_units)

    for sent in train_sentences:
        n = len(sent)

        for i, u in enumerate(sent):
            fk = fragment_key(u)
            target = residual_lm_key(u)
            fragment_candidates[fk][target] += 1

            if i > 0:
                lk = phrase_left_key(sent[i - 1], u)
                left_phrase_candidates[lk][target] += 1

            if i + 1 < n:
                rk = phrase_right_key(u, sent[i + 1])
                right_phrase_candidates[rk][target] += 1

        for k in range(n - 1):
            sig2 = (unit_signature(sent[k]), unit_signature(sent[k + 1]))
            compound2[sig2] += 1

        for k in range(n - 2):
            sig3 = (
                unit_signature(sent[k]),
                unit_signature(sent[k + 1]),
                unit_signature(sent[k + 2]),
            )
            compound3[sig3] += 1

        for i, u in enumerate(sent):
            if not is_anchor(u, short_high_freq):
                continue

            anchor = anchor_signature(u)

            for offset in range(-LEFT_RADIUS, RIGHT_RADIUS + 1):
                if offset == 0:
                    continue

                j = i + offset
                if j < 0 or j >= n:
                    continue

                neighbor = sent[j]

                pat = (
                    anchor,
                    offset,
                    unit_signature(neighbor),
                )
                local_patterns[pat] += 1

    return {
        "freq": freq,
        "short_high_freq": short_high_freq,
        "local_patterns": local_patterns,
        "compound2": compound2,
        "compound3": compound3,
        "fragment_candidates": fragment_candidates,
        "left_phrase_candidates": left_phrase_candidates,
        "right_phrase_candidates": right_phrase_candidates,
        "subword_vocab": subword_vocab,
        "subword_counts": subword_counts,
        "subword_split_index": subword_split_index,
        "subword_single_index": subword_single_index,
        "train_sentence_count": len(train_sentences),
    }


def evaluate_recoverability_action(u, sent, idx, knowledge, max_candidates=MAX_CANDIDATES):
    target = residual_lm_key(u)
    best = None

    # A. existing hand-built fragment recovery
    fk = fragment_key(u)
    r = candidate_rank_and_size(
        knowledge["fragment_candidates"].get(fk),
        target,
    )
    if r is not None and r["rank"] is not None and r["candidate_size"] <= max_candidates:
        best = {
            "action": "fragment",
            "rank": r["rank"],
            "candidate_size": r["candidate_size"],
            "prob": r["prob"],
            "count": r["count"],
        }

    # B. data-driven subword recovery
    sub_action = best_subword_action(u, knowledge, max_candidates=max_candidates)
    if sub_action is not None:
        if best is None or sub_action["candidate_size"] < best["candidate_size"]:
            best = sub_action

    # C. phrase recovery
    if idx > 0:
        lk = phrase_left_key(sent[idx - 1], u)
        r = candidate_rank_and_size(
            knowledge["left_phrase_candidates"].get(lk),
            target,
        )
        if r is not None and r["rank"] is not None and r["candidate_size"] <= max_candidates:
            cand = {
                "action": "left_phrase",
                "rank": r["rank"],
                "candidate_size": r["candidate_size"],
                "prob": r["prob"],
                "count": r["count"],
            }
            if best is None or cand["candidate_size"] < best["candidate_size"]:
                best = cand

    if idx + 1 < len(sent):
        rk = phrase_right_key(u, sent[idx + 1])
        r = candidate_rank_and_size(
            knowledge["right_phrase_candidates"].get(rk),
            target,
        )
        if r is not None and r["rank"] is not None and r["candidate_size"] <= max_candidates:
            cand = {
                "action": "right_phrase",
                "rank": r["rank"],
                "candidate_size": r["candidate_size"],
                "prob": r["prob"],
                "count": r["count"],
            }
            if best is None or cand["candidate_size"] < best["candidate_size"]:
                best = cand

    return best


def recover_sentence(sent, knowledge):
    short_high_freq = knowledge["short_high_freq"]
    local_patterns = knowledge["local_patterns"]
    compound2 = knowledge["compound2"]
    compound3 = knowledge["compound3"]

    n = len(sent)
    recovered = [False] * n
    anchors = []

    for i, u in enumerate(sent):
        if is_anchor(u, short_high_freq):
            anchors.append(i)
            recovered[i] = True

    for i in anchors:
        anchor = anchor_signature(sent[i])

        for offset in range(-LEFT_RADIUS, RIGHT_RADIUS + 1):
            if offset == 0:
                continue

            j = i + offset
            if j < 0 or j >= n:
                continue

            neighbor = sent[j]

            pat = (
                anchor,
                offset,
                unit_signature(neighbor),
            )

            if local_patterns.get(pat, 0) > 0:
                recovered[j] = True

    for i in range(n - 1):
        sig2 = (unit_signature(sent[i]), unit_signature(sent[i + 1]))
        if compound2.get(sig2, 0) > 0:
            recovered[i] = True
            recovered[i + 1] = True

    for i in range(n - 2):
        sig3 = (
            unit_signature(sent[i]),
            unit_signature(sent[i + 1]),
            unit_signature(sent[i + 2]),
        )
        if compound3.get(sig3, 0) > 0:
            recovered[i] = True
            recovered[i + 1] = True
            recovered[i + 2] = True

    residual_items = []

    for i, ok in enumerate(recovered):
        if ok:
            continue

        action = evaluate_recoverability_action(
            sent[i],
            sent,
            i,
            knowledge,
        )

        residual_items.append({
            "unit": sent[i],
            "idx": i,
            "action": action,
        })

    residual_chars = sum(len(item["unit"]["s"]) for item in residual_items)

    return {
        "unit_count": n,
        "anchor_count": len(anchors),
        "recovered_count": sum(recovered),
        "residual_count": n - sum(recovered),
        "residual_chars": residual_chars,
        "residual_items": residual_items,
        "residual_examples": [
            item["unit"]["s"]
            for item in residual_items
        ][:20],
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("loading train units...")
    train_units = load_units(TRAIN_PATHS, MAX_TRAIN_UNITS)

    print("loading val units...")
    val_units = load_units(VAL_PATHS, MAX_VAL_UNITS)

    print("building train knowledge...")
    knowledge = build_train_knowledge(train_units)

    print("splitting val sentences...")
    val_sentences = split_sentences(val_units)

    totals = Counter()
    examples = []
    action_examples = []

    for sent in val_sentences:
        r = recover_sentence(sent, knowledge)

        totals["sentence_count"] += 1
        totals["unit_count"] += r["unit_count"]
        totals["anchor_unit_count"] += r["anchor_count"]
        totals["recovered_unit_count"] += r["recovered_count"]
        totals["residual_unit_count"] += r["residual_count"]
        totals["residual_chars"] += r["residual_chars"]

        for item in r["residual_items"]:
            action = item["action"]

            if action is not None:
                totals["recoverable_residual_count"] += 1
                totals[f"action_{action['action']}"] += 1

                if len(action_examples) < 80:
                    row = {
                        "unit": item["unit"]["s"],
                        "idx": item["idx"],
                        "action": action,
                    }
                    action_examples.append(row)
            else:
                totals["fallback_residual_count"] += 1

        if len(examples) < 30 and r["residual_count"] > 0:
            examples.append({
                "sentence_preview": " ".join(u["s"] for u in sent[:30]),
                "unit_count": r["unit_count"],
                "anchor_count": r["anchor_count"],
                "recovered_count": r["recovered_count"],
                "residual_count": r["residual_count"],
                "residual_examples": r["residual_examples"],
            })

    unit_count = totals["unit_count"] or 1
    residual_count = totals["residual_unit_count"] or 1

    result = {
        "status": "OK",
        "probe": "residual_subword_probe_v1",
        "train_paths": TRAIN_PATHS,
        "val_paths": VAL_PATHS,
        "max_train_units": MAX_TRAIN_UNITS,
        "max_val_units": MAX_VAL_UNITS,
        "left_radius": LEFT_RADIUS,
        "right_radius": RIGHT_RADIUS,
        "max_candidates": MAX_CANDIDATES,
        "subword_min_len": SUBWORD_MIN_LEN,
        "subword_max_len": SUBWORD_MAX_LEN,
        "subword_min_freq": SUBWORD_MIN_FREQ,
        "train_sentence_count": knowledge["train_sentence_count"],
        "train_short_high_freq_count": len(knowledge["short_high_freq"]),
        "train_local_pattern_count": len(knowledge["local_patterns"]),
        "train_compound2_count": len(knowledge["compound2"]),
        "train_compound3_count": len(knowledge["compound3"]),
        "train_fragment_key_count": len(knowledge["fragment_candidates"]),
        "train_left_phrase_key_count": len(knowledge["left_phrase_candidates"]),
        "train_right_phrase_key_count": len(knowledge["right_phrase_candidates"]),
        "train_subword_vocab_count": len(knowledge["subword_vocab"]),
        "train_subword_split_key_count": len(knowledge["subword_split_index"]),
        "train_subword_single_key_count": len(knowledge["subword_single_index"]),
        "sentence_count": totals["sentence_count"],
        "unit_count": totals["unit_count"],
        "anchor_unit_count": totals["anchor_unit_count"],
        "recovered_unit_count": totals["recovered_unit_count"],
        "residual_unit_count": totals["residual_unit_count"],
        "residual_chars": totals["residual_chars"],
        "anchor_coverage_rate": totals["anchor_unit_count"] / unit_count,
        "recovered_unit_rate": totals["recovered_unit_count"] / unit_count,
        "residual_unit_rate": totals["residual_unit_count"] / unit_count,
        "recoverability_gated": {
            "recoverable_residual_count": totals["recoverable_residual_count"],
            "fallback_residual_count": totals["fallback_residual_count"],
            "recoverable_residual_rate": totals["recoverable_residual_count"] / residual_count,
            "action_fragment": totals["action_fragment"],
            "action_subword_split": totals["action_subword_split"],
            "action_subword_single": totals["action_subword_single"],
            "action_left_phrase": totals["action_left_phrase"],
            "action_right_phrase": totals["action_right_phrase"],
        },
        "action_examples": action_examples,
        "examples": examples,
    }

    out_path = OUT_DIR / "residual_subword_probe_v1.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
