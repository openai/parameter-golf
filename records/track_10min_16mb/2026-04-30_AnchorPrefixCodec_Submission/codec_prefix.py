import json
import math
import importlib.util
from collections import Counter, defaultdict
from pathlib import Path

try:
    import sentencepiece as spm
except ImportError as e:
    raise SystemExit("missing sentencepiece: pip install sentencepiece") from e


BASE_PATH = Path(__file__).with_name("codec_base.py")

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
TOKENIZER_PATH = REPO_ROOT / "data" / "tokenizers" / "fineweb_1024_bpe.model"

OUT_DIR = Path(__file__).with_name("records")
OUT_PATH = OUT_DIR / "fallback_subword_lm_v4_prefix_retained.json"

LM_ALPHA = 0.05
PREFIX_KEEP_VALUES = [0, 1, 2, 3]

BITS_ANCHOR = 6.0
BITS_STRUCTURE_RECOVERED = 2.0
BITS_ACTION_TAG = 2.0
AVG_CHOICE_BITS_CANDIDATE_LE_3 = math.log2(3)

BOW2 = ("BOW2",)
BOW1 = ("BOW1",)
EOW = ("EOW",)


def load_base_module():
    spec = importlib.util.spec_from_file_location("residual_subword_probe_v1", BASE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_sp():
    sp = spm.SentencePieceProcessor()
    sp.load(str(TOKENIZER_PATH))
    return sp


def sp_tokens(sp, text):
    return tuple(sp.encode(text, out_type=int))


def word_piece_sequence(sp, text):
    return [("SP", tok) for tok in sp_tokens(sp, text)] + [EOW]


def recover_sentence(sent, knowledge, m):
    short_high_freq = knowledge["short_high_freq"]
    local_patterns = knowledge["local_patterns"]
    compound2 = knowledge["compound2"]
    compound3 = knowledge["compound3"]

    n = len(sent)
    recovered = [False] * n
    anchors = []

    for i, u in enumerate(sent):
        if m.is_anchor(u, short_high_freq):
            anchors.append(i)
            recovered[i] = True

    for i in anchors:
        anchor = m.anchor_signature(sent[i])
        for offset in range(-m.LEFT_RADIUS, m.RIGHT_RADIUS + 1):
            if offset == 0:
                continue
            j = i + offset
            if 0 <= j < n:
                pat = (anchor, offset, m.unit_signature(sent[j]))
                if local_patterns.get(pat, 0) > 0:
                    recovered[j] = True

    for i in range(n - 1):
        sig2 = (m.unit_signature(sent[i]), m.unit_signature(sent[i + 1]))
        if compound2.get(sig2, 0) > 0:
            recovered[i] = True
            recovered[i + 1] = True

    for i in range(n - 2):
        sig3 = (
            m.unit_signature(sent[i]),
            m.unit_signature(sent[i + 1]),
            m.unit_signature(sent[i + 2]),
        )
        if compound3.get(sig3, 0) > 0:
            recovered[i] = True
            recovered[i + 1] = True
            recovered[i + 2] = True

    fallback_items = []
    recoverable_items = []
    action_counts = Counter()
    recoverable_residual_count = 0

    for i, ok in enumerate(recovered):
        if ok:
            continue

        action = m.evaluate_recoverability_action(sent[i], sent, i, knowledge)

        if action is not None:
            recoverable_residual_count += 1
            action_counts[f"action_{action['action']}"] += 1
            recoverable_items.append({"unit": sent[i], "idx": i, "action": action})
        else:
            fallback_items.append({"unit": sent[i], "idx": i})

    return {
        "unit_count": n,
        "anchor_count": len(anchors),
        "recovered_non_anchor": max(0, sum(recovered) - len(anchors)),
        "recoverable_residual_count": recoverable_residual_count,
        "fallback_items": fallback_items,
        "recoverable_items": recoverable_items,
        "fallback_count": len(fallback_items),
        "action_counts": action_counts,
    }


def build_piece_ngram_lm(train_units, sp):
    unigram = Counter()
    bigram = defaultdict(Counter)
    trigram = defaultdict(Counter)
    totals = Counter()

    for u in train_units:
        seq = word_piece_sequence(sp, u["s"])

        prev2 = BOW2
        prev1 = BOW1

        for piece in seq:
            trigram[(prev2, prev1)][piece] += 1
            bigram[prev1][piece] += 1
            unigram[piece] += 1

            prev2, prev1 = prev1, piece
            totals["piece_train_tokens"] += 1

        totals["word_train_items"] += 1

    return {
        "unigram": unigram,
        "bigram": bigram,
        "trigram": trigram,
        "vocab_size": max(1, len(unigram)),
        "unigram_total": sum(unigram.values()),
        "totals": totals,
    }


def prob_from_counter(counter, piece, vocab_size):
    if not counter:
        return None

    total = sum(counter.values())
    count = counter.get(piece, 0)

    if count <= 0:
        return None

    return (count + LM_ALPHA) / (total + LM_ALPHA * vocab_size)


def score_suffix_with_prefix(sp, lm, text, prefix_keep):
    seq = word_piece_sequence(sp, text)
    pieces_only = [p for p in seq if p != EOW]

    # Keep at most existing non-EOW pieces.
    keep = min(prefix_keep, len(pieces_only))

    # Prefix is assumed transmitted/retained, so not charged here.
    prefix = pieces_only[:keep]
    suffix = pieces_only[keep:] + [EOW]

    prev2 = BOW2
    prev1 = BOW1

    # Feed retained prefix into LM state without charging bits.
    for p in prefix:
        prev2, prev1 = prev1, p

    bits = 0.0
    src = Counter()

    for piece in suffix:
        p = prob_from_counter(lm["trigram"].get((prev2, prev1)), piece, lm["vocab_size"])
        if p is not None:
            source = "trigram"
        else:
            p = prob_from_counter(lm["bigram"].get(prev1), piece, lm["vocab_size"])
            if p is not None:
                source = "bigram"
            else:
                p = prob_from_counter(lm["unigram"], piece, lm["vocab_size"])
                if p is not None:
                    source = "unigram"
                else:
                    p = LM_ALPHA / (lm["unigram_total"] + LM_ALPHA * lm["vocab_size"])
                    source = "unseen"

        bits += -math.log2(max(p, 1e-12))
        src[source] += 1
        prev2, prev1 = prev1, piece

    return {
        "bits": bits,
        "source": dict(src),
        "full_piece_count": len(seq),
        "retained_piece_count": keep,
        "predicted_piece_count": len(suffix),
        "full_seq": seq,
        "retained_prefix": prefix,
        "predicted_suffix": suffix,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    m = load_base_module()
    sp = load_sp()

    print("loading train units...")
    train_units = m.load_units(m.TRAIN_PATHS, m.MAX_TRAIN_UNITS)

    print("loading val units...")
    val_units = m.load_units(m.VAL_PATHS, m.MAX_VAL_UNITS)

    print("building structure/subword knowledge...")
    knowledge = m.build_train_knowledge(train_units)

    print("building all-unit SP piece ngram LM...")
    lm = build_piece_ngram_lm(train_units, sp)

    print("splitting val sentences...")
    val_sentences = m.split_sentences(val_units)

    totals = Counter()
    case_bits = {k: 0.0 for k in PREFIX_KEEP_VALUES}
    case_source = {k: Counter() for k in PREFIX_KEEP_VALUES}
    case_predicted_pieces = {k: 0 for k in PREFIX_KEEP_VALUES}
    examples = []

    for sent in val_sentences:
        rec = recover_sentence(sent, knowledge, m)

        totals["unit_count"] += rec["unit_count"]
        totals["anchor_count"] += rec["anchor_count"]
        totals["recovered_non_anchor"] += rec["recovered_non_anchor"]
        totals["recoverable_residual_count"] += rec["recoverable_residual_count"]
        totals["fallback_count"] += rec["fallback_count"]

        for k, v in rec["action_counts"].items():
            totals[k] += v

        for item in rec["fallback_items"]:
            u = item["unit"]

            per_item = {}
            for keep in PREFIX_KEEP_VALUES:
                scored = score_suffix_with_prefix(sp, lm, u["s"], keep)
                case_bits[keep] += scored["bits"]
                case_predicted_pieces[keep] += scored["predicted_piece_count"]

                for src_name, src_count in scored["source"].items():
                    case_source[keep][src_name] += src_count

                per_item[str(keep)] = {
                    "bits": scored["bits"],
                    "retained_piece_count": scored["retained_piece_count"],
                    "predicted_piece_count": scored["predicted_piece_count"],
                    "retained_prefix": [str(x) for x in scored["retained_prefix"]],
                    "predicted_suffix": [str(x) for x in scored["predicted_suffix"]],
                    "source": scored["source"],
                }

            if len(examples) < 80:
                examples.append({
                    "unit": u["s"],
                    "idx": item["idx"],
                    "full_sp_seq": [str(x) for x in word_piece_sequence(sp, u["s"])],
                    "cases": per_item,
                })

    anchor_bits = totals["anchor_count"] * BITS_ANCHOR
    structure_bits = totals["recovered_non_anchor"] * BITS_STRUCTURE_RECOVERED
    recoverable_bits = totals["recoverable_residual_count"] * (
        BITS_ACTION_TAG + AVG_CHOICE_BITS_CANDIDATE_LE_3
    )

    unit_count = max(1, totals["unit_count"])
    fallback_count = max(1, totals["fallback_count"])

    cases = {}
    for keep in PREFIX_KEEP_VALUES:
        total_bits = anchor_bits + structure_bits + recoverable_bits + case_bits[keep]

        cases[str(keep)] = {
            "prefix_keep": keep,
            "fallback_bits": case_bits[keep],
            "fallback_avg_bits_per_word": case_bits[keep] / fallback_count,
            "fallback_avg_bits_per_predicted_piece": case_bits[keep] / max(1, case_predicted_pieces[keep]),
            "predicted_piece_count": case_predicted_pieces[keep],
            "total_bits": total_bits,
            "bits_per_unit": total_bits / unit_count,
            "source": dict(case_source[keep]),
        }

    result = {
        "status": "OK",
        "probe": "fallback_subword_lm_v4_prefix_retained",
        "tokenizer_path": str(TOKENIZER_PATH),
        "train_units": len(train_units),
        "val_units": len(val_units),
        "lm_train": {
            "word_train_items": lm["totals"]["word_train_items"],
            "piece_train_tokens": lm["totals"]["piece_train_tokens"],
            "vocab_size": lm["vocab_size"],
            "unigram_count": len(lm["unigram"]),
            "bigram_state_count": len(lm["bigram"]),
            "trigram_state_count": len(lm["trigram"]),
        },
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
        "base_bits": {
            "anchor_bits": anchor_bits,
            "structure_bits": structure_bits,
            "recoverable_residual_bits": recoverable_bits,
        },
        "cases": cases,
        "examples": examples,
    }

    OUT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"[saved] {OUT_PATH}")


if __name__ == "__main__":
    main()



