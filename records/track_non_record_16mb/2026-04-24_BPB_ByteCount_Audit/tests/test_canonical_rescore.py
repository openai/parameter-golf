"""Tests for canonical_rescore.py — the BPB byte-count audit tool."""

import sys
from pathlib import Path

import pytest

# canonical_rescore.py lives at the submission folder root (one level up from tests/).
sys.path.insert(0, str(Path(__file__).parent.parent))

import canonical_rescore as cr  # noqa: E402

SUBMISSION_ROOT = Path(__file__).parent.parent
PARAMETER_GOLF = Path("/workspace/parameter-golf")
CANONICAL_TRAIN_SCRIPT = (
    PARAMETER_GOLF
    / "records"
    / "track_10min_16mb"
    / "2026-04-18_SP8192_MPSGD_QKGain525"
    / "train_gpt.py"
)
BUGGY_FIXTURE = SUBMISSION_ROOT / "tests" / "fixtures" / "buggy_train_gpt.py"
BUGGY_BYTE_TOKEN_FIXTURE = SUBMISSION_ROOT / "tests" / "fixtures" / "buggy_byte_token.py"
BUGGY_MISSING_IS_UNUSED_FIXTURE = SUBMISSION_ROOT / "tests" / "fixtures" / "buggy_missing_is_unused.py"
BUGGY_TRIPLE_FIXTURE = SUBMISSION_ROOT / "tests" / "fixtures" / "buggy_triple.py"
TOKENIZER = PARAMETER_GOLF / "data" / "tokenizers" / "fineweb_8192_bpe.model"
VAL_DATA = str(PARAMETER_GOLF / "data" / "datasets" / "fineweb10B_sp8192" / "fineweb_val_*.bin")

# Smaller subset to keep tests fast (~1s); the full audit uses all 40M tokens.
SMOKE_TOKENS = 200_000


@pytest.fixture(scope="module")
def luts():
    """Build the canonical LUTs once per module."""
    if not TOKENIZER.exists():
        pytest.skip(f"Tokenizer missing: {TOKENIZER}")
    return cr.build_canonical_luts(TOKENIZER)


@pytest.fixture(scope="module")
def val_tokens():
    if not Path(VAL_DATA.replace("*", "000000")).exists():
        pytest.skip(f"Val data missing under: {VAL_DATA}")
    return cr.load_val_tokens(VAL_DATA)


# ---------------------------------------------------------------------------
# Static LUT classification
# ---------------------------------------------------------------------------


def test_canonical_pr1727_classifies_as_correct():
    if not CANONICAL_TRAIN_SCRIPT.exists():
        pytest.skip(f"Canonical script missing: {CANONICAL_TRAIN_SCRIPT}")
    src = CANONICAL_TRAIN_SCRIPT.read_text()
    assert cr.classify_lut(src) == "CORRECT"


def test_buggy_fixture_classifies_as_buggy():
    src = BUGGY_FIXTURE.read_text()
    assert cr.classify_lut(src) == "BUGGY"


def test_obfuscated_pattern_classifies_as_obfuscated():
    src = (
        "import lzma, base64\n"
        "exec(lzma.decompress(base64.b85decode('BLOB')))\n"
    )
    assert cr.classify_lut(src) == "OBFUSCATED"


def test_lzma_import_alone_does_not_trigger_obfuscated():
    """PR #1727 imports lzma for artifact compression but is not obfuscated.

    The three-variant classifier requires all canonical properties to match
    for a CORRECT verdict, so the synthetic source below includes the
    sp.is_byte branch and the full boundary predicate alongside the
    canonical leading-space assignment.
    """
    src = (
        "import lzma\n"
        "def build_sentencepiece_luts(sp, vocab, device):\n"
        "    for token_id in range(vocab):\n"
        "        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):\n"
        "            continue\n"
        "        if sp.is_byte(token_id):\n"
        "            base_bytes_np[token_id] = 1\n"
        "            continue\n"
        "        piece = sp.id_to_piece(token_id)\n"
        "        base_bytes_np[token_id] = len(piece.encode('utf-8'))\n"
    )
    assert cr.classify_lut(src) == "CORRECT"


def test_unknown_pattern_classifies_as_unknown():
    src = "def build_sentencepiece_luts(sp, vocab, device):\n    return None\n"
    assert cr.classify_lut(src) == "UNKNOWN"


def test_buggy_pattern_with_extra_whitespace():
    src = "base_bytes_np[token_id] = len( piece.encode('utf-8') ) + 1\n"
    assert cr.classify_lut(src) == "BUGGY"


# ---------------------------------------------------------------------------
# Three-variant deviation detection (leading_space_plus_one,
# byte_token_wrong_size, missing_is_unused)
# ---------------------------------------------------------------------------


def test_detector_byte_token_bug():
    """sp.is_byte branch sized by len(piece.encode('utf-8')) instead of 1."""
    src = BUGGY_BYTE_TOKEN_FIXTURE.read_text()
    status, deviations = cr.classify_lut_detailed(src)
    assert status == "BUGGY"
    assert "byte_token_wrong_size" in deviations
    # Only the byte-token bug is present — +1 was reverted and is_unused is intact.
    assert "leading_space_plus_one" not in deviations
    assert "missing_is_unused" not in deviations


def test_detector_missing_is_unused():
    """Boundary predicate omits sp.is_unused."""
    src = BUGGY_MISSING_IS_UNUSED_FIXTURE.read_text()
    status, deviations = cr.classify_lut_detailed(src)
    assert status == "BUGGY"
    assert "missing_is_unused" in deviations
    assert "leading_space_plus_one" not in deviations
    assert "byte_token_wrong_size" not in deviations


def test_detector_triple_bug():
    """All three bugs present simultaneously (yahya010 train_gdn_7k.py case)."""
    src = BUGGY_TRIPLE_FIXTURE.read_text()
    status, deviations = cr.classify_lut_detailed(src)
    assert status == "BUGGY"
    assert set(deviations) == {
        "leading_space_plus_one",
        "byte_token_wrong_size",
        "missing_is_unused",
    }


def test_canonical_not_flagged():
    """Regression: PR #1727's canonical train_gpt.py still CORRECT under the
    stricter three-variant classifier."""
    if not CANONICAL_TRAIN_SCRIPT.exists():
        pytest.skip(f"Canonical script missing: {CANONICAL_TRAIN_SCRIPT}")
    src = CANONICAL_TRAIN_SCRIPT.read_text()
    status, deviations = cr.classify_lut_detailed(src)
    assert status == "CORRECT"
    assert deviations == []


def test_original_buggy_still_detected():
    """Regression: the original +1-only fixture is still BUGGY, with
    deviations = ['leading_space_plus_one'] (no others)."""
    src = BUGGY_FIXTURE.read_text()
    status, deviations = cr.classify_lut_detailed(src)
    assert status == "BUGGY"
    assert deviations == ["leading_space_plus_one"]


def test_classify_lut_backcompat_returns_string():
    """The single-return classify_lut still exists for callers that only need
    the status string."""
    assert cr.classify_lut(BUGGY_FIXTURE.read_text()) == "BUGGY"
    assert cr.classify_lut(BUGGY_BYTE_TOKEN_FIXTURE.read_text()) == "BUGGY"
    assert cr.classify_lut(BUGGY_MISSING_IS_UNUSED_FIXTURE.read_text()) == "BUGGY"
    assert cr.classify_lut(BUGGY_TRIPLE_FIXTURE.read_text()) == "BUGGY"


# ---------------------------------------------------------------------------
# Byte counting math
# ---------------------------------------------------------------------------


def test_byte_count_canonical_matches_eval_logic_on_synthetic_data():
    """Tiny synthetic case: verify canonical sum matches a hand-computed value."""
    import numpy as np

    # vocab: 0=boundary, 1=byte, 2='hi' no leading space (2 bytes), 3='▁the' (3 bytes + maybe space)
    base_bytes = np.array([0, 1, 2, 3], dtype=np.int32)
    has_leading_space = np.array([False, False, False, True], dtype=bool)
    is_boundary = np.array([True, False, False, False], dtype=bool)

    # Need at least seq_len-stride+2 tokens for one window. Use seq_len=8, stride=2.
    # 10 tokens.
    val_tokens = np.array([0, 2, 3, 2, 3, 2, 3, 0, 2, 3], dtype=np.uint16)
    seq_len, stride = 8, 2

    counts = cr.compute_byte_counts(
        val_tokens, base_bytes, has_leading_space, is_boundary, seq_len, stride
    )

    # Hand calc: scored y positions tile val_tokens[1:total_tokens+1] = val_tokens[1:10] = [2,3,2,3,2,3,0,2,3]
    # base_bytes sum: 2+3+2+3+2+3+0+2+3 = 20
    # leading_space[y] mask: [F,T,F,T,F,T,F,F,T] → 4 leading-space tokens
    # prev (x) tokens: val_tokens[0:9] = [0,2,3,2,3,2,3,0,2]
    # is_boundary[x] = [T,F,F,F,F,F,F,T,F]
    # ~is_boundary[x] = [F,T,T,T,T,T,T,F,T]
    # ls & ~pb = [F,T,F,T,F,T,F,F,T] → 4 ones
    # canonical = 20 + 4 = 24
    # buggy = 24 + (leading_space count = 4) = 28
    assert counts.canonical_byte_count == 24
    assert counts.buggy_byte_count == 28
    assert counts.leading_space_token_count == 4


def test_byte_count_inflation_ratio_real_data(luts, val_tokens):
    """On the real fineweb_val data subset, the inflation ratio matches yahya's report."""
    base_bytes, has_leading_space, is_boundary = luts
    subset = val_tokens[:SMOKE_TOKENS]
    counts = cr.compute_byte_counts(
        subset, base_bytes, has_leading_space, is_boundary, seq_len=2048, stride=64
    )
    ratio = counts.buggy_byte_count / counts.canonical_byte_count
    # yahya reports 1.1746 on full val; subsets vary slightly but should land near 1.17.
    assert 1.10 < ratio < 1.25, f"unexpected inflation ratio {ratio:.4f}"


# ---------------------------------------------------------------------------
# Scoring-mode variants (see audit/methodology.md §4)
# ---------------------------------------------------------------------------


def test_scoring_mode_sliding_window_boundary_masked(luts, val_tokens):
    """Default mode — matches PR #1727's eval_val_sliding. Should be ~1.1671."""
    base_bytes, has_leading_space, is_boundary = luts
    counts = cr.compute_byte_counts(
        val_tokens, base_bytes, has_leading_space, is_boundary,
        seq_len=2048, stride=64,
        scoring_mode="sliding-window-boundary-masked",
    )
    ratio = counts.buggy_byte_count / counts.canonical_byte_count
    assert 1.166 <= ratio <= 1.168, f"sliding-window ratio {ratio:.6f} outside [1.166, 1.168]"
    assert counts.num_windows > 0


def test_scoring_mode_all_tokens_boundary_masked(luts, val_tokens):
    """Flat 1:N slice with boundary mask. Identical to sliding-window on SP8192 val
    because the last trimmed window covers all tokens and no boundary tokens
    (control/unknown/unused) appear as predecessors in fineweb val."""
    base_bytes, has_leading_space, is_boundary = luts
    counts = cr.compute_byte_counts(
        val_tokens, base_bytes, has_leading_space, is_boundary,
        seq_len=2048, stride=64,
        scoring_mode="all-tokens-boundary-masked",
    )
    ratio = counts.buggy_byte_count / counts.canonical_byte_count
    assert 1.166 <= ratio <= 1.168, f"all-tokens (masked) ratio {ratio:.6f} outside [1.166, 1.168]"
    assert counts.num_windows == 0  # not window-based


def test_scoring_mode_all_tokens_no_mask(luts, val_tokens):
    """Flat 1:N slice, boundary mask replaced by all-ones. On SP8192 fineweb val
    this is empirically equal to the masked variants because (ls & is_boundary[x])
    is zero on this stream — see methodology.md §4 for the residual-gap
    analysis vs yahya's 1.1746."""
    base_bytes, has_leading_space, is_boundary = luts
    counts = cr.compute_byte_counts(
        val_tokens, base_bytes, has_leading_space, is_boundary,
        seq_len=2048, stride=64,
        scoring_mode="all-tokens-no-mask",
    )
    ratio = counts.buggy_byte_count / counts.canonical_byte_count
    # Empirical: 1.1671 on SP8192 val (same as masked variants). The 1.173-1.176
    # range would be expected if yahya's 1.1746 were a pure no-mask artefact;
    # since it is not, the residual-gap explanation (yahya used a different LUT
    # with byte-token and is_unused handling bugs) is documented in methodology.
    assert 1.166 <= ratio <= 1.168, f"all-tokens no-mask ratio {ratio:.6f} outside [1.166, 1.168]"


def test_scoring_mode_unknown_raises(luts, val_tokens):
    base_bytes, has_leading_space, is_boundary = luts
    import pytest as _pt
    with _pt.raises(ValueError):
        cr.compute_byte_counts(
            val_tokens, base_bytes, has_leading_space, is_boundary,
            seq_len=2048, stride=64, scoring_mode="not-a-real-mode",
        )


# ---------------------------------------------------------------------------
# End-to-end rescore
# ---------------------------------------------------------------------------


def test_rescore_canonical_pr1727():
    if not CANONICAL_TRAIN_SCRIPT.exists() or not TOKENIZER.exists():
        pytest.skip("Canonical PR #1727 train_gpt.py or tokenizer missing")
    result = cr.rescore(
        train_script=CANONICAL_TRAIN_SCRIPT,
        tokenizer=TOKENIZER,
        val_data=VAL_DATA,
        seq_len=2048,
        stride=64,
        reported_bpb=1.07217,
        pr_number=1727,
        max_val_tokens=SMOKE_TOKENS,
    )
    assert result["lut_status"] == "CORRECT"
    # CORRECT scripts get an applied ratio of exactly 1.0
    assert result["inflation_ratio"] == 1.0
    assert result["inferred_canonical_bpb"] == pytest.approx(1.07217)
    # 1.07217 < 1.0738, so it passes the merged-SOTA threshold
    assert result["passes_merged_sota_threshold"] is True


def test_rescore_buggy_fixture():
    if not TOKENIZER.exists():
        pytest.skip("Tokenizer missing")
    result = cr.rescore(
        train_script=BUGGY_FIXTURE,
        tokenizer=TOKENIZER,
        val_data=VAL_DATA,
        seq_len=2048,
        stride=64,
        reported_bpb=1.02840,  # PR #1758's reported BPB
        pr_number=1758,
        max_val_tokens=SMOKE_TOKENS,
    )
    assert result["lut_status"] == "BUGGY"
    ratio = result["inflation_ratio"]
    assert ratio is not None
    assert 1.10 < ratio < 1.25, f"unexpected inflation ratio {ratio:.4f}"
    expected_inferred = 1.02840 * ratio
    assert result["inferred_canonical_bpb"] == pytest.approx(expected_inferred)
