"""Sprint 002 smoke harness.

Validates that the ablation harness toggles in train_gpt.py resolve to the
expected branches without launching training. No CUDA, no torchrun.

Toggles covered:
  A1  QUANTIZE_WEIGHTS={int8|none}        (artifact path + dequant branch)
  A2  NUM_KV_HEADS                        (Hyperparameters field)
  A3  SDPA_BACKEND={flash|math|cudnn|mem_efficient}
  A4  OPTIMIZER={muon|adamw}
  A5  QUANT_SCHEME={per_row|per_tensor}   (exercises quantize_float_tensor directly)
  A6  TIE_EMBEDDINGS                      (Hyperparameters field)

Also verifies:
  - record_run.py can parse the log shape that train_gpt.py emits
    (against a synthesized log fragment, so we catch regressions before H100 spend).
  - aggregate.py reads runs.jsonl + computes per-row stats + Welch's t correctly
    (against a synthesized 6-row JSONL, so the paper-table generator is validated
    end-to-end before any real seeds land).

What this smoke does NOT do:
  - launch CUDA training (those toggles are exercised end-to-end on the burst H100)
  - validate numerical equivalence between roundtrips

Run from the repo root:
    python smoke_sprint002.py

Exit code 0 on full pass; nonzero on any failure.
"""
from __future__ import annotations

import importlib
import os
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
TRAIN_GPT_SRC = REPO / "train_gpt.py"


def _ok(msg: str) -> None:
    print(f"  PASS  {msg}")


def _fail(msg: str) -> None:
    print(f"  FAIL  {msg}", file=sys.stderr)


def _reload_train_gpt():
    """Reload train_gpt so module-level imports re-evaluate. Hyperparameters
    reads env vars at *class definition* time, so a stale import would lock in
    whatever environment the previous reload saw."""
    sys.path.insert(0, str(REPO))
    if "train_gpt" in sys.modules:
        return importlib.reload(sys.modules["train_gpt"])
    return importlib.import_module("train_gpt")


def check_source_tokens() -> None:
    """Cheap sanity check that the four new toggles still appear in source."""
    print("[1/8] source-level toggle presence ...")
    src = TRAIN_GPT_SRC.read_text(encoding="utf-8")
    expected = [
        ('os.environ.get("QUANTIZE_WEIGHTS"', "A1 QUANTIZE_WEIGHTS"),
        ('os.environ.get("QUANT_SCHEME"', "A5 QUANT_SCHEME"),
        ('os.environ.get("SDPA_BACKEND"', "A3 SDPA_BACKEND"),
        ('os.environ.get("OPTIMIZER"', "A4 OPTIMIZER"),
    ]
    for needle, label in expected:
        if needle not in src:
            _fail(f"missing {label} env-var read in train_gpt.py")
            sys.exit(2)
        _ok(f"{label} env-var read present")


def check_hyperparameters_a2_a6() -> None:
    """A2 (NUM_KV_HEADS) and A6 (TIE_EMBEDDINGS) flip Hyperparameters fields directly."""
    print("[2/8] Hyperparameters env-var resolution (A2, A6) ...")

    # Default.
    for k in ("NUM_KV_HEADS", "TIE_EMBEDDINGS", "NUM_HEADS"):
        os.environ.pop(k, None)
    train_gpt = _reload_train_gpt()
    h = train_gpt.Hyperparameters()
    if h.num_kv_heads != 4 or h.num_heads != 8 or h.tie_embeddings is not True:
        _fail(f"defaults wrong: kv={h.num_kv_heads} heads={h.num_heads} tie={h.tie_embeddings}")
        sys.exit(3)
    _ok(f"defaults: num_heads=8, num_kv_heads=4, tie_embeddings=True")

    # A2: NUM_KV_HEADS=8 (MHA, equal to num_heads).
    os.environ["NUM_KV_HEADS"] = "8"
    train_gpt = _reload_train_gpt()
    h = train_gpt.Hyperparameters()
    if h.num_kv_heads != 8:
        _fail(f"A2 NUM_KV_HEADS=8 not picked up, got {h.num_kv_heads}")
        sys.exit(3)
    _ok("A2: NUM_KV_HEADS=8 -> Hyperparameters.num_kv_heads = 8 (MHA)")
    os.environ.pop("NUM_KV_HEADS", None)

    # A6: TIE_EMBEDDINGS=0.
    os.environ["TIE_EMBEDDINGS"] = "0"
    train_gpt = _reload_train_gpt()
    h = train_gpt.Hyperparameters()
    if h.tie_embeddings is not False:
        _fail(f"A6 TIE_EMBEDDINGS=0 not picked up, got {h.tie_embeddings}")
        sys.exit(3)
    _ok("A6: TIE_EMBEDDINGS=0 -> Hyperparameters.tie_embeddings = False")
    os.environ.pop("TIE_EMBEDDINGS", None)


def check_quant_scheme_a5() -> None:
    """A5: exercise quantize_float_tensor directly under both schemes."""
    print("[3/8] QUANT_SCHEME branch in quantize_float_tensor (A5) ...")
    import torch

    os.environ.pop("QUANT_SCHEME", None)
    train_gpt = _reload_train_gpt()

    t = torch.randn(8, 16)  # 2D matrix

    # per_row default
    q_row, s_row = train_gpt.quantize_float_tensor(t)
    if q_row.dtype != torch.int8 or q_row.shape != t.shape:
        _fail(f"per_row q shape/dtype: {q_row.shape} {q_row.dtype}")
        sys.exit(4)
    if s_row.ndim != 1 or s_row.numel() != t.shape[0]:
        _fail(f"per_row scale should be 1D of len {t.shape[0]}, got ndim={s_row.ndim} numel={s_row.numel()}")
        sys.exit(4)
    _ok(f"per_row default: q={tuple(q_row.shape)}/int8, scale={tuple(s_row.shape)} ({s_row.dtype})")

    # per_tensor branch
    os.environ["QUANT_SCHEME"] = "per_tensor"
    q_tens, s_tens = train_gpt.quantize_float_tensor(t)
    if q_tens.dtype != torch.int8 or q_tens.shape != t.shape:
        _fail(f"per_tensor q shape/dtype: {q_tens.shape} {q_tens.dtype}")
        sys.exit(4)
    if s_tens.ndim != 0:
        _fail(f"per_tensor scale should be 0-dim scalar, got ndim={s_tens.ndim}")
        sys.exit(4)
    _ok(f"per_tensor: q={tuple(q_tens.shape)}/int8, scale=scalar ({s_tens.dtype})")

    # Bad value rejected
    os.environ["QUANT_SCHEME"] = "bogus"
    try:
        train_gpt.quantize_float_tensor(t)
    except ValueError as e:
        if "QUANT_SCHEME" not in str(e):
            _fail(f"unexpected ValueError text: {e}")
            sys.exit(4)
        _ok("bogus QUANT_SCHEME rejected with ValueError")
    else:
        _fail("bogus QUANT_SCHEME silently accepted")
        sys.exit(4)
    os.environ.pop("QUANT_SCHEME", None)


def check_main_inline_validators() -> None:
    """A1 (QUANTIZE_WEIGHTS), A3 (SDPA_BACKEND), A4 (OPTIMIZER) are validated
    inline in main(). We can't run main without CUDA, but we can scan the source
    to confirm the fail-fast `not in {...}` guards exist for each."""
    print("[4/8] inline validator presence for A1, A3, A4 ...")
    src = TRAIN_GPT_SRC.read_text(encoding="utf-8")
    patterns = [
        (r'QUANTIZE_WEIGHTS.*?{[^}]*"int8"[^}]*"none"[^}]*}', "A1 QUANTIZE_WEIGHTS={int8,none}"),
        (r'SDPA_BACKEND.*?{[^}]*"flash"[^}]*"math"[^}]*}', "A3 SDPA_BACKEND={flash,math,...}"),
        (r'OPTIMIZER.*?{[^}]*"muon"[^}]*"adamw"[^}]*}', "A4 OPTIMIZER={muon,adamw}"),
    ]
    for pattern, label in patterns:
        if not re.search(pattern, src, flags=re.DOTALL):
            _fail(f"missing inline guard for {label}")
            sys.exit(5)
        _ok(f"{label} guard present")


def check_artifact_path_branch() -> None:
    """A1 changes the artifact filename. Confirm both candidates appear in source."""
    print("[5/8] artifact path branch (A1) ...")
    src = TRAIN_GPT_SRC.read_text(encoding="utf-8")
    if "final_model.int8.ptz" not in src:
        _fail("int8 artifact path missing")
        sys.exit(6)
    if "final_model.raw.ptz" not in src:
        _fail("raw artifact path missing")
        sys.exit(6)
    _ok("both artifact paths reachable in source")


def check_momentum_warmup_gated() -> None:
    """A4: Muon momentum warmup ramp must be gated on optimizer_choice == 'muon'."""
    print("[6/8] Muon momentum-warmup ramp gated on optimizer_choice (A4) ...")
    src = TRAIN_GPT_SRC.read_text(encoding="utf-8")
    pattern = r'if optimizer_choice == "muon":\s*\n\s*frac = min\(step / args\.muon_momentum_warmup_steps'
    if not re.search(pattern, src):
        _fail("momentum warmup ramp is not gated; AdamW path would dirty param groups")
        sys.exit(7)
    _ok("momentum warmup ramp gated correctly")


def check_record_run_parser() -> None:
    """Round-trip a synthesized train_gpt.py log through record_run.py --dry-run.
    Fails if record_run can't extract bpb/train_s/eval_s/artifact_bytes/world_size."""
    print("[7/8] record_run.py parses synthesized log shape ...")
    import json
    import subprocess
    import tempfile

    record_script = REPO / "record_run.py"
    if not record_script.exists():
        _fail(f"missing {record_script}")
        sys.exit(8)

    # Mimics the lines train_gpt.py actually emits (see lines 924, 1073, 1144, 1171).
    fake_log = (
        "world_size:8 grad_accum_steps:1\n"
        "step:200/20000 train_loss:7.2143 train_time:9852ms step_avg:49.26ms\n"
        "step:20000/20000 train_loss:3.2917 train_time:597432ms step_avg:29.87ms\n"
        "Serialized model int8+zlib: 14823104 bytes (payload:14400000 raw_torch:15000000 payload_ratio:1.05x)\n"
        "Total submission size int8+zlib: 15123104 bytes\n"
        "final_int8_zlib_roundtrip val_loss:3.2917 val_bpb:0.9485 eval_time:412350ms\n"
        "final_int8_zlib_roundtrip_exact val_loss:3.29170000 val_bpb:0.94850000\n"
    )
    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "b0_seed1337.txt"
        log_path.write_text(fake_log, encoding="utf-8")
        result = subprocess.run(
            [sys.executable, str(record_script), str(log_path),
             "--row", "B0", "--dry-run"],
            capture_output=True, text=True, cwd=str(REPO),
        )
    if result.returncode != 0:
        _fail(f"record_run.py exited {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}")
        sys.exit(8)
    try:
        row = json.loads(result.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError) as e:
        _fail(f"could not parse record_run.py stdout as JSON: {e}\nraw:\n{result.stdout}")
        sys.exit(8)

    expected = {"bpb": 0.9485, "val_loss": 3.2917, "eval_s": 412.35,
                "train_s": 597.432, "artifact_bytes": 15123104, "world_size": 8,
                "stopped_early": False, "row": "B0", "seed": 1337,
                "config": {}, "run_id": "b0_seed1337"}
    for k, v in expected.items():
        if row.get(k) != v:
            _fail(f"record_run.py row[{k!r}] = {row.get(k)!r}, expected {v!r}")
            sys.exit(8)
    if "config_hash" not in row or len(row["config_hash"]) != 12:
        _fail(f"config_hash malformed: {row.get('config_hash')!r}")
        sys.exit(8)
    _ok(f"parsed bpb={row['bpb']} train_s={row['train_s']:.1f} eval_s={row['eval_s']:.1f} "
        f"size={row['artifact_bytes']:,} world_size={row['world_size']}")
    _ok(f"row schema complete: {sorted(row.keys())}")


def check_aggregate() -> None:
    """Round-trip a synthesized 6-row JSONL through aggregate.py.

    Synthesizes 3 B0 seeds + 3 A1 seeds with deliberate separation so Welch's t
    is significant, then asserts (a) the JSON dump shape, (b) computed means,
    (c) Welch's t direction + finite p, (d) markdown rendering doesn't crash.
    """
    print("[8/8] aggregate.py reads runs.jsonl + computes per-row stats ...")
    import json
    import subprocess
    import tempfile

    aggregate_script = REPO / "aggregate.py"
    if not aggregate_script.exists():
        _fail(f"missing {aggregate_script}")
        sys.exit(9)

    # Synthesize 3 B0 seeds (mean bpb 0.9485) and 3 A1 seeds (mean bpb 0.9620,
    # i.e. removing INT8 quant hurts compression). Variance kept small so t is
    # deterministically large and p < 0.01 — the test's job is to catch broken
    # math, not exercise edge cases.
    b0_seeds = [(0.9482, 3.2914), (0.9487, 3.2920), (0.9486, 3.2917)]
    a1_seeds = [(0.9618, 3.3415), (0.9622, 3.3420), (0.9620, 3.3418)]

    def fake_row(row, seed, bpb, val_loss, config, train_s, eval_s, bytes_):
        return {
            "row": row, "run_id": f"{row.lower()}_seed{seed}", "seed": seed,
            "config": config, "config_hash": "deadbeefcafe"[:12],
            "bpb": bpb, "val_loss": val_loss,
            "train_s": train_s, "eval_s": eval_s, "artifact_bytes": bytes_,
            "world_size": 8, "stopped_early": False,
            "git_commit": "abc123" * 6, "log_path": f"logs/{row.lower()}_seed{seed}.txt",
            "recorded_at": "2026-04-26T12:00:00+00:00",
        }

    rows: list[dict] = []
    for i, (bpb, vl) in enumerate(b0_seeds, 1):
        rows.append(fake_row("B0", 1337 * i, bpb, vl, {}, 597.4, 412.3, 15_123_104))
    for i, (bpb, vl) in enumerate(a1_seeds, 1):
        rows.append(fake_row("A1", 1337 * i, bpb, vl,
                             {"QUANTIZE_WEIGHTS": "none"}, 595.1, 410.8, 30_240_000))

    with tempfile.TemporaryDirectory() as tmp:
        jsonl_path = Path(tmp) / "runs.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

        # 1. JSON output round-trip.
        result = subprocess.run(
            [sys.executable, str(aggregate_script), "--input", str(jsonl_path), "--json"],
            capture_output=True, text=True, cwd=str(REPO),
        )
        if result.returncode != 0:
            _fail(f"aggregate.py --json exited {result.returncode}\n"
                  f"stderr:\n{result.stderr}")
            sys.exit(9)

        try:
            agg = json.loads(result.stdout)
        except json.JSONDecodeError as e:
            _fail(f"aggregate.py --json output not parseable: {e}\nraw:\n{result.stdout}")
            sys.exit(9)

        if agg.get("baseline") != "B0" or agg.get("metric") != "bpb":
            _fail(f"unexpected header: {agg.get('baseline')} / {agg.get('metric')}")
            sys.exit(9)
        if set(agg["rows"].keys()) != {"B0", "A1"}:
            _fail(f"expected rows={{B0,A1}}, got {sorted(agg['rows'].keys())}")
            sys.exit(9)

        b0 = agg["rows"]["B0"]
        a1 = agg["rows"]["A1"]
        # Means within 5e-4 of expected.
        b0_expected = sum(s[0] for s in b0_seeds) / 3
        a1_expected = sum(s[0] for s in a1_seeds) / 3
        if abs(b0["bpb_mean"] - b0_expected) > 5e-4:
            _fail(f"B0 bpb_mean off: got {b0['bpb_mean']}, expected ~{b0_expected}")
            sys.exit(9)
        if abs(a1["bpb_mean"] - a1_expected) > 5e-4:
            _fail(f"A1 bpb_mean off: got {a1['bpb_mean']}, expected ~{a1_expected}")
            sys.exit(9)
        if b0.get("n") != 3 or a1.get("n") != 3:
            _fail(f"expected n=3 each, got B0={b0.get('n')} A1={a1.get('n')}")
            sys.exit(9)
        if b0.get("welch_t_vs_baseline") is not None:
            _fail("baseline row should have welch_t_vs_baseline=None")
            sys.exit(9)

        wt = a1.get("welch_t_vs_baseline")
        if wt is None or wt.get("p_two_sided") is None:
            _fail(f"A1 welch_t_vs_baseline malformed: {wt}")
            sys.exit(9)
        # A1 has higher BPB than B0 (worse compression without quantization),
        # so delta should be positive and t should be positive too.
        if not (wt["delta"] > 0):
            _fail(f"expected positive Δ (A1 worse than B0), got {wt['delta']}")
            sys.exit(9)
        if not (wt["t"] > 0):
            _fail(f"expected positive t (A1 mean > B0 mean), got {wt['t']}")
            sys.exit(9)
        if not (0.0 < wt["p_two_sided"] < 0.05):
            _fail(f"expected p<0.05 with this synthetic separation, got {wt['p_two_sided']}")
            sys.exit(9)
        _ok(f"B0 mean={b0['bpb_mean']:.4f}, A1 mean={a1['bpb_mean']:.4f}, "
            f"Δ={wt['delta']:+.4f}, t={wt['t']:+.2f}, df={wt['df']:.1f}, "
            f"p={wt['p_two_sided']:.4g}")

        # 2. Markdown render (must not crash, must contain the row labels).
        md = subprocess.run(
            [sys.executable, str(aggregate_script), "--input", str(jsonl_path), "--markdown"],
            capture_output=True, text=True, cwd=str(REPO),
        )
        if md.returncode != 0:
            _fail(f"aggregate.py --markdown exited {md.returncode}\nstderr:\n{md.stderr}")
            sys.exit(9)
        if "| B0 |" not in md.stdout or "| A1 |" not in md.stdout:
            _fail(f"markdown output missing row rows:\n{md.stdout}")
            sys.exit(9)
        _ok("markdown render contains both B0 and A1 rows")

        # 3. Plain-text render (default mode, must not crash).
        txt = subprocess.run(
            [sys.executable, str(aggregate_script), "--input", str(jsonl_path)],
            capture_output=True, text=True, cwd=str(REPO),
        )
        if txt.returncode != 0:
            _fail(f"aggregate.py default exited {txt.returncode}\nstderr:\n{txt.stderr}")
            sys.exit(9)
        if "B0" not in txt.stdout or "A1" not in txt.stdout:
            _fail("plain-text render missing rows")
            sys.exit(9)
        _ok("plain-text render contains both B0 and A1 rows")


def main() -> int:
    print("=" * 60)
    print("Sprint 002 smoke: ablation harness toggle verification")
    print("=" * 60)
    t0 = time.perf_counter()
    check_source_tokens()
    check_hyperparameters_a2_a6()
    check_quant_scheme_a5()
    check_main_inline_validators()
    check_artifact_path_branch()
    check_momentum_warmup_gated()
    check_record_run_parser()
    check_aggregate()
    print("=" * 60)
    print(f"ALL CHECKS PASSED in {time.perf_counter() - t0:.2f}s")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
