#!/usr/bin/env python3
"""CPU-only pre-H100 checks for Parameter Golf records.

Checks:
- Python bytecode compile of the target train script.
- Exact sliding-window target coverage on adversarial synthetic lengths.
- Tokenizer metadata NPZ shape/scalar sanity without importing numpy/torch.
- Submission byte accounting for code, optional tokenizer files, and model artifact.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import py_compile
import struct
import sys
import zipfile
from pathlib import Path


DEFAULT_LIMIT_BYTES = 16_000_000
META_KEYS = {
    "format_version",
    "tokenizer_kind",
    "source_model_name",
    "vocab_size",
    "base_bytes",
    "has_leading_space",
    "is_boundary_token",
}


class SmokeError(RuntimeError):
    pass


def resolve(path: str | Path | None, base: Path) -> Path | None:
    if path is None or str(path) == "":
        return None
    p = Path(path)
    return p if p.is_absolute() else base / p


def dedupe(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    out: list[Path] = []
    for path in paths:
        key = path.resolve() if path.exists() else path.absolute()
        if key not in seen:
            seen.add(key)
            out.append(path)
    return out


def compile_script(train_script: Path) -> dict[str, int]:
    if not train_script.is_file():
        raise SmokeError(f"missing train script: {train_script}")
    py_compile.compile(str(train_script), doraise=True)
    source = train_script.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(train_script))
    has_main = any(isinstance(node, ast.FunctionDef) and node.name == "main" for node in tree.body)
    guard_count = source.count('__name__ == "__main__"') + source.count("__name__ == '__main__'")
    return {
        "code_bytes": len(source.encode("utf-8")),
        "torch_compile_refs": source.count("torch.compile"),
        "has_main": int(has_main),
        "main_guard_refs": guard_count,
    }


def dtype_itemsize(descr: str) -> tuple[str, int, str]:
    endian = descr[0] if descr and descr[0] in "<>|=" else "|"
    body = descr[1:] if endian != "|" else descr.lstrip("|")
    if body in {"b1", "?"}:
        return "b", 1, endian
    if not body:
        raise SmokeError(f"unsupported dtype descriptor {descr!r}")
    kind = body[0]
    try:
        size = int(body[1:])
    except ValueError as exc:
        raise SmokeError(f"unsupported dtype descriptor {descr!r}") from exc
    if kind == "U":
        return kind, size * 4, endian
    if kind in {"S", "i", "u", "f", "c"}:
        return kind, size, endian
    raise SmokeError(f"unsupported dtype descriptor {descr!r}")


def parse_npy(raw: bytes, name: str) -> dict[str, object]:
    if not raw.startswith(b"\x93NUMPY"):
        raise SmokeError(f"{name}: not an NPY payload")
    major = raw[6]
    if major == 1:
        header_len = struct.unpack("<H", raw[8:10])[0]
        offset = 10
    elif major in {2, 3}:
        header_len = struct.unpack("<I", raw[8:12])[0]
        offset = 12
    else:
        raise SmokeError(f"{name}: unsupported NPY version {major}")
    header = ast.literal_eval(raw[offset : offset + header_len].decode("latin1"))
    descr = str(header["descr"])
    if "O" in descr:
        raise SmokeError(f"{name}: object arrays are not allowed")
    shape = tuple(int(x) for x in header["shape"])
    elems = math.prod(shape) if shape else 1
    kind, itemsize, endian = dtype_itemsize(descr)
    payload = raw[offset + header_len :]
    expected = elems * itemsize
    if len(payload) < expected:
        raise SmokeError(f"{name}: short NPY payload, expected {expected} bytes")
    return {
        "name": name,
        "descr": descr,
        "shape": shape,
        "elems": elems,
        "kind": kind,
        "itemsize": itemsize,
        "endian": endian,
        "payload": payload[:expected],
    }


def scalar(array: dict[str, object]) -> int | str | bool:
    if array["elems"] != 1:
        raise SmokeError(f"{array['name']}: expected scalar")
    payload = array["payload"]
    kind = str(array["kind"])
    itemsize = int(array["itemsize"])
    endian = str(array["endian"])
    byteorder = "big" if endian == ">" else "little"
    if kind == "U":
        return bytes(payload).decode("utf-32be" if endian == ">" else "utf-32le").rstrip("\x00")
    if kind == "S":
        return bytes(payload).split(b"\x00", 1)[0].decode("utf-8")
    if kind in {"i", "u"}:
        return int.from_bytes(bytes(payload[:itemsize]), byteorder=byteorder, signed=(kind == "i"))
    if kind == "b":
        return bool(bytes(payload[:1])[0])
    raise SmokeError(f"{array['name']}: unsupported scalar dtype {array['descr']}")


def int_array_stats(array: dict[str, object]) -> dict[str, int]:
    kind = str(array["kind"])
    if kind not in {"i", "u", "b"}:
        raise SmokeError(f"{array['name']}: expected int/bool array")
    itemsize = int(array["itemsize"])
    endian = str(array["endian"])
    byteorder = "big" if endian == ">" else "little"
    payload = bytes(array["payload"])
    values = [
        int.from_bytes(payload[i : i + itemsize], byteorder=byteorder, signed=(kind == "i"))
        for i in range(0, len(payload), itemsize)
    ]
    if not values:
        return {"min": 0, "max": 0, "nonzero": 0, "true": 0}
    return {
        "min": min(values),
        "max": max(values),
        "nonzero": sum(1 for v in values if v != 0),
        "true": sum(1 for v in values if v),
    }


def load_tokenizer_meta(meta_path: Path, vocab_size_arg: int | None, tokenizer_path: Path | None) -> dict[str, object]:
    if not meta_path.is_file():
        raise SmokeError(f"missing tokenizer metadata: {meta_path}")
    with zipfile.ZipFile(meta_path) as zf:
        arrays = {
            Path(name).stem: parse_npy(zf.read(name), name)
            for name in zf.namelist()
            if name.endswith(".npy")
        }
    missing = sorted(META_KEYS - set(arrays))
    if missing:
        raise SmokeError(f"{meta_path}: missing metadata arrays: {', '.join(missing)}")

    format_version = int(scalar(arrays["format_version"]))
    meta_vocab_size = int(scalar(arrays["vocab_size"]))
    tokenizer_kind = str(scalar(arrays["tokenizer_kind"]))
    source_model_name = str(scalar(arrays["source_model_name"]))
    expected_vocab = vocab_size_arg or meta_vocab_size
    if format_version < 1:
        raise SmokeError(f"{meta_path}: unsupported format_version={format_version}")
    if meta_vocab_size <= 0:
        raise SmokeError(f"{meta_path}: invalid vocab_size={meta_vocab_size}")

    for key in ("base_bytes", "has_leading_space", "is_boundary_token"):
        elems = int(arrays[key]["elems"])
        if elems < expected_vocab:
            raise SmokeError(f"{meta_path}: {key} has {elems} entries, expected >= {expected_vocab}")

    if tokenizer_path is not None and tokenizer_path.exists():
        if Path(source_model_name).name != tokenizer_path.name:
            raise SmokeError(
                f"{meta_path}: source_model_name={source_model_name!r} "
                f"does not match tokenizer file {tokenizer_path.name!r}"
            )

    base_stats = int_array_stats(arrays["base_bytes"])
    leading_stats = int_array_stats(arrays["has_leading_space"])
    boundary_stats = int_array_stats(arrays["is_boundary_token"])
    if base_stats["nonzero"] <= 0:
        raise SmokeError(f"{meta_path}: base_bytes has no nonzero byte lengths")

    return {
        "format_version": format_version,
        "tokenizer_kind": tokenizer_kind,
        "source_model_name": source_model_name,
        "vocab_size": meta_vocab_size,
        "expected_vocab": expected_vocab,
        "base_entries": int(arrays["base_bytes"]["elems"]),
        "base_nonzero": base_stats["nonzero"],
        "base_max": base_stats["max"],
        "leading_true": leading_stats["true"],
        "boundary_true": boundary_stats["true"],
        "bytes": meta_path.stat().st_size,
    }


def exact_ranges(total: int, seq_len: int, stride: int) -> list[tuple[int, int, int, int]]:
    if total <= 0 or seq_len <= 0 or stride <= 0:
        raise SmokeError("total, seq_len, and stride must be positive")
    if stride > seq_len:
        raise SmokeError(f"stride={stride} cannot exceed seq_len={seq_len}")
    last_start = max(total - seq_len, 0)
    starts = list(range(0, last_start + 1, stride))
    if not starts or starts[-1] != last_start:
        starts.append(last_start)
    starts = sorted(set(starts))
    covered = 0
    ranges: list[tuple[int, int, int, int]] = []
    for ws in starts:
        end = min(ws + seq_len, total)
        score_start = max(covered, ws)
        if score_start < end:
            ranges.append((ws, end, score_start, end))
            covered = end
        if covered == total:
            break
    return ranges


def legacy_ranges(total: int, seq_len: int, stride: int) -> list[tuple[int, int, int, int]]:
    if total <= 0 or seq_len <= 0 or stride <= 0:
        raise SmokeError("total, seq_len, and stride must be positive")
    starts = [ws for ws in range(0, total, stride) if min(ws + seq_len, total) - ws >= 1]
    ranges: list[tuple[int, int, int, int]] = []
    for ws in starts:
        end = min(ws + seq_len, total)
        wlen = end - ws
        local_start = 0 if ws == 0 else max(wlen - stride, 0)
        ranges.append((ws, end, ws + local_start, end))
    return ranges


def validate_ranges(total: int, ranges: list[tuple[int, int, int, int]]) -> None:
    counts = [0] * total
    context = [-1] * total
    for ws, end, score_start, score_end in ranges:
        if not (0 <= ws <= score_start <= score_end <= end <= total):
            raise SmokeError(
                f"bad range total={total}: window=({ws},{end}) score=({score_start},{score_end})"
            )
        for token_idx in range(score_start, score_end):
            counts[token_idx] += 1
            context[token_idx] = token_idx - ws + 1
    for token_idx, count in enumerate(counts):
        if count != 1:
            raise SmokeError(f"coverage token={token_idx} count={count} total={total} ranges={ranges[:6]}")
    starts = [(ws, end) for ws, end, _, _ in ranges]
    for token_idx, ctx in enumerate(context):
        best = max((token_idx - ws + 1 for ws, end in starts if ws <= token_idx < end), default=-1)
        if ctx != best:
            raise SmokeError(f"context token={token_idx} got={ctx} best={best} total={total}")


def smoke_sliding(policy: str) -> dict[str, int | str]:
    cases: set[tuple[int, int, int]] = {
        (1, 16, 4),
        (10, 16, 4),
        (17, 16, 7),
        (31, 16, 7),
        (100, 16, 7),
        (102, 16, 7),
        (257, 64, 16),
        (8193, 128, 64),
        (16385, 256, 64),
    }
    for total in range(1, 96):
        for seq_len in (1, 2, 3, 4, 5, 8, 16, 31):
            for stride in (1, 2, 3, 4, 7, 8, 16, 31):
                if stride <= seq_len:
                    cases.add((total, seq_len, stride))
    maker = exact_ranges if policy == "exact" else legacy_ranges
    max_total = 0
    max_windows = 0
    for total, seq_len, stride in sorted(cases):
        ranges = maker(total, seq_len, stride)
        try:
            validate_ranges(total, ranges)
        except SmokeError as exc:
            raise SmokeError(f"case total={total} seq_len={seq_len} stride={stride}: {exc}") from exc
        max_total = max(max_total, total)
        max_windows = max(max_windows, len(ranges))
    return {"policy": policy, "cases": len(cases), "max_total": max_total, "max_windows": max_windows}


def artifact_accounting(
    train_script: Path,
    code_bytes: int,
    artifacts: list[Path],
    includes: list[Path],
    limit_bytes: int,
    submission_json: Path | None,
    require_artifact: bool,
) -> dict[str, int | str]:
    missing = [str(p) for p in artifacts + includes if not p.exists()]
    if missing:
        raise SmokeError(f"missing counted files: {', '.join(missing)}")
    artifact_bytes = sum(p.stat().st_size for p in artifacts)
    include_bytes = sum(p.stat().st_size for p in includes)
    total = code_bytes + artifact_bytes + include_bytes
    if require_artifact and artifact_bytes <= 0:
        raise SmokeError("no model artifact counted; pass --artifact final_model.*.ptz")
    if total > limit_bytes:
        raise SmokeError(f"artifact accounting exceeds limit: total={total} limit={limit_bytes}")

    declared = ""
    if submission_json is not None and submission_json.is_file():
        data = json.loads(submission_json.read_text(encoding="utf-8"))
        if "bytes_total" in data:
            declared_bytes = int(data["bytes_total"])
            if declared_bytes > limit_bytes:
                raise SmokeError(f"{submission_json}: bytes_total={declared_bytes} exceeds {limit_bytes}")
            if declared_bytes < code_bytes + include_bytes:
                raise SmokeError(
                    f"{submission_json}: bytes_total={declared_bytes} is below known code+include bytes "
                    f"{code_bytes + include_bytes}"
                )
            declared = str(declared_bytes)

    return {
        "code_bytes": code_bytes,
        "artifact_bytes": artifact_bytes,
        "include_bytes": include_bytes,
        "total_counted": total,
        "limit_bytes": limit_bytes,
        "remaining": limit_bytes - total,
        "declared_bytes_total": declared,
        "mode": "strict" if require_artifact else "budget",
        "train_script": str(train_script),
    }


def print_result(name: str, ok: bool, payload: dict[str, object] | str) -> None:
    status = "OK" if ok else "FAIL"
    if isinstance(payload, str):
        print(f"{status} {name}: {payload}")
    else:
        details = " ".join(f"{k}={v}" for k, v in payload.items())
        print(f"{status} {name}: {details}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run CPU-only pre-H100 Parameter Golf smoke checks.")
    p.add_argument("--record-dir", default=".", help="Record/candidate directory. Defaults to cwd.")
    p.add_argument("--train-script", default="train_gpt.py", help="Train script path relative to record dir.")
    p.add_argument("--tokenizer-meta", default="", help="Tokenizer .meta.npz path. Defaults to candidate.meta.npz if present.")
    p.add_argument("--tokenizer-path", default="", help="Tokenizer file path for metadata source-name check.")
    p.add_argument("--vocab-size", type=int, default=0, help="Expected vocab size. Defaults to metadata vocab_size.")
    p.add_argument("--allow-missing-meta", action="store_true", help="Do not fail if tokenizer metadata is absent.")
    p.add_argument("--coverage-policy", choices=("exact", "legacy"), default="exact")
    p.add_argument("--artifact", action="append", default=[], help="Compressed model artifact to count. Repeatable.")
    p.add_argument("--include", action="append", default=[], help="Extra counted dependency file. Repeatable.")
    p.add_argument("--no-auto-include-tokenizer", action="store_true", help="Do not auto-count tokenizer/meta files.")
    p.add_argument("--limit-bytes", type=int, default=DEFAULT_LIMIT_BYTES)
    p.add_argument("--submission-json", default="", help="submission.json path. Defaults to record-dir/submission.json if present.")
    p.add_argument("--require-artifact", action="store_true", help="Fail unless at least one model artifact is counted.")
    return p


def main() -> int:
    args = build_parser().parse_args()
    record_dir = Path(args.record_dir).resolve()
    train_script = resolve(args.train_script, record_dir)
    assert train_script is not None

    meta_path = resolve(args.tokenizer_meta, record_dir)
    if meta_path is None and (record_dir / "candidate.meta.npz").is_file():
        meta_path = record_dir / "candidate.meta.npz"
    tokenizer_path = resolve(args.tokenizer_path, record_dir)
    if tokenizer_path is None:
        for name in ("candidate.vocab", "candidate.model"):
            if (record_dir / name).is_file():
                tokenizer_path = record_dir / name
                break
    submission_json = resolve(args.submission_json, record_dir)
    if submission_json is None and (record_dir / "submission.json").is_file():
        submission_json = record_dir / "submission.json"

    artifact_paths = [resolve(p, record_dir) for p in args.artifact]
    artifact_paths = [p for p in artifact_paths if p is not None]
    if not artifact_paths:
        for name in ("final_model.int6.ptz", "final_model.int8.ptz", "final_model.pt"):
            candidate = record_dir / name
            if candidate.is_file():
                artifact_paths.append(candidate)
    include_paths = [resolve(p, record_dir) for p in args.include]
    include_paths = [p for p in include_paths if p is not None]
    if not args.no_auto_include_tokenizer:
        include_paths.extend([p for p in (meta_path, tokenizer_path) if p is not None and p.exists()])
    include_paths = [p for p in dedupe(include_paths) if p.resolve() != train_script.resolve()]
    artifact_paths = dedupe(artifact_paths)

    failures = 0
    compile_info: dict[str, int] = {"code_bytes": 0}
    try:
        compile_info = compile_script(train_script)
        print_result("compile", True, compile_info)
    except Exception as exc:
        failures += 1
        print_result("compile", False, str(exc))

    try:
        coverage_info = smoke_sliding(args.coverage_policy)
        print_result("sliding_coverage", True, coverage_info)
    except Exception as exc:
        failures += 1
        print_result("sliding_coverage", False, str(exc))

    try:
        if meta_path is None:
            if args.allow_missing_meta:
                print_result("tokenizer_meta", True, "missing allowed")
            else:
                raise SmokeError("no tokenizer metadata found; pass --tokenizer-meta")
        else:
            meta_info = load_tokenizer_meta(
                meta_path,
                args.vocab_size or None,
                tokenizer_path,
            )
            print_result("tokenizer_meta", True, meta_info)
    except Exception as exc:
        failures += 1
        print_result("tokenizer_meta", False, str(exc))

    try:
        accounting = artifact_accounting(
            train_script,
            int(compile_info.get("code_bytes", 0)),
            artifact_paths,
            include_paths,
            args.limit_bytes,
            submission_json,
            args.require_artifact,
        )
        print_result("artifact_accounting", True, accounting)
    except Exception as exc:
        failures += 1
        print_result("artifact_accounting", False, str(exc))

    if failures:
        print(f"FAIL pre_h100_smoke failures={failures}")
        return 1
    print("PASS pre_h100_smoke")
    return 0


if __name__ == "__main__":
    sys.exit(main())
