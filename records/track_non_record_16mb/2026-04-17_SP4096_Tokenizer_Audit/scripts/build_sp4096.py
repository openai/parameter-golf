"""Offline-capable wrapper around the canonical parameter-golf tokenizer exporter."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
from pathlib import Path

WRAPPER_DIR = Path(__file__).resolve().parent


def find_repo_root(start: Path) -> Path:
    current = start.resolve()
    for candidate in (current, *current.parents):
        if (candidate / "data" / "download_hf_docs_and_tokenize.py").is_file():
            return candidate
    raise FileNotFoundError(f"Could not locate repo root from {start}")


REPO_ROOT = find_repo_root(WRAPPER_DIR)
DEFAULT_CANON_DIR = REPO_ROOT / "data"
CANON_DIR = Path(os.environ.get("TOKENIZER_4K_CANON_DIR", str(DEFAULT_CANON_DIR))).expanduser().resolve()
CANON_SCRIPT = CANON_DIR / "download_hf_docs_and_tokenize.py"
DEFAULT_DATA_ROOT = Path(os.environ.get("TOKENIZER_4K_DATA_ROOT", str(REPO_ROOT / "data"))).expanduser().resolve()
DEFAULT_DOCS_PATH = Path(os.environ.get("TOKENIZER_4K_DOCS_PATH", str(DEFAULT_DATA_ROOT / "docs_selected.jsonl"))).expanduser().resolve()
DEFAULT_OUTPUT_ROOT = Path(os.environ.get("MATCHED_FINEWEB_OUTPUT_ROOT", str(DEFAULT_DATA_ROOT))).expanduser().resolve()
EXPECTED_SHA = "84386dfa7b339a5d4831d5273c4a2028b78b60670d3a235633a8520545d19bc7"
DEFAULT_SPEC = WRAPPER_DIR / "tokenizer_specs.sp4096.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_or_link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def usage_probe_path(path: Path) -> Path:
    if path.exists():
        return path
    parent = path.parent
    while not parent.exists() and parent != parent.parent:
        parent = parent.parent
    return parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build sp4096 artifacts via the canonical tokenizer exporter")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Directory where tokenizer + shard artifacts are written")
    parser.add_argument("--docs-path", default=str(DEFAULT_DOCS_PATH), help="Verified local docs_selected.jsonl source")
    parser.add_argument("--sidecar-path", default=None, help="Optional docs_selected.source_manifest.json source")
    parser.add_argument("--tokenizer-config", default=str(DEFAULT_SPEC), help="sp4096 tokenizer config JSON")
    parser.add_argument("--expected-docs-sha", default=EXPECTED_SHA, help="Expected sha256 for docs_selected.jsonl")
    return parser


def resolve_sidecar(docs_path: Path, sidecar_path: str | None) -> Path | None:
    if sidecar_path is not None:
        path = Path(sidecar_path).expanduser().resolve()
        return path if path.exists() else None
    candidate = docs_path.with_name("docs_selected.source_manifest.json")
    return candidate if candidate.exists() else None


def preflight(*, output_root: Path, docs_path: Path, sidecar_path: Path | None, tokenizer_config: Path, expected_sha: str) -> None:
    import sentencepiece as spm

    print("=" * 72, flush=True)
    print("sp4096 build wrapper preflight", flush=True)
    print(f"canon_script  : {CANON_SCRIPT}", flush=True)
    print(f"output_root  : {output_root}", flush=True)
    print(f"docs         : {docs_path} ({docs_path.stat().st_size:,} bytes)", flush=True)
    print(f"sidecar      : {sidecar_path if sidecar_path is not None else '<missing>'}", flush=True)
    print(f"tokenizer_cfg: {tokenizer_config}", flush=True)
    print(f"sp version   : {spm.__version__}", flush=True)
    print(f"cpu_count    : {os.cpu_count()}", flush=True)
    print(f"free_gb      : {shutil.disk_usage(usage_probe_path(output_root)).free / 1024 ** 3:.1f}", flush=True)
    print(
        f"THREADS/BS   : {os.environ.get('MATCHED_FINEWEB_TOKENIZER_THREADS', '<unset>')}"
        f" / {os.environ.get('MATCHED_FINEWEB_SP_BATCH_SIZE', '<unset>')}",
        flush=True,
    )
    print("computing docs_sha256 (~3 min on 48 GB)...", flush=True)
    got = sha256_file(docs_path)
    print(f"docs_sha256  : {got}", flush=True)
    if got != expected_sha:
        sys.exit(f"ABORT: docs_sha256 mismatch; expected {expected_sha}")
    print("docs_sha256 OK", flush=True)
    print("=" * 72, flush=True)


def main() -> None:
    args = build_parser().parse_args()
    output_root = Path(args.output_root).expanduser().resolve()
    docs_path = Path(args.docs_path).expanduser().resolve()
    tokenizer_config = Path(args.tokenizer_config).expanduser().resolve()
    sidecar_path = resolve_sidecar(docs_path, args.sidecar_path)

    if not docs_path.is_file():
        sys.exit(f"ABORT: docs path missing: {docs_path}")
    if not tokenizer_config.is_file():
        sys.exit(f"ABORT: tokenizer config missing: {tokenizer_config}")
    if not CANON_SCRIPT.is_file():
        sys.exit(f"ABORT: canonical script missing: {CANON_SCRIPT}")

    preflight(
        output_root=output_root,
        docs_path=docs_path,
        sidecar_path=sidecar_path,
        tokenizer_config=tokenizer_config,
        expected_sha=args.expected_docs_sha,
    )

    sys.path.insert(0, str(CANON_DIR))
    import download_hf_docs_and_tokenize as canon

    local_sources = {canon.DOCS_FILENAME: docs_path}
    if sidecar_path is not None and sidecar_path.is_file():
        local_sources[canon.SIDECAR_FILENAME] = sidecar_path

    def local_copy(*, repo_id, remote_root, filename, destination):
        del repo_id, remote_root
        src = local_sources.get(filename)
        if src is None or not src.is_file():
            return False
        copy_or_link(src, Path(destination))
        return True

    canon.copy_from_hf_cache = local_copy

    sys.argv = [
        "download_hf_docs_and_tokenize.py",
        "--output-root",
        str(output_root),
        "--tokenizer-config",
        str(tokenizer_config),
        "--skip-byte",
    ]
    print(f"invoking canonical main() argv={sys.argv}", flush=True)
    canon.main()
    print("sp4096 build wrapper DONE", flush=True)


if __name__ == "__main__":
    main()
