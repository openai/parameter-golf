#!/usr/bin/env python3
"""Build compact competition submission script from verbose source."""
from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import io
import lzma
import tokenize
from pathlib import Path


class _StripDocstrings(ast.NodeTransformer):
    """Remove leading docstrings from modules/classes/functions."""

    @staticmethod
    def _strip(body: list[ast.stmt]) -> list[ast.stmt]:
        if not body:
            return body
        first = body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            return body[1:]
        return body

    def visit_Module(self, node: ast.Module) -> ast.AST:
        self.generic_visit(node)
        node.body = self._strip(node.body)
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        self.generic_visit(node)
        node.body = self._strip(node.body)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        self.generic_visit(node)
        node.body = self._strip(node.body)
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        self.generic_visit(node)
        node.body = self._strip(node.body)
        return node


def _strip_comments(src: str) -> str:
    out: list[tokenize.TokenInfo] = []
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.COMMENT:
            continue
        out.append(tok)
    code = tokenize.untokenize(out)
    lines = code.splitlines()
    compact_lines: list[str] = []
    blank = False
    for ln in lines:
        if ln.strip():
            compact_lines.append(ln.rstrip())
            blank = False
        elif not blank:
            compact_lines.append("")
            blank = True
    if compact_lines and compact_lines[-1] != "":
        compact_lines.append("")
    return "\n".join(compact_lines)


def minify_python(src_text: str) -> str:
    tree = ast.parse(src_text)
    tree = _StripDocstrings().visit(tree)
    ast.fix_missing_locations(tree)
    unparsed = ast.unparse(tree)
    return _strip_comments(unparsed)


def _wrapper(codec: str, compressed: bytes, filename: str) -> str:
    if codec == "raw":
        payload = compressed
        return (
            f"import lzma\n"
            f"p={payload!r}\n"
            f"exec(compile(lzma.decompress(p),{filename!r},'exec'))\n"
        )
    if codec == "b64":
        payload = base64.b64encode(compressed)
        return (
            f"import lzma,base64 as b\n"
            f"p={payload!r}\n"
            f"exec(compile(lzma.decompress(b.b64decode(p)),{filename!r},'exec'))\n"
        )
    if codec == "b85":
        payload = base64.b85encode(compressed)
        return (
            f"import lzma,base64 as b\n"
            f"p={payload!r}\n"
            f"exec(compile(lzma.decompress(b.b85decode(p)),{filename!r},'exec'))\n"
        )
    raise ValueError(f"unsupported codec: {codec}")


def _select_wrapper(compressed: bytes, filename: str, codec: str) -> tuple[str, str]:
    if codec != "auto":
        w = _wrapper(codec, compressed, filename)
        return codec, w
    options = {c: _wrapper(c, compressed, filename) for c in ("raw", "b64", "b85")}
    best_codec = min(options, key=lambda c: len(options[c].encode("utf-8")))
    return best_codec, options[best_codec]


def build_submission(
    source: Path,
    output: Path,
    lzma_preset: int,
    minify: bool,
    codec: str,
) -> dict[str, object]:
    raw_src = source.read_text(encoding="utf-8")
    build_src = minify_python(raw_src) if minify else raw_src
    compressed = lzma.compress(build_src.encode("utf-8"), preset=lzma_preset)
    selected_codec, wrapper = _select_wrapper(compressed, source.name, codec)
    output.write_text(wrapper, encoding="utf-8")
    return {
        "source_sha256": hashlib.sha256(raw_src.encode("utf-8")).hexdigest(),
        "build_sha256": hashlib.sha256(build_src.encode("utf-8")).hexdigest(),
        "source_bytes": len(raw_src.encode("utf-8")),
        "build_bytes": len(build_src.encode("utf-8")),
        "compressed_bytes": len(compressed),
        "output_bytes": len(wrapper.encode("utf-8")),
        "codec": selected_codec,
        "minify": minify,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build compressed train_gpt.py from verbose source")
    p.add_argument("--source", default="train_gpt_verbose.py", help="Source Python file")
    p.add_argument("--output", default="train_gpt.py", help="Generated submission file")
    p.add_argument("--lzma-preset", type=int, default=9, choices=range(10), help="LZMA preset (0-9)")
    p.add_argument("--codec", choices=["auto", "raw", "b64", "b85"], default="auto")
    p.add_argument("--no-minify", action="store_true", help="Disable AST/token minification")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    stats = build_submission(
        source=Path(args.source),
        output=Path(args.output),
        lzma_preset=args.lzma_preset,
        minify=not args.no_minify,
        codec=args.codec,
    )
    print(f"Built {args.output} from {args.source}")
    for k in (
        "source_sha256",
        "build_sha256",
        "source_bytes",
        "build_bytes",
        "compressed_bytes",
        "output_bytes",
        "codec",
        "minify",
    ):
        print(f"{k}={stats[k]}")


if __name__ == "__main__":
    main()
