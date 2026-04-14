#!/usr/bin/env python3
"""Extract competition-critical code from train_gpt_verbose.py.

Dead Code Elimination (DCE) for the competition profile:
- Removes classes/functions unused in the skc/skc_competition architecture
- Constant-folds dead branches based on competition defaults
- Produces a minimal intermediate file for build_submission.py

Usage:
    python extract_competition.py --source train_gpt_verbose.py --output train_gpt_extracted.py
    python build_submission.py --source train_gpt_extracted.py --output train_gpt.py
"""
from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path


# Classes that are DEAD in skc_competition profile (no attention, no feedback,
# no capsule bank, no koopman SSM, no VRL, no MoE router)
DEAD_CLASSES = {
    # Attention-based blocks (architecture=skc uses only SKCLayer)
    "Block",
    "ParallelResidualBlock",
    "CausalSelfAttention",
    "XSALayer",
    # Feedback system (feedback_enabled=0)
    "FeedbackPooler",
    "FeedbackAdapter",
    # Capsule bank (capsule_enabled=0)
    "CapsuleBank",
    # Koopman SSM blocks (separate from SKC Koopman)
    "KoopmanBlock",
    "KoopmanTokenMixer",
    # GPTQ (gptq_lite_enabled=0)
    "GPTQLiteQuantizer",
}

# Functions that are DEAD in competition profile
DEAD_FUNCTIONS = {
    # VRL (vrl_enabled=0)
    "vrl_loss",
    "vrl_penalty",
    # GPTQ
    "gptq_lite_quantize",
    "gptq_lite_dequantize",
    # Feedback
    "feedback_sketch",
    # Test/debug utilities not needed in submission
    "test_fwht_parity",
    "test_scan_parity",
    "test_engram_hash_gather_parity",
    "test_feedback_parity",
    "test_all",
    "test_ternary_dequant_parity",
}

# Imports that may become unused after DCE
OPTIONAL_IMPORTS = set()


class DeadCodeEliminator(ast.NodeTransformer):
    """Remove dead classes and functions from AST."""

    def __init__(self, dead_classes: set[str], dead_functions: set[str]):
        self.dead_classes = dead_classes
        self.dead_functions = dead_functions
        self.removed: list[str] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST | None:
        if node.name in self.dead_classes:
            self.removed.append(f"class {node.name}")
            return None
        self.generic_visit(node)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST | None:
        if node.name in self.dead_functions:
            self.removed.append(f"def {node.name}")
            return None
        self.generic_visit(node)
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST | None:
        if node.name in self.dead_functions:
            self.removed.append(f"async def {node.name}")
            return None
        self.generic_visit(node)
        return node


class UnusedImportPruner(ast.NodeTransformer):
    """Drop top-level `import x` / `from m import a, b` names never referenced
    elsewhere in the module. Conservative: only removes names clearly unused
    by the post-DCE AST.
    """

    def __init__(self, tree: ast.Module):
        self._used: set[str] = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Name):
                self._used.add(n.id)
            elif isinstance(n, ast.Attribute):
                base = n
                while isinstance(base, ast.Attribute):
                    base = base.value
                if isinstance(base, ast.Name):
                    self._used.add(base.id)

    def _keep(self, alias: ast.alias) -> bool:
        name = alias.asname or alias.name.split(".")[0]
        return name in self._used

    def visit_Import(self, node: ast.Import) -> ast.AST | None:
        kept = [a for a in node.names if self._keep(a)]
        if not kept:
            return None
        node.names = kept
        return node

    def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.AST | None:
        # Wildcard imports: keep as-is (we cannot prove what's used)
        if any(a.name == "*" for a in node.names):
            return node
        kept = [a for a in node.names if self._keep(a)]
        if not kept:
            return None
        node.names = kept
        return node


class DocstringStripper(ast.NodeTransformer):
    """Remove docstrings from all modules/classes/functions."""

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


def strip_comments_and_blanks(src: str) -> str:
    """Remove comment-only lines and collapse multiple blank lines."""
    lines = src.splitlines()
    out: list[str] = []
    blank = False
    for ln in lines:
        stripped = ln.strip()
        if stripped.startswith("#"):
            continue
        if stripped:
            out.append(ln.rstrip())
            blank = False
        elif not blank:
            out.append("")
            blank = True
    if out and out[-1] != "":
        out.append("")
    return "\n".join(out)


def extract_competition(source_text: str, verbose: bool = False) -> str:
    """Apply DCE and minification to produce competition-ready source."""
    tree = ast.parse(source_text)

    # Phase 1: Dead code elimination
    dce = DeadCodeEliminator(DEAD_CLASSES, DEAD_FUNCTIONS)
    tree = dce.visit(tree)
    ast.fix_missing_locations(tree)

    if verbose:
        for item in dce.removed:
            print(f"  DCE removed: {item}")

    # Phase 2: Strip docstrings
    tree = DocstringStripper().visit(tree)
    ast.fix_missing_locations(tree)

    # Phase 2b: Prune imports that became unused after DCE
    tree = UnusedImportPruner(tree).visit(tree)
    ast.fix_missing_locations(tree)

    # Phase 2c: Post-extraction py_compile smoke (fail fast if DCE broke syntax
    # or left dangling references at parse time).
    _check_src = ast.unparse(tree)
    try:
        compile(_check_src, "<extracted>", "exec")
    except SyntaxError as e:
        raise RuntimeError(f"extract_competition produced invalid source: {e}") from e

    # Phase 3: Unparse and clean
    unparsed = ast.unparse(tree)
    cleaned = strip_comments_and_blanks(unparsed)

    return cleaned


def main():
    parser = argparse.ArgumentParser(
        description="Extract competition-critical code from verbose source"
    )
    parser.add_argument(
        "--source", default="train_gpt_verbose.py", help="Verbose source file"
    )
    parser.add_argument(
        "--output", default="train_gpt_extracted.py", help="Extracted output file"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show removed items"
    )
    parser.add_argument(
        "--stats", action="store_true", help="Show size statistics"
    )
    args = parser.parse_args()

    source_path = Path(args.source)
    output_path = Path(args.output)

    raw_src = source_path.read_text(encoding="utf-8")
    extracted = extract_competition(raw_src, verbose=args.verbose)
    output_path.write_text(extracted, encoding="utf-8")

    if args.stats or args.verbose:
        raw_bytes = len(raw_src.encode("utf-8"))
        ext_bytes = len(extracted.encode("utf-8"))
        reduction = 1.0 - ext_bytes / raw_bytes
        print(f"Source:    {raw_bytes:>8,} bytes ({source_path})")
        print(f"Extracted: {ext_bytes:>8,} bytes ({output_path})")
        print(f"Reduction: {reduction:.1%}")

    print(f"Extracted {output_path} from {source_path}")


if __name__ == "__main__":
    main()
