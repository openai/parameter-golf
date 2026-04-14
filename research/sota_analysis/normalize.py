#!/usr/bin/env python3
"""AST-normalize every extracted record to a canonical format so that
one-liner-compressed sources diff meaningfully against the baseline."""
import ast, pathlib, sys

SRC = pathlib.Path("/Users/william/Desktop/parameter-golf/sota_analysis/records")
DST = pathlib.Path("/Users/william/Desktop/parameter-golf/sota_analysis/records_normalized")
DST.mkdir(exist_ok=True)

for f in sorted(SRC.glob("*.py")):
    text = f.read_text()
    try:
        tree = ast.parse(text)
        norm = ast.unparse(tree)
    except SyntaxError as e:
        # Rare: a wrapper still present.  Just copy.
        norm = f"# SYNTAX ERROR: {e}\n{text}"
    (DST / f.name).write_text(norm)
    print(f"{f.name:65s} {norm.count(chr(10))+1:5d} lines")
