#!/usr/bin/env python3
import ast, sys
from pathlib import Path

SRC = Path("train_breadcrumb_recur_ema_stochdepth.py")
DST = Path("train_breadcrumb_recur_ema_stochdepth_stepbound.py")

if not SRC.exists():
    print(f"ERROR: {SRC} not found")
    sys.exit(1)

src = SRC.read_text()

A1_OLD = '    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))'
A1_NEW = A1_OLD + '\n    max_steps = int(os.environ.get("MAX_STEPS", 0))'

A2_OLD = '''        if stop_after_step is None and reached_cap:
            stop_after_step = step'''
A2_NEW = '''        reached_step_cap = args.max_steps > 0 and step >= args.max_steps
        if stop_after_step is None and (reached_cap or reached_step_cap):
            stop_after_step = step'''

if A1_OLD not in src or A2_OLD not in src:
    print("ERROR: anchors not found")
    sys.exit(1)

patched = src.replace(A1_OLD, A1_NEW).replace(A2_OLD, A2_NEW)
ast.parse(patched)
DST.write_text(patched)
print(f"patched: {len(patched.splitlines())} lines")
