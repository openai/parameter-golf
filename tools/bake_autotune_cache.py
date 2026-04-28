#!/usr/bin/env python3
from __future__ import annotations

import argparse
import tarfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Bundle TorchInductor autotune cache artifacts for reuse.')
    p.add_argument('--cache-dir', default='/tmp/pg_inductor_cache', help='Root TORCHINDUCTOR_CACHE_DIR to scan.')
    p.add_argument('--output', default='autotune_cache_bundle.tar.xz', help='Output tar.xz path.')
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    if not cache_dir.exists():
        raise FileNotFoundError(f'cache directory does not exist: {cache_dir}')

    matches = sorted(cache_dir.rglob('autotune_cache.pkl'))
    if not matches:
        raise RuntimeError(f'no autotune_cache.pkl files found under {cache_dir}')

    out_path = Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out_path, mode='w:xz') as tf:
        for f in matches:
            tf.add(f, arcname=f.relative_to(cache_dir))

    print(f'wrote:{out_path}')
    print(f'files:{len(matches)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
