#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spectral_flood_walk_v1_sizes import estimate_v1a_sizes


class Defaults:
    embed_dim = 256
    num_layers = 6
    ff_mult = 4
    pos_buckets = 256
    semantic_layers = "2,4"
    use_semantic_memory = True
    pk_num_subkeys = 64
    pk_key_dim = 16
    pk_code_dim = 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate V1a artifact and semantic-memory sizes")
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--embed-dim", type=int, default=Defaults.embed_dim)
    parser.add_argument("--num-layers", type=int, default=Defaults.num_layers)
    parser.add_argument("--ff-mult", type=int, default=Defaults.ff_mult)
    parser.add_argument("--pos-buckets", type=int, default=Defaults.pos_buckets)
    parser.add_argument("--semantic-layers", default=Defaults.semantic_layers)
    parser.add_argument("--use-semantic-memory", action=argparse.BooleanOptionalAction, default=Defaults.use_semantic_memory)
    parser.add_argument("--pk-num-subkeys", type=int, default=Defaults.pk_num_subkeys)
    parser.add_argument("--pk-key-dim", type=int, default=Defaults.pk_key_dim)
    parser.add_argument("--pk-code-dim", type=int, default=Defaults.pk_code_dim)
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sizes = estimate_v1a_sizes(
        vocab_size=args.vocab_size,
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        ff_mult=args.ff_mult,
        pos_buckets=args.pos_buckets,
        semantic_layers=args.semantic_layers,
        use_semantic_memory=args.use_semantic_memory,
        pk_num_subkeys=args.pk_num_subkeys,
        pk_key_dim=args.pk_key_dim,
        pk_code_dim=args.pk_code_dim,
    )
    if args.as_json:
        print(json.dumps(sizes, indent=2))
        return
    for key, value in sizes.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
