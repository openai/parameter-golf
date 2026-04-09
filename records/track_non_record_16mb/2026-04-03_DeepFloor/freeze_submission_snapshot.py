#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


RECORD_DIR = Path(__file__).resolve().parent
REPO_ROOT = RECORD_DIR.parents[2]
DEFAULT_SOURCE = REPO_ROOT / "spectral_flood_walk_v3.py"
DEFAULT_OUTPUT = RECORD_DIR / "deepfloor_snapshot.py"
HEADER_LINES = [
    "# Frozen DeepFloor snapshot for submission packaging.\n",
    "# Regenerate this file with freeze_submission_snapshot.py when the source changes.\n",
    "\n",
]
V0_IMPORT = "from spectral_flood_walk_v0 import maybe_reset_cuda_peak_memory, maybe_sync_cuda\n"
V2A_IMPORT = "from spectral_flood_walk_v2a import batch_from_starts, build_lm_starts\n"
V0_HELPERS = """def maybe_sync_cuda(device: torch.device) -> None:\n    if device.type == \"cuda\":\n        torch.cuda.synchronize(device)\n\n\ndef maybe_reset_cuda_peak_memory(device: torch.device) -> None:\n    if device.type == \"cuda\":\n        torch.cuda.reset_peak_memory_stats(device)\n\n\n"""
V2A_HELPERS = """def build_lm_starts(num_tokens: int, seq_len: int, stride: int) -> list[int]:\n    stop = num_tokens - seq_len - 1\n    if stop <= 0:\n        return []\n    return list(range(0, stop, stride))\n\n\ndef batch_from_starts(\n    tokens: torch.Tensor,\n    starts: list[int],\n    seq_len: int,\n    device: torch.device,\n) -> tuple[torch.Tensor, torch.Tensor]:\n    windows = [tokens[start : start + seq_len + 1] for start in starts]\n    batch = torch.stack(windows).to(device=device, dtype=torch.long)\n    return batch[:, :-1], batch[:, 1:]\n\n\n"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze the repo-root DeepFloor implementation into the record folder")
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def add_header(text: str) -> str:
    text = text.replace(V0_IMPORT, V0_HELPERS)
    text = text.replace(V2A_IMPORT, V2A_HELPERS)
    lines = text.splitlines(keepends=True)
    if lines and lines[0].startswith("#!"):
        return lines[0] + "".join(HEADER_LINES) + "".join(lines[1:])
    return "".join(HEADER_LINES) + text


def main() -> None:
    args = parse_args()
    source_path = Path(args.source).resolve()
    output_path = Path(args.output).resolve()
    source_text = source_path.read_text(encoding="utf-8")
    output_path.write_text(add_header(source_text), encoding="utf-8")
    print(f"wrote {output_path} from {source_path}")


if __name__ == "__main__":
    main()
