#!/usr/bin/env python3
"""Build a compact architecture flow graph from Shroud trace events."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="Path to Shroud JSONL trace")
    p.add_argument("--output", required=True, help="Output JSON path")
    return p.parse_args()


def _as_float(v: object, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _as_int(v: object, default: int = 0) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _node_key(stage: str, loop: int, block: int) -> str:
    return f"{stage}|L{loop}|B{block}"


def build_graph(input_path: Path) -> dict:
    nodes: dict[str, dict] = {}
    edge_bins: dict[tuple[str, str, str], dict] = {}
    event_counts: defaultdict[str, int] = defaultdict(int)

    def upsert_node(stage: str, loop: int, block: int, *, source: str) -> str:
        key = _node_key(stage, loop, block)
        if key not in nodes:
            nodes[key] = {
                "id": key,
                "stage": stage,
                "loop": loop,
                "block": block,
                "source": source,
            }
        return key

    def add_edge(src: str, dst: str, kind: str, magnitude: float) -> None:
        k = (src, dst, kind)
        bin_ = edge_bins.get(k)
        if bin_ is None:
            bin_ = {
                "src": src,
                "dst": dst,
                "kind": kind,
                "count": 0,
                "magnitude_sum": 0.0,
                "magnitude_avg": 0.0,
            }
            edge_bins[k] = bin_
        bin_["count"] += 1
        bin_["magnitude_sum"] += float(magnitude)
        bin_["magnitude_avg"] = bin_["magnitude_sum"] / max(1, bin_["count"])

    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ev = json.loads(line)
            et = str(ev.get("type", ""))
            event_counts[et] += 1

            if et == "activation":
                stage = str(ev.get("stage", "unknown"))
                loop = _as_int(ev.get("loop"), -1)
                block = _as_int(ev.get("block"), -1)
                upsert_node(stage, loop, block, source="activation")
                continue

            if et == "head_event":
                loop = _as_int(ev.get("loop"), 0)
                block = _as_int(ev.get("block"), 0)
                head = _as_int(ev.get("head"), 0)
                kv_head = _as_int(ev.get("kv_head"), 0)
                q = upsert_node(f"head_q:{head}", loop, block, source="head_event")
                kv = upsert_node(f"head_kv:{kv_head}", loop, block, source="head_event")
                crawler = upsert_node("crawler", loop, block, source="head_event")
                add_edge(q, kv, "head_q_to_kv", _as_float(ev.get("qk_align"), 0.0))
                add_edge(kv, crawler, "head_kv_to_crawler", _as_float(ev.get("transfer"), 0.0))
                continue

            if et == "flow_event":
                kind = str(ev.get("kind", "flow"))
                src_stage = str(ev.get("src_stage", "unknown"))
                dst_stage = str(ev.get("dst_stage", "unknown"))
                src_loop = _as_int(ev.get("loop"), -1)
                dst_loop = _as_int(ev.get("dst_loop"), src_loop)
                src_block = _as_int(ev.get("block"), -1)
                dst_block = _as_int(ev.get("dst_block"), src_block)
                src = upsert_node(src_stage, src_loop, src_block, source="flow_event")
                dst = upsert_node(dst_stage, dst_loop, dst_block, source="flow_event")
                add_edge(src, dst, kind, _as_float(ev.get("magnitude"), 0.0))
                continue

    edges = sorted(
        edge_bins.values(),
        key=lambda e: (-(e["magnitude_sum"]), -(e["count"]), e["src"], e["dst"]),
    )
    return {
        "meta": {
            "source": str(input_path),
            "events": dict(sorted(event_counts.items())),
            "nodes": len(nodes),
            "edges": len(edges),
        },
        "nodes": sorted(nodes.values(), key=lambda n: (n["stage"], n["loop"], n["block"])),
        "edges": edges,
    }


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    graph = build_graph(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(graph, indent=2), encoding="utf-8")
    print(f"wrote architecture flow graph to {output_path}")
    print(
        f"nodes={graph['meta']['nodes']} edges={graph['meta']['edges']} "
        f"events={graph['meta']['events']}"
    )


if __name__ == "__main__":
    main()
