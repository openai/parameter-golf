#!/usr/bin/env python3
"""Convert Shroud JSONL traces into a compact point-cloud JSON for the viewer."""

from __future__ import annotations

import argparse
import colorsys
import json
import math
from pathlib import Path


STAGE_ANGLE = {
    "embed": 0.15,
    "encoder": 0.85,
    "inst": 1.55,
    "crawler": 2.25,
    "delta": 3.05,
    "decoder": 3.95,
    "head_in": 4.75,
}

# Complementary-style palette anchors (teal/orange + violet/lime accents)
STAGE_HUE = {
    "embed": 0.56,
    "encoder": 0.59,
    "inst": 0.08,
    "crawler": 0.11,
    "head_q": 0.57,
    "head_kv": 0.08,
    "delta": 0.43,
    "decoder": 0.86,
    "head_in": 0.91,
    "compression": 0.31,
    "metric": 0.67,
}


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _hsl_rgb(h: float, s: float, l: float) -> tuple[float, float, float]:
    r, g, b = colorsys.hls_to_rgb(h % 1.0, _clamp(l, 0.0, 1.0), _clamp(s, 0.0, 1.0))
    return float(r), float(g), float(b)


def _inject_flow_fields(point: dict, *, stage: str, loop: int, block: int, head: int = -1, kv_head: int = -1,
                       qk_align: float = 0.0, rms: float = 0.0, std: float = 0.0, amax: float = 0.0,
                       q_rms: float = 0.0, k_rms: float = 0.0, v_rms: float = 0.0,
                       step: int = 0, micro_step: int = 0, kind: str = "",
                       transfer: float = 0.0, attn_entropy: float = 0.0, attn_lag: float = 0.0,
                       recent_mass: float = 0.0, attn_peak: float = 0.0, out_rms: float = 0.0,
                       token_count: int = 0) -> None:
    point["stage"] = stage
    point["loop"] = loop
    point["block"] = block
    point["head"] = head
    point["kv_head"] = kv_head
    point["qk_align"] = qk_align
    point["rms"] = rms
    point["std"] = std
    point["amax"] = amax
    point["q_rms"] = q_rms
    point["k_rms"] = k_rms
    point["v_rms"] = v_rms
    point["step"] = step
    point["micro_step"] = micro_step
    point["kind"] = kind
    point["energy"] = _clamp(abs(rms) + abs(std) + abs(amax), 0.0, 6.0)
    point["transfer"] = transfer
    point["attn_entropy"] = attn_entropy
    point["attn_lag"] = attn_lag
    point["recent_mass"] = recent_mass
    point["attn_peak"] = attn_peak
    point["out_rms"] = out_rms
    point["token_count"] = token_count


def _activation_to_point(ev: dict, step_norm: float) -> dict:
    stage = str(ev.get("stage", "encoder"))
    step = _safe_int(ev.get("step"), 0)
    micro = _safe_int(ev.get("micro_step"), 0)
    loop = _safe_int(ev.get("loop"), -1)
    block = _safe_int(ev.get("block"), -1)
    rms = _safe_float(ev.get("rms"), 0.0)
    std = _safe_float(ev.get("std"), 0.0)
    amax = _safe_float(ev.get("amax"), 0.0)
    theta = STAGE_ANGLE.get(stage, 0.0) + 0.17 * max(loop, 0) + 0.05 * micro
    radius = 1.4 + 0.55 * max(block + 1, 1) + 0.30 * math.log1p(abs(std) * 20.0)
    x = radius * math.cos(theta)
    z = radius * math.sin(theta) + 0.6 * math.tanh(rms * 2.0)
    y = (step / step_norm) + 0.06 * micro
    hue_base = STAGE_HUE.get(stage, 0.58)
    hue = (hue_base + 0.07 * math.tanh(std * 3.0) - 0.04 * math.tanh(rms * 2.0)) % 1.0
    sat = _clamp(0.65 + 0.25 * math.tanh(amax * 0.8), 0.25, 0.95)
    lum = _clamp(0.44 + 0.20 * math.tanh(rms * 3.0), 0.22, 0.72)
    r, g, b = _hsl_rgb(hue, sat, lum)
    point = {
        "x": x,
        "y": y,
        "z": z,
        "r": r,
        "g": g,
        "b": b,
        "label": f"{stage} s={step} m={micro} l={loop} b={block} rms={rms:.4f} std={std:.4f}",
    }
    _inject_flow_fields(
        point,
        stage=stage,
        loop=loop,
        block=block,
        head=-1,
        kv_head=-1,
        qk_align=0.0,
        rms=rms,
        std=std,
        amax=amax,
        step=step,
        micro_step=micro,
        kind="activation",
    )
    return point


def _event_to_point(ev: dict, idx: int, step_norm: float) -> dict | None:
    et = ev.get("type")
    if et == "activation":
        return _activation_to_point(ev, step_norm)
    if et == "compression":
        blob = _safe_float(ev.get("quant_blob_bytes"), 0.0)
        raw = _safe_float(ev.get("quant_raw_bytes"), max(blob, 1.0))
        ratio = _safe_float(ev.get("compression_ratio"), raw / max(blob, 1.0))
        theta = 5.25 + 0.03 * idx
        radius = 3.5 + 0.28 * math.log1p(max(blob, 1.0))
        x = radius * math.cos(theta)
        z = radius * math.sin(theta) + 0.35 * ratio
        y = 0.2 * idx
        r, g, b = _hsl_rgb(STAGE_HUE["compression"], 0.80, 0.50)
        point = {
            "x": x,
            "y": y,
            "z": z,
            "r": r,
            "g": g,
            "b": b,
            "label": f"compression ratio={ratio:.3f} raw={int(raw)} blob={int(blob)}",
            "compression_ratio": ratio,
            "quant_raw_bytes": int(raw),
            "quant_blob_bytes": int(blob),
        }
        _inject_flow_fields(
            point,
            stage="compression",
            loop=0,
            block=0,
            qk_align=0.0,
            step=0,
            micro_step=0,
            kind="compression",
        )
        return point
    if et == "metric":
        bpb = float(ev.get("val_bpb", 0.0))
        theta = 4.7 + 0.04 * idx
        radius = 2.4 + 3.0 * bpb
        x = radius * math.cos(theta)
        z = radius * math.sin(theta)
        y = 0.5 + 0.1 * idx
        r, g, b = _hsl_rgb(STAGE_HUE["metric"], 0.74, 0.52)
        point = {
            "x": x,
            "y": y,
            "z": z,
            "r": r,
            "g": g,
            "b": b,
            "label": f"metric {ev.get('stage','')} bpb={bpb:.6f}",
            "bpb": bpb,
        }
        _inject_flow_fields(
            point,
            stage="metric",
            loop=0,
            block=0,
            qk_align=0.0,
            step=0,
            micro_step=0,
            kind="metric",
        )
        return point
    return None


def _head_event_to_flow(ev: dict, step_norm: float) -> tuple[dict, dict, dict]:
    step = _safe_int(ev.get("step"), 0)
    micro = _safe_int(ev.get("micro_step"), 0)
    loop = _safe_int(ev.get("loop"), 0)
    block = _safe_int(ev.get("block"), 0)
    head = _safe_int(ev.get("head"), 0)
    kv_head = _safe_int(ev.get("kv_head"), 0)
    n_heads = max(1, _safe_int(ev.get("num_heads"), 1))
    n_kv = max(1, _safe_int(ev.get("num_kv_heads"), 1))
    q_rms = _safe_float(ev.get("q_rms"), 0.0)
    k_rms = _safe_float(ev.get("k_rms"), 0.0)
    v_rms = _safe_float(ev.get("v_rms"), 0.0)
    align = _clamp(_safe_float(ev.get("qk_align"), 0.0), 0.0, 1.0)
    transfer = _safe_float(ev.get("transfer"), 0.0)
    attn_entropy = _safe_float(ev.get("attn_entropy"), 0.0)
    attn_lag = _safe_float(ev.get("attn_lag"), 0.0)
    recent_mass = _clamp(_safe_float(ev.get("recent_mass"), 0.0), 0.0, 1.0)
    attn_peak = _clamp(_safe_float(ev.get("attn_peak"), 0.0), 0.0, 1.0)
    out_rms = _safe_float(ev.get("out_rms"), 0.0)
    token_count = _safe_int(ev.get("token_count"), 0)
    y = (step / step_norm) + 0.05 * micro
    base = STAGE_ANGLE.get("crawler", 2.25) + 0.22 * loop + 0.08 * block
    theta_q = base + (2.0 * math.pi * (head / n_heads))
    theta_k = base + 0.45 + (2.0 * math.pi * (kv_head / n_kv))
    r_q = 4.2 + 0.30 * block + 0.50 * math.tanh(q_rms * 2.0)
    r_k = 3.3 + 0.24 * block + 0.40 * math.tanh(k_rms * 2.0)
    qx = r_q * math.cos(theta_q)
    qz = r_q * math.sin(theta_q) + 0.22 * math.tanh(q_rms * 2.0)
    kx = r_k * math.cos(theta_k)
    kz = r_k * math.sin(theta_k) - 0.16 * math.tanh(v_rms * 2.0)
    q_color = _hsl_rgb(0.57 + 0.05 * align, 0.78, 0.47 + 0.15 * align)
    k_color = _hsl_rgb(0.08 - 0.03 * align, 0.80, 0.45 + 0.12 * align)
    edge_color = _hsl_rgb(0.50 - 0.42 * align, 0.82, 0.52)
    q_point = {
        "x": qx,
        "y": y,
        "z": qz,
        "r": q_color[0],
        "g": q_color[1],
        "b": q_color[2],
        "label": (
            f"Q loop={loop} block={block} head={head} "
            f"q_rms={q_rms:.4f} qk_align={align:.4f}"
        ),
    }
    _inject_flow_fields(
        q_point,
        stage="crawler",
        loop=loop,
        block=block,
        head=head,
        kv_head=kv_head,
        qk_align=align,
        q_rms=q_rms,
        step=step,
        micro_step=micro,
        kind="head_q",
        transfer=transfer,
        attn_entropy=attn_entropy,
        attn_lag=attn_lag,
        recent_mass=recent_mass,
        attn_peak=attn_peak,
        out_rms=out_rms,
        token_count=token_count,
    )
    k_point = {
        "x": kx,
        "y": y,
        "z": kz,
        "r": k_color[0],
        "g": k_color[1],
        "b": k_color[2],
        "label": (
            f"KV loop={loop} block={block} kv_head={kv_head} "
            f"k_rms={k_rms:.4f} v_rms={v_rms:.4f}"
        ),
    }
    _inject_flow_fields(
        k_point,
        stage="crawler",
        loop=loop,
        block=block,
        head=head,
        kv_head=kv_head,
        qk_align=align,
        k_rms=k_rms,
        v_rms=v_rms,
        step=step,
        micro_step=micro,
        kind="head_kv",
        transfer=transfer,
        attn_entropy=attn_entropy,
        attn_lag=attn_lag,
        recent_mass=recent_mass,
        attn_peak=attn_peak,
        out_rms=out_rms,
        token_count=token_count,
    )
    entropy_norm = _clamp(attn_entropy / max(1e-6, math.log(max(2, token_count))), 0.0, 1.0)
    lag_norm = math.tanh(attn_lag / max(1.0, float(token_count)))
    flow = _clamp(
        0.28
        + 0.42 * align
        + 0.20 * math.tanh(transfer * 1.8)
        + 0.12 * recent_mass
        + 0.08 * (1.0 - entropy_norm)
        + 0.06 * lag_norm,
        0.05,
        1.5,
    )
    phase = (0.017 * step + 0.11 * head + 0.07 * loop + 0.03 * block) % 1.0
    edge = {
        "r": edge_color[0],
        "g": edge_color[1],
        "b": edge_color[2],
        "alpha": 0.20 + 0.55 * align,
        "flow": flow,
        "phase": phase,
        "loop": loop,
        "block": block,
        "head": head,
        "kv_head": kv_head,
        "step": step,
        "qk_align": align,
        "transfer": transfer,
        "attn_entropy": attn_entropy,
        "attn_lag": attn_lag,
        "recent_mass": recent_mass,
        "attn_peak": attn_peak,
        "out_rms": out_rms,
        "token_count": token_count,
    }
    return q_point, k_point, edge


def _stage_to_anchor(
    stage: str,
    *,
    step: int,
    micro: int,
    loop: int,
    block: int,
    head: int,
    kv_head: int,
    idx: int,
    step_norm: float,
) -> tuple[float, float, float, tuple[float, float, float]]:
    stage_l = str(stage or "crawler").lower()
    y = (step / max(step_norm, 1.0)) + 0.05 * micro + 0.02 * max(loop, 0)
    if stage_l == "head_q":
        base = STAGE_ANGLE.get("crawler", 2.25) + 0.22 * max(loop, 0) + 0.08 * max(block, 0)
        theta = base + (2.0 * math.pi * (max(head, 0) % 16) / 16.0)
        radius = 4.2 + 0.30 * max(block, 0)
        x = radius * math.cos(theta)
        z = radius * math.sin(theta)
    elif stage_l == "head_kv":
        base = STAGE_ANGLE.get("crawler", 2.25) + 0.22 * max(loop, 0) + 0.08 * max(block, 0)
        theta = base + 0.45 + (2.0 * math.pi * (max(kv_head, 0) % 8) / 8.0)
        radius = 3.35 + 0.24 * max(block, 0)
        x = radius * math.cos(theta)
        z = radius * math.sin(theta)
    elif stage_l == "compression":
        theta = 5.2 + 0.02 * (idx % 31)
        radius = 4.9
        x = radius * math.cos(theta)
        z = radius * math.sin(theta) + 0.4
        y = 1.2 + 0.05 * (idx % 9)
    elif stage_l == "metric":
        theta = 4.6 + 0.02 * (idx % 31)
        radius = 5.5
        x = radius * math.cos(theta)
        z = radius * math.sin(theta)
        y = 1.6 + 0.05 * (idx % 7)
    else:
        theta = STAGE_ANGLE.get(stage_l, STAGE_ANGLE.get("crawler", 2.25))
        theta += 0.16 * max(loop, 0) + 0.06 * max(block, 0) + 0.015 * (idx % 17)
        radius = 2.0 + 0.5 * max(block + 1, 1)
        x = radius * math.cos(theta)
        z = radius * math.sin(theta)
    hue = STAGE_HUE.get(stage_l, 0.58)
    sat = 0.78 if stage_l in ("head_q", "head_kv", "crawler", "compression") else 0.72
    lum = 0.50 if stage_l in ("compression", "metric") else 0.46
    color = _hsl_rgb(hue, sat, lum)
    return x, y, z, color


def _flow_event_to_pair(ev: dict, idx: int, step_norm: float) -> tuple[dict, dict, dict]:
    step = _safe_int(ev.get("step"), 0)
    micro = _safe_int(ev.get("micro_step"), 0)
    loop = _safe_int(ev.get("loop"), 0)
    block = _safe_int(ev.get("block"), 0)
    src_loop = _safe_int(ev.get("src_loop"), loop)
    src_block = _safe_int(ev.get("src_block"), block)
    dst_loop = _safe_int(ev.get("dst_loop"), loop)
    dst_block = _safe_int(ev.get("dst_block"), block)
    head = _safe_int(ev.get("head"), -1)
    kv_head = _safe_int(ev.get("kv_head"), -1)
    src_stage = str(ev.get("src_stage", "crawler")).lower()
    dst_stage = str(ev.get("dst_stage", "crawler")).lower()
    kind = str(ev.get("kind", "flow"))
    magnitude = _safe_float(
        ev.get("magnitude"),
        _safe_float(ev.get("flow"), _safe_float(ev.get("transfer"), _safe_float(ev.get("qk_align"), 0.12))),
    )
    flow = _clamp(max(0.02, magnitude), 0.02, 1.9)
    src_xyz = _stage_to_anchor(
        src_stage,
        step=step,
        micro=micro,
        loop=src_loop,
        block=src_block,
        head=head,
        kv_head=kv_head,
        idx=idx,
        step_norm=step_norm,
    )
    dst_xyz = _stage_to_anchor(
        dst_stage,
        step=step,
        micro=micro,
        loop=dst_loop,
        block=dst_block,
        head=head,
        kv_head=kv_head,
        idx=idx + 1,
        step_norm=step_norm,
    )
    sx, sy, sz, src_color = src_xyz
    dx, dy, dz, dst_color = dst_xyz
    src_point = {
        "x": sx,
        "y": sy,
        "z": sz,
        "r": src_color[0],
        "g": src_color[1],
        "b": src_color[2],
        "label": f"{src_stage} flow_src step={step} loop={src_loop} block={src_block} kind={kind}",
    }
    _inject_flow_fields(
        src_point,
        stage=src_stage,
        loop=src_loop,
        block=src_block,
        head=head,
        kv_head=kv_head,
        qk_align=_safe_float(ev.get("qk_align"), 0.0),
        step=step,
        micro_step=micro,
        kind=f"flow_src:{kind}",
        transfer=_safe_float(ev.get("transfer"), 0.0),
        attn_entropy=_safe_float(ev.get("attn_entropy"), 0.0),
        attn_lag=_safe_float(ev.get("attn_lag"), 0.0),
        recent_mass=_safe_float(ev.get("recent_mass"), 0.0),
        attn_peak=_safe_float(ev.get("attn_peak"), 0.0),
        out_rms=_safe_float(ev.get("out_rms"), 0.0),
        token_count=_safe_int(ev.get("token_count"), 0),
    )
    dst_point = {
        "x": dx,
        "y": dy,
        "z": dz,
        "r": dst_color[0],
        "g": dst_color[1],
        "b": dst_color[2],
        "label": f"{dst_stage} flow_dst step={step} loop={dst_loop} block={dst_block} kind={kind}",
    }
    _inject_flow_fields(
        dst_point,
        stage=dst_stage,
        loop=dst_loop,
        block=dst_block,
        head=head,
        kv_head=kv_head,
        qk_align=_safe_float(ev.get("qk_align"), 0.0),
        step=step,
        micro_step=micro,
        kind=f"flow_dst:{kind}",
        transfer=_safe_float(ev.get("transfer"), 0.0),
        attn_entropy=_safe_float(ev.get("attn_entropy"), 0.0),
        attn_lag=_safe_float(ev.get("attn_lag"), 0.0),
        recent_mass=_safe_float(ev.get("recent_mass"), 0.0),
        attn_peak=_safe_float(ev.get("attn_peak"), 0.0),
        out_rms=_safe_float(ev.get("out_rms"), 0.0),
        token_count=_safe_int(ev.get("token_count"), 0),
    )
    h_mid = (STAGE_HUE.get(src_stage, 0.58) + STAGE_HUE.get(dst_stage, 0.58)) * 0.5
    edge_color = _hsl_rgb(h_mid, 0.80, 0.54)
    edge = {
        "r": edge_color[0],
        "g": edge_color[1],
        "b": edge_color[2],
        "alpha": _clamp(0.12 + 0.48 * math.tanh(flow * 0.9), 0.10, 0.88),
        "flow": flow,
        "phase": (0.021 * step + 0.07 * block + 0.11 * max(head, 0) + 0.031 * (idx % 29)) % 1.0,
        "loop": loop,
        "block": block,
        "head": head,
        "kv_head": kv_head,
        "step": step,
        "qk_align": _safe_float(ev.get("qk_align"), 0.0),
        "transfer": _safe_float(ev.get("transfer"), 0.0),
        "attn_entropy": _safe_float(ev.get("attn_entropy"), 0.0),
        "attn_lag": _safe_float(ev.get("attn_lag"), 0.0),
        "recent_mass": _safe_float(ev.get("recent_mass"), 0.0),
        "attn_peak": _safe_float(ev.get("attn_peak"), 0.0),
        "out_rms": _safe_float(ev.get("out_rms"), 0.0),
        "token_count": _safe_int(ev.get("token_count"), 0),
        "flow_kind": kind,
        "src_stage": src_stage,
        "dst_stage": dst_stage,
        "src_key": str(ev.get("src_key", f"S:{src_stage}:L{src_loop}:B{src_block}")),
        "dst_key": str(ev.get("dst_key", f"S:{dst_stage}:L{dst_loop}:B{dst_block}")),
        "magnitude": flow,
    }
    return src_point, dst_point, edge


def _read_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def build_points(events: list[dict], max_points: int, max_edges: int) -> tuple[list[dict], list[dict]]:
    steps = [int(e.get("step", 0)) for e in events if e.get("type") in ("activation", "head_event", "flow_event")]
    step_norm = float(max(max(steps, default=1), 1))
    pts: list[dict] = []
    edges: list[dict] = []
    for idx, ev in enumerate(events):
        if ev.get("type") == "flow_event":
            if str(ev.get("kind", "")) == "head_q_to_kv":
                # Avoid duplicating the canonical head_event Q↔KV edge already emitted below.
                continue
            if len(pts) + 2 > max_points or len(edges) >= max_edges:
                continue
            src_point, dst_point, edge = _flow_event_to_pair(ev, idx=idx, step_norm=step_norm)
            si = len(pts)
            di = si + 1
            pts.append(src_point)
            pts.append(dst_point)
            edges.append({"i": si, "j": di, **edge})
            continue
        if ev.get("type") == "head_event":
            if len(pts) + 2 > max_points or len(edges) >= max_edges:
                continue
            q_point, k_point, edge = _head_event_to_flow(ev, step_norm=step_norm)
            qi = len(pts)
            ki = qi + 1
            pts.append(q_point)
            pts.append(k_point)
            edges.append({"i": qi, "j": ki, **edge})
            continue
        if len(pts) >= max_points:
            continue
        p = _event_to_point(ev, idx=idx, step_norm=step_norm)
        if p is not None:
            pts.append(p)
    return pts, edges


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input Shroud JSONL trace")
    ap.add_argument("--output", required=True, help="Output points JSON path")
    ap.add_argument("--max-points", type=int, default=120_000)
    ap.add_argument("--max-edges", type=int, default=120_000)
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    events = _read_jsonl(in_path)
    points, edges = build_points(
        events,
        max_points=max(1, args.max_points),
        max_edges=max(1, args.max_edges),
    )
    payload = {
        "meta": {
            "source": str(in_path),
            "events": len(events),
            "points": len(points),
            "edges": len(edges),
            "palette": "complementary-gradients",
            "layout": "polar-flow",
        },
        "points": points,
        "edges": edges,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {len(points)} points to {out_path}")


if __name__ == "__main__":
    main()
