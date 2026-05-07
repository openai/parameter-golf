from __future__ import annotations

import argparse
import html
import re
from dataclasses import dataclass
from pathlib import Path


TRAIN_RE = re.compile(r"step:(\d+)/\d+\s+train_loss:([0-9.]+)")
VAL_RE = re.compile(r"step:(\d+)/\d+\s+val_loss:([0-9.]+)\s+val_bpb:([0-9.]+)")
FINAL_Q_RE = re.compile(r"final_int8_zlib_roundtrip(?:_exact)? val_loss:([0-9.]+) val_bpb:([0-9.]+)")
FINAL_TTT_RE = re.compile(r"final_int8_ttt_adapter kind:([a-z_]+) val_loss:([0-9.]+) val_bpb:([0-9.]+)")
COLOR_CYCLE = [
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#9467bd",
    "#ff7f0e",
    "#17becf",
    "#8c564b",
    "#e377c2",
]


@dataclass
class Series:
    slug: str
    description: str
    log_path: Path
    train_points: list[tuple[int, float]]
    val_points: list[tuple[int, float]]
    val_bpbs: list[tuple[int, float]]
    final_roundtrip_bpb: float | None
    final_ttt_kind: str | None
    final_ttt_bpb: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an SVG loss graph from ablation logs.")
    parser.add_argument("--log-dir", type=Path, default=Path("ablations") / "logs")
    parser.add_argument("--output", type=Path, default=Path("ablations") / "ablations_loss.svg")
    parser.add_argument("--summary", type=Path, default=Path("ablations") / "ablations_summary.tsv")
    return parser.parse_args()


def parse_log(log_path: Path) -> Series:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    description = ""
    for line in text.splitlines():
        if line.startswith("# description="):
            description = line.split("=", 1)[1]
            break
    train_points = [(int(step), float(loss)) for step, loss in TRAIN_RE.findall(text)]
    val_points = [(int(step), float(loss)) for step, loss, _ in VAL_RE.findall(text)]
    val_bpbs = [(int(step), float(bpb)) for step, _, bpb in VAL_RE.findall(text)]
    roundtrip_matches = FINAL_Q_RE.findall(text)
    ttt_matches = FINAL_TTT_RE.findall(text)
    return Series(
        slug=log_path.stem,
        description=description or log_path.stem,
        log_path=log_path,
        train_points=train_points,
        val_points=val_points,
        val_bpbs=val_bpbs,
        final_roundtrip_bpb=float(roundtrip_matches[-1][1]) if roundtrip_matches else None,
        final_ttt_kind=ttt_matches[-1][0] if ttt_matches else None,
        final_ttt_bpb=float(ttt_matches[-1][2]) if ttt_matches else None,
    )


def axis_bounds(series_list: list[Series], attr: str) -> tuple[float, float] | None:
    values: list[float] = []
    for series in series_list:
        points = getattr(series, attr)
        values.extend(value for _, value in points)
    if not values:
        return None
    low = min(values)
    high = max(values)
    if low == high:
        pad = max(abs(low) * 0.05, 0.1)
        return low - pad, high + pad
    pad = (high - low) * 0.08
    return low - pad, high + pad


def x_bounds(series_list: list[Series]) -> tuple[int, int]:
    xs = [step for series in series_list for points in (series.train_points, series.val_points) for step, _ in points]
    if not xs:
        return 0, 1
    return 0, max(xs)


def scale_x(step: float, x_min: float, x_max: float, left: float, width: float) -> float:
    span = max(x_max - x_min, 1.0)
    return left + ((step - x_min) / span) * width


def scale_y(value: float, y_min: float, y_max: float, top: float, height: float) -> float:
    span = max(y_max - y_min, 1e-9)
    return top + height - ((value - y_min) / span) * height


def build_polyline(points: list[tuple[int, float]], x_min: float, x_max: float, y_min: float, y_max: float, left: float, top: float, width: float, height: float) -> str:
    return " ".join(
        f"{scale_x(step, x_min, x_max, left, width):.2f},{scale_y(value, y_min, y_max, top, height):.2f}"
        for step, value in points
    )


def build_panel(series_list: list[Series], attr: str, title: str, left: float, top: float, width: float, height: float, x_min: float, x_max: float) -> str:
    bounds = axis_bounds(series_list, attr)
    if bounds is None:
        return (
            f"<g><rect x='{left}' y='{top}' width='{width}' height='{height}' fill='white' stroke='#cccccc'/>"
            f"<text x='{left + 12}' y='{top + 24}' font-size='18' font-family='sans-serif'>{html.escape(title)}</text>"
            f"<text x='{left + width / 2:.2f}' y='{top + height / 2:.2f}' text-anchor='middle' font-size='16' fill='#666'>No data yet</text></g>"
        )

    y_min, y_max = bounds
    parts = [
        f"<g><rect x='{left}' y='{top}' width='{width}' height='{height}' fill='white' stroke='#cccccc'/>",
        f"<text x='{left + 12}' y='{top + 24}' font-size='18' font-family='sans-serif'>{html.escape(title)}</text>",
    ]
    for tick_idx in range(5):
        frac = tick_idx / 4
        y_value = y_min + frac * (y_max - y_min)
        y = scale_y(y_value, y_min, y_max, top, height)
        parts.append(f"<line x1='{left}' y1='{y:.2f}' x2='{left + width}' y2='{y:.2f}' stroke='#eeeeee'/>")
        parts.append(f"<text x='{left - 10}' y='{y + 4:.2f}' text-anchor='end' font-size='12' fill='#555'>{y_value:.3f}</text>")
    for tick_idx in range(6):
        frac = tick_idx / 5
        x_value = x_min + frac * (x_max - x_min)
        x = scale_x(x_value, x_min, x_max, left, width)
        parts.append(f"<line x1='{x:.2f}' y1='{top}' x2='{x:.2f}' y2='{top + height}' stroke='#f3f3f3'/>")
        parts.append(f"<text x='{x:.2f}' y='{top + height + 18}' text-anchor='middle' font-size='12' fill='#555'>{int(round(x_value))}</text>")

    for idx, series in enumerate(series_list):
        points = getattr(series, attr)
        if not points:
            continue
        color = COLOR_CYCLE[idx % len(COLOR_CYCLE)]
        polyline = build_polyline(points, x_min, x_max, y_min, y_max, left, top, width, height)
        parts.append(f"<polyline fill='none' stroke='{color}' stroke-width='2.25' points='{polyline}'/>")
        end_x = scale_x(points[-1][0], x_min, x_max, left, width)
        end_y = scale_y(points[-1][1], y_min, y_max, top, height)
        parts.append(f"<circle cx='{end_x:.2f}' cy='{end_y:.2f}' r='3.5' fill='{color}'/>")
    parts.append("</g>")
    return "".join(parts)


def legend(series_list: list[Series], left: float, top: float) -> str:
    parts = [f"<g><text x='{left}' y='{top}' font-size='18' font-family='sans-serif'>Ablations</text>"]
    y = top + 28
    for idx, series in enumerate(series_list):
        color = COLOR_CYCLE[idx % len(COLOR_CYCLE)]
        bpb_bits: list[str] = []
        if series.final_roundtrip_bpb is not None:
            bpb_bits.append(f"roundtrip={series.final_roundtrip_bpb:.4f}")
        if series.final_ttt_bpb is not None and series.final_ttt_kind is not None:
            bpb_bits.append(f"ttt[{series.final_ttt_kind}]={series.final_ttt_bpb:.4f}")
        metrics = " | ".join(bpb_bits) if bpb_bits else "pending"
        parts.append(f"<line x1='{left}' y1='{y - 4}' x2='{left + 18}' y2='{y - 4}' stroke='{color}' stroke-width='3'/>")
        parts.append(f"<text x='{left + 26}' y='{y}' font-size='13' font-family='sans-serif'>{html.escape(series.slug)} - {html.escape(metrics)}</text>")
        y += 20
    parts.append("</g>")
    return "".join(parts)


def build_svg(series_list: list[Series]) -> str:
    width = 1500
    height = 920
    left = 85
    panel_width = 980
    panel_height = 300
    x_min, x_max = x_bounds(series_list)
    train_panel = build_panel(series_list, "train_points", "Train loss vs step", left, 70, panel_width, panel_height, x_min, x_max)
    val_panel = build_panel(series_list, "val_points", "Validation loss vs step", left, 450, panel_width, panel_height, x_min, x_max)
    return (
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>"
        "<rect width='100%' height='100%' fill='#fafafa'/>"
        "<text x='85' y='36' font-size='28' font-family='sans-serif'>Random-map adapter ablations</text>"
        "<text x='85' y='780' font-size='13' fill='#555' font-family='sans-serif'>Step</text>"
        f"{train_panel}{val_panel}{legend(series_list, 1110, 92)}"
        "</svg>"
    )


def write_summary(series_list: list[Series], output_path: Path) -> None:
    lines = ["slug\tdescription\ttrain_points\tval_points\tfinal_roundtrip_bpb\tfinal_ttt_kind\tfinal_ttt_bpb\tlog_path"]
    for series in series_list:
        lines.append(
            "\t".join(
                [
                    series.slug,
                    series.description.replace("\t", " "),
                    str(len(series.train_points)),
                    str(len(series.val_points)),
                    f"{series.final_roundtrip_bpb:.8f}" if series.final_roundtrip_bpb is not None else "",
                    series.final_ttt_kind or "",
                    f"{series.final_ttt_bpb:.8f}" if series.final_ttt_bpb is not None else "",
                    str(series.log_path),
                ]
            )
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.log_dir.mkdir(parents=True, exist_ok=True)
    series_list = [parse_log(path) for path in sorted(args.log_dir.glob("*.log"))]
    svg = build_svg(series_list)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(svg, encoding="utf-8")
    write_summary(series_list, args.summary)
    print(f"Wrote {args.output}")
    print(f"Wrote {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
