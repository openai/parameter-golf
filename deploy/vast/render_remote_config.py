from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


def parse_scalar(raw: str) -> Any:
    return yaml.safe_load(raw)


def render_remote_config(
    config: dict[str, Any],
    *,
    workdir: Path,
    python_bin: Path,
    gpus: int,
    logs_dir: Path,
    output_root: Path | None = None,
    activate_script: Path | None = None,
    fixed_env_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rendered = dict(config)
    runner = dict(rendered.get("runner", {}))
    search = dict(rendered.get("search", {}))

    script_path = Path(str(runner["script_path"]))
    if not script_path.is_absolute():
        script_path = (workdir / script_path).resolve()

    logs_path = logs_dir if logs_dir.is_absolute() else (workdir / logs_dir).resolve()
    if output_root is None:
        current_output_root = Path(str(search.get("output_root", f"search_runs/{Path(str(script_path)).stem}")))
        output_root = current_output_root if current_output_root.is_absolute() else (workdir / current_output_root).resolve()
    elif not output_root.is_absolute():
        output_root = (workdir / output_root).resolve()

    runner["workdir"] = str(workdir)
    runner["script_path"] = str(script_path)
    runner["python_bin"] = str(python_bin)
    runner["activate_script"] = str(activate_script) if activate_script is not None else None
    runner["gpus"] = int(gpus)
    runner["logs_dir"] = str(logs_path)

    search["output_root"] = str(output_root)
    rendered["runner"] = runner
    rendered["search"] = search

    if fixed_env_overrides:
        fixed_env = dict(rendered.get("fixed_env", {}))
        fixed_env.update(fixed_env_overrides)
        rendered["fixed_env"] = fixed_env

    return rendered


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render a remote-ready Parameter Golf search config")
    parser.add_argument("--input", required=True, help="Input YAML config")
    parser.add_argument("--output", required=True, help="Output YAML config")
    parser.add_argument("--workdir", required=True, help="Remote workdir root")
    parser.add_argument("--python-bin", required=True, help="Remote Python binary to use")
    parser.add_argument("--gpus", type=int, required=True, help="Remote GPU count")
    parser.add_argument("--logs-dir", default="logs", help="Remote logs directory")
    parser.add_argument("--output-root", default=None, help="Override search.output_root")
    parser.add_argument("--activate-script", default=None, help="Optional activation script")
    parser.add_argument(
        "--set-fixed-env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override one fixed_env key using YAML scalar parsing",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    config = yaml.safe_load(input_path.read_text(encoding="utf-8"))

    overrides: dict[str, Any] = {}
    for item in args.set_fixed_env:
        if "=" not in item:
            raise ValueError(f"Invalid fixed-env override: {item!r}")
        key, raw_value = item.split("=", 1)
        overrides[key] = parse_scalar(raw_value)

    rendered = render_remote_config(
        config,
        workdir=Path(args.workdir),
        python_bin=Path(args.python_bin),
        gpus=args.gpus,
        logs_dir=Path(args.logs_dir),
        output_root=Path(args.output_root) if args.output_root else None,
        activate_script=Path(args.activate_script) if args.activate_script else None,
        fixed_env_overrides=overrides,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(rendered, sort_keys=False), encoding="utf-8")


if __name__ == "__main__":
    main()
