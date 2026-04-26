from __future__ import annotations

import argparse
from pathlib import Path

from experiment_env import guard_experiment_env

guard_experiment_env(script_name="export_minimal_real_cael_checkpoint", require_torch=True)

import torch
from prototype.model import RoutedTinyLM
from prototype.train import ShardTokenLoader, choose_device, set_seed
from run_understanding_gate_probe_eval import make_config


DEFAULT_OUTPUT = Path("/Users/seryn/Documents/Parameter_Golf/repo/three_layer_prototype_builder_assets/minimal_real_cael_checkpoint.pt")
DEFAULT_TRAIN_PATTERN = "/Users/seryn/Documents/Parameter_Golf/repo/curriculum_ablation/corpora/baseline_plain_order/datasets/fineweb10B_sp1024/fineweb_train_*.bin"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a minimal real-Cael checkpoint for builder integration.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-pattern", type=str, default=DEFAULT_TRAIN_PATTERN)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--train-steps", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=32)
    parser.add_argument("--shared-repeats", type=int, default=2)
    parser.add_argument("--router-temperature", type=float, default=1.5)
    parser.add_argument("--op3-hidden-dim", type=int, default=96)
    parser.add_argument("--binary-forward", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)
    model = RoutedTinyLM(make_config(args))
    device = choose_device(args.device)
    model.to(device)
    if args.train_steps > 0:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        loader = ShardTokenLoader(args.train_pattern, args.batch_size, args.seq_len, device)
        model.train()
        for _ in range(args.train_steps):
            inputs, targets = loader.next_batch()
            optimizer.zero_grad(set_to_none=True)
            loss, _stats = model(inputs, targets, binary_forward=args.binary_forward)
            loss.backward()
            optimizer.step()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output)
    print(args.output)


if __name__ == "__main__":
    main()
