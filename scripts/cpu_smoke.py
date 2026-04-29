import argparse
import glob
import importlib
import json
import math
import os
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
DEFAULT_TRAIN_FILES = os.path.join(DEFAULT_DATA_PATH, "fineweb_train_*.bin")
DEFAULT_VAL_FILES = os.path.join(DEFAULT_DATA_PATH, "fineweb_val_*.bin")
DEFAULT_TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
DEFAULT_RESULTS_DIR = "./results/cpu_smoke"


def resolve_repo_path(path_value):
    raw_path = os.fspath(path_value)
    if os.path.isabs(raw_path):
        return Path(raw_path)
    return Path(os.path.abspath(os.path.join(str(REPO_ROOT), raw_path)))


def resolve_repo_glob(pattern):
    raw_pattern = os.fspath(pattern)
    if os.path.isabs(raw_pattern):
        return raw_pattern
    return os.path.abspath(os.path.join(str(REPO_ROOT), raw_pattern))


def require_existing_file(path_value, label):
    path = resolve_repo_path(path_value)
    if not path.is_file():
        raise FileNotFoundError(
            "Missing {label} file: {path}. Pass an explicit path or set the matching environment variable.".format(
                label=label,
                path=path,
            )
        )
    return path


def require_matching_files(pattern, label):
    resolved_pattern = resolve_repo_glob(pattern)
    matches = [Path(path) for path in sorted(glob.glob(resolved_pattern))]
    if not matches:
        raise FileNotFoundError(
            "Missing {label} files: no files matched {pattern}. Pass an explicit glob or set DATA_PATH.".format(
                label=label,
                pattern=resolved_pattern,
            )
        )
    return matches


def write_summary(results_dir, mode, payload):
    output_dir = resolve_repo_path(results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output_path = output_dir / "{mode}_{timestamp}.json".format(mode=mode, timestamp=timestamp)
    data = {
        "generated_at": timestamp,
        "mode": mode,
        "repo_root": str(REPO_ROOT),
    }
    data.update(payload)
    with output_path.open("w") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return output_path


def positive_int(value, name):
    if value <= 0:
        raise ValueError("{name} must be positive, got {value}".format(name=name, value=value))


def load_train_gpt_module():
    repo_root = str(REPO_ROOT)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    try:
        return importlib.import_module("train_gpt")
    except SyntaxError as exc:
        raise RuntimeError(
            "Unable to import train_gpt.py with this interpreter. Use a Python version compatible with train_gpt.py."
        )
    except ImportError as exc:
        raise RuntimeError(
            "Unable to import train_gpt.py dependencies. Make sure the same environment used for train_gpt.py is active."
        )


def build_model(args, tg, device):
    model = tg.GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
    )
    return model.to(device)


def ensure_vocab_covers_batch(tokens, vocab_size, label):
    max_token_id = int(tokens.max().item())
    if max_token_id >= vocab_size:
        raise ValueError(
            "{label} contains token id {token_id}, but vocab_size is {vocab_size}. Increase --vocab-size to cover the dataset tokenizer ids.".format(
                label=label,
                token_id=max_token_id,
                vocab_size=vocab_size,
            )
        )


def run_train(args):
    positive_int(args.steps, "steps")
    positive_int(args.seq_len, "seq_len")
    positive_int(args.batch_size_seqs, "batch_size_seqs")
    require_matching_files(args.train_files, "training shard")

    tg = load_train_gpt_module()
    import torch

    device = torch.device("cpu")
    torch.manual_seed(args.seed)

    loader = tg.DistributedTokenLoader(resolve_repo_glob(args.train_files), 0, 1, device)
    model = build_model(args, tg, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    batch_tokens = args.batch_size_seqs * args.seq_len
    losses = []

    model.train()
    for step_idx in range(args.steps):
        x, y = loader.next_batch(batch_tokens, args.seq_len, 1)
        ensure_vocab_covers_batch(y, args.vocab_size, "training batch")
        optimizer.zero_grad()
        loss = model(x, y)
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach().cpu().item())
        losses.append(loss_value)
        print("train step {step}/{total} loss={loss:.6f}".format(step=step_idx + 1, total=args.steps, loss=loss_value))

    summary_path = write_summary(
        results_dir=args.results_dir,
        mode="train",
        payload={
            "batch_size_seqs": args.batch_size_seqs,
            "device": str(device),
            "final_loss": losses[-1],
            "initial_loss": losses[0],
            "losses": losses,
            "lr": args.lr,
            "matched_train_files": len(require_matching_files(args.train_files, "training shard")),
            "model_dim": args.model_dim,
            "num_heads": args.num_heads,
            "num_kv_heads": args.num_kv_heads,
            "num_layers": args.num_layers,
            "seq_len": args.seq_len,
            "steps": args.steps,
            "train_files": resolve_repo_glob(args.train_files),
            "vocab_size": args.vocab_size,
        },
    )
    print("wrote train summary to {path}".format(path=summary_path))
    return summary_path


def run_eval(args):
    positive_int(args.batches, "batches")
    positive_int(args.seq_len, "seq_len")
    positive_int(args.batch_size_seqs, "batch_size_seqs")
    tokenizer_path = require_existing_file(args.tokenizer_path, "tokenizer")
    require_matching_files(args.val_files, "validation shard")

    tg = load_train_gpt_module()
    import torch

    device = torch.device("cpu")
    torch.manual_seed(args.seed)

    sp = tg.spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    if args.vocab_size < int(sp.vocab_size()):
        raise ValueError(
            "vocab_size ({vocab_size}) must be at least the tokenizer vocab size ({tokenizer_vocab}).".format(
                vocab_size=args.vocab_size,
                tokenizer_vocab=int(sp.vocab_size()),
            )
        )
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = tg.build_sentencepiece_luts(
        sp,
        args.vocab_size,
        device,
    )
    val_tokens = tg.load_validation_tokens(resolve_repo_glob(args.val_files), args.seq_len)
    total_seqs = (val_tokens.numel() - 1) // args.seq_len
    max_seqs = args.batches * args.batch_size_seqs
    seqs_to_eval = min(total_seqs, max_seqs)
    if seqs_to_eval <= 0:
        raise ValueError("Validation data is too short for seq_len={seq_len}".format(seq_len=args.seq_len))

    model = build_model(args, tg, device)
    model.eval()

    val_loss_sum = 0.0
    val_token_count = 0
    val_byte_count = 0
    batch_losses = []
    batches_ran = 0

    with torch.no_grad():
        for seq_start in range(0, seqs_to_eval, args.batch_size_seqs):
            seq_end = min(seq_start + args.batch_size_seqs, seqs_to_eval)
            raw_start = seq_start * args.seq_len
            raw_end = seq_end * args.seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64)
            x = local[:-1].reshape(-1, args.seq_len)
            y = local[1:].reshape(-1, args.seq_len)
            ensure_vocab_covers_batch(y, args.vocab_size, "validation batch")
            batch_loss = model(x, y)
            batch_loss_value = float(batch_loss.detach().cpu().item())
            batch_losses.append(batch_loss_value)
            token_count = int(y.numel())
            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int64)
            token_bytes = token_bytes + (
                has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]
            ).to(dtype=torch.int64)
            byte_count = int(token_bytes.sum().item())

            val_loss_sum += batch_loss_value * token_count
            val_token_count += token_count
            val_byte_count += byte_count
            batches_ran += 1
            print("eval batch {batch} loss={loss:.6f}".format(batch=batches_ran, loss=batch_loss_value))
            if batches_ran >= args.batches:
                break

    smoke_val_loss = val_loss_sum / float(val_token_count)
    smoke_val_bpb = (smoke_val_loss / math.log(2.0)) * (float(val_token_count) / float(val_byte_count))

    summary_path = write_summary(
        results_dir=args.results_dir,
        mode="eval",
        payload={
            "batch_losses": batch_losses,
            "batch_size_seqs": args.batch_size_seqs,
            "batches_ran": batches_ran,
            "batches_requested": args.batches,
            "device": str(device),
            "matched_val_files": len(require_matching_files(args.val_files, "validation shard")),
            "model_dim": args.model_dim,
            "num_heads": args.num_heads,
            "num_kv_heads": args.num_kv_heads,
            "num_layers": args.num_layers,
            "seq_len": args.seq_len,
            "smoke_val_bpb": smoke_val_bpb,
            "smoke_val_loss": smoke_val_loss,
            "tokenizer_path": str(tokenizer_path),
            "val_files": resolve_repo_glob(args.val_files),
            "val_token_count": val_token_count,
            "val_byte_count": val_byte_count,
            "vocab_size": args.vocab_size,
        },
    )
    print(
        "eval summary loss={loss:.6f} bpb={bpb:.6f} wrote {path}".format(
            loss=smoke_val_loss,
            bpb=smoke_val_bpb,
            path=summary_path,
        )
    )
    return summary_path


def add_model_args(parser):
    parser.add_argument("--seed", type=int, default=int(os.environ.get("SEED", 1337)))
    parser.add_argument("--vocab-size", type=int, default=int(os.environ.get("VOCAB_SIZE", 1024)))
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--model-dim", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-kv-heads", type=int, default=2)
    parser.add_argument("--mlp-mult", type=int, default=2)
    parser.add_argument("--tie-embeddings", dest="tie_embeddings", action="store_true")
    parser.add_argument("--no-tie-embeddings", dest="tie_embeddings", action="store_false")
    parser.set_defaults(tie_embeddings=bool(int(os.environ.get("TIE_EMBEDDINGS", "1"))))
    parser.add_argument("--tied-embed-init-std", type=float, default=float(os.environ.get("TIED_EMBED_INIT_STD", 0.005)))
    parser.add_argument("--logit-softcap", type=float, default=float(os.environ.get("LOGIT_SOFTCAP", 30.0)))
    parser.add_argument("--rope-base", type=float, default=float(os.environ.get("ROPE_BASE", 10000.0)))
    parser.add_argument("--qk-gain-init", type=float, default=float(os.environ.get("QK_GAIN_INIT", 1.5)))


def build_parser():
    parser = argparse.ArgumentParser(description="Local CPU smoke harness for tiny train/eval checks.")
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train", help="Run a tiny CPU training smoke.")
    train_parser.add_argument("--train-files", default=DEFAULT_TRAIN_FILES)
    train_parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    train_parser.add_argument("--steps", type=int, default=2)
    train_parser.add_argument("--seq-len", type=int, default=32)
    train_parser.add_argument("--batch-size-seqs", type=int, default=2)
    train_parser.add_argument("--lr", type=float, default=1e-3)
    add_model_args(train_parser)

    eval_parser = subparsers.add_parser("eval", help="Run a tiny CPU eval smoke.")
    eval_parser.add_argument("--val-files", default=DEFAULT_VAL_FILES)
    eval_parser.add_argument("--tokenizer-path", default=DEFAULT_TOKENIZER_PATH)
    eval_parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    eval_parser.add_argument("--batches", type=int, default=2)
    eval_parser.add_argument("--seq-len", type=int, default=32)
    eval_parser.add_argument("--batch-size-seqs", type=int, default=2)
    add_model_args(eval_parser)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 2
    try:
        if args.command == "train":
            run_train(args)
        elif args.command == "eval":
            run_eval(args)
        else:
            raise ValueError("Unknown command: {command}".format(command=args.command))
    except Exception as exc:
        sys.stderr.write("ERROR: {message}\n".format(message=exc))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())