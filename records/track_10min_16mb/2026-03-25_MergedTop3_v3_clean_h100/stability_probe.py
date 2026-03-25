import argparse
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F


def maybe_init_dist() -> tuple[bool, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    distributed = world_size > 1
    if distributed and not dist.is_initialized():
        dist.init_process_group("nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    return distributed, rank, world_size


def probe_attention(device: torch.device) -> None:
    batch = 2
    seqlen = 128
    num_heads = 8
    num_kv_heads = 4
    head_dim = 64
    dtype = torch.bfloat16

    q = torch.randn(batch, seqlen, num_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch, seqlen, num_kv_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch, seqlen, num_kv_heads, head_dim, device=device, dtype=dtype)

    flash_backend = None
    flash_func = None
    if os.environ.get("DISABLE_FLASH_ATTN", "0") != "1":
        for name in ("flash_attn_interface", "flash_attn"):
            try:
                mod = __import__(name, fromlist=["flash_attn_func"])
                flash_func = getattr(mod, "flash_attn_func")
                flash_backend = name
                break
            except Exception:
                pass

    if flash_func is not None:
        try:
            out = flash_func(q, k, v, causal=True)
            assert out.shape == q.shape
        except RuntimeError as exc:
            if "no kernel image is available" not in str(exc):
                raise
            flash_backend = None
            flash_func = None

    require_flash = os.environ.get("REQUIRE_FLASH_ATTN", "0") == "1"
    if require_flash and flash_func is None:
        raise SystemExit("attention probe failed: flash attention required but unavailable")

    if flash_func is None:
        q_t = q.transpose(1, 2)
        k_t = k.transpose(1, 2).repeat_interleave(num_heads // num_kv_heads, dim=1)
        v_t = v.transpose(1, 2).repeat_interleave(num_heads // num_kv_heads, dim=1)
        out = F.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=True).transpose(1, 2)
        assert out.shape == q.shape

    print(f"attention_probe_ok backend={flash_backend or 'sdpa_fallback'}")


def probe_collective(device: torch.device, distributed: bool, rank: int, world_size: int) -> None:
    t = torch.tensor([rank + 1.0], device=device)
    if distributed:
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        expected = world_size * (world_size + 1) / 2
        if abs(t.item() - expected) > 1e-5:
            raise SystemExit(f"collective probe failed: got {t.item()} expected {expected}")
        print(f"collective_probe_ok sum={t.item():.1f}")
    else:
        print("collective_probe_ok single_process")


def probe_checkpoint(device: torch.device, rank: int, checkpoint_dir: Path) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt = checkpoint_dir / "stability_probe.pt"
    if rank == 0:
        payload = {
            "tensor": torch.arange(16, device=device, dtype=torch.float32).cpu(),
            "meta": {"kind": "stability_probe"},
        }
        torch.save(payload, ckpt)
    if dist.is_initialized():
        dist.barrier()
    payload = torch.load(ckpt, map_location="cpu")
    assert payload["tensor"].shape == (16,)
    print("checkpoint_probe_ok")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", default=".")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("stability probe failed: cuda unavailable")

    distributed, rank, world_size = maybe_init_dist()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    print(f"probe_start rank={rank} world_size={world_size} device={device}")
    probe_attention(device)
    probe_collective(device, distributed, rank, world_size)
    probe_checkpoint(device, rank, Path(args.checkpoint_dir))
    torch.cuda.synchronize(device)
    print(f"probe_done rank={rank}")

    if distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
