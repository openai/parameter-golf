from __future__ import annotations

import os
import socket

import torch
import torch.distributed as dist


def main() -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    tensor = torch.tensor([rank + 1.0], device="cuda")
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    expected = world_size * (world_size + 1) / 2
    if abs(float(tensor.item()) - float(expected)) > 1e-5:
        raise RuntimeError(f"Unexpected all_reduce result {tensor.item()} != {expected}")

    props = torch.cuda.get_device_properties(local_rank)
    print(
        "ddp_smoke_rank "
        f"rank={rank} local_rank={local_rank} host={socket.gethostname()} "
        f"device={torch.cuda.get_device_name(local_rank)!r} "
        f"cc={props.major}.{props.minor} "
        f"mem_total={props.total_memory}"
    )
    dist.barrier()
    if rank == 0:
        print(f"ddp_smoke:ok world_size={world_size} reduced_sum={tensor.item():.1f}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
