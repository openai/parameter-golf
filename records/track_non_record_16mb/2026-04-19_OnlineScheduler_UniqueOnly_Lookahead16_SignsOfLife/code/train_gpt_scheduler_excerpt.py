# Excerpt from a modified train_gpt.py showing the UniqueLookaheadScheduler
# and SchedulerBackedDistributedLoader classes that implement the 8xH100
# scheduler path (SCHEDULER_POLICY=score_unique_only, separate scheduler rank).
# This is the real-machine path intended for the compute grant run.
# The local 6/6 ablation used prototype_train_local_ablation.py instead.

# --- from train_gpt.py line 535 ---
class UniqueLookaheadScheduler:
    def __init__(self, pattern: str, lookahead: int):
        self.stream = TokenStream(pattern)
        self.lookahead = lookahead
        self.buffers: dict[int, list[tuple[float, Tensor]]] = {}

    def _score(self, chunk: Tensor) -> float:
        local = chunk.to(dtype=torch.int64, device="cpu")
        if local.numel() == 0:
            return 0.0
        return float(torch.unique(local).numel()) / float(local.numel())

    def _fill(self, chunk_size: int) -> None:
        buffer = self.buffers.setdefault(chunk_size, [])
        while len(buffer) < self.lookahead:
            chunk = self.stream.take(chunk_size)
            buffer.append((self._score(chunk), chunk))

    def next_chunk(self, chunk_size: int) -> Tensor:
        self._fill(chunk_size)
        buffer = self.buffers[chunk_size]
        pick_idx = max(range(len(buffer)), key=lambda idx: buffer[idx][0])
        _, chunk = buffer.pop(pick_idx)
        return chunk


# --- from train_gpt.py line 561 ---
class SchedulerBackedDistributedLoader:
    def __init__(
        self,
        pattern: str,
        rank: int,
        train_ranks: int,
        scheduler_rank: int,
        device: torch.device,
        seq_len: int,
        global_batch_tokens: int,
        lookahead: int,
    ):
        self.rank = rank
        self.train_ranks = train_ranks
        self.scheduler_rank = scheduler_rank
        self.device = device
        self.seq_len = seq_len
        self.global_batch_tokens = global_batch_tokens
        self.local_seq_counts = split_sequence_counts(global_batch_tokens, seq_len, train_ranks)
        self.local_token_counts = [count * seq_len for count in self.local_seq_counts]
        self.local_token_count = 0 if rank >= train_ranks else self.local_token_counts[rank]
        self.local_span = self.local_token_count + 1
        self.scheduler = UniqueLookaheadScheduler(pattern, lookahead) if rank == scheduler_rank else None

    def next_batch(self, *_args) -> tuple[Tensor | None, Tensor | None]:
        if self.rank == self.scheduler_rank:
            if self.scheduler is None:
                raise RuntimeError("Scheduler rank is missing scheduler state")
            for target_rank in range(self.train_ranks):
                flat = self.scheduler.next_chunk(self.local_token_counts[target_rank] + 1).to(
                    self.device, dtype=torch.int64, non_blocking=True
                )
                dist.send(flat, dst=target_rank)
            return None, None
        if self.rank >= self.train_ranks:
            return None, None
        flat = torch.empty((self.local_span,), device=self.device, dtype=torch.int64)
        dist.recv(flat, src=self.scheduler_rank)
        x = flat[:-1].reshape(-1, self.seq_len)
        y = flat[1:].reshape(-1, self.seq_len)
        return x, y
