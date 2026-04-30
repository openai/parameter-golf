from  __future__  import annotations

import glob
import math
import os
import time
from pathlib import Path

import numpy as np
import torch


def load_data_shard(file: Path) -> np.ndarray:
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    header_bytes = 256 * np.dtype("<i4").itemsize
    return np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)


def to_numpy_counts(counts: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(counts, torch.Tensor):
        return counts.detach().cpu().numpy().astype(np.float64, copy=False)
    return counts.astype(np.float64, copy=False)


# -------------------------------------------------------
# BIGRAM (unchanged from original)
# -------------------------------------------------------

def build_bigram_matrix(
    shard_pattern: str,
    vocab_size: int = 1024,
    laplace_alpha: float = 0.01,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    raw_counts = np.zeros((vocab_size, vocab_size), dtype=np.float64)
    unigram_counts = np.zeros(vocab_size, dtype=np.float64)
    files = sorted(glob.glob(shard_pattern))
    if not files:
        raise FileNotFoundError(f"No shards matched: {shard_pattern}")

    for index, file_name in enumerate(files, start=1):
        tokens = load_data_shard(Path(file_name))
        valid_tokens = tokens[tokens < vocab_size].astype(np.int32, copy=False)
        if valid_tokens.size:
            unigram_counts += np.bincount(valid_tokens, minlength=vocab_size)
        prev = tokens[:-1].astype(np.int32, copy=False)
        curr = tokens[1:].astype(np.int32, copy=False)
        mask = (prev < vocab_size) & (curr < vocab_size)
        flat = prev[mask] * vocab_size + curr[mask]
        if flat.size:
            raw_counts += np.bincount(flat, minlength=vocab_size * vocab_size).reshape(vocab_size, vocab_size)
        if index % 10 == 0 or index == len(files):
            print(
                f"bigram scan {index}/{len(files)} shards "
                f"counts={raw_counts.sum():.0f}",
                flush=True,
            )

    counts = raw_counts + laplace_alpha
    row_sums = counts.sum(axis=1, keepdims=True)
    col_sums = counts.sum(axis=0)
    log_probs = np.log(counts / row_sums).astype(np.float32)
    stats = {
        "log_probs": torch.from_numpy(log_probs),
        "counts": torch.from_numpy(counts.astype(np.float32)),
        "counts_raw": torch.from_numpy(raw_counts.astype(np.float32)),
        "unigram": torch.from_numpy(unigram_counts.astype(np.float32)),
        "row_sums": torch.from_numpy(row_sums.squeeze(-1).astype(np.float32)),
        "col_sums": torch.from_numpy(col_sums.astype(np.float32)),
        "row_sums_raw": torch.from_numpy(raw_counts.sum(axis=1).astype(np.float32)),
        "col_sums_raw": torch.from_numpy(raw_counts.sum(axis=0).astype(np.float32)),
        "alpha": torch.tensor(laplace_alpha, dtype=torch.float32),
    }
    return torch.from_numpy(log_probs), stats


def compress_svd(matrix: torch.Tensor, rank: int = 128) -> dict[str, torch.Tensor]:
    max_rank = min(rank, matrix.size(0), matrix.size(1))
    U, S, Vt = torch.linalg.svd(matrix.float(), full_matrices=False)
    US = (U[:, :max_rank] * S[:max_rank].unsqueeze(0)).to(torch.float16).contiguous()
    Vt_r = Vt[:max_rank, :].to(torch.float16).contiguous()
    recon = US.float() @ Vt_r.float()
    mse = float((matrix.float() - recon).pow(2).mean().item())
    max_err = float((matrix.float() - recon).abs().max().item())
    kept_var = float(S[:max_rank].pow(2).sum().item())
    total_var = float(S.pow(2).sum().item())
    print(
        f"rank-{max_rank} SVD mse={mse:.6f} max_err={max_err:.4f} "
        f"var_explained={100.0 * kept_var / max(total_var, 1e-12):.2f}%",
        flush=True,
    )
    return {
        "bigram_US": US,
        "bigram_Vt": Vt_r,
        "bigram_rank": torch.tensor(max_rank, dtype=torch.int32),
    }


def build_pmi_lens_basis_from_counts(
    counts: np.ndarray | torch.Tensor,
    rank: int = 32,
    alpha: float = 0.01,
    clip: float = 8.0,
    ppmi: bool = False,
) -> dict[str, object]:
    C = to_numpy_counts(counts)
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError(f"Expected square 2D counts matrix, got {tuple(C.shape)}")

    C = 0.5 * (C + C.T)
    C = C + alpha
    total = float(C.sum())
    Pij = C / max(total, 1e-12)
    Pi = np.sum(Pij, axis=1, keepdims=True)
    Pj = np.sum(Pij, axis=0, keepdims=True)
    K = np.log(Pij) - np.log(Pi) - np.log(Pj)
    K = np.clip(K, -clip, clip)
    if ppmi:
        K = np.maximum(K, 0.0)
    K = K - K.mean(axis=0, keepdims=True)
    K = K - K.mean(axis=1, keepdims=True)

    Kt = torch.from_numpy(K).float()
    max_rank = min(rank, Kt.size(0), Kt.size(1))
    U, S, _ = torch.linalg.svd(Kt, full_matrices=False)
    E = U[:, :max_rank] * S[:max_rank].clamp_min(0).sqrt().unsqueeze(0)
    E = torch.nn.functional.rms_norm(E, (max_rank,))
    recon = E @ E.T
    mse = float((Kt - recon).pow(2).mean().item())
    kind = "sym_count_ppmi_double_centered" if ppmi else "sym_count_pmi_double_centered"
    print(
        f"PMI-LENS rank={max_rank} ppmi={ppmi} clip={clip:.2f} mse={mse:.6f}",
        flush=True,
    )
    return {
        "pmi_embed": E.to(torch.float16).contiguous(),
        "pmi_rank": torch.tensor(max_rank, dtype=torch.int32),
        "pmi_clip": torch.tensor(clip, dtype=torch.float32),
        "pmi_ppmi": torch.tensor(int(ppmi), dtype=torch.int32),
        "pmi_kind": kind,
        "pmi_recon_mse": torch.tensor(mse, dtype=torch.float32),
    }


def build_directional_pmi_lens_basis_from_counts(
    counts: np.ndarray | torch.Tensor,
    rank: int = 32,
    alpha: float = 0.01,
    clip: float = 8.0,
) -> dict[str, object]:
    C = to_numpy_counts(counts)
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError(f"Expected square 2D counts matrix, got {tuple(C.shape)}")

    C = C + alpha
    row_sums = C.sum(axis=1, keepdims=True)
    col_sums = C.sum(axis=0, keepdims=True)
    total = float(C.sum())
    D = np.log(C) - np.log(row_sums) - (np.log(col_sums) - math.log(max(total, 1e-12)))
    D = np.clip(D, -clip, clip)

    Dt = torch.from_numpy(D).float()
    max_rank = min(rank, Dt.size(0), Dt.size(1))
    U, S, Vt = torch.linalg.svd(Dt, full_matrices=False)
    sigma = S[:max_rank].clamp_min(0).sqrt()
    E_key = U[:, :max_rank] * sigma.unsqueeze(0)
    E_query = Vt[:max_rank, :].T * sigma.unsqueeze(0)
    E_key = torch.nn.functional.rms_norm(E_key, (max_rank,))
    E_query = torch.nn.functional.rms_norm(E_query, (max_rank,))
    recon = E_key @ E_query.T
    mse = float((Dt - recon).pow(2).mean().item())
    print(
        f"directional PMI-LENS rank={max_rank} clip={clip:.2f} mse={mse:.6f}",
        flush=True,
    )
    return {
        "pmi_key_embed": E_key.to(torch.float16).contiguous(),
        "pmi_query_embed": E_query.to(torch.float16).contiguous(),
        "pmi_rank": torch.tensor(max_rank, dtype=torch.int32),
        "pmi_clip": torch.tensor(clip, dtype=torch.float32),
        "pmi_kind": "directional_count_pmi",
        "pmi_recon_mse": torch.tensor(mse, dtype=torch.float32),
    }


# -------------------------------------------------------
# TRIGRAM
# -------------------------------------------------------

def build_trigram_counts(
    shard_pattern: str,
    vocab_size: int = 1024,
    device: str = "cpu",
) -> torch.Tensor:
    """Accumulate trigram counts (a, b, c) across all shards.

    Uses GPU scatter_add_ when device='cuda' — ~100x faster than numpy bincount.
    Returns float32 tensor of shape (V, V, V).  VRAM/RAM ≈ 4 GB for V=1024.
    """
    V = vocab_size
    counts = torch.zeros(V * V * V, dtype=torch.float32, device=device)
    files = sorted(glob.glob(shard_pattern))
    if not files:
        raise FileNotFoundError(f"No shards matched: {shard_pattern}")

    t0 = time.time()
    for index, file_name in enumerate(files, start=1):
        tokens_np = load_data_shard(Path(file_name))
        tokens = torch.from_numpy(tokens_np.astype(np.int64, copy=False))
        a = tokens[:-2]
        b = tokens[1:-1]
        c = tokens[2:]
        mask = (a < V) & (b < V) & (c < V)
        a, b, c = a[mask], b[mask], c[mask]
        flat = (a * (V * V) + b * V + c).to(device=device)
        ones = torch.ones_like(flat, dtype=torch.float32)
        counts.scatter_add_(0, flat, ones)
        elapsed = time.time() - t0
        if index % 5 == 0 or index == len(files):
            print(
                f"trigram scan {index}/{len(files)} shards "
                f"total_trigrams={counts.sum().item():.0f} elapsed={elapsed:.1f}s",
                flush=True,
            )

    return counts.reshape(V, V, V)


def compute_trigram_residual(
    trigram_counts: torch.Tensor,
    bigram_logprior: torch.Tensor,
    vocab_size: int = 1024,
    laplace_alpha: float = 0.01,
    min_count: int = 5,
    shrink_k: float = 50.0,
    device: str = "cpu",
) -> torch.Tensor:
    """Compute residual[a,b,c] = log P(c|a,b) - log P(c|b).

    Entries with fewer than min_count raw trigram occurrences are zeroed
    to suppress Laplace-dominated noise.  All computation on device.
    """
    raw = trigram_counts.to(device=device, dtype=torch.float32)
    smoothed = raw + laplace_alpha
    pair_sums = smoothed.sum(dim=-1, keepdim=True)
    trigram_logprob = torch.log(smoothed / pair_sums)

    # bigram[b, c] = log P(c|b), broadcast along dim 0 (the "a" axis)
    bigram = bigram_logprior.to(device=device, dtype=torch.float32)
    residual = trigram_logprob - bigram.unsqueeze(0)

    mask = raw >= min_count
    pair_raw = raw.sum(dim=-1, keepdim=True)
    if shrink_k > 0.0:
        shrink = pair_raw / (pair_raw + shrink_k)
        residual = shrink * residual

    masked = torch.where(mask, residual, torch.zeros_like(residual))
    denom = mask.sum(dim=-1, keepdim=True).clamp_min(1)
    mean = masked.sum(dim=-1, keepdim=True) / denom
    residual = torch.where(mask, residual - mean, torch.zeros_like(residual))

    nonzero_frac = float((residual.abs() > 1e-6).sum().item()) / residual.numel()
    low_frac = float((raw < min_count).sum().item()) / raw.numel()
    print(
        f"trigram residual shape={tuple(residual.shape)} "
        f"range=[{residual.min().item():.4f},{residual.max().item():.4f}] "
        f"nonzero_frac={100.0 * nonzero_frac:.1f}% "
        f"zeroed_low_count={100.0 * low_frac:.1f}% "
        f"shrink_k={shrink_k:.1f}",
        flush=True,
    )
    return residual


def cp_als(
    X: torch.Tensor,
    rank: int = 64,
    n_iters: int = 50,
    eps: float = 1e-8,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """CP decomposition via ALS with HOSVD initialization.

    X: (V, V, V) tensor.
    Returns A, B, C each of shape (V, R) such that
    X ≈ sum_r A[:,r] ⊗ B[:,r] ⊗ C[:,r]  (with lambdas absorbed into C).
    """
    V = X.size(0)
    R = rank
    X = X.to(device=device, dtype=torch.float32)

    print(f"CP-ALS: V={V} R={R} iters={n_iters} device={device}", flush=True)
    t0 = time.time()

    # HOSVD initialization — truncated SVD of each mode unfolding
    A = torch.svd_lowrank(X.reshape(V, V * V), q=R)[0]           # (V, R)
    B = torch.svd_lowrank(X.permute(1, 0, 2).reshape(V, V * V), q=R)[0]
    C = torch.svd_lowrank(X.permute(2, 0, 1).reshape(V, V * V), q=R)[0]

    print(f"HOSVD init done in {time.time() - t0:.1f}s", flush=True)

    lambdas = torch.ones(R, device=device)
    for it in range(n_iters):
        # --- Mode 0: update A ---
        # T[i,j,r] = sum_k X[i,j,k] * C[k,r]
        T = X.reshape(V * V, V) @ C                    # (V*V, R)
        T = T.reshape(V, V, R)
        A_new = (T * B.unsqueeze(0)).sum(dim=1)         # (V, R)
        gram = (B.T @ B) * (C.T @ C) + eps * torch.eye(R, device=device)
        A = A_new @ torch.linalg.inv(gram)
        lambda_a = A.norm(dim=0).clamp_min(eps)
        A = A / lambda_a

        # --- Mode 1: update B ---
        # reuse T from mode-0 (T[i,j,r] = sum_k X[i,j,k]*C[k,r])
        B_new = (T * A.unsqueeze(1)).sum(dim=0)         # (V, R)
        gram = (A.T @ A) * (C.T @ C) + eps * torch.eye(R, device=device)
        B = B_new @ torch.linalg.inv(gram)
        lambda_b = B.norm(dim=0).clamp_min(eps)
        B = B / lambda_b

        # --- Mode 2: update C ---
        # S[j,k,r] = sum_i X[i,j,k] * A[i,r]  — computed per-j to avoid 4GB copy
        S = torch.zeros(V, V, R, device=device)
        for j in range(V):
            S[j] = X[:, j, :].T @ A                     # (V, R)
        C_new = (S * B.unsqueeze(1)).sum(dim=0)          # (V, R)
        gram = (A.T @ A) * (B.T @ B) + eps * torch.eye(R, device=device)
        C = C_new @ torch.linalg.inv(gram)
        lambda_c = C.norm(dim=0).clamp_min(eps)
        C = C / lambda_c
        lambdas = lambda_a * lambda_b * lambda_c

        if (it + 1) % 10 == 0 or it == 0:
            # Sampled reconstruction error — full recon is V^3 memory
            # Sample 1024 random (a,b) pairs and compare slices
            sample_a = torch.randint(0, V, (64,), device=device)
            sample_b = torch.randint(0, V, (64,), device=device)
            X_sample = X[sample_a, sample_b, :]                  # (64, V)
            # Recon: sum_r A[a,r] * B[b,r] * C[:,r]^T
            coeffs = A[sample_a] * B[sample_b]                   # (64, R)
            recon_sample = coeffs @ (C * lambdas).T               # (64, V)
            mse = (X_sample - recon_sample).pow(2).mean().item()
            print(
                f"  ALS iter {it+1}/{n_iters} sampled_mse={mse:.6f} "
                f"lambda_range=[{lambdas.min().item():.4f},{lambdas.max().item():.4f}] "
                f"time={time.time()-t0:.1f}s",
                flush=True,
            )

    # Absorb lambdas into C
    C_scaled = C * lambdas.unsqueeze(0)
    return A.cpu(), B.cpu(), C_scaled.cpu()


def main() -> None:
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    shard_pattern = os.path.join(data_path, "fineweb_train_*.bin")
    vocab_size = int(os.environ.get("VOCAB_SIZE", "1024"))
    laplace_alpha = float(os.environ.get("BIGRAM_LAPLACE_ALPHA", "0.01"))
    svd_rank = int(os.environ.get("BIGRAM_SVD_RANK", "128"))
    pmi_rank = int(os.environ.get("PMI_RANK", "32"))
    pmi_alpha = float(os.environ.get("PMI_ALPHA", str(laplace_alpha)))
    pmi_clip = float(os.environ.get("PMI_CLIP", "8.0"))
    build_ppmi = os.environ.get("BUILD_PPMI", "1") == "1"
    build_directional_pmi = os.environ.get("BUILD_DIRECTIONAL_PMI", "1") == "1"
    trigram_rank = int(os.environ.get("TRIGRAM_RANK", "64"))
    trigram_min_count = int(os.environ.get("TRIGRAM_MIN_COUNT", "5"))
    trigram_shrink_k = float(os.environ.get("TRIGRAM_SHRINK_K", "50"))
    trigram_als_iters = int(os.environ.get("TRIGRAM_ALS_ITERS", "50"))
    cp_device = os.environ.get("CP_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

    script_dir = Path(__file__).resolve().parent
    full_out = Path(os.environ.get("BIGRAM_OUT", str(script_dir / "bigram_logprior.pt")))
    bigram_stats_out = Path(os.environ.get("BIGRAM_STATS_OUT", str(script_dir / "bigram_stats.pt")))
    lowrank_out = Path(os.environ.get("BIGRAM_LOWRANK_OUT", str(script_dir / "bigram_lowrank.pt")))
    pmi_out = Path(os.environ.get("PMI_OUT", str(script_dir / f"pmi_embed_r{pmi_rank}.pt")))
    ppmi_out = Path(os.environ.get("PPMI_OUT", str(script_dir / f"ppmi_embed_r{pmi_rank}.pt")))
    directional_pmi_out = Path(
        os.environ.get("DIRECTIONAL_PMI_OUT", str(script_dir / f"directional_pmi_embed_r{pmi_rank}.pt"))
    )
    trigram_out = Path(os.environ.get("TRIGRAM_OUT", str(script_dir / "trigram_cp.pt")))

    # ---- BIGRAM ----
    print(
        f"building bigram prior vocab={vocab_size} alpha={laplace_alpha} pattern={shard_pattern}",
        flush=True,
    )
    bigram, bigram_stats = build_bigram_matrix(
        shard_pattern,
        vocab_size=vocab_size,
        laplace_alpha=laplace_alpha,
    )

    probs = bigram.exp()
    entropy = -(probs * bigram).sum(dim=-1)
    print(
        f"bigram shape={tuple(bigram.shape)} range=[{bigram.min().item():.3f},{bigram.max().item():.3f}] "
        f"mean_entropy={entropy.mean().item():.4f} nats "
        f"({entropy.mean().item() / math.log(2.0):.4f} bits)",
        flush=True,
    )

    torch.save(bigram, full_out)
    print(f"saved {full_out} bytes={full_out.stat().st_size}", flush=True)

    torch.save(bigram_stats, bigram_stats_out)
    print(f"saved {bigram_stats_out} bytes={bigram_stats_out.stat().st_size}", flush=True)

    lowrank = compress_svd(bigram, rank=svd_rank)
    torch.save(lowrank, lowrank_out)
    print(f"saved {lowrank_out} bytes={lowrank_out.stat().st_size}", flush=True)

    raw_bigram_counts = bigram_stats["counts_raw"]

    print(
        f"\nbuilding PMI-LENS bases rank={pmi_rank} alpha={pmi_alpha} clip={pmi_clip}",
        flush=True,
    )
    pmi_obj = build_pmi_lens_basis_from_counts(
        raw_bigram_counts,
        rank=pmi_rank,
        alpha=pmi_alpha,
        clip=pmi_clip,
        ppmi=False,
    )
    torch.save(pmi_obj, pmi_out)
    print(f"saved {pmi_out} bytes={pmi_out.stat().st_size}", flush=True)

    if build_ppmi:
        ppmi_obj = build_pmi_lens_basis_from_counts(
            raw_bigram_counts,
            rank=pmi_rank,
            alpha=pmi_alpha,
            clip=pmi_clip,
            ppmi=True,
        )
        torch.save(ppmi_obj, ppmi_out)
        print(f"saved {ppmi_out} bytes={ppmi_out.stat().st_size}", flush=True)

    if build_directional_pmi:
        directional_pmi_obj = build_directional_pmi_lens_basis_from_counts(
            raw_bigram_counts,
            rank=pmi_rank,
            alpha=pmi_alpha,
            clip=pmi_clip,
        )
        torch.save(directional_pmi_obj, directional_pmi_out)
        print(f"saved {directional_pmi_out} bytes={directional_pmi_out.stat().st_size}", flush=True)

    # ---- TRIGRAM ----
    print(
        f"\nbuilding trigram residual vocab={vocab_size} rank={trigram_rank} "
        f"min_count={trigram_min_count} shrink_k={trigram_shrink_k} device={cp_device}",
        flush=True,
    )

    trigram_counts = build_trigram_counts(shard_pattern, vocab_size=vocab_size, device=cp_device)
    print(
        f"trigram counts: total={trigram_counts.sum().item():.0f} "
        f"nonzero={int((trigram_counts > 0).sum().item())}/{trigram_counts.numel()} "
        f"({100.0 * (trigram_counts > 0).float().mean().item():.1f}%)",
        flush=True,
    )

    residual = compute_trigram_residual(
        trigram_counts, bigram, vocab_size=vocab_size,
        laplace_alpha=laplace_alpha, min_count=trigram_min_count,
        shrink_k=trigram_shrink_k,
        device=cp_device,
    )
    del trigram_counts  # free 4GB

    A, B, C = cp_als(
        residual, rank=trigram_rank, n_iters=trigram_als_iters,
        device=cp_device,
    )
    del residual  # free 4GB

    trigram_obj = {
        "trigram_u": A.to(torch.float16).contiguous(),   # (V, R) — indexed by t-2
        "trigram_v": B.to(torch.float16).contiguous(),   # (V, R) — indexed by t-1
        "trigram_w": C.to(torch.float16).contiguous(),   # (V, R) — indexed by target
        "trigram_rank": torch.tensor(trigram_rank, dtype=torch.int32),
    }
    torch.save(trigram_obj, trigram_out)
    print(
        f"saved {trigram_out} bytes={trigram_out.stat().st_size} "
        f"factors: u={tuple(A.shape)} v={tuple(B.shape)} w={tuple(C.shape)}",
        flush=True,
    )


if __name__ == "__main__":
    main()