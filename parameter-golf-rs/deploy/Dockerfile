# Multi-stage build for Parameter Golf competition.
# Target: Modal H100x8 with CUDA 12.6 + NCCL 2.21
#
# Stage 1: Build the Rust binaries with CUDA feature
# Stage 2: Minimal runtime with just binaries + CUDA libs
#
# NOTE: cudarc 0.16.6 requires cublasGemmGroupedBatchedEx which was
# introduced in cuBLAS 12.5. We use CUDA 12.6 to ensure full API coverage.

# === Build stage ===
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04 AS builder

# Install Rust toolchain
RUN apt-get update && apt-get install -y curl build-essential pkg-config libssl-dev && \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Install NCCL dev and cuDNN dev (matching CUDA 12.6)
RUN apt-get install -y --allow-change-held-packages libnccl2 libnccl-dev libcudnn9-cuda-12 libcudnn9-dev-cuda-12 || apt-get install -y --allow-change-held-packages libnccl2 libnccl-dev libcudnn8-dev

WORKDIR /build

# Copy manifests first for dependency caching
COPY Cargo.toml Cargo.lock ./
COPY crates/ crates/

# Build all binaries with CUDA feature
RUN cargo build --release --features cuda -p pg-train -p pg-bench

# === Runtime stage ===
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --allow-change-held-packages libnccl2 libnccl-dev libcudnn9-cuda-12 || apt-get install -y --allow-change-held-packages libnccl2 libnccl-dev libcudnn8 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/target/release/pg-train /usr/local/bin/pg-train
COPY --from=builder /build/target/release/smoke /usr/local/bin/pg-smoke
COPY --from=builder /build/target/release/gemm-bench /usr/local/bin/pg-gemm-bench
COPY --from=builder /build/target/release/nccl-bench /usr/local/bin/pg-nccl-bench
COPY --from=builder /build/target/release/preliminary /usr/local/bin/pg-preliminary
COPY --from=builder /build/target/release/parity-forward /usr/local/bin/pg-parity-forward
COPY specs /specs
# Data directory mount point
# VOLUME /data removed since Modal handles volumes via @app.function(volumes={...})

ENV RUST_LOG=info
ENV CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# No ENTRYPOINT — Modal manages its own entrypoint to run Python function code.
# Rust binaries are invoked via subprocess.run() in the Modal functions.
