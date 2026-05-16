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

# Modal image builds occasionally hit transient Ubuntu mirror/DNS failures.
# Keep retry policy in the image rather than requiring manual reruns.
RUN printf 'Acquire::Retries "5";\nAcquire::http::Timeout "30";\nAcquire::https::Timeout "30";\n' \
    > /etc/apt/apt.conf.d/80-retries

# Install the minimal Rust toolchain. Modal builders have previously inherited a
# default rustup profile that pulled clippy/rustfmt and caused avoidable network
# failures during H100 validation image rebuilds.
RUN apt-get update && apt-get install -y --no-install-recommends curl build-essential pkg-config libssl-dev ca-certificates && \
    export RUSTUP_INIT_SKIP_PATH_CHECK=yes && \
    for attempt in 1 2 3 4 5; do \
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --profile minimal --default-toolchain stable && break; \
        rm -rf /root/.rustup/downloads/*.partial; \
        if [ "$attempt" = "5" ]; then exit 1; fi; \
        sleep $((attempt * 5)); \
    done
ENV PATH="/root/.cargo/bin:${PATH}"

# Install NCCL/cuDNN matching the CUDA 12.8 base image. Falling through to the
# generic packages is preferable to silently upgrading NCCL to a CUDA 13.x build.
RUN apt-get install -y --allow-change-held-packages \
    libnccl2=2.25.1-1+cuda12.8 \
    libnccl-dev=2.25.1-1+cuda12.8 \
    libcudnn9-cuda-12 \
    libcudnn9-dev-cuda-12 || \
    apt-get install -y --allow-change-held-packages libnccl2 libnccl-dev libcudnn8-dev

# cuDNN's graph/frontend SDPA API is distributed as NVIDIA's header-only
# cudnn-frontend project, not by the CUDA apt package on this base image.
RUN curl -L https://github.com/NVIDIA/cudnn-frontend/archive/refs/tags/v1.22.0.tar.gz \
    | tar xz -C /opt && \
    cp -r /opt/cudnn-frontend-1.22.0/include/* /usr/local/include/

WORKDIR /build

# The repository is mounted into /build by Modal at function startup and
# compiled into a persistent /build/target volume. Do not copy source or run
# `cargo fetch` in the image: doing so invalidates the whole CUDA/Rust base
# image on every kernel edit and makes H100 iteration hostage to image rebuilds.

# === Runtime stage ===
# Use the builder image as the runtime while iterating so the remote function
# can compile pg-train after startup. This intentionally trades image size for
# reliable validation.
FROM builder

COPY specs /specs
# Data directory mount point
# VOLUME /data removed since Modal handles volumes via @app.function(volumes={...})

ENV RUST_LOG=info
ENV CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# No ENTRYPOINT — Modal manages its own entrypoint to run Python function code.
# Rust binaries are invoked via subprocess.run() in the Modal functions.
