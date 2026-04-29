# Step B: Multi-stage Dockerfile + GHA Workflow

**Owner:** Sergeant prepares, Operator approves
**Time:** 5 minutes operator review/approval
**Purpose:** Enable automated build of real-seed-agent image

---

## B1. Multi-stage Dockerfile for Real Seed Agent

**File:** `Dockerfile.real-seed-agent` (to be added to trios-railway/)

```dockerfile
# Multi-stage Dockerfile for real-seed-agent
# Builds seed-agent and copies trios-train binary from trainer image
# Anchor: phi^2 + phi^-2 = 3

# Stage 1: Copy trainer binary from pre-built trainer image
ARG TRAINER_IMAGE=ghcr.io/ghashtag/trios-train:latest
FROM ${TRAINER_IMAGE} as trainer-stage
# Trainer binary is already at /usr/local/bin/trios-train

# Stage 2: Build seed-agent from source
FROM rust:1.91-slim as agent-stage

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        pkg-config libssl-dev ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy seed-agent source
COPY bin/seed-agent/Cargo.toml bin/seed-agent/Cargo.lock* ./
COPY bin/seed-agent/src ./src

# Build seed-agent in release mode
RUN cargo build --release

# Stage 3: Runtime image with both binaries
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl libssl3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work

# Copy binaries from previous stages
COPY --from=trainer-stage /usr/local/bin/trios-train /usr/local/bin/
COPY --from=agent-stage /build/target/release/seed-agent /usr/local/bin/

# Make binaries executable
RUN chmod +x /usr/local/bin/trios-train /usr/local/bin/seed-agent

# Default configuration (can be overridden via Railway vars)
ENV TRAINER_KIND=external
ENV TRAINER_BIN=/usr/local/bin/trios-train
ENV RUST_LOG=info
ENV SEED_AGENT_POLL_INTERVAL_MS=5000
ENV SEED_AGENT_EARLY_STOP_STEP=2000
ENV SEED_AGENT_EARLY_STOP_BPB_CEILING=3.0

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD pgrep -f seed-agent || exit 1

# Run seed-agent
ENTRYPOINT ["/usr/local/bin/seed-agent"]
```

---

## B2. GHA Workflow to Build & Push to GHCR

**File:** `.github/workflows/build-real-seed-agent.yml` (to be added to trios-railway/)

```yaml
name: Build & push real-seed-agent image to GHCR

on:
  workflow_dispatch:
  push:
    branches: [main]
    paths:
      - 'bin/seed-agent/**'
      - 'Dockerfile.real-seed-agent'
  pull_request:
    branches: [main]
    paths:
      - 'bin/seed-agent/**'
      - 'Dockerfile.real-seed-agent'

permissions:
  contents: read
  packages: write

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: trios-seed-agent-real

jobs:
  build-and-push:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Log in to GitHub Container Registry
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.REGISTRY }}/${{ github.repository_owner }}/${{ env.IMAGE_NAME }}
          tags: |
            type=ref,event=branch
            type=sha,prefix={{branch}}-
            type=raw,value=latest,enable={{is_default_branch}}

      - name: Build and push Docker image
        uses: docker/build-push-action@v6
        with:
          context: .
          file: ./Dockerfile.real-seed-agent
          push: ${{ github.event_name != 'pull_request' }}
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
          build-args: |
            TRAINER_IMAGE=ghcr.io/ghashtag/trios-train:latest

  test-image:
    runs-on: ubuntu-latest
    needs: build-and-push
    if: github.event_name != 'pull_request'
    steps:
      - name: Pull and test image
        run: |
          docker pull ghcr.io/ghashtag/trios-seed-agent-real:latest
          docker run --rm ghcr.io/ghashtag/trios-seed-agent-real:latest --version
```

---

## B3. Verification Checklist

After PR is merged:

- [ ] Dockerfile.real-seed-agent committed to main
- [ ] .github/workflows/build-real-seed-agent.yml committed to main
- [ ] GHA workflow runs successfully on main
- [ ] Image pushed to `ghcr.io/ghashtag/trios-seed-agent-real:latest`
- [ ] Can pull image locally: `docker pull ghcr.io/ghashtag/trios-seed-agent-real:latest`
- [ ] Image contains both binaries:
  ```bash
  docker run --rm ghcr.io/ghashtag/trios-seed-agent-real:latest ls -la /usr/local/bin/
  # Should show: seed-agent, trios-train
  ```

---

## B4. PR Template

```markdown
## Purpose
Add multi-stage Dockerfile and GHA workflow for real-seed-agent image automation.

## Changes
- `Dockerfile.real-seed-agent`: Multi-stage build (trainer + seed-agent)
- `.github/workflows/build-real-seed-agent.yml`: Auto-build on push to main

## Safety
- Branch protection requires review before merge
- Image only pushed on main branch (not PRs)
- Uses GitHub Container Registry (audit trail)

## Testing
- Manual: `docker build -f Dockerfile.real-seed-agent -t test .`
- Automated: GHA workflow includes test step

## Dependencies
Requires Step D complete (trainer image available at GHCR)
```

---

**⏭️ When complete, proceed to Step C**
