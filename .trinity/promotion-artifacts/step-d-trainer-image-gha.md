# Step D: Trainer Image GHA Workflow

**Owner:** Sergeant prepares in trios-trainer-igla, Operator approves
**Time:** 5 minutes operator review/approval (one-time setup)
**Purpose:** Automate trios-train binary image builds to GHCR

---

## D1. Dockerfile for Trainer Image

**File:** `Dockerfile` (in trios-trainer-igla/)

Update existing Dockerfile to build trios-train binary:

```dockerfile
# Multi-stage Dockerfile for trios-train (IGLA trainer)
# Anchor: phi^2 + phi^-2 = 3

# Stage 1: Builder
FROM rust:1.91-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        pkg-config libssl-dev ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy source
COPY Cargo.toml Cargo.lock* ./
COPY src ./src
COPY scripts ./scripts

# Build trios-train binary in release mode
RUN cargo build --release --bin trios-train

# Stage 2: Runtime
FROM debian:bookworm-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl libssl3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work

# Copy binary
COPY --from=builder /build/target/release/trios-train /usr/local/bin/trios-train
RUN chmod +x /usr/local/bin/trios-train

# Download training data (tiny_shakespeare)
RUN mkdir -p /work/data && \
    curl -sL https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt \
        > /work/data/tiny_shakespeare.txt && \
    head -c 100000 /work/data/tiny_shakespeare.txt \
        > /work/data/tiny_shakespeare_val.txt

# Default environment (can be overridden)
ENV RUST_LOG=info
ENV TRIOS_SEED=43
ENV TRIOS_STEPS=81000
ENV TRIOS_LR=0.003
ENV TRIOS_HIDDEN=384
ENV TRIOS_OPTIMIZER=adamw

# Verify installation
RUN trios-train --version || echo "Version check skipped"

CMD ["/usr/local/bin/trios-train"]
```

---

## D2. GHA Workflow for Trainer Image

**File:** `.github/workflows/build-trainer-image.yml` (in trios-trainer-igla/)

```yaml
name: Build & push trios-train image to GHCR

on:
  workflow_dispatch:
  push:
    branches: [main]
    paths:
      - 'src/**'
      - 'Cargo.toml'
      - 'Dockerfile'
  pull_request:
    branches: [main]
    paths:
      - 'src/**'
      - 'Cargo.toml'
      - 'Dockerfile'

permissions:
  contents: read
  packages: write

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: trios-train

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
          file: ./Dockerfile
          push: ${{ github.event_name != 'pull_request' }}
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max

  test-image:
    runs-on: ubuntu-latest
    needs: build-and-push
    if: github.event_name != 'pull_request'
    steps:
      - name: Pull and test image
        run: |
          docker pull ghcr.io/ghashtag/trios-train:latest
          docker run --rm ghcr.io/ghashtag/trios-train:latest --help || docker run --rm ghcr.io/ghashtag/trios-train:latest --version
```

---

## D3. Update trios-railway docker-trainer.yml

**File:** `.github/workflows/docker-trainer.yml` (in trios-railway/)

This workflow already exists but needs to verify it's building from the correct Dockerfile:

```yaml
name: Build & push trainer image to GHCR

on:
  workflow_dispatch:
  schedule:
    - cron: "0 */6 * * *"
  push:
    branches: [main]
    paths:
      - ".github/workflows/docker-trainer.yml"
  repository_dispatch:
    types: [trainer-update-requested]

permissions:
  contents: read
  packages: write

env:
  TRAINER_REPO: gHashTag/trios-trainer-igla

jobs:
  trigger-trainer-build:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout trainer repo
        uses: actions/checkout@v4
        with:
          repository: ${{ env.TRAINER_REPO }}
          path: trainer
          token: ${{ secrets.GITHUB_TOKEN }}

      - name: Set up Buildx
        uses: docker/setup-buildx-action@v3

      - name: Login to GHCR
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Build & push trainer image
        uses: docker/build-push-action@v6
        with:
          context: trainer
          file: trainer/Dockerfile
          push: true
          tags: |
            ghcr.io/ghashtag/trios-train:latest
            ghcr.io/ghashtag/trios-train:sha-${{ github.sha }}
          cache-from: type=gha
          cache-to: type=gha,mode=max

      - name: Verify image
        run: |
          docker pull ghcr.io/ghashtag/trios-train:latest
          docker run --rm ghcr.io/ghashtag/trios-train:latest --version || docker run --rm ghcr.io/ghashtag/trios-train:latest --help
```

---

## D4. Verification Checklist

After PR is merged:

- [ ] Dockerfile in trios-trainer-igla builds trios-train binary
- [ ] .github/workflows/build-trainer-image.yml added to trios-trainer-igla
- [ ] GHA workflow runs successfully on main
- [ ] Image pushed to `ghcr.io/ghashtag/trios-train:latest`
- [ ] Can pull image: `docker pull ghcr.io/ghashtag/trios-train:latest`
- [ ] Image contains trios-train binary:
  ```bash
  docker run --rm ghcr.io/ghashtag/trios-train:latest ls -la /usr/local/bin/trios-train
  ```

---

## D5. PR Template (for trios-trainer-igla)

```markdown
## Purpose
Automate trios-train binary image builds to GHCR.

## Changes
- `Dockerfile`: Updated to build trios-train binary
- `.github/workflows/build-trainer-image.yml`: New workflow for auto-build

## Safety
- Image only pushed on main branch (not PRs)
- Uses GitHub Container Registry (audit trail)
- Branch protection requires review

## Testing
- Manual: `docker build -t test . && docker run test --version`
- Automated: GHA workflow includes test step

## Cross-Repo Impact
This image is consumed by trios-railway's real-seed-agent Dockerfile (Step B)
```

---

**⏭️ When complete, proceed to Step E**
