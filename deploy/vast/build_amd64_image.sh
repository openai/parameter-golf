#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE_TAG="${PG_IMAGE_TAG:-}"
BASE_IMAGE="${PG_BASE_IMAGE:-nvcr.io/nvidia/pytorch:25.12-py3}"
PUSH_FLAG="${PG_PUSH_IMAGE:-1}"

docker_hub_repo_json() {
    local image_ref="$1"
    python3 - "$image_ref" <<'PY'
import json
import re
import sys

ref = sys.argv[1]
ref = ref.split("@", 1)[0]
last_slash = ref.rfind("/")
last_colon = ref.rfind(":")
if last_colon > last_slash:
    ref = ref[:last_colon]
parts = ref.split("/")
if len(parts) == 2:
    registry = "docker.io"
    namespace, repo = parts
elif len(parts) >= 3 and ("." in parts[0] or ":" in parts[0] or parts[0] == "localhost"):
    registry = parts[0]
    namespace = parts[1]
    repo = "/".join(parts[2:])
else:
    raise SystemExit(1)
print(json.dumps({"registry": registry, "namespace": namespace, "repo": repo}))
PY
}

require_current_builder_supports_amd64() {
    if ! docker buildx inspect >/tmp/pg_buildx_inspect.txt 2>&1; then
        cat /tmp/pg_buildx_inspect.txt >&2 || true
        echo "error: docker buildx inspect failed; select or create a buildx builder first" >&2
        exit 1
    fi
    if ! grep -q 'linux/amd64' /tmp/pg_buildx_inspect.txt; then
        cat /tmp/pg_buildx_inspect.txt >&2
        echo "error: current buildx builder does not advertise linux/amd64 support" >&2
        exit 1
    fi
}

require_docker_hub_push_prereqs() {
    local image_ref="$1"
    local repo_json registry namespace repo

    repo_json="$(docker_hub_repo_json "$image_ref")" || {
        echo "error: expected a fully-qualified image tag like docker.io/<namespace>/<repo>:<tag>" >&2
        exit 1
    }
    registry="$(python3 -c 'import json,sys; print(json.loads(sys.stdin.read())["registry"])' <<<"$repo_json")"
    namespace="$(python3 -c 'import json,sys; print(json.loads(sys.stdin.read())["namespace"])' <<<"$repo_json")"
    repo="$(python3 -c 'import json,sys; print(json.loads(sys.stdin.read())["repo"])' <<<"$repo_json")"

    if [[ "$registry" != "docker.io" && "$registry" != "index.docker.io" ]]; then
        return 0
    fi

    if ! python3 - <<'PY'
import json
import os
import sys

cfg_path = os.path.expanduser("~/.docker/config.json")
try:
    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)
except FileNotFoundError:
    sys.exit(1)

auths = cfg.get("auths", {})
targets = {
    "https://index.docker.io/v1/",
    "index.docker.io",
    "docker.io",
    "registry-1.docker.io",
}

if any(key in auths for key in targets):
    sys.exit(0)
if cfg.get("credsStore"):
    sys.exit(0)
if any(key in cfg.get("credHelpers", {}) for key in targets):
    sys.exit(0)
sys.exit(1)
PY
    then
        echo "error: no Docker Hub login is visible from this WSL/docker CLI environment" >&2
        echo "hint: run 'docker login -u $namespace' in WSL before building" >&2
        exit 1
    fi

    if ! python3 - "$namespace" "$repo" <<'PY'
import json
import sys
import urllib.error
import urllib.request

namespace, repo = sys.argv[1], sys.argv[2]
url = f"https://hub.docker.com/v2/repositories/{namespace}/{repo}/"
try:
    with urllib.request.urlopen(url, timeout=20) as resp:
        if resp.status == 200:
            sys.exit(0)
except urllib.error.HTTPError as exc:
    if exc.code == 404:
        sys.exit(2)
    raise
except Exception:
    sys.exit(3)
sys.exit(4)
PY
    then
        rc=$?
        if [[ "$rc" == "2" ]]; then
            echo "error: Docker Hub repo $namespace/$repo does not exist yet" >&2
            echo "hint: create https://hub.docker.com/repository/docker/$namespace/$repo first" >&2
        else
            echo "error: could not verify Docker Hub repo $namespace/$repo" >&2
        fi
        exit 1
    fi
}

if [[ -z "$IMAGE_TAG" ]]; then
    echo "error: set PG_IMAGE_TAG, e.g. ghcr.io/you/parameter-golf-vast:latest" >&2
    exit 1
fi

cd "$ROOT_DIR"
require_current_builder_supports_amd64
if [[ "$PUSH_FLAG" == "1" ]]; then
    require_docker_hub_push_prereqs "$IMAGE_TAG"
fi

CMD=(
    docker buildx build
    --platform linux/amd64
    --build-arg "BASE_IMAGE=$BASE_IMAGE"
    -f deploy/vast/Dockerfile.amd64
    -t "$IMAGE_TAG"
)

if [[ "$PUSH_FLAG" == "1" ]]; then
    CMD+=(--push)
else
    CMD+=(--load)
fi

CMD+=(.)
echo "Running: ${CMD[*]}"
"${CMD[@]}"
