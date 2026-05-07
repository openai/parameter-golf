"""Pod-side self-termination helpers for RunPod pods.

Provides a bash preamble and environment-variable helpers so that
every pod launched by this repository's tooling will terminate itself
after a hard deadline, independent of the HPC session that created it.

Hard deadline: 12 minutes (720 seconds) by default.
Retrieval buffer: 2 minutes (120 seconds) — callers should finish
data download at least this long before the hard deadline fires.

Mechanism:
    1. A background subshell sleeps for PGOLF_HARD_DEADLINE_SEC seconds.
    2. On wake-up it calls RunPod's GraphQL ``podTerminate`` mutation
       using ``curl``, authenticated with RUNPOD_API_KEY.
    3. As a last-resort fallback it sends ``kill 1`` to stop PID 1
       (the container init process), which RunPod treats as pod exit.

Environment variables consumed on the pod:
    PGOLF_HARD_DEADLINE_SEC  – seconds until self-termination (default 720)
    RUNPOD_API_KEY           – bearer token for the terminate mutation
    RUNPOD_POD_ID            – injected automatically by RunPod runtime
"""

# 12-minute hard wall-clock budget for any pod.
POD_HARD_DEADLINE_SECONDS = 720

# Callers should finish retrieval this many seconds before the
# hard deadline fires.  2 minutes is conservative for Jupyter
# download of logs + small artifacts.
RETRIEVAL_BUFFER_SECONDS = 120


def selfterm_env_dict(api_key, deadline_sec=POD_HARD_DEADLINE_SECONDS):
    """Return env-var dict to pass to RunPod ``create_pod``.

    Parameters
    ----------
    api_key : str
        RunPod API bearer token (never written to disk).
    deadline_sec : int
        Hard pod lifetime in seconds.  Default 720 (12 min).

    Returns
    -------
    dict
        Keys suitable for merging into a pod's env mapping.
    """
    return {
        "RUNPOD_API_KEY": api_key,
        "PGOLF_HARD_DEADLINE_SEC": str(int(deadline_sec)),
    }


def selfterm_bash_preamble():
    r"""Return a bash snippet that arms pod-side self-termination.

    The snippet must be inserted **before** the user payload in any
    job wrapper script.  It launches a background subshell that:

    * sleeps for ``$PGOLF_HARD_DEADLINE_SEC`` seconds (default 720),
    * calls RunPod's GraphQL terminate mutation via ``curl``,
    * falls back to ``kill 1`` if the API call fails.

    The snippet is safe to embed under ``set +e`` or ``set -o pipefail``
    and does not ``set -e`` itself.
    """
    return _SELFTERM_PREAMBLE


# ---------------------------------------------------------------------------
# The actual bash snippet — kept as a module-level constant so it is
# easy to inspect and test.  The triple-quoted string is *not* an
# f-string; all ``$`` references are shell variables.
# ---------------------------------------------------------------------------
_SELFTERM_PREAMBLE = r"""
# ── Pod-side self-termination (independent of HPC session) ──────
(
  _deadline="${PGOLF_HARD_DEADLINE_SEC:-720}"
  _pod_id="${RUNPOD_POD_ID:-}"
  _api_key="${RUNPOD_API_KEY:-}"
  echo "[pgolf-selfterm] Self-termination armed: ${_deadline}s deadline (pod=${_pod_id})"
  sleep "$_deadline"
  echo "[pgolf-selfterm] DEADLINE REACHED (${_deadline}s). Terminating pod ${_pod_id}..."
  if [ -n "$_api_key" ] && [ -n "$_pod_id" ]; then
    curl -sS --max-time 30 -X POST https://api.runpod.io/graphql \
      -H "Content-Type: application/json" \
      -H "Authorization: Bearer ${_api_key}" \
      -d "{\"query\": \"mutation { podTerminate(input: { podId: \\\"${_pod_id}\\\" }) }\"}" \
      || echo "[pgolf-selfterm] curl terminate failed, falling back to kill 1"
  else
    echo "[pgolf-selfterm] Missing API_KEY or POD_ID, falling back to kill 1"
  fi
  sleep 10
  kill 1 2>/dev/null || true
) &
# ── End self-termination preamble ───────────────────────────────
"""
