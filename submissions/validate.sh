#!/usr/bin/env bash
# validate.sh — pre-submission checklist enforcer
# Usage: bash submissions/validate.sh records/track_10min_16mb/YYYY-MM-DD_Name_8xH100/
set -euo pipefail

RECORDS_DIR="${1:-}"
[[ -n "${RECORDS_DIR}" ]] || { echo "Usage: bash submissions/validate.sh <records_dir>"; exit 1; }
[[ -d "${RECORDS_DIR}" ]] || { echo "ERROR: directory not found: ${RECORDS_DIR}"; exit 1; }

PASS=0
FAIL=0

ok()   { echo "  [OK]   $*"; PASS=$((PASS+1)); }
fail() { echo "  [FAIL] $*"; FAIL=$((FAIL+1)); }
warn() { echo "  [WARN] $*"; }

echo ""
echo "======================================"
echo "  SUBMISSION VALIDATION"
echo "  ${RECORDS_DIR}"
echo "======================================"
echo ""

# ── 1. Required files ──────────────────────────────────────────────
echo "[ Required files ]"

for f in submission.json train_gpt.py README.md; do
    if [[ -f "${RECORDS_DIR}/${f}" ]]; then
        ok "${f} exists"
    else
        fail "${f} MISSING"
    fi
done

# At least one seed log required; seed 444 + 300 strongly recommended
LOGS_FOUND=0
for seed in 444 300 42; do
    log="${RECORDS_DIR}/train_seed${seed}.log"
    if [[ -f "${log}" ]]; then
        ok "train_seed${seed}.log exists"
        LOGS_FOUND=$((LOGS_FOUND+1))
    else
        if [[ "${seed}" == "444" || "${seed}" == "300" ]]; then
            fail "train_seed${seed}.log MISSING (required)"
        fi
    fi
done
[[ ${LOGS_FOUND} -ge 2 ]] || fail "Need at least seed=444 and seed=300 logs"

echo ""

# ── 2. submission.json validation ─────────────────────────────────
echo "[ submission.json ]"

JSON="${RECORDS_DIR}/submission.json"
if [[ ! -f "${JSON}" ]]; then
    fail "submission.json not found — skipping JSON checks"
else
    # Valid JSON?
    if python3 -c "import json,sys; json.load(open('${JSON}'))" 2>/dev/null; then
        ok "valid JSON"
    else
        fail "INVALID JSON — fix syntax errors first"
    fi

    # Required top-level fields
    for field in author github_id name blurb date val_bpb bytes_total bytes_code hardware; do
        val=$(python3 -c "import json; d=json.load(open('${JSON}')); print(d.get('${field}','MISSING'))" 2>/dev/null || echo "MISSING")
        if [[ "${val}" == "MISSING" || "${val}" == "None" ]]; then
            fail "field '${field}' missing"
        elif [[ "${val}" == "FILL_IN"* || "${val}" == "0" ]]; then
            fail "field '${field}' not filled in (value: ${val})"
        else
            ok "field '${field}' = ${val}"
        fi
    done

    # bytes_total <= 16MB
    BYTES=$(python3 -c "import json; print(json.load(open('${JSON}')).get('bytes_total', 0))" 2>/dev/null || echo "0")
    if [[ "${BYTES}" -gt 16000000 ]]; then
        fail "bytes_total=${BYTES} EXCEEDS 16MB limit (16,000,000 bytes)"
    elif [[ "${BYTES}" -gt 15500000 ]]; then
        warn "bytes_total=${BYTES} is close to 16MB limit — double-check"
        PASS=$((PASS+1))
    elif [[ "${BYTES}" -gt 0 ]]; then
        ok "bytes_total=${BYTES} is legal (≤ 16MB)"
    else
        fail "bytes_total=0 — not filled in"
    fi

    # bytes_code cross-check with log
    CODE_BYTES=$(python3 -c "import json; print(json.load(open('${JSON}')).get('bytes_code', 0))" 2>/dev/null || echo "0")
    if [[ "${CODE_BYTES}" -gt 0 ]]; then
        # Check if log confirms the code size
        LOG_CODE=""
        for seed in 444 300 42; do
            logfile="${RECORDS_DIR}/train_seed${seed}.log"
            if [[ -f "${logfile}" ]]; then
                LOG_CODE=$(grep -oP 'Code size:\s*\K[0-9]+' "${logfile}" | head -1 || true)
                [[ -n "${LOG_CODE}" ]] && break
            fi
        done
        if [[ -n "${LOG_CODE}" ]]; then
            if [[ "${LOG_CODE}" == "${CODE_BYTES}" ]]; then
                ok "bytes_code=${CODE_BYTES} matches log Code size: ${LOG_CODE}"
            else
                fail "bytes_code=${CODE_BYTES} DOES NOT MATCH log 'Code size: ${LOG_CODE}' — wrong train_gpt.py?"
            fi
        else
            warn "Could not find 'Code size:' in logs — verify bytes_code=${CODE_BYTES} manually"
        fi
    fi

    # val_bpb_exact cross-check with log
    for seed in 444 300; do
        field="seed_${seed}"
        EXACT=$(python3 -c "import json; d=json.load(open('${JSON}')); s=d.get('${field}',{}); print(s.get('val_bpb_exact','MISSING'))" 2>/dev/null || echo "MISSING")
        logfile="${RECORDS_DIR}/train_seed${seed}.log"
        if [[ "${EXACT}" == "MISSING" || "${EXACT}" == "0.0" || "${EXACT}" == "0" ]]; then
            fail "seed_${seed}.val_bpb_exact not filled in"
        elif [[ -f "${logfile}" ]]; then
            LOG_BPB=$(grep -oP 'final_sliding_window_exact val_bpb=\K[0-9.]+' "${logfile}" | tail -1 || true)
            if [[ -n "${LOG_BPB}" ]]; then
                if [[ "${LOG_BPB}" == "${EXACT}" ]]; then
                    ok "seed_${seed} val_bpb_exact=${EXACT} matches log"
                else
                    fail "seed_${seed} val_bpb_exact=${EXACT} DOES NOT MATCH log final_sw val_bpb=${LOG_BPB}"
                fi
            else
                warn "seed_${seed}: could not find 'final_sliding_window_exact' in log — verify manually"
            fi
        else
            warn "seed_${seed}: log not found, cannot cross-check val_bpb_exact"
        fi
    done
fi

echo ""

# ── 3. Branch check ────────────────────────────────────────────────
echo "[ Git branch ]"

CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
if [[ "${CURRENT_BRANCH}" == "TEST_LAB" ]]; then
    ok "on TEST_LAB — remember to create submission/<name> branch before pushing"
elif [[ "${CURRENT_BRANCH}" == submission/* ]]; then
    ok "on submission branch: ${CURRENT_BRANCH}"
else
    warn "on branch '${CURRENT_BRANCH}' — make sure you create a submission/<name> branch before pushing to fork1"
fi

# Confirm records dir is committed
if git ls-files --error-unmatch "${RECORDS_DIR}/submission.json" &>/dev/null; then
    ok "submission.json is tracked by git"
else
    warn "submission.json is not yet committed — do 'git add ${RECORDS_DIR}/' first"
fi

echo ""

# ── 4. Summary ─────────────────────────────────────────────────────
echo "======================================"
if [[ ${FAIL} -eq 0 ]]; then
    echo "  RESULT: ALL CHECKS PASSED (${PASS} ok)"
    echo ""
    echo "  Next step:"
    echo "    git checkout -b submission/<name>"
    echo "    git add ${RECORDS_DIR}/"
    echo "    git commit -m 'Add <Name> submission — <BPB> BPB, <size>MB'"
    echo "    git push fork1 submission/<name>"
    echo "    # then: read submissions/PROTOCOL.md Step 5 for gh pr create"
else
    echo "  RESULT: ${FAIL} FAILURE(S), ${PASS} passed"
    echo ""
    echo "  Fix all [FAIL] items before proceeding."
    echo "  See submissions/PROTOCOL.md for details."
fi
echo "======================================"
echo ""

[[ ${FAIL} -eq 0 ]]
