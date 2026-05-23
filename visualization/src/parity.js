// Browser-side parity walker.
//
// Loads:
//   - model.bin/json (converted MDLM weights)
//   - tokens.json    (canonical Transformer abstract tokenization)
//   - masks.bin      (deterministic K=64 masks)
//   - mdlm_trace_k00.bin/.json (every intermediate from one Python pass)
//
// Runs mdlmForward in JS with the same inputs, intercepting each capture
// with a callback that diffs against the corresponding golden tensor and
// prints {name, max abs diff, OK/WARN/FAIL}.

import { loadModel, loadTrace, loadTokens, loadMasks } from "./loader.js";
import { mdlmForward } from "./mdlm.js";
import { maxAbsDiff } from "./ops.js";

const STATUS = document.getElementById("status");
const LOG = document.getElementById("log");

// Relative-diff thresholds (matches scripts/node_parity.js): residual stream
// magnitudes blow up across layers, so absolute diff is misleading — what
// matters is diff / max_abs(reference). 1e-5 ≈ fp32 ULP, 1e-3 = "noticeable
// but harmless drift from differing reduction orders".
const REL_OK = 1e-5;
const REL_WARN = 1e-3;

function logRow(name, diffInfo) {
  const row = document.createElement("div");
  row.className = "row";
  let cls = "ok";
  let badge = "OK";
  let detail = "";
  if (!diffInfo.ok) {
    cls = "fail";
    badge = "FAIL";
    detail = diffInfo.reason;
  } else {
    if (diffInfo.rel > REL_WARN) { cls = "fail"; badge = "FAIL"; }
    else if (diffInfo.rel > REL_OK) { cls = "warn"; badge = "WARN"; }
    detail = `abs=${diffInfo.max.toExponential(2)} rel=${diffInfo.rel.toExponential(2)} refMax=${diffInfo.maxAbsRef.toExponential(2)}`;
  }
  row.innerHTML = `<span class="name">${name}</span><span class="diff ${cls}">${badge}</span><span>${detail}</span>`;
  LOG.appendChild(row);
  return cls;
}

async function runParity() {
  STATUS.textContent = "Loading model + goldens…";
  LOG.innerHTML = "";

  const t0 = performance.now();
  const [{ tensors: weights, manifest: modelManifest },
         tokens,
         masks,
         { tensors: traceTensors, manifest: traceManifest }] = await Promise.all([
    loadModel("models/mdlm"),
    loadTokens("goldens/tokens.json"),
    loadMasks("goldens/masks.bin"),
    loadTrace("goldens/mdlm_trace_k00.bin", "goldens/mdlm_trace_k00.json"),
  ]);

  const tLoaded = performance.now();
  STATUS.textContent = `Loaded in ${((tLoaded - t0) / 1000).toFixed(2)}s. Running JS forward…`;

  const arch = modelManifest.arch;
  const k = traceManifest.k ?? 0;
  console.log("arch:", arch, "k:", k);

  // The mask for pass k is row k of masks.masks
  const n = masks.n;
  const maskRow = masks.masks.slice(k * n, (k + 1) * n);

  // Token IDs as a plain array
  const tokenIds = tokens.token_ids;
  if (tokenIds.length !== n) {
    throw new Error(`token count ${tokenIds.length} != mask width ${n}`);
  }

  // Walk: each call to cap diffs against the trace.
  let okCount = 0, warnCount = 0, failCount = 0;
  let firstFail = null;
  const checkedNames = new Set();

  function cap(name, jsTensor) {
    const golden = traceTensors.get(name);
    if (!golden) {
      // Unknown captures — likely cos/sin which we don't capture per-block in
      // identical form; mark info-only.
      logRow(name, { ok: false, reason: "no golden (skipped)" });
      return;
    }
    checkedNames.add(name);
    const d = maxAbsDiff(jsTensor, golden);
    const cls = logRow(name, d);
    if (cls === "ok") okCount++;
    else if (cls === "warn") warnCount++;
    else { failCount++; if (firstFail === null) firstFail = name; }
  }

  const tForwardStart = performance.now();
  try {
    mdlmForward(weights, arch, tokenIds, maskRow, cap);
  } catch (e) {
    STATUS.innerHTML = `<span class="fail">forward threw:</span> ${e.message}`;
    console.error(e);
    return;
  }
  const tForwardEnd = performance.now();

  const total = okCount + warnCount + failCount;
  const missing = traceTensors.size - checkedNames.size;
  STATUS.innerHTML =
    `Forward: ${((tForwardEnd - tForwardStart) / 1000).toFixed(2)}s. ` +
    `Checked ${total} tensors. ` +
    `<span class="ok">OK ${okCount}</span> · ` +
    `<span class="warn">WARN ${warnCount}</span> · ` +
    `<span class="fail">FAIL ${failCount}</span>` +
    (missing > 0 ? ` · ${missing} golden tensors uncaptured.` : "") +
    (firstFail ? ` First fail: <code>${firstFail}</code>.` : "");
}

document.getElementById("run").addEventListener("click", runParity);
