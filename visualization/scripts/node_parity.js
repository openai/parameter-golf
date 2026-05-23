#!/usr/bin/env node
// Run the JS MDLM forward against the saved trace from the command line.
// Same code path as the browser version (parity.js), but loads files via fs
// so we can iterate without a browser.
//
// Usage:
//   node visualization/scripts/node_parity.js [--k 0]

import { readFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");

// Dynamic import so the module path is relative to this file location.
const { mdlmForward } = await import(`${ROOT}/src/mdlm.js`);
const { maxAbsDiff } = await import(`${ROOT}/src/ops.js`);
const { loadModelFromDir } = await import(`${ROOT}/scripts/loader_node.js`);

async function loadTrace(binPath, jsonPath) {
  const manifest = JSON.parse(await readFile(jsonPath, "utf-8"));
  const buf = await readFile(binPath);
  const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
  const tensors = new Map();
  for (const entry of manifest.tensors) {
    const view = new Float32Array(ab, entry.offset, entry.nelems);
    tensors.set(entry.name, { data: new Float32Array(view), shape: entry.shape });
  }
  return { manifest, tensors };
}

async function loadMasks(path) {
  const buf = await readFile(path);
  const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
  const dv = new DataView(ab);
  const magic = new TextDecoder().decode(new Uint8Array(ab, 0, 8));
  if (magic !== "PGMASK\0\0") throw new Error(`bad magic: ${JSON.stringify(magic)}`);
  const K = dv.getUint32(8, true);
  const n = dv.getUint32(12, true);
  const eps = dv.getFloat32(16, true);
  const tValues = new Float32Array(ab, 24, K).slice();
  const masks = new Uint8Array(ab, 24 + 4 * K, K * n).slice();
  return { K, n, eps, tValues, masks };
}

const k = (() => {
  const i = process.argv.indexOf("--k");
  return i >= 0 ? parseInt(process.argv[i + 1], 10) : 0;
})();

console.log(`loading model + goldens (k=${k})...`);
const t0 = performance.now();
const [{ tensors: weights, manifest: modelManifest },
       tokens,
       masks,
       { tensors: traceTensors }] = await Promise.all([
  loadModelFromDir(`${ROOT}/public/models/mdlm`),
  JSON.parse(await readFile(`${ROOT}/goldens/tokens.json`, "utf-8")),
  loadMasks(`${ROOT}/goldens/masks.bin`),
  loadTrace(`${ROOT}/goldens/mdlm_trace_k${String(k).padStart(2, "0")}.bin`,
            `${ROOT}/goldens/mdlm_trace_k${String(k).padStart(2, "0")}.json`),
]);
const tLoaded = performance.now();
console.log(`  loaded in ${((tLoaded - t0) / 1000).toFixed(2)}s`);
console.log(`  arch: ${JSON.stringify(modelManifest.arch)}`);
console.log(`  goldens trace tensors: ${traceTensors.size}`);

const arch = modelManifest.arch;
const n = masks.n;
const maskRow = masks.masks.slice(k * n, (k + 1) * n);
const tokenIds = tokens.token_ids;
if (tokenIds.length !== n) {
  throw new Error(`token count ${tokenIds.length} != mask width ${n}`);
}

// Relative-diff thresholds: absolute diff matters less than diff relative to
// the magnitude of the reference tensor (residuals blow up across layers).
const REL_OK = 1e-5;     // ~ fp32 ULP — torch and JS produce essentially the same value
const REL_WARN = 1e-3;   // tolerable noise from differing reduction orders
let ok = 0, warn = 0, fail = 0, missing = 0;
let firstFail = null;
const checked = new Set();
const failedRows = [];
const allRows = [];

function fmt(d) {
  if (!d.ok) return `ERR ${d.reason}`;
  return `abs=${d.max.toExponential(2)}  rel=${d.rel.toExponential(2)}  refMax=${d.maxAbsRef.toExponential(2)}`;
}

function cap(name, jsTensor) {
  const golden = traceTensors.get(name);
  if (!golden) {
    missing++;
    if (process.env.VERBOSE) console.log(`     -- ${name}: no golden`);
    return;
  }
  checked.add(name);
  const d = maxAbsDiff(jsTensor, golden);
  allRows.push([name, d]);
  if (!d.ok) {
    fail++; firstFail ??= name;
    failedRows.push([name, fmt(d)]);
    return;
  }
  if (d.rel > REL_WARN) {
    fail++; firstFail ??= name;
    failedRows.push([name, fmt(d)]);
  } else if (d.rel > REL_OK) {
    warn++;
    if (process.env.VERBOSE) console.log(`WARN ${name}: ${fmt(d)}`);
  } else {
    ok++;
    if (process.env.VERBOSE) console.log(`  OK ${name}: ${fmt(d)}`);
  }
}

console.log("\nrunning JS forward...");
const tFwd0 = performance.now();
mdlmForward(weights, arch, tokenIds, maskRow, cap);
const tFwd1 = performance.now();
console.log(`forward: ${((tFwd1 - tFwd0) / 1000).toFixed(2)}s`);
console.log(`\nresults: OK=${ok} WARN=${warn} FAIL=${fail} (golden tensors uncaptured: ${traceTensors.size - checked.size})`);
if (firstFail) console.log(`first failure: ${firstFail}`);
console.log("\nfailed rows (first 20):");
for (const [n, d] of failedRows.slice(0, 20)) {
  console.log(`  ${n.padEnd(40)} ${d}`);
}

// Highlight the final logits diff explicitly — that's what the BPB ultimately depends on.
const final = allRows.find(([n]) => n === "logits");
if (final) {
  const d = final[1];
  console.log(`\nfinal logits: ${fmt(d)}`);
}

// --- Compute NLL contributions for this pass and diff vs the topk golden ---
// For pass k, the per-position NLL contribution is -log_softmax(logits)[true_id]
// at each MASKED position. We diff against the saved mdlm_topk.json for this k.
const { logSoftmax } = await import(`${ROOT}/src/ops.js`);
const finalLogits = (() => {
  // Re-run forward without capturing intermediates (or just reuse the one we
  // already ran above by capturing the "logits" tensor). Simpler: re-run with
  // a single tensor capture.
  let out = null;
  mdlmForward(weights, arch, tokenIds, maskRow, (n, t) => {
    if (n === "logits") out = t;
  });
  return out;
})();
// finalLogits shape [1, n, V]; squeeze to [n, V]
const V = arch.vocab_size;
const lp = logSoftmax({ data: finalLogits.data, shape: [n, V] });

const topkGoldens = JSON.parse(await readFile(`${ROOT}/goldens/mdlm_topk.json`, "utf-8"));
const passEntry = topkGoldens.passes.find((p) => p.k === k);
let maxNllAbs = 0, maxNllRel = 0, count = 0;
let maxLogitAbs = 0, maxLogitRel = 0;
for (const e of passEntry.entries) {
  const ourNll = -lp.data[e.pos * V + e.true_id];
  const dAbs = Math.abs(ourNll - e.nll_nats);
  const dRel = e.nll_nats !== 0 ? dAbs / Math.abs(e.nll_nats) : dAbs;
  if (dAbs > maxNllAbs) maxNllAbs = dAbs;
  if (dRel > maxNllRel) maxNllRel = dRel;
  // Also check the true_logit recorded in the topk golden
  const ourTrueLogit = finalLogits.data[e.pos * V + e.true_id];
  const lAbs = Math.abs(ourTrueLogit - e.true_logit);
  const lRel = e.true_logit !== 0 ? lAbs / Math.abs(e.true_logit) : lAbs;
  if (lAbs > maxLogitAbs) maxLogitAbs = lAbs;
  if (lRel > maxLogitRel) maxLogitRel = lRel;
  count++;
}
console.log(`\nper-pass NLL parity over ${count} masked positions:`);
console.log(`  NLL  max abs=${maxNllAbs.toExponential(3)}  max rel=${maxNllRel.toExponential(3)}`);
console.log(`  true_logit max abs=${maxLogitAbs.toExponential(3)}  max rel=${maxLogitRel.toExponential(3)}`);

// --- Top-10 ID parity ---
let topIdMatches = 0, topIdMismatches = 0;
for (const e of passEntry.entries) {
  // Take top-10 of our row
  const row = lp.data.subarray(e.pos * V, (e.pos + 1) * V);
  const idx = Array.from({ length: V }, (_, i) => i);
  // Use the raw logits from finalLogits to mimic the golden ordering exactly
  const rowL = finalLogits.data.subarray(e.pos * V, (e.pos + 1) * V);
  idx.sort((a, b) => rowL[b] - rowL[a]);
  const ourTop = idx.slice(0, e.ids.length);
  for (let i = 0; i < ourTop.length; i++) {
    if (ourTop[i] === e.ids[i]) topIdMatches++;
    else topIdMismatches++;
  }
}
console.log(`top-10 IDs: ${topIdMatches} matches, ${topIdMismatches} mismatches ` +
            `(out of ${count * (passEntry.entries[0]?.ids?.length ?? 10)})`);

process.exit(fail > 0 ? 1 : 0);
