#!/usr/bin/env node
// Node parity test for the AR forward.
//
// AR receives tokens[:n-1] as input (matching stage_ar in generate_goldens.py),
// then we diff every captured intermediate against ar_trace.bin.

import { readFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const { arForward } = await import(`${ROOT}/src/ar.js`);
const { maxAbsDiff, logSoftmax } = await import(`${ROOT}/src/ops.js`);
const { loadModelFromDir } = await import(`${ROOT}/scripts/loader_node.js`);
async function loadTrace(binPath, jsonPath) {
  const manifest = JSON.parse(await readFile(jsonPath, "utf-8"));
  const buf = await readFile(binPath);
  const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
  const tensors = new Map();
  for (const e of manifest.tensors) {
    const view = new Float32Array(ab, e.offset, e.nelems);
    tensors.set(e.name, { data: new Float32Array(view), shape: e.shape });
  }
  return { manifest, tensors };
}

const t0 = performance.now();
const [{ tensors: weights, manifest: modelManifest }, tokens, { tensors: traceTensors }] =
  await Promise.all([
    loadModelFromDir(`${ROOT}/public/models/ar`),
    JSON.parse(await readFile(`${ROOT}/goldens/tokens.json`, "utf-8")),
    loadTrace(`${ROOT}/goldens/ar_trace.bin`, `${ROOT}/goldens/ar_trace.json`),
  ]);
console.log(`loaded in ${((performance.now() - t0) / 1000).toFixed(2)}s; arch=${JSON.stringify(modelManifest.arch)}`);

const arch = modelManifest.arch;
const n = tokens.n_tokens;
const inputIds = tokens.token_ids.slice(0, n - 1);

const REL_OK = 1e-5, REL_WARN = 1e-3;
let ok = 0, warn = 0, fail = 0;
let firstFail = null;
const checked = new Set();
const failedRows = [];

function fmt(d) {
  if (!d.ok) return `ERR ${d.reason}`;
  return `abs=${d.max.toExponential(2)} rel=${d.rel.toExponential(2)} refMax=${d.maxAbsRef.toExponential(2)}`;
}
function cap(name, t) {
  const golden = traceTensors.get(name);
  if (!golden) return;
  checked.add(name);
  const d = maxAbsDiff(t, golden);
  if (!d.ok || d.rel > REL_WARN) { fail++; firstFail ??= name; failedRows.push([name, fmt(d)]); }
  else if (d.rel > REL_OK) warn++;
  else ok++;
}

console.log(`\nrunning AR forward on ${inputIds.length} tokens...`);
const tFwd = performance.now();
const finalLogits = arForward(weights, arch, inputIds, cap);
console.log(`forward: ${((performance.now() - tFwd) / 1000).toFixed(2)}s`);

console.log(`\nresults: OK=${ok} WARN=${warn} FAIL=${fail} (golden uncaptured: ${traceTensors.size - checked.size})`);
if (firstFail) console.log(`first fail: ${firstFail}`);
for (const [n, d] of failedRows.slice(0, 20)) console.log(`  ${n.padEnd(40)} ${d}`);

// Diff per-token NLL and top-10 against ar_topk.json
const arTop = JSON.parse(await readFile(`${ROOT}/goldens/ar_topk.json`, "utf-8"));
const V = arch.vocab_size;
const lp = logSoftmax({ data: finalLogits.data, shape: [inputIds.length, V] });

let maxNllAbs = 0, maxNllRel = 0;
let topMatches = 0, topMismatches = 0;
let totalChecked = 0;
for (const e of arTop.tokens) {
  if (e.predicted_from === null) continue;
  const row = e.predicted_from;
  const ourNllNats = -lp.data[row * V + e.true_id];
  const refNllBits = e.bits;
  const ourNllBits = ourNllNats / Math.log(2);
  const dAbs = Math.abs(ourNllBits - refNllBits);
  const dRel = refNllBits !== 0 ? dAbs / Math.abs(refNllBits) : dAbs;
  if (dAbs > maxNllAbs) maxNllAbs = dAbs;
  if (dRel > maxNllRel) maxNllRel = dRel;
  // Top-10 match
  const rowL = finalLogits.data.subarray(row * V, (row + 1) * V);
  const idx = Array.from({ length: V }, (_, i) => i);
  idx.sort((a, b) => rowL[b] - rowL[a]);
  const ourTop = idx.slice(0, e.ids.length);
  for (let i = 0; i < ourTop.length; i++) {
    if (ourTop[i] === e.ids[i]) topMatches++;
    else topMismatches++;
  }
  totalChecked++;
}
console.log(`\nper-token NLL parity over ${totalChecked} positions:`);
console.log(`  max abs (bits) = ${maxNllAbs.toExponential(3)}`);
console.log(`  max rel        = ${maxNllRel.toExponential(3)}`);
console.log(`top-10 IDs: ${topMatches} matches, ${topMismatches} mismatches`);

process.exit(fail > 0 ? 1 : 0);
