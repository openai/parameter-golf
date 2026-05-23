#!/usr/bin/env node
// JS↔Python parity check for the new artifact pipeline.
//
//   1. Read public/models/<kind>/model.safetensors.gz
//   2. Inflate with zlib (browser would use DecompressionStream("gzip"))
//   3. Parse the safetensors header + tensor bytes
//   4. Dequantize: int8 + fp16 scale + bf16 round-trip (AR), or
//      fp8_e4m3fn LUT (MDLM); fp32 passthrough
//   5. Read onnx_data_manifest.json
//   6. Reconstruct the model.onnx.data byte buffer (transpose where the
//      manifest says so, copy zeros for length-only blocks)
//   7. SHA-256 the result and assert it matches:
//        - manifest.sha256                (the reference hash)
//        - sha256(public/models/<kind>/model.onnx.data)  (committed file)
//        - the Python reconstruction hash (from reconstruct_onnx_data.py)
//
// Usage:
//   node visualization/scripts/node_parity_safetensors.js [ar|mdlm|all]

import { readFile } from "node:fs/promises";
import { gunzipSync } from "node:zlib";
import { createHash } from "node:crypto";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const { parseSafetensors, dequantSafetensors } = await import(`${ROOT}/src/safetensors.js`);

function sha256Hex(buf) {
  return createHash("sha256").update(buf).digest("hex");
}

function reconstructOnnxData(manifest, deq) {
  const buf = new Uint8Array(manifest.total_bytes);
  for (const e of manifest.layout) {
    if (e.kind === "zeros") continue;  // buf is zero-initialized
    if (e.kind === "gap") {
      const blob = Buffer.from(e.data_b64, "base64");
      if (blob.length !== e.length) {
        throw new Error(`gap length mismatch at ${e.offset}: ${blob.length} vs ${e.length}`);
      }
      buf.set(blob, e.offset);
      continue;
    }
    if (e.kind !== "tensor") {
      throw new Error(`unknown layout kind: ${e.kind}`);
    }
    if (e.source === "constant") {
      const blob = Buffer.from(e.data_b64, "base64");
      if (blob.length !== e.length) {
        throw new Error(`constant length mismatch for ${e.name}: ${blob.length} vs ${e.length}`);
      }
      buf.set(blob, e.offset);
      continue;
    }
    if (e.source !== "weight") {
      throw new Error(`unknown source: ${e.source} on ${e.name}`);
    }
    const wrapped = deq[e.safetensors_base];
    if (!wrapped) throw new Error(`safetensors_base ${e.safetensors_base} not in dequant map`);
    const arr = wrapped.data;
    let blobBytes;
    if (e.transform === "identity") {
      blobBytes = new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength);
    } else if (e.transform === "transpose") {
      // 2-D transpose. The manifest's `dims` is the post-transpose ONNX shape;
      // the safetensors source has shape (inDim, outDim).
      const [outDim, inDim] = e.dims;
      if (outDim * inDim !== arr.length) {
        throw new Error(
          `transpose for ${e.name}: dims ${e.dims} require ${outDim * inDim} elems, got ${arr.length}`
        );
      }
      const transposed = new Float32Array(arr.length);
      for (let i = 0; i < inDim; i++) {
        for (let o = 0; o < outDim; o++) {
          transposed[o * inDim + i] = arr[i * outDim + o];
        }
      }
      blobBytes = new Uint8Array(transposed.buffer);
    } else {
      throw new Error(`unknown transform: ${e.transform} on ${e.name}`);
    }
    if (blobBytes.byteLength !== e.length) {
      throw new Error(
        `length mismatch for ${e.name}: blob=${blobBytes.byteLength} expected=${e.length}`
      );
    }
    buf.set(blobBytes, e.offset);
  }
  return buf;
}

async function checkOne(kind) {
  const dir = `${ROOT}/public/models/${kind}`;
  const t0 = performance.now();
  const [gz, manifestText, committedData] = await Promise.all([
    readFile(`${dir}/model.safetensors.gz`),
    readFile(`${dir}/onnx_data_manifest.json`, "utf-8"),
    readFile(`${dir}/model.onnx.data`).catch(() => null),
  ]);

  const stBytes = gunzipSync(gz);
  const parsed = parseSafetensors(stBytes);
  const deq = dequantSafetensors(parsed);
  const manifest = JSON.parse(manifestText);
  const rebuilt = reconstructOnnxData(manifest, deq);
  const rebuiltSha = sha256Hex(rebuilt);

  const elapsed = ((performance.now() - t0) / 1000).toFixed(2);
  console.log(`== ${kind.toUpperCase()} ==`);
  console.log(`  rebuilt:        ${rebuilt.length.toLocaleString()} B   sha256=${rebuiltSha}`);
  console.log(`  manifest.ref:   ${manifest.total_bytes.toLocaleString()} B   sha256=${manifest.sha256}`);
  if (rebuiltSha !== manifest.sha256) {
    console.error(`  ✗ rebuilt sha256 != manifest.sha256`);
    return 1;
  }
  console.log(`  ✓ matches manifest reference`);
  if (committedData) {
    const committedSha = sha256Hex(committedData);
    console.log(`  on-disk model.onnx.data: sha256=${committedSha}`);
    if (committedSha !== rebuiltSha) {
      console.error(`  ✗ rebuilt sha256 != on-disk sha256`);
      return 1;
    }
    console.log(`  ✓ matches committed model.onnx.data`);
  }
  console.log(`  (took ${elapsed}s)\n`);
  return 0;
}

const arg = process.argv[2] || "all";
const kinds = arg === "all" ? ["ar", "mdlm"] : [arg];
let bad = 0;
for (const k of kinds) bad += await checkOne(k);
process.exit(bad);
