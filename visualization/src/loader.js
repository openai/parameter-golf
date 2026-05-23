// Load a model artifact pair:
//   <baseUrl>/model.json              — small (a few hundred bytes), {arch, label, ...}
//   <baseUrl>/model.safetensors.gz    — ≤16 MiB, gzipped safetensors with the
//                                       quantized state dict (int8 + fp16 scale
//                                       + fp32 passthrough for AR; raw fp8_e4m3fn
//                                       + fp32 passthrough for MDLM)
//
// Returns:
//   { manifest, tensors: { [name]: { data: Float32Array, shape: number[] } } }
//
// `tensors` is the dequantized state dict, byte-for-byte identical to what
// `pack_safetensors.py` and `reconstruct_onnx_data.py` produce. The hand-JS
// forward (src/ar.js, src/mdlm.js) consumes it directly; the ORT path
// reconstructs `model.onnx.data` from these tensors plus the
// `onnx_data_manifest.json` (see src/onnx_data.js).

import { parseSafetensors, dequantSafetensors } from "./safetensors.js";

async function fetchOk(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`failed to fetch ${url}: ${r.status}`);
  return r;
}

export async function loadModel(baseUrl) {
  const manifest = await (await fetchOk(`${baseUrl}/model.json`)).json();

  const gzRes = await fetchOk(`${baseUrl}/model.safetensors.gz`);
  // DecompressionStream("gzip") is the boring path: built-in to all modern
  // browsers since 2022 (Chrome 80, Firefox 113, Safari 16.4) and globally
  // available in Node 22+.
  const decompressed = gzRes.body.pipeThrough(new DecompressionStream("gzip"));
  const stBytes = new Uint8Array(await new Response(decompressed).arrayBuffer());
  const parsed = parseSafetensors(stBytes);
  const tensors = dequantSafetensors(parsed);

  return { manifest, tensors };
}

// Load a trace bundle (trace.bin + trace.json). Returns a Map of name -> {data, shape}.
export async function loadTrace(binUrl, jsonUrl) {
  const manifest = await (await fetch(jsonUrl)).json();
  const buf = await (await fetch(binUrl)).arrayBuffer();
  const out = new Map();
  for (const entry of manifest.tensors) {
    const nElems = entry.nelems;
    const view = new Float32Array(buf, entry.offset, nElems);
    out.set(entry.name, { data: new Float32Array(view), shape: entry.shape });
  }
  return { manifest, tensors: out };
}

// Load a tokens.json (the saved tokenization of the canonical input).
export async function loadTokens(url) {
  return await (await fetch(url)).json();
}

// Load masks.bin (header + t_values + KxN bool bytes). Returns
// {K, n, eps, tValues: Float32Array, masks: Uint8Array(K*n)}
export async function loadMasks(url) {
  const buf = await (await fetch(url)).arrayBuffer();
  const dv = new DataView(buf);
  // magic "PGMASK\0\0" (8B), uint32 K (LE), uint32 n (LE), float32 eps (LE), float32 _ (LE)
  const magic = new TextDecoder().decode(new Uint8Array(buf, 0, 8));
  if (magic !== "PGMASK\0\0") throw new Error(`bad magic: ${JSON.stringify(magic)}`);
  const K = dv.getUint32(8, true);
  const n = dv.getUint32(12, true);
  const eps = dv.getFloat32(16, true);
  const tValues = new Float32Array(buf, 24, K).slice();
  const masks = new Uint8Array(buf, 24 + 4 * K, K * n).slice();
  return { K, n, eps, tValues, masks };
}
