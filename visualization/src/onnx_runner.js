// Wraps onnxruntime-web. Same models, same weights as the hand-JS forward;
// the JS-bound graph runner is much faster (~21× vs hand-JS for AR).
//
// Artifact pipeline:
//
//   - Disk: model.onnx (graph, ~1 MB) + model.safetensors.gz (≤16 MiB) +
//           onnx_data_manifest.json (~88 KB).
//   - Browser: dequantize safetensors once (loader.js) → reconstruct
//     model.onnx.data into a Uint8Array (onnx_data.js) → hand it to
//     ort.InferenceSession.create({ externalData: [{ path, data }] }).
//   - The same dequantized Float32Arrays back the hand-JS forward, so
//     we pay the dequant cost once.
//
// EP fallback: WebGPU can fail at session-create time *or* at run time
// (lost device, shader compile failure, OOM, driver bug). We try the
// preferred EP order at create time, and on a run-time error we tear the
// session down and rebuild on the next EP. The rebuild is gated so we don't
// thrash if WASM also fails.

// IMPORTANT: keep this import first — ort_setup.js pins wasmPaths *before*
// transformers.js's module-level defaulting code can run. See
// src/ort_setup.js for the rationale.
import { ort } from "./ort_setup.js";
import { reconstructOnnxData } from "./onnx_data.js";

// (wasmPaths is pinned in src/ort_setup.js — must run before transformers.js)

const wantedEPs = (() => {
  const params = typeof window !== "undefined" ? new URLSearchParams(window.location.search) : null;
  const ep = params?.get("ep");
  if (ep === "webgpu") return ["webgpu", "wasm"];
  if (ep === "wasm") return ["wasm"];
  return ["wasm"];
})();

// Per-kind session state. We cache the ONNX graph + reconstructed external
// data buffer so a runtime fallback only pays for the InferenceSession.create
// call, not the dequant.
const sessions = {};
const epAttempts = {};   // kind → number of EPs already tried (0..wantedEPs.length)

export function getActiveEP() {
  return Object.values(sessions)[0]?._ep ?? wantedEPs[0];
}

async function _materialize(kind, baseUrl, tensors) {
  // Returns { onnxBuf, externalData } — the inputs we need to recreate the
  // session on a different EP without re-fetching anything from the network
  // or re-running the dequant.
  if (sessions[kind]?._materialized) return sessions[kind]._materialized;
  const [onnxBuf, manifest] = await Promise.all([
    fetch(`${baseUrl}/model.onnx`).then((r) => r.arrayBuffer()),
    fetch(`${baseUrl}/onnx_data_manifest.json`).then((r) => r.json()),
  ]);
  const onnxDataBytes = reconstructOnnxData(manifest, tensors);
  return { onnxBuf, onnxDataBytes };
}

async function _createOnEP(onnxBuf, onnxDataBytes, ep) {
  return await ort.InferenceSession.create(onnxBuf, {
    executionProviders: [ep],
    externalData: [{ data: onnxDataBytes, path: "model.onnx.data" }],
  });
}

/** Create (or fetch from cache) an ORT session for `kind`. Tries `wantedEPs`
 *  in order until one succeeds. */
export async function loadSession(kind, baseUrl, tensors) {
  if (sessions[kind]) return sessions[kind];

  const materialized = await _materialize(kind, baseUrl, tensors);

  let session = null;
  let chosen = null;
  let i = 0;
  for (; i < wantedEPs.length; i++) {
    const ep = wantedEPs[i];
    try {
      session = await _createOnEP(materialized.onnxBuf, materialized.onnxDataBytes, ep);
      chosen = ep;
      break;
    } catch (e) {
      console.warn(`ORT EP ${ep} failed at create for ${kind}:`, e.message ?? e);
    }
  }
  if (!session) throw new Error(`could not create ORT session for ${kind}`);
  session._ep = chosen;
  session._materialized = materialized;
  console.log(`ORT session for ${kind}: ${chosen}`);
  sessions[kind] = session;
  epAttempts[kind] = i + 1;  // index of next EP to try if run-time fallback fires
  return session;
}

/** Run the session. On error, if the active EP is not the last one in
 *  wantedEPs, tear down and rebuild on the next EP, then retry once. */
async function _runWithFallback(kind, baseUrl, tensors, feeds, outName) {
  const session = await loadSession(kind, baseUrl, tensors);
  try {
    const out = await session.run(feeds);
    return { data: out[outName].data, dims: out[outName].dims };
  } catch (e) {
    const failedEP = session._ep;
    const nextIdx = epAttempts[kind];
    if (nextIdx >= wantedEPs.length) {
      // No more EPs to try.
      throw e;
    }
    console.warn(
      `ORT run failed on ${failedEP} for ${kind}; falling back to ${wantedEPs[nextIdx]}:`,
      e.message ?? e
    );
    // Keep the materialized graph + external data; only the session gets rebuilt.
    const materialized = session._materialized;
    try { await session.release?.(); } catch {}
    sessions[kind] = null;

    const newSession = await _createOnEP(
      materialized.onnxBuf, materialized.onnxDataBytes, wantedEPs[nextIdx],
    );
    newSession._ep = wantedEPs[nextIdx];
    newSession._materialized = materialized;
    sessions[kind] = newSession;
    epAttempts[kind] = nextIdx + 1;
    console.log(`ORT session for ${kind} re-created on ${newSession._ep}`);
    const out = await newSession.run(feeds);
    return { data: out[outName].data, dims: out[outName].dims };
  }
}

export async function arForwardOnnx(tokenIds, tensors) {
  const T = tokenIds.length;
  const ids = new ort.Tensor(
    "int64", BigInt64Array.from(tokenIds.map((v) => BigInt(v))), [1, T]
  );
  return _runWithFallback("ar", "models/ar", tensors, { input_ids: ids }, "logits");
}

export async function mdlmForwardOnnx(tokenIds, mask, tensors) {
  const T = tokenIds.length;
  const ids = new ort.Tensor(
    "int64", BigInt64Array.from(tokenIds.map((v) => BigInt(v))), [1, T]
  );
  const maskTensor = new ort.Tensor(
    "int64",
    BigInt64Array.from(mask.length === T ? mask : new Uint8Array(T), (v) => BigInt(v)),
    [1, T]
  );
  return _runWithFallback(
    "mdlm", "models/mdlm", tensors,
    { input_ids: ids, mask: maskTensor }, "logits"
  );
}
