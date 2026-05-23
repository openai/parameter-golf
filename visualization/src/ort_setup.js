// Side-effect-only module: pin ORT-Web's wasm + WebGPU env *before* anything
// imports `@huggingface/transformers` or `onnxruntime-web`. Must be the first
// import in any file that touches either, because transformers.js has
// module-level code (`backends/onnx.js`) that defaults `ONNX_ENV.wasm.wasmPaths`
// to the asyncify wasm on cdn.jsdelivr.net the moment its module is evaluated.
// Asyncify ships only the GBQ schema, no kernel — see WASM EP and
// GatherBlockQuantized in PROGRESS.md.

import * as ort from "onnxruntime-web/webgpu";

// Pin to the jsep wasm — has GBQ kernels for both WASM and WebGPU EPs.
//   ort-wasm-simd-threaded.wasm           12 MB  plain CPU EP, has GBQ
//   ort-wasm-simd-threaded.jsep.wasm      24 MB  JSEP, has GBQ on both EPs
//   ort-wasm-simd-threaded.asyncify.wasm  22 MB  schema only, no GBQ kernel
//   ort-wasm-simd-threaded.jspi.wasm      13 MB  schema only, no GBQ kernel
const _ortWasmBase = new URL("./onnx-wasm/", document.baseURI).href;
ort.env.wasm.wasmPaths = {
  wasm: _ortWasmBase + "ort-wasm-simd-threaded.jsep.wasm",
  mjs:  _ortWasmBase + "ort-wasm-simd-threaded.jsep.mjs",
};

// Block the wasted asyncify/jspi prefetch. `onnxruntime-web/webgpu`'s bundle
// pre-fetches `ort-wasm-simd-threaded.asyncify.{wasm,mjs}` from cdn.jsdelivr
// regardless of how we pin wasmPaths (the bundle has its own fallback path).
// The asyncify wasm doesn't ship the GatherBlockQuantized kernel and we run
// on the jsep wasm anyway, so the prefetch is ~22 MB of pure waste. Short-
// circuit it with a synthetic 404 — the bundle logs a warning and proceeds.
if (typeof window !== "undefined") {
  const _origFetch = window.fetch.bind(window);
  window.fetch = function (input, init) {
    const url = typeof input === "string" ? input : (input && input.url) || "";
    if (
      url.includes("ort-wasm-simd-threaded.asyncify") ||
      url.includes("ort-wasm-simd-threaded.jspi")
    ) {
      return Promise.resolve(new Response("", {
        status: 404,
        statusText: "intentionally blocked: jsep wasm pinned, asyncify unused",
      }));
    }
    return _origFetch(input, init);
  };
}

export { ort };
