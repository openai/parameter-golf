// SmolLM2-135M-Instruct via Transformers.js, used to compute per-token NLL
// on a text input → bits-per-byte for the visualization.
//
// Built by scripts/quantize_smollm2.py following the recipe from sidechat's
// SmolLM2-360M-quantization.md: int4 weights (uint8 packed) for both
// `MatMulNBits` (attention/MLP) and `GatherBlockQuantized` (embedding +
// lm_head, byte-aliased so the embedding is on disk exactly once). All
// other tensors are fp32 — there is no fp64 anywhere.
//
// 77 MB on disk vs HF's pre-built model_q4f16.onnx (117 MB); BPB on the
// canonical Transformer abstract: 1.03 vs HF's 0.95 — the embedding's int4
// rounding error compounds through the network.
//
// Runtime: works on either ORT-Web EP (WASM CPU or WebGPU). The plain
// `ort-wasm-simd-threaded.wasm` registers the GatherBlockQuantized kernel
// for the CPU EP just fine; transformers.js's default-for-`device:'wasm'`
// asyncify wasm is the one that lacks it. We pin ORT to the jsep wasm in
// onnx_runner.js so both EPs see the kernel.
//
// Strategy:
//   - Load model_q4f16.onnx + tokenizer from /models/smollm2-135m/.
//   - Tokenize text via the model's own HF tokenizer (NOT our SP tokenizer).
//   - Run a single forward pass.
//   - For each token i in 1..n-1, NLL = -log_softmax(logits[i-1])[token_id_i].
//   - Map per-token NLL onto bytes via tokenizer offsets.

// IMPORTANT: keep this side-effect import first — ort_setup.js pins
// wasmPaths to the jsep wasm. transformers.js's `backends/onnx.js` has
// module-level code that, on first import, defaults ONNX_ENV.wasm.wasmPaths
// to the asyncify wasm on cdn.jsdelivr.net if it isn't already set. We need
// to set it first. See src/ort_setup.js + PROGRESS.md "WASM EP and GBQ".
import "./ort_setup.js";

import { AutoTokenizer, AutoModelForCausalLM, env } from "@huggingface/transformers";
import { topKFromLogits, rankOfId } from "./topk.js";

// Tell transformers.js to load from our local path rather than the HF CDN.
// transformers.js resolves `localModelPath + model_id + "/" + file` against
// `document.baseURI`, so a relative `./models/` works at any sub-path.
env.allowRemoteModels = false;
env.allowLocalModels = true;
env.localModelPath = "./models/";
// Don't write to the browser cache — we already have everything on disk.
env.useBrowserCache = false;
env.useCustomCache = false;
env.useFSCache = false;

const MODEL_ID = "smollm2-135m";  // → /models/smollm2-135m/

let _tokenizer = null;
let _model = null;
let _loadingPromise = null;

export function smollm2State() {
  return {
    tokenizerLoaded: !!_tokenizer,
    modelLoaded: !!_model,
  };
}

// Generation reaches in for the loaded singletons; loadSmolLM2 must have
// already populated them.
export function _modelRef() { return _model; }
export function _tokenizerRef() { return _tokenizer; }

// Cache decoded pieces by id for the popover. SmolLM2's vocab is ~49k; on a
// few hundred tokens we'll only ever decode the actual + top-K we surface,
// so this is at most ~10× hot tokens.
const _smollm2PieceCache = new Map();
export function smollm2Piece(id) {
  if (_smollm2PieceCache.has(id)) return _smollm2PieceCache.get(id);
  if (!_tokenizer) throw new Error("call loadSmolLM2 first");
  const piece = _tokenizer.decode([id], { skip_special_tokens: false });
  _smollm2PieceCache.set(id, piece);
  return piece;
}

async function _loadOnDevice(device, onProgress) {
  return await AutoModelForCausalLM.from_pretrained(MODEL_ID, {
    // dtype: "q4" picks model_q4.onnx (int4 weights, fp32 activations).
    dtype: "q4",
    device,
    // Our weights are split into 4 external-data chunks
    // (model_q4.onnx_data, _data_1, _data_2, _data_3) — matches the
    // ChunkWriter convention in scripts/quantize_smollm2.py.
    use_external_data_format: 4,
    progress_callback: (p) => onProgress({ status: "model", info: p }),
  });
}

export async function loadSmolLM2(opts = {}) {
  if (_loadingPromise) return _loadingPromise;
  _loadingPromise = (async () => {
    const onProgress = opts.onProgress ?? (() => {});
    onProgress({ status: "tokenizer", info: `loading ${MODEL_ID} tokenizer` });
    _tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID, {
      progress_callback: (p) => onProgress({ status: "tokenizer", info: p }),
    });

    // WebGPU is opt-in (?ep=webgpu); default to WASM (which now has GBQ
    // because we pin ORT to the jsep wasm in onnx_runner.js). If WebGPU is
    // requested but unavailable, fall back to WASM transparently.
    const preferred = opts.device ?? "wasm";
    onProgress({ status: "model", info: `loading ${MODEL_ID} (q4, ${preferred})` });
    try {
      _model = await _loadOnDevice(preferred, onProgress);
      _model._device = preferred;
    } catch (e) {
      if (preferred !== "wasm") {
        console.warn(`SmolLM2 ${preferred} load failed, falling back to wasm:`, e.message ?? e);
        onProgress({ status: "model", info: `${preferred} failed; loading on wasm` });
        _model = await _loadOnDevice("wasm", onProgress);
        _model._device = "wasm";
      } else {
        throw e;
      }
    }
    console.log(`SmolLM2 device: ${_model._device}`);
    onProgress({ status: "ready", info: "ready" });
    return { tokenizer: _tokenizer, model: _model };
  })();
  return _loadingPromise;
}

// Per-byte bits using the SmolLM2 model. Returns:
//   {
//     n_bytes, n_tokens,
//     bpb,
//     per_byte: Float64Array of length n_bytes,
//     per_token_bits: number[] of length n_tokens (first token = log2(vocab)),
//     token_offsets: [start_byte, end_byte] per token,
//   }
const _SMOLLM2_HEARTBEAT_MS = 30;
const _POPOVER_K = 10;
const _POPOVER_THRESH = 1e-3;
async function _smollm2Yield() {
  await new Promise((r) => setTimeout(r, 0));
}

export async function smollm2BPB(text, opts = {}) {
  if (!_model || !_tokenizer) throw new Error("call loadSmolLM2 first");

  const enc = _tokenizer(text, {
    return_offsets_mapping: true,
    return_tensor: false,
  });
  // Some tokenizers don't expose offsets via the call API. Fall back to encode().
  let inputIds = enc.input_ids;
  let offsets = enc.offset_mapping;
  if (!Array.isArray(inputIds)) {
    // Tensor returned; pull data
    inputIds = Array.from(inputIds.data ?? inputIds);
  }
  if (!offsets) {
    // Recompute via the tokenizer's encode method on the original text.
    const eo = _tokenizer.encode(text, { add_special_tokens: false });
    inputIds = eo.tokens?.ids ?? eo.ids ?? eo;
    offsets = eo.offsets ?? null;
  }

  const n = inputIds.length;
  const textBytes = new TextEncoder().encode(text);
  const nBytes = textBytes.length;
  const vocabSize = _model.config.vocab_size;

  if (n < 2) {
    const per_byte = new Float64Array(nBytes).fill(8.0);
    return {
      n_bytes: nBytes, n_tokens: n,
      bpb: 8.0,
      per_byte, per_token_bits: [Math.log2(vocabSize)],
      token_offsets: offsets ?? [],
      vocab_size: vocabSize, per_token: [],
    };
  }

  // Run a single causal forward pass on the whole sequence.
  // We use the model's `forward` directly with input_ids tensor.
  // Transformers.js expects an object input.
  const { Tensor } = await import("@huggingface/transformers");
  const idsTensor = new Tensor("int64",
    BigInt64Array.from(inputIds.map((v) => BigInt(v))),
    [1, n]);
  // SmolLM2's exported ONNX requires attention_mask and position_ids inputs.
  // For a single non-padded sequence we pass all-1 attention and 0..n-1 positions.
  const attnMask = new Tensor("int64",
    BigInt64Array.from({ length: n }, () => 1n), [1, n]);
  const posIds = new Tensor("int64",
    BigInt64Array.from({ length: n }, (_, i) => BigInt(i)), [1, n]);
  const out = await _model({
    input_ids: idsTensor,
    attention_mask: attnMask,
    position_ids: posIds,
  });
  // out.logits has shape [1, n, vocab_size], dtype float32 typically
  const logits = out.logits;  // Transformers Tensor
  const logitsData = logits.data;  // Float32Array of length n*V
  const V = logits.dims[2];
  if (V !== vocabSize) {
    console.warn(`logits vocab ${V} != config ${vocabSize}, using logits dim`);
  }

  // Pre-compute the byte span for each token once. With offsets we use them;
  // otherwise we decode token by token and lay them out sequentially.
  const spans = new Array(n);  // [startB, endB] per token, or null
  if (offsets && offsets.length === n) {
    for (let ti = 0; ti < n; ti++) {
      const [startC, endC] = offsets[ti];
      if (startC === endC) { spans[ti] = null; continue; }
      const startB = new TextEncoder().encode(text.slice(0, startC)).length;
      const endB = new TextEncoder().encode(text.slice(0, endC)).length;
      spans[ti] = endB > startB ? [startB, Math.min(endB, nBytes)] : null;
    }
  } else {
    let cursor = 0;
    for (let ti = 0; ti < n; ti++) {
      const piece = _tokenizer.decode([inputIds[ti]], { skip_special_tokens: false });
      const blen = new TextEncoder().encode(piece).length;
      if (blen <= 0) { spans[ti] = null; continue; }
      const startB = cursor, endB = Math.min(cursor + blen, nBytes);
      spans[ti] = [startB, endB];
      cursor += blen;
    }
  }

  // Walk tokens left-to-right. For each token, compute log_softmax of its
  // predictor row, write the per-byte rate into perByte, and emit a
  // throttled onProgress so the bar lights up + flashes for that token.
  const perByte = new Float64Array(nBytes);
  const covered = new Uint8Array(nBytes);
  const tokenBits = new Array(n);
  tokenBits[0] = Math.log2(V);
  // Per-token rows for the popover; first token has no predictive row.
  const perToken = new Array(n);
  perToken[0] = {
    byteStart: spans[0]?.[0] ?? 0,
    byteEnd: spans[0]?.[1] ?? 0,
    id: inputIds[0], piece: smollm2Piece(inputIds[0]),
    bits: tokenBits[0], rank: null, topk: null, aboveThresh: null,
    firstToken: true,
  };
  // The very first token is uniform-prior; if it has a span, paint it.
  if (spans[0]) {
    const [s, e] = spans[0];
    const bpe = tokenBits[0] / (e - s);
    for (let b = s; b < e; b++) { perByte[b] = bpe; covered[b] = 1; }
  }

  const baseLabel = "SmolLM2-135M";
  const baseColor = "#22c55e";
  const snapshot = (kDone) => {
    let total = 0;
    let cov = 0;
    for (let b = 0; b < nBytes; b++) { total += perByte[b]; if (covered[b]) cov++; }
    return {
      bpb: cov > 0 ? total / cov : 0,
      per_byte: perByte,
      n_tokens: kDone,
      label: baseLabel,
      color: baseColor,
      vocab_size: V,
      per_token: perToken,
      progress: { k: kDone, K: n, kind: "tokens" },
    };
  };

  // Per-token bookkeeping is microseconds; a wall-clock heartbeat alone won't
  // fire often enough to show progress. Force a token-count batch too.
  const BATCH = Math.max(1, Math.ceil((n - 1) / 30));
  let lastTick = performance.now();
  for (let i = 1; i < n; i++) {
    const row = logitsData.subarray((i - 1) * V, i * V);
    const { topk, aboveThresh, lse } = topKFromLogits(row, V, _POPOVER_K, _POPOVER_THRESH);
    const logp = row[inputIds[i]] - lse;
    tokenBits[i] = -logp / Math.log(2);
    const rank = rankOfId(row, V, inputIds[i]);

    perToken[i] = {
      byteStart: spans[i]?.[0] ?? 0,
      byteEnd: spans[i]?.[1] ?? 0,
      id: inputIds[i], piece: smollm2Piece(inputIds[i]),
      bits: tokenBits[i], rank, topk, aboveThresh,
    };

    const sp = spans[i];
    if (sp) {
      const [s, e] = sp;
      const bpe = tokenBits[i] / (e - s);
      for (let b = s; b < e; b++) { perByte[b] = bpe; covered[b] = 1; }
    }

    if ((i % BATCH === 0) || (performance.now() - lastTick >= _SMOLLM2_HEARTBEAT_MS)) {
      opts.onProgress?.(snapshot(i + 1));
      await _smollm2Yield();
      lastTick = performance.now();
    }
  }

  // Final pass: fill any uncovered bytes with the nearest neighbour, matching
  // the prior behaviour.
  let lastVal = 0;
  for (let b = 0; b < nBytes; b++) {
    if (covered[b]) lastVal = perByte[b];
    else perByte[b] = lastVal;
  }

  let totalBits = 0;
  for (let b = 0; b < nBytes; b++) totalBits += perByte[b];
  const bpb = nBytes > 0 ? totalBits / nBytes : 0;

  return {
    n_bytes: nBytes,
    n_tokens: n,
    bpb,
    per_byte: perByte,
    per_token_bits: tokenBits,
    token_offsets: offsets,
    vocab_size: V,
    per_token: perToken,
  };
}
