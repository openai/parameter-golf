// Generative side of the visualizer. Four samplers:
//   gzipGenerate    — byte-level compression sampler over /[a-z .]/, P ∝ 2^(-bits).
//   arGenerate      — causal AR, no KV cache (re-forward each step). O(N²).
//   mdlmGenerate    — bidirectional MDLM, iterative denoising at T = L,
//                     confidence-order single-position-per-step.
//   smollm2Generate — direct ORT-Web with manual KV cache.
//
// All four call opts.onToken(piece, fullText) per emitted/committed token so
// the UI can stream text into the output row with the same flash effect used
// in BPB mode.

import { loadTokenizer, encode, decode, getAllPieces, pieceForId } from "./tokenizer.js";
import { arForwardOnnx, mdlmForwardOnnx } from "./onnx_runner.js";
import { loadSmolLM2, smollm2Piece } from "./smollm2.js";
import { topKFromLogits, rankOfId } from "./topk.js";

const _GEN_K = 10;
const _GEN_THRESH = 1e-3;

async function _yield() { await new Promise((r) => setTimeout(r, 0)); }

// Length of deflate-raw output for the input bytes (zero if empty input).
async function _deflateLen(bytes) {
  if (bytes.length === 0) return 0;
  const cs = new CompressionStream("deflate-raw");
  const w = cs.writable.getWriter();
  w.write(bytes);
  w.close();
  const r = cs.readable.getReader();
  let total = 0;
  while (true) {
    const { value, done } = await r.read();
    if (done) break;
    total += value.length;
  }
  return total;
}

// Sample from a logits row with the standard "fix the repetition loop" knobs:
//   - repetitionPenalty  ≈ 1.15 (HF-style: divide positive logits / multiply
//                                 negative logits at recently-seen IDs).
//   - topP               ≈ 0.9  (nucleus — keep smallest set whose cumulative
//                                 prob ≥ p, then sample inside that set).
//   - temperature        ≈ 0.8  (0 ⇒ argmax of the post-penalty/post-top-p
//                                 distribution).
//
// Order matters: penalize first (operates on raw logits), then temperature
// scale, then nucleus truncation, then sample. Mirrors transformers.js's
// LogitsProcessor pipeline.
function sampleLogits(row, opts = {}) {
  const temperature = opts.temperature ?? 1.0;
  const topP = opts.topP ?? 1.0;
  const repetitionPenalty = opts.repetitionPenalty ?? 1.0;
  const recentIds = opts.recentIds;
  const V = row.length;

  // Copy so we don't mutate the model's logits buffer.
  const logits = new Float32Array(V);
  for (let i = 0; i < V; i++) logits[i] = row[i];

  // Repetition penalty.
  if (repetitionPenalty !== 1.0 && recentIds && recentIds.length > 0) {
    const seen = new Set(recentIds);
    for (const id of seen) {
      if (id < 0 || id >= V) continue;
      logits[id] = logits[id] >= 0 ? logits[id] / repetitionPenalty
                                    : logits[id] * repetitionPenalty;
    }
  }

  if (temperature <= 0) {
    // Greedy argmax over the (penalized) logits.
    let best = -Infinity, bestI = 0;
    for (let i = 0; i < V; i++) if (logits[i] > best) { best = logits[i]; bestI = i; }
    return bestI;
  }

  // Apply temperature, then softmax to probabilities.
  const inv = 1 / temperature;
  let m = -Infinity;
  for (let i = 0; i < V; i++) { logits[i] *= inv; if (logits[i] > m) m = logits[i]; }
  let s = 0;
  const probs = new Float64Array(V);
  for (let i = 0; i < V; i++) { probs[i] = Math.exp(logits[i] - m); s += probs[i]; }
  for (let i = 0; i < V; i++) probs[i] /= s;

  // Top-p nucleus truncation (only if topP < 1).
  if (topP < 1.0 && topP > 0) {
    // Sort indices by descending prob; walk cumulative; zero out the tail.
    const order = new Int32Array(V);
    for (let i = 0; i < V; i++) order[i] = i;
    // Partial sort would be cheaper, but for V=1024 (AR/MDLM) and V≈49k
    // (SmolLM2) full sort is still microseconds.
    const idxArr = Array.from(order);
    idxArr.sort((a, b) => probs[b] - probs[a]);
    let cum = 0;
    let cutoff = idxArr.length;
    for (let i = 0; i < idxArr.length; i++) {
      cum += probs[idxArr[i]];
      if (cum >= topP) { cutoff = i + 1; break; }
    }
    const keep = new Uint8Array(V);
    for (let i = 0; i < cutoff; i++) keep[idxArr[i]] = 1;
    let sKeep = 0;
    for (let i = 0; i < V; i++) { if (!keep[i]) probs[i] = 0; else sKeep += probs[i]; }
    if (sKeep > 0) for (let i = 0; i < V; i++) probs[i] /= sKeep;
  }

  // Inverse-CDF sample.
  const r = Math.random();
  let acc = 0;
  for (let i = 0; i < V; i++) { acc += probs[i]; if (acc >= r) return i; }
  return V - 1;
}

// ---- AR ----
export async function arGenerate(weights, arch, promptText, opts = {}) {
  const { maxTokens = 64, temperature = 1.0, topP = 1.0, repetitionPenalty = 1.0, onToken, signal } = opts;
  // Penalize tokens within this window of "recently emitted" — both the prompt
  // (so the model isn't free to immediately echo it back) and the new tail.
  // 64 is a reasonable default; the HF default is 1.0 (off) but for small
  // models 1.15 is the typical low-touch start.
  const RECENT_WINDOW = 128;
  // AR was also trained on 1024-token windows. Truncate from the head so the
  // active prefix during generation never exceeds the trained context.
  const MAX_LEN = 1024;
  await loadTokenizer("tokenizer/fineweb_1024_bpe.model");
  const maxPromptIds = MAX_LEN - maxTokens;
  if (maxPromptIds <= 0) {
    throw new Error(`maxTokens=${maxTokens} leaves no room in the 1024-token AR window`);
  }
  const promptIdsFull = encode(promptText);
  const truncatedFromHead = Math.max(0, promptIdsFull.length - maxPromptIds);
  const promptIds = truncatedFromHead > 0
    ? promptIdsFull.slice(-maxPromptIds)
    : promptIdsFull;
  if (truncatedFromHead > 0) {
    console.warn(
      `AR: prompt truncated to last ${maxPromptIds} tokens ` +
      `(was ${promptIdsFull.length}, dropped ${truncatedFromHead} from head)`
    );
  }
  let ids = [...promptIds];
  // Decode prompt+tail together each step and slice off the prompt prefix —
  // SP's DecodeIds drops the leading ▁ marker on the very first piece, so
  // decoding only the tail loses the space between prompt and first new token
  // (you'd see "isa must" instead of "is a must"). Decoding the full ids
  // canonically resolves the boundary.
  let fullDecoded = decode(ids);
  const promptDecodedLen = fullDecoded.length;
  onToken?.("", fullDecoded, { phase: "prompt", k: 0, K: maxTokens, promptLen: promptDecodedLen });
  for (let step = 0; step < maxTokens; step++) {
    if (signal?.aborted) break;
    const fl = await arForwardOnnx(ids, weights);
    const V = arch.vocab_size;
    const T = ids.length;
    const lastRow = fl.data.subarray((T - 1) * V, T * V);
    const recentIds = ids.slice(Math.max(0, ids.length - RECENT_WINDOW));
    const tokenId = sampleLogits(lastRow, { temperature, topP, repetitionPenalty, recentIds });
    // Top-K is computed on the *raw* logits (no rep penalty / temperature /
    // top-p applied). The popover surfaces what the model actually thought,
    // not the post-processed sampling distribution.
    const { topk, aboveThresh, lse } = topKFromLogits(lastRow, V, _GEN_K, _GEN_THRESH);
    const rank = rankOfId(lastRow, V, tokenId);
    const chosenBits = -(lastRow[tokenId] - lse) / Math.log(2);
    ids.push(tokenId);
    const newFull = decode(ids);
    const piece = newFull.slice(fullDecoded.length);
    fullDecoded = newFull;
    onToken?.(piece, fullDecoded, {
      phase: "step", k: step + 1, K: maxTokens, promptLen: promptDecodedLen,
      tokenizer: "fineweb_sp", vocabSize: V,
      chosen: { id: tokenId, piece: pieceForId(tokenId), rank, bits: chosenBits },
      topk, aboveThresh,
    });
    await _yield();
  }
  return { promptText, generated: fullDecoded.slice(promptDecodedLen), ids };
}

// ---- MDLM ----
//
// Confidence-order, one-position-per-step iterative denoising. T defaults to L
// (one position revealed per pass). At each step:
//   1. forward(prompt + current_state) where MASK positions still have masks
//   2. for each still-masked position, compute top-1 prob; pick the most
//      confident position
//   3. sample (or argmax) a token there, commit it, never revisit
//
// The MDLM ONNX takes (input_ids[1,T], mask[1,T]). For positions where mask=1
// the model substitutes the [MASK] embedding internally — the input_id at
// those positions is ignored, so we just put 0 there.
export async function mdlmGenerate(weights, arch, promptText, opts = {}) {
  const { maxTokens = 64, temperature = 0, topP = 1.0, repetitionPenalty = 1.0, onToken, signal } = opts;
  await loadTokenizer("tokenizer/fineweb_1024_bpe.model");
  // MDLM was trained on 1024-token windows. Generation needs prompt + L mask
  // slots to fit within the window — if the prompt is longer than (1024 - L),
  // truncate from the LEFT (keep the tail) so the masks always sit at the
  // far right where the model expects them. The tokenizer's job is encode-
  // only here; we slice on token IDs.
  const MAX_LEN = 1024;
  const L = maxTokens;
  const maxPromptIds = MAX_LEN - L;
  if (maxPromptIds <= 0) {
    throw new Error(`maxTokens=${L} leaves no room in the 1024-token MDLM window`);
  }
  const promptIdsFull = encode(promptText);
  const truncatedFromHead = Math.max(0, promptIdsFull.length - maxPromptIds);
  const promptIds = truncatedFromHead > 0
    ? promptIdsFull.slice(-maxPromptIds)
    : promptIdsFull;
  if (truncatedFromHead > 0) {
    console.warn(
      `MDLM: prompt truncated to last ${maxPromptIds} tokens ` +
      `(was ${promptIdsFull.length}, dropped ${truncatedFromHead} from head)`
    );
  }
  const total = promptIds.length + L;
  // ids: prompt ids followed by L placeholder zeros (will be ignored where masked).
  const ids = new Array(total);
  for (let i = 0; i < promptIds.length; i++) ids[i] = promptIds[i];
  for (let i = promptIds.length; i < total; i++) ids[i] = 0;
  // mask[i] = 1 ⇒ position i is currently masked.
  const mask = new Uint8Array(total);
  for (let i = promptIds.length; i < total; i++) mask[i] = 1;
  const revealed = new Array(L).fill(null);  // ids committed at each suffix position
  // Per-position popover payload, built up alongside `revealed` so the suffix
  // plan we ship in onToken can carry per-token top-K to the renderer without
  // a side-channel.
  const revealedData = new Array(L).fill(null);

  // Decode prompt alone for the canonical boundary (same trick as arGenerate).
  const promptDecoded = decode(promptIds);
  onToken?.("", promptDecoded, { phase: "prompt", k: 0, K: L, promptLen: promptDecoded.length });
  let prevText = "";
  for (let step = 0; step < L; step++) {
    if (signal?.aborted) break;
    const fl = await mdlmForwardOnnx(ids, mask, weights);
    const V = arch.vocab_size;
    // Pick the most confident still-masked position. Confidence = max
    // softmax prob at that row, computed via log-sum-exp.
    let bestPos = -1, bestProb = -Infinity, bestRow = null;
    for (let i = promptIds.length; i < total; i++) {
      if (!mask[i]) continue;
      const row = fl.data.subarray(i * V, (i + 1) * V);
      let m = -Infinity;
      for (let j = 0; j < V; j++) if (row[j] > m) m = row[j];
      let s = 0;
      for (let j = 0; j < V; j++) s += Math.exp(row[j] - m);
      const top1Prob = 1 / s;  // Math.exp(m - (m + Math.log(s)))
      if (top1Prob > bestProb) { bestProb = top1Prob; bestPos = i; bestRow = row; }
    }
    if (bestPos < 0) break;
    // For MDLM, "recent" = prompt + everything committed so far in the suffix.
    const committed = revealed.filter((x) => x != null);
    const recentIds = [...promptIds, ...committed];
    const tokenId = sampleLogits(bestRow, { temperature, topP, repetitionPenalty, recentIds });
    const { topk, aboveThresh, lse } = topKFromLogits(bestRow, V, _GEN_K, _GEN_THRESH);
    const rank = rankOfId(bestRow, V, tokenId);
    const chosenBits = -(bestRow[tokenId] - lse) / Math.log(2);
    ids[bestPos] = tokenId;
    mask[bestPos] = 0;
    revealed[bestPos - promptIds.length] = tokenId;
    revealedData[bestPos - promptIds.length] = {
      chosen: { id: tokenId, piece: pieceForId(tokenId), rank, bits: chosenBits },
      topk, aboveThresh,
      tokenizer: "fineweb_sp", vocabSize: V,
    };

    // Build the per-position render plan. Runs of consecutive revealed tokens
    // are decoded with the prompt as context (so SP's leading ▁ on the first
    // piece of the run survives — without that we'd see "isrecorded" instead
    // of "is recorded"). Within each run we split the decoded text per-token
    // by walking prefix decodes; this gives every committed slot its own
    // span for the popover while preserving inter-token spacing.
    const plan = [];
    let runIds = [], runStart = -1;
    const flushRun = () => {
      if (runIds.length === 0) return;
      let prevLen = 0;
      for (let i = 0; i < runIds.length; i++) {
        const decoded = decode([...promptIds, ...runIds.slice(0, i + 1)])
          .slice(promptDecoded.length);
        const text = decoded.slice(prevLen);
        const pos = runStart + i;
        const data = revealedData[pos] || {};
        plan.push({ kind: "tok", text, pos,
                    chosen: data.chosen, topk: data.topk,
                    aboveThresh: data.aboveThresh,
                    tokenizer: data.tokenizer, vocabSize: data.vocabSize });
        prevLen = decoded.length;
      }
      runIds = [];
      runStart = -1;
    };
    for (let i = 0; i < L; i++) {
      const t = revealed[i];
      if (t == null) {
        flushRun();
        plan.push({ kind: "placeholder", text: " _", pos: i });
      } else {
        if (runIds.length === 0) runStart = i;
        runIds.push(t);
      }
    }
    flushRun();
    const suffixText = plan.map((p) => p.text).join("");
    const fullText = promptDecoded + suffixText;
    const piece = suffixText.slice(prevText.length);
    prevText = suffixText;
    onToken?.(piece, fullText, {
      phase: "step", k: step + 1, K: L,
      promptLen: promptDecoded.length,
      revealedPos: bestPos - promptIds.length,
      tokenizer: "fineweb_sp", vocabSize: V,
      chosen: { id: tokenId, piece: pieceForId(tokenId), rank, bits: chosenBits },
      topk, aboveThresh,
      suffixPlan: plan,
    });
    await _yield();
  }
  return { promptText, generated: prevText, ids: revealed.filter((x) => x != null) };
}

// ---- SmolLM2 ----
//
// We can't use `model.generate({ streamer })` for streaming because the WASM
// EP's `session.run` returns a Promise wrapping a *synchronous* WASM call.
// The await in transformers.js's gen loop yields to microtasks only, and
// setTimeout-based DOM updates run as macrotasks — microtasks always
// pre-empt, so the streamer's callbacks pile up and the browser only repaints
// after generate() fully returns ("renders all at once").
//
// Roll our own forward → sample → emit → setTimeout(0) loop instead. We lose
// transformers.js's KV-cache management (every step does a full forward on
// the growing prefix; O(N²) work overall), but gain real task yields between
// tokens, which is what makes the streaming visible. For 64-token demos at
// 135M params with int4 weights this is fine; if generation length grows
// past ~256 we'd want to wire up KV cache by hand.
export async function smollm2Generate(promptText, opts = {}) {
  const { maxTokens = 64, temperature = 1.0, topP = 1.0, repetitionPenalty = 1.0,
          onToken, onLoadingProgress, signal } = opts;
  const RECENT_WINDOW = 128;
  const transformers = await import("@huggingface/transformers");
  const Tensor = transformers.Tensor;
  await loadSmolLM2({ onProgress: onLoadingProgress });
  const { _model_ref, _tokenizer_ref } = await _getSmolLM2Refs();
  const tokenizer = _tokenizer_ref;
  const model = _model_ref;

  // Tokenize the prompt to token IDs (plain numbers, no tensors).
  const enc = tokenizer(promptText, { return_tensor: false });
  let ids = Array.from(enc.input_ids ?? []);
  if (ids.length === 0) {
    onToken?.("", promptText, { phase: "prompt", k: 0, K: maxTokens, promptLen: promptText.length });
    return { promptText, generated: "" };
  }

  const promptLen = promptText.length;
  onToken?.("", promptText, { phase: "prompt", k: 0, K: maxTokens, promptLen });

  // EOS handling: tokenizer exposes eos_token_id as either a number or a list.
  const eosRaw = tokenizer.eos_token_id ?? model.config?.eos_token_id;
  const eosSet = new Set(Array.isArray(eosRaw) ? eosRaw : (eosRaw != null ? [eosRaw] : []));

  // KV-cache plumbing. Our ONNX was re-exported via
  // scripts/export_smollm2_with_kv.py — that wrapper monkey-patches optimum's
  // `patched_dynamic_layer_update` to always trace the
  // `torch.cat([self.keys, key_states], dim=-2)` branch, so each layer's
  // `present.{i}.{key,value}` is now CUMULATIVE (shape
  // [B, H, past_seq + new_seq, D]). That means we can simply pass them
  // straight back as `past_key_values.{i}.{key,value}` for the next step
  // without any JS-side concatenation.
  //
  // Step 0: prefill with empty past_kvs ([1,3,0,64]) + full prompt.
  // Step k (k>0): single new token + position_ids=[kvLen] + attn covering
  // kvLen+1 + past_kvs from previous step's present.
  const cfg = model.config ?? {};
  const numLayers = cfg.num_hidden_layers ?? 30;
  const numKvHeads = cfg.num_key_value_heads ?? 3;
  const headDim = (cfg.hidden_size ?? 576) / (cfg.num_attention_heads ?? 9);

  function emptyKvShape() { return [1, numKvHeads, 0, headDim]; }
  function makeEmptyKvs() {
    const kvs = {};
    for (let i = 0; i < numLayers; i++) {
      kvs[`past_key_values.${i}.key`] =
        new Tensor("float32", new Float32Array(0), emptyKvShape());
      kvs[`past_key_values.${i}.value`] =
        new Tensor("float32", new Float32Array(0), emptyKvShape());
    }
    return kvs;
  }
  function presentToPast(out) {
    const kvs = {};
    for (let i = 0; i < numLayers; i++) {
      const k = out[`present.${i}.key`];
      const v = out[`present.${i}.value`];
      if (!k || !v) return null;
      kvs[`past_key_values.${i}.key`] = k;
      kvs[`past_key_values.${i}.value`] = v;
    }
    return kvs;
  }

  let pastKvs = makeEmptyKvs();
  let kvLen = 0;
  let cachedOK = true;
  let prevSuffix = "";
  for (let step = 0; step < maxTokens; step++) {
    if (signal?.aborted) break;
    let inputIds, positionIds, attnMask, totalLen;
    if (step === 0) {
      inputIds = ids;
      const T = ids.length;
      positionIds = Array.from({ length: T }, (_, i) => BigInt(i));
      attnMask = Array.from({ length: T }, () => 1n);
      totalLen = T;
    } else if (cachedOK) {
      inputIds = [ids[ids.length - 1]];
      positionIds = [BigInt(kvLen)];
      attnMask = Array.from({ length: kvLen + 1 }, () => 1n);
      totalLen = kvLen + 1;
    } else {
      inputIds = ids;
      const T = ids.length;
      positionIds = Array.from({ length: T }, (_, i) => BigInt(i));
      attnMask = Array.from({ length: T }, () => 1n);
      totalLen = T;
    }
    const T = inputIds.length;
    const idsTensor = new Tensor("int64",
      BigInt64Array.from(inputIds.map((v) => BigInt(v))), [1, T]);
    const posT = new Tensor("int64", BigInt64Array.from(positionIds), [1, T]);
    const attnT = new Tensor("int64", BigInt64Array.from(attnMask), [1, totalLen]);
    // CRITICAL: transformers.js's decoder_forward destructures
    // `model_inputs.past_key_values` and then calls addPastKeyValues. If
    // past_key_values is undefined/empty, addPastKeyValues silently
    // overwrites any flat past_key_values.X.{key,value} entries in the feeds
    // with zero-filled empty tensors. So we must pass the cache as the
    // single `past_key_values` field — a bare object with own enumerable
    // keys works (Object.assigned onto feeds).
    const out = await model({
      input_ids: idsTensor,
      attention_mask: attnT,
      position_ids: posT,
      ...(cachedOK ? { past_key_values: pastKvs } : {}),
    });
    const logits = out.logits;
    const V = logits.dims[2];
    const Tout = logits.dims[1];
    const lastRow = logits.data.subarray((Tout - 1) * V, Tout * V);
    const recentIds = ids.slice(Math.max(0, ids.length - RECENT_WINDOW));
    const tokenId = sampleLogits(lastRow, { temperature, topP, repetitionPenalty, recentIds });
    const { topk, aboveThresh, lse } = topKFromLogits(lastRow, V, _GEN_K, _GEN_THRESH);
    const rank = rankOfId(lastRow, V, tokenId);
    const chosenBits = -(lastRow[tokenId] - lse) / Math.log(2);
    if (eosSet.has(tokenId)) break;
    ids.push(tokenId);
    if (cachedOK) {
      const next = presentToPast(out);
      if (next) {
        pastKvs = next;
        kvLen = totalLen;
      } else {
        console.warn("SmolLM2: present.* outputs missing; falling back to no-cache decode.");
        cachedOK = false;
      }
    }
    const suffix = tokenizer.decode(ids.slice(ids.length - (step + 1)), {
      skip_special_tokens: true,
    });
    const piece = suffix.slice(prevSuffix.length);
    prevSuffix = suffix;
    onToken?.(piece, promptText + suffix, {
      phase: "step", k: step + 1, K: maxTokens, promptLen,
      tokenizer: "smollm2_hf", vocabSize: V,
      chosen: { id: tokenId, piece: smollm2Piece(tokenId), rank, bits: chosenBits },
      topk, aboveThresh,
    });
    await new Promise((r) => setTimeout(r, 0));  // task yield ⇒ browser repaints
  }
  return { promptText, generated: prevSuffix };
}

// Pry out the singletons stored in smollm2.js without exposing them publicly.
// loadSmolLM2 already populated them; we just reach them through a side-effect
// helper added to that module.
async function _getSmolLM2Refs() {
  const mod = await import("./smollm2.js");
  return {
    _model_ref: mod._modelRef(),
    _tokenizer_ref: mod._tokenizerRef(),
  };
}

// ---- gzip ----
//
// Compression-as-language-model. We commit to "gzip is the model" and frame
// generation as: at each step, pick a variable-length continuation (1–5
// bytes) drawn from a pool of linguistically-real fragments, scored by
// gzip's per-byte bit cost. Two problems with naive enumeration:
//   (1) `_deflateLen` returns BYTES — bit-level differences round away,
//       making most random byte sequences score identically.
//   (2) The byte cost is non-monotonic in input length (adding more can
//       compress to fewer bytes), so brute-force enumeration over the
//       28-letter alphabet picks encoder-quirk wins like "mmea" over real
//       prompt repetitions.
//
// Fix: restrict candidates to byte sequences that are linguistically real:
//   - Every length-n substring of (prompt + already-generated bytes), for
//     n ∈ {1..5}. These are LZ77 back-reference targets.
//   - Every BPE token from the FineWeb 1024-vocab tokenizer that decodes to
//     1–5 bytes. Captures common English fragments — " the", "tion", "ing"
//     — that gzip's bit-cost wouldn't necessarily favor on novel substrings,
//     but are linguistically natural.
// Random-noise n-grams like "mmea"/"aaaa" never enter the pool.
//
// Variable lengths compete via per-byte cost: bpb = 8·delta / length.
// Candidates with cheap LZ77 matches (low bpb) win regardless of length;
// among ties (e.g. slack-absorbed cost-0 candidates), sort by descending
// length so we prefer committing more bytes at once when free. Sampling
// uses logits = −bpb·ln(2) with temperature/top-p/rep-penalty applied
// uniformly. Each chunk commits its chosen length (1–5 bytes), and the
// loop runs until maxTokens bytes have been generated.

// Convert a sentencepiece piece to its byte form. ▁ is SP's word-boundary
// marker — render it as a leading space.
function _pieceBytes(piece) {
  return new TextEncoder().encode(piece.replace(/▁/g, " "));
}

const MAX_LEN = 5;

// DRY ("Don't Repeat Yourself") sampler — modern n-gram-aware repetition
// control. For each candidate, find the longest sequence (q bytes of context
// tail + p bytes of candidate prefix) that already appears earlier in the
// search window; subtract `multiplier * base^(match_len - allowed)` from the
// logit. We restrict the search window to the GENERATED portion only, so
// prompt-substring back-references aren't double-penalized.
const DRY_BASE = 1.75;
const DRY_ALLOWED_LENGTH = 2;
const DRY_MAX_BACKWARD = 16;

function _dryMatchLen(genStr, candStr, maxBack) {
  const N = genStr.length;
  if (N === 0) return 0;
  let bestN = 0;
  const candLen = candStr.length;
  const qLimit = Math.min(maxBack, N);
  for (let q = 0; q <= qLimit; q++) {
    if (q + candLen <= bestN) continue;
    const tail = q > 0 ? genStr.slice(N - q) : "";
    for (let p = candLen; p >= 1; p--) {
      if (q + p <= bestN) break;
      const queryStr = tail + candStr.slice(0, p);
      // Match must end before the trailing q bytes (else it's the cursor itself).
      // Since query is q+p > q bytes, indexOf can't return position N-q.
      if (genStr.indexOf(queryStr) >= 0) {
        bestN = q + p;
        break;
      }
    }
  }
  return bestN;
}

export async function gzipGenerate(promptText, opts = {}) {
  const { maxTokens = 64, temperature = 1.0, topP = 1.0,
          repetitionPenalty = 1.0, onToken, signal } = opts;
  const RECENT_WINDOW = 128;

  await loadTokenizer("tokenizer/fineweb_1024_bpe.model");

  const seedText = promptText.toLowerCase();
  const seedBytes = Array.from(new TextEncoder().encode(seedText));
  const bytes = [...seedBytes];

  const promptLen = seedText.length;
  onToken?.("", seedText, { phase: "prompt", k: 0, K: maxTokens, promptLen });

  // Static portion of the candidate pool: every BPE piece that decodes to
  // 1..MAX_LEN bytes. Pieces longer than MAX_LEN are skipped entirely (we
  // don't truncate, since a truncated piece isn't a real linguistic unit).
  const tokenSeeds = [];
  const tokenKeys = new Set();
  for (const piece of getAllPieces()) {
    const b = _pieceBytes(piece);
    if (b.length < 1 || b.length > MAX_LEN) continue;
    const key = String.fromCharCode(...b);
    if (tokenKeys.has(key)) continue;
    tokenKeys.add(key);
    tokenSeeds.push(b);
  }

  let prevSuffix = "";
  let chunkCount = 0;

  while (bytes.length - seedBytes.length < maxTokens) {
    if (signal?.aborted) break;
    if (chunkCount++ > maxTokens + 8) break; // safety: each chunk emits ≥1 byte

    const ctx = new Uint8Array(bytes);
    const cPrefix = await _deflateLen(ctx);

    // DRY multiplier maps the UI's "rep penalty" slider [1..2] onto DRY's
    // multiplier strength. 1.0 = off, 2.0 = strong (mult=1.0).
    const dryMultiplier = repetitionPenalty > 1.0 ? (repetitionPenalty - 1.0) : 0;
    const genBytes = bytes.slice(seedBytes.length);
    const genStr = (dryMultiplier > 0 && genBytes.length > 0)
      ? String.fromCharCode.apply(null, genBytes) : "";

    // Build pool: tokenizer pieces (static) + every length-n substring of
    // bytes-so-far for n ∈ {1..MAX_LEN}. Dedupe by exact byte key.
    const candidates = [];
    const stepKeys = new Set(tokenKeys);
    for (const t of tokenSeeds) candidates.push(t);
    for (let n = 1; n <= MAX_LEN; n++) {
      for (let i = 0; i + n <= bytes.length; i++) {
        let key = "";
        for (let j = 0; j < n; j++) key += String.fromCharCode(bytes[i + j]);
        if (stepKeys.has(key)) continue;
        stepKeys.add(key);
        const sub = new Uint8Array(n);
        for (let j = 0; j < n; j++) sub[j] = bytes[i + j];
        candidates.push(sub);
      }
    }

    // Score each: bpb = 8·delta_bytes / cand.length. Apply DRY penalty by
    // looking up the longest match of (gen-tail + cand-prefix) in genStr.
    // Length bonus (LENGTH_ALPHA · len) tiebreaks toward longer candidates
    // when bpb is tied at zero — without this, byte-quantization produces
    // hundreds of tied candidates at logit=0 and sampling treats them
    // uniformly, fragmenting the output into 1-byte chunks at low temp.
    // SCORE_SCALE = 20 makes the sampling 1/20 as hot at any given T — gzip's
    // logit gaps are small in nats (1 byte of cost = ln(2) nats), so by the
    // time temperature is applied the distribution is much flatter than an
    // LLM's. Multiplying logits by 20 stretches the gaps so T's effect feels
    // closer to LLM behavior.
    const LENGTH_ALPHA = 0.1;
    const SCORE_SCALE = 20;
    const C = candidates.length;
    const logits = new Float32Array(C);
    const buf = new Uint8Array(ctx.length + MAX_LEN);
    buf.set(ctx);
    for (let i = 0; i < C; i++) {
      const cand = candidates[i];
      for (let j = 0; j < cand.length; j++) buf[ctx.length + j] = cand[j];
      const cFull = await _deflateLen(buf.subarray(0, ctx.length + cand.length));
      const deltaBytes = Math.max(0, cFull - cPrefix);
      const bpb = (deltaBytes * 8) / cand.length;
      let logit = -bpb * Math.LN2 + LENGTH_ALPHA * cand.length;
      if (dryMultiplier > 0 && genStr.length > 0) {
        const candStr = String.fromCharCode.apply(null, Array.from(cand));
        const matchLen = _dryMatchLen(genStr, candStr, DRY_MAX_BACKWARD);
        if (matchLen > DRY_ALLOWED_LENGTH) {
          logit -= dryMultiplier * Math.pow(DRY_BASE, matchLen - DRY_ALLOWED_LENGTH);
        }
      }
      logits[i] = logit * SCORE_SCALE;
    }

    // Sample over the candidate pool by per-byte cost.
    let chosenIdx;
    if (temperature <= 0) {
      // Greedy: highest logit. Tie-break by longer length (more bytes / fewer
      // re-scoring rounds when free).
      let best = -Infinity, bestLen = 0;
      chosenIdx = 0;
      for (let i = 0; i < C; i++) {
        if (logits[i] > best ||
            (logits[i] === best && candidates[i].length > bestLen)) {
          best = logits[i]; bestLen = candidates[i].length; chosenIdx = i;
        }
      }
    } else {
      const inv = 1 / temperature;
      let m = -Infinity;
      for (let i = 0; i < C; i++) { logits[i] *= inv; if (logits[i] > m) m = logits[i]; }
      const probs = new Float64Array(C);
      let s = 0;
      for (let i = 0; i < C; i++) { probs[i] = Math.exp(logits[i] - m); s += probs[i]; }
      for (let i = 0; i < C; i++) probs[i] /= s;

      if (topP < 1.0 && topP > 0) {
        const order = Array.from({ length: C }, (_, i) => i);
        order.sort((a, b) => probs[b] - probs[a]);
        let cum = 0, cutoff = C;
        for (let i = 0; i < C; i++) {
          cum += probs[order[i]];
          if (cum >= topP) { cutoff = i + 1; break; }
        }
        const keep = new Uint8Array(C);
        for (let i = 0; i < cutoff; i++) keep[order[i]] = 1;
        let sKeep = 0;
        for (let i = 0; i < C; i++) {
          if (!keep[i]) probs[i] = 0;
          else sKeep += probs[i];
        }
        if (sKeep > 0) for (let i = 0; i < C; i++) probs[i] /= sKeep;
      }

      const r = Math.random();
      let acc = 0;
      chosenIdx = C - 1;
      for (let i = 0; i < C; i++) {
        acc += probs[i];
        if (acc >= r) { chosenIdx = i; break; }
      }
    }

    const chosen = candidates[chosenIdx];
    for (let j = 0; j < chosen.length; j++) bytes.push(chosen[j]);

    const suffix = new TextDecoder().decode(
      new Uint8Array(bytes.slice(seedBytes.length))
    );
    const piece = suffix.slice(prevSuffix.length);
    prevSuffix = suffix;
    onToken?.(piece, seedText + suffix, {
      phase: "step", k: bytes.length - seedBytes.length, K: maxTokens, promptLen,
    });
    await _yield();
  }
  return { promptText: seedText, generated: prevSuffix };
}
