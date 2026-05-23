// Per-method "compute per-byte BPB on this text" wrappers. Each returns
// { bpb, per_byte: Float64Array of length n_bytes, n_tokens, label, color }.
//
// Heavy loops (windowed gzip per-byte, K-sample MDLM) accept opts.onProgress
// — called with a partial result object of the same shape every ~30 ms so
// the UI can repaint per-byte bars as work proceeds. Internally they yield
// to the event loop on the same cadence so the page stays interactive.

const HEARTBEAT_MS = 30;

// AR and MDLM were trained on isolated 1024-token windows. For longer inputs,
// chop into independent non-overlapping chunks and score each on its own —
// this matches the training distribution exactly (same chunk size, same
// bidirectional/causal context inside each), with no scoring-window or
// double-counting decisions to make. The cost is that boundary tokens lose
// cross-chunk context, identical to how every standard chunked-perplexity
// implementation behaves.
const CHUNK_SIZE = 1024;

function _chunks(n) {
  const out = [];
  for (let start = 0; start < n; start += CHUNK_SIZE) {
    out.push({ start, end: Math.min(start + CHUNK_SIZE, n) });
  }
  if (out.length === 0) out.push({ start: 0, end: 0 });
  return out;
}

async function _yield() {
  // Hand the event loop a chance to repaint and run other tasks. setTimeout(0)
  // is the boring way; rAF would tie us to vsync (~16 ms) which can be too
  // tight for inner loops we don't want to gate on the compositor.
  await new Promise((r) => setTimeout(r, 0));
}

import { loadTokenizer, encodeFull, pieceForId } from "./tokenizer.js";
import { mdlmForward } from "./mdlm.js";
import { arForward } from "./ar.js";
import { logSoftmax } from "./ops.js";
import { loadSmolLM2, smollm2BPB } from "./smollm2.js";
import { arForwardOnnx, mdlmForwardOnnx } from "./onnx_runner.js";
import { topKFromLogits, rankOfId } from "./topk.js";

// Top-K + above-threshold settings for the popover. K=10 follows the spec.
// PROB_THRESH = 1e-3 (anything ≥ 0.1% counts as a plausible alternative);
// counter intuition: for a sharp distribution this is a small handful, for a
// flat one it can run into the thousands — exactly what we want to surface.
const POPOVER_K = 10;
const POPOVER_THRESH = 1e-3;

// Toggle backends. ONNX is faster (WASM-SIMD optimized matmul) but the
// hand-rolled JS forward is the parity oracle.
export const BACKEND = { ar: "onnx", mdlm: "onnx" };

function logitsFromAr(weights, arch, tokenIds) {
  if (BACKEND.ar === "onnx") {
    return arForwardOnnx(tokenIds, weights);  // returns {data, dims}
  }
  const t = arForward(weights, arch, tokenIds);
  return Promise.resolve({ data: t.data, dims: t.shape });
}

function logitsFromMdlm(weights, arch, tokenIds, mask) {
  if (BACKEND.mdlm === "onnx") {
    return mdlmForwardOnnx(tokenIds, mask, weights);
  }
  const t = mdlmForward(weights, arch, tokenIds, mask);
  return Promise.resolve({ data: t.data, dims: t.shape });
}

// ---- gzip (DEFLATE via CompressionStream) ----
//
// Marginal bits/byte against the full prefix: for each byte i we compress
// text[0..i+1) and subtract the compressed size of text[0..i). The delta (in
// bits) is what DEFLATE spent on byte i given everything before it. No
// attribution windowing — repeated paragraphs anywhere earlier in the text
// reduce later bytes' cost (deflate's own back-reference window is 32 KiB).
// Cost is O(N²) bytes compressed; for an N≈3300 input that's ~22 MB, still
// well under a second on modern browsers.

async function deflateLen(bytes) {
  const cs = new CompressionStream("deflate-raw");
  const writer = cs.writable.getWriter();
  writer.write(bytes);
  writer.close();
  const reader = cs.readable.getReader();
  let total = 0;
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    total += value.length;
  }
  return total;
}

export async function gzipBPB(textBytes, opts = {}) {
  const totalLen = await deflateLen(textBytes);
  const totalBPB = (totalLen * 8) / textBytes.length;

  const n = textBytes.length;
  const perByte = new Float64Array(n);
  let bytesDone = 0;
  const baseResult = () => ({
    bpb: totalBPB, per_byte: perByte, n_tokens: 0,
    label: "gzip", color: "#6b7280",
    progress: { k: bytesDone, K: n, kind: "bytes" },
  });

  let lastTick = performance.now();
  let cPrefix = 0;  // bytes for text[0..i); seeded at i=0 with the empty deflate.
  for (let i = 0; i < n; i++) {
    const full = textBytes.subarray(0, i + 1);
    const cFull = await deflateLen(full);
    perByte[i] = Math.max(0, (cFull - cPrefix) * 8);
    cPrefix = cFull;  // tomorrow's prefix is today's full prefix.
    bytesDone = i + 1;
    const now = performance.now();
    if (now - lastTick >= HEARTBEAT_MS) {
      opts.onProgress?.(baseResult());
      await _yield();
      lastTick = performance.now();
    }
  }
  return baseResult();
}

// ---- shared: spread per-token bits onto bytes ----
function spreadTokenBitsToBytes(tokenBits, byteLens, nBytes) {
  const out = new Float64Array(nBytes);
  let cursor = 0;
  for (let i = 0; i < tokenBits.length; i++) {
    const bits = tokenBits[i];
    const blen = byteLens[i];
    if (blen > 0) {
      const per = bits / blen;
      const end = Math.min(cursor + blen, nBytes);
      for (let b = cursor; b < end; b++) out[b] = per;
      cursor += blen;
    }
  }
  return out;
}

// ---- AR ----
export async function arBPB(weights, arch, text, opts = {}) {
  await loadTokenizer("tokenizer/fineweb_1024_bpe.model");
  const { ids: tokenIds, byteLens } = encodeFull(text);
  const n = tokenIds.length;
  const textBytes = new TextEncoder().encode(text);
  const nBytes = textBytes.length;
  const baseLabel = arch.label ?? "AR";
  if (n < 2) {
    return { bpb: 8.0, per_byte: new Float64Array(nBytes).fill(8.0),
             n_tokens: n, label: baseLabel, color: "#3b82f6",
             vocab_size: arch.vocab_size, per_token: [], tokenizer: "fineweb_sp" };
  }

  const V = arch.vocab_size;
  const perByte = new Float64Array(nBytes);
  // Per-token byte spans — populated alongside writeToken so the popover can
  // resolve "which token is this character in?" without reconstructing it.
  const perToken = new Array(n).fill(null);
  let cursor = 0;
  const writeToken = (tokenIdx, bits, extra) => {
    const blen = byteLens[tokenIdx];
    const start = cursor;
    if (blen > 0) {
      const per = bits / blen;
      const end = Math.min(cursor + blen, nBytes);
      for (let b = cursor; b < end; b++) perByte[b] = per;
      cursor += blen;
    }
    perToken[tokenIdx] = {
      byteStart: start, byteEnd: cursor,
      id: tokenIds[tokenIdx], piece: pieceForId(tokenIds[tokenIdx]),
      bits, ...extra,
    };
  };

  const snapshot = (tokensDone) => {
    let total = 0;
    for (let b = 0; b < nBytes; b++) total += perByte[b];
    const cov = Math.max(1, cursor);
    return { bpb: total / cov, per_byte: perByte, n_tokens: tokensDone,
             label: baseLabel, color: "#3b82f6",
             vocab_size: V, per_token: perToken, tokenizer: "fineweb_sp",
             progress: { k: tokensDone, K: n, kind: "tokens" } };
  };

  const BATCH = Math.max(1, Math.ceil((n - 1) / 30));
  let lastTick = performance.now();
  // Independent 1024-token chunks. For inputs ≤ 1024 tokens this is the
  // exact same path as before. The first token of each chunk has no in-chunk
  // left context and is charged the uniform-prior cost — same convention as
  // token 0 of the document had previously.
  for (const chunk of _chunks(n)) {
    if (chunk.end - chunk.start < 1) continue;
    writeToken(chunk.start, Math.log2(V), { rank: null, topk: null, aboveThresh: null, firstOfChunk: true });
    if (chunk.end - chunk.start < 2) continue;

    // Feed tokens[start..end-2], get logits for positions [start+1..end-1].
    const inputIds = tokenIds.slice(chunk.start, chunk.end - 1);
    const fl = await logitsFromAr(weights, arch, inputIds);
    const rawShape = [chunk.end - chunk.start - 1, V];
    const lp = logSoftmax({ data: fl.data, shape: rawShape });
    for (let i = chunk.start + 1; i < chunk.end; i++) {
      const relPos = i - chunk.start;            // 1..chunkLen-1
      const lpRow = lp.data.subarray((relPos - 1) * V, relPos * V);
      const rawRow = fl.data.subarray((relPos - 1) * V, relPos * V);
      const nllNats = -lpRow[tokenIds[i]];
      // topKFromLogits accepts raw logits; using rawRow is equivalent and
      // avoids the redundant LSE we'd compute on lpRow (whose values are
      // already log-probs, but the helper does its own LSE pass).
      const { topk, aboveThresh } = topKFromLogits(rawRow, V, POPOVER_K, POPOVER_THRESH);
      const rank = rankOfId(rawRow, V, tokenIds[i]);
      writeToken(i, nllNats / Math.log(2), { rank, topk, aboveThresh });
      if ((i % BATCH === 0) || (performance.now() - lastTick >= HEARTBEAT_MS)) {
        opts.onProgress?.(snapshot(i + 1));
        await _yield();
        lastTick = performance.now();
      }
    }
  }

  let total = 0;
  for (let i = 0; i < nBytes; i++) total += perByte[i];
  return { bpb: total / nBytes, per_byte: perByte, n_tokens: n,
           label: baseLabel, color: "#3b82f6",
           vocab_size: V, per_token: perToken, tokenizer: "fineweb_sp" };
}

// ---- MDLM ----
//
// K-step any-order autoregressive (AOAR) BPB with MaskGIT-style parallel
// commit. We start with all positions masked and run exactly K diffusion
// steps. After step k (0-indexed) we want round((k+1)·N/K) positions
// committed; the new positions are picked by descending top-1 confidence
// from the current forward pass. The committed positions' bits are
// −log₂ p_θ(x_i | x_committed_before_this_step), which is exact by
// construction (and the popover top-K is the same distribution).
//
// K = N → pure AOAR (one position per step).
// K = 1 → single forward, all positions committed in parallel from prompt
//   only (Hoogeboom 2021 ARDM "in-parallel" scoring; tightly equivalent to
//   the all-masked single-pass option).
// In between: parallel commit, with a small bias from the conditional-
// independence assumption between sibling commits in the same step.
export async function mdlmBPB(weights, arch, text, opts = {}) {
  const K = opts.K ?? 16;
  const forceFirstUnmasked = !!opts.forceFirstUnmasked;
  await loadTokenizer("tokenizer/fineweb_1024_bpe.model");
  const { ids: tokenIds, byteLens } = encodeFull(text);
  const n = tokenIds.length;
  const textBytes = new TextEncoder().encode(text);
  const nBytes = textBytes.length;
  const baseLabel = arch.label ?? "MDLM";
  if (n < 1) {
    return { bpb: 8.0, per_byte: new Float64Array(nBytes).fill(8.0),
             n_tokens: n, label: baseLabel, color: "#f97316",
             vocab_size: arch.vocab_size, per_token: [], tokenizer: "fineweb_sp" };
  }

  const V = arch.vocab_size;

  // Per-token rows: byte spans + identity, populated as commits land.
  const perToken = new Array(n);
  {
    let cursor = 0;
    for (let i = 0; i < n; i++) {
      perToken[i] = {
        byteStart: cursor,
        byteEnd: Math.min(cursor + byteLens[i], nBytes),
        id: tokenIds[i], piece: pieceForId(tokenIds[i]),
        bits: 0, rank: null, topk: null, aboveThresh: null,
        commitStep: null, committedAt: null,  // committedAt = positions committed before this one
      };
      cursor += byteLens[i];
    }
  }

  // Running snapshot. Match AR's convention: bpb = (committed bits) /
  // (committed bytes). Uncommitted tokens contribute 0 bits and are excluded
  // from the denominator entirely, so the running bpb is always a valid
  // average over what's actually been scored. Bars for uncommitted bytes
  // stay at 0 opacity (faint) — same as AR's running visualization.
  function snapshot(stepDone, committedSoFar) {
    const tokenBits = perToken.map((t) => t.bits);
    const pb = spreadTokenBitsToBytes(tokenBits, byteLens, nBytes);
    let totalBits = 0;
    let committedBytes = 0;
    for (let i = 0; i < n; i++) {
      if (perToken[i].committedAt != null) {
        totalBits += tokenBits[i];
        committedBytes += byteLens[i];
      }
    }
    return { bpb: committedBytes > 0 ? totalBits / committedBytes : 0,
             per_byte: pb, n_tokens: n,
             label: baseLabel, color: "#f97316",
             vocab_size: V, per_token: perToken, tokenizer: "fineweb_sp",
             progress: { k: stepDone, K, kind: "diffusion steps" } };
  }

  let lastTick = performance.now();

  // Walk chunks independently. Each chunk runs its own K-step diffusion: at
  // step k we want round((k+1)·cLen/K) positions committed; the new ones are
  // the top-(target − already_committed) by confidence in the current pass.
  for (const c of _chunks(n)) {
    const cLen = c.end - c.start;
    if (cLen < 1) continue;
    const chunkIds = tokenIds.slice(c.start, c.end);
    const mask = new Uint8Array(cLen).fill(1);
    let committedInChunk = 0;
    let totalCommittedAcrossChunks = 0;

    for (let step = 0; step < K; step++) {
      const targetCommitted = Math.round(((step + 1) * cLen) / K);
      const toCommit = Math.max(0, Math.min(cLen, targetCommitted) - committedInChunk);
      if (toCommit === 0) continue;

      const fl = await logitsFromMdlm(weights, arch, chunkIds, mask);

      // Score every still-masked position by top-1 confidence.
      const candidates = [];
      for (let i = 0; i < cLen; i++) {
        if (!mask[i]) continue;
        const row = fl.data.subarray(i * V, (i + 1) * V);
        let m = -Infinity;
        for (let j = 0; j < V; j++) if (row[j] > m) m = row[j];
        let s = 0;
        for (let j = 0; j < V; j++) s += Math.exp(row[j] - m);
        const top1Prob = 1 / s;
        candidates.push({ i, top1Prob, row });
      }
      // Sort by descending confidence.
      candidates.sort((a, b) => b.top1Prob - a.top1Prob);

      // Pick which positions to commit this step.
      const pickIdx = new Set();
      for (let r = 0; r < Math.min(toCommit, candidates.length); r++) {
        pickIdx.add(candidates[r].i);
      }
      // Force-decode the leftmost still-masked position. If it's not already
      // in the commit set, evict the lowest-confidence pick for it.
      if (forceFirstUnmasked) {
        let leftmost = -1;
        for (let i = 0; i < cLen; i++) {
          if (mask[i]) { leftmost = i; break; }
        }
        if (leftmost >= 0 && !pickIdx.has(leftmost)) {
          // Find the picked candidate with the lowest confidence and replace it.
          let worst = null;
          for (const cand of candidates) {
            if (!pickIdx.has(cand.i)) continue;
            if (!worst || cand.top1Prob < worst.top1Prob) worst = cand;
          }
          if (worst) pickIdx.delete(worst.i);
          pickIdx.add(leftmost);
        }
      }

      // Score and commit each picked position. The model's distribution at
      // pick time is conditioned on x_committed_before_this_step, so the
      // popover top-K matches: bits = -log2 p_actual = -log2 of one of the
      // entries in topk (or beyond, if rank > popover-K). Within a step,
      // give parallel commits sequential `committedAt` values in descending-
      // confidence order so each row gets a distinct "committed #N" — they
      // all share `commitStep` because they were scored under the same
      // mask, but the linear order across the chunk stays unique.
      let pickedThisStep = 0;
      for (const cand of candidates) {
        if (!pickIdx.has(cand.i)) continue;
        const { i, row } = cand;
        const { topk, aboveThresh, lse } =
          topKFromLogits(row, V, POPOVER_K, POPOVER_THRESH);
        const actualLp = row[chunkIds[i]] - lse;
        const aoarBits = -actualLp / Math.log(2);
        const rank = rankOfId(row, V, chunkIds[i]);

        const tokIdx = c.start + i;
        perToken[tokIdx].bits = aoarBits;
        perToken[tokIdx].rank = rank;
        perToken[tokIdx].topk = topk;
        perToken[tokIdx].aboveThresh = aboveThresh;
        perToken[tokIdx].commitStep = step;
        perToken[tokIdx].committedAt = totalCommittedAcrossChunks +
                                       committedInChunk + pickedThisStep;
        pickedThisStep += 1;
      }
      // Apply the unmasks after scoring so within-step commits all see the
      // same context (parallel-commit semantics, identical for all of them).
      for (const cand of candidates) {
        if (!pickIdx.has(cand.i)) continue;
        mask[cand.i] = 0;
        committedInChunk += 1;
      }

      if (performance.now() - lastTick >= HEARTBEAT_MS) {
        opts.onProgress?.(snapshot(step + 1, committedInChunk));
        await _yield();
        lastTick = performance.now();
      }
    }
    totalCommittedAcrossChunks += committedInChunk;
  }

  const tokenBits = perToken.map((t) => t.bits);
  const perByte = spreadTokenBitsToBytes(tokenBits, byteLens, nBytes);
  let total = 0;
  for (let i = 0; i < nBytes; i++) total += perByte[i];
  return { bpb: total / nBytes, per_byte: perByte, n_tokens: n,
           label: baseLabel, color: "#f97316",
           vocab_size: V, per_token: perToken, tokenizer: "fineweb_sp",
           progress: { k: K, K, kind: "diffusion steps" } };
}

// ---- SmolLM2 ----
export async function smollm2BPBWrap(text, opts = {}) {
  await loadSmolLM2();
  const r = await smollm2BPB(text, { onProgress: opts.onProgress });
  return {
    bpb: r.bpb, per_byte: r.per_byte, n_tokens: r.n_tokens,
    label: "SmolLM2-135M", color: "#22c55e",
    vocab_size: r.vocab_size, per_token: r.per_token, tokenizer: "smollm2_hf",
  };
}
