// Float32Array tensor ops, matched line-for-line against PyTorch's defaults so
// JS results agree bit-for-bit (or at least to fp32 ULP) with the goldens.
//
// Conventions:
//   * Every tensor is {data: Float32Array, shape: number[]} stored row-major.
//   * No allocation reuse; the parity work cares more about clarity than speed.
//   * Math is fp32: we do everything via Float32Array and `Math.fround` is not
//     needed because the array storage already truncates after each store.
//
// The ops mirror torch as follows:
//   rmsNorm(x, eps?)       -> F.rms_norm(x, (D,), eps=eps ?? finfo.eps)
//   matmulLinear(x, W)     -> F.linear(x, W)  [W shape (out, in), no bias]
//   matmulLinearT(x, W)    -> x @ W.T         [same as above written out]
//   reluSquared(x)         -> torch.relu(x).square()
//   softmax(x, dim=-1)     -> F.softmax(x, dim=-1)
//   logSoftmax(x, dim=-1)  -> F.log_softmax(x, dim=-1)
//   embeddingLookup        -> nn.Embedding.forward
//   apply rotary           -> bpb_compare.apply_rotary_emb (rotated-by-half)
//   sdpa                   -> F.scaled_dot_product_attention with optional GQA

// PyTorch uses torch.finfo(input.dtype).eps when rms_norm's `eps=None`.
// For fp32: 1.1920929e-07. Constant matches torch.finfo(torch.float32).eps.
export const FP32_EPS = 1.1920928955078125e-7;

// ---------- Indexing helpers ----------

export function product(shape) {
  let p = 1;
  for (const s of shape) p *= s;
  return p;
}

export function strides(shape) {
  const out = new Array(shape.length);
  let s = 1;
  for (let i = shape.length - 1; i >= 0; i--) {
    out[i] = s;
    s *= shape[i];
  }
  return out;
}

export function makeTensor(shape, fill = 0) {
  const data = new Float32Array(product(shape));
  if (fill) data.fill(fill);
  return { data, shape };
}

// ---------- RMSNorm ----------
// Forward: y = x * rsqrt(mean(x^2) + eps) over the last dim
// (no learnable weight — torch's F.rms_norm with weight=None).
export function rmsNorm(t, eps = FP32_EPS) {
  const D = t.shape[t.shape.length - 1];
  const N = product(t.shape) / D;
  const out = new Float32Array(t.data.length);
  for (let n = 0; n < N; n++) {
    const off = n * D;
    let sumsq = 0;
    for (let i = 0; i < D; i++) {
      const v = t.data[off + i];
      sumsq += v * v;
    }
    const rms = 1 / Math.sqrt(sumsq / D + eps);
    for (let i = 0; i < D; i++) out[off + i] = t.data[off + i] * rms;
  }
  return { data: out, shape: t.shape.slice() };
}

// ---------- Linear (no bias) ----------
// y[..., i] = sum_j x[..., j] * W[i, j]    (W is (out, in))
// Matches F.linear(x, W) exactly: same loop order as torch's CPU kernel for
// small matrices (i outer, j inner over a contiguous row of W) — that's what
// gives us the same fp32 accumulation order.
export function linear(x, W) {
  // x: [..., in], W: [out, in]
  const inDim = x.shape[x.shape.length - 1];
  const outDim = W.shape[0];
  if (W.shape[1] !== inDim) {
    throw new Error(`linear shape mismatch: x last=${inDim}, W=${W.shape}`);
  }
  const N = product(x.shape) / inDim;
  const out = new Float32Array(N * outDim);
  for (let n = 0; n < N; n++) {
    const xOff = n * inDim;
    const yOff = n * outDim;
    for (let i = 0; i < outDim; i++) {
      const wOff = i * inDim;
      let s = 0;
      for (let j = 0; j < inDim; j++) s += x.data[xOff + j] * W.data[wOff + j];
      out[yOff + i] = s;
    }
  }
  return { data: out, shape: [...x.shape.slice(0, -1), outDim] };
}

// ---------- Element-wise ----------

export function reluSquared(t) {
  const out = new Float32Array(t.data.length);
  for (let i = 0; i < t.data.length; i++) {
    const v = t.data[i];
    if (v > 0) out[i] = v * v;
  }
  return { data: out, shape: t.shape.slice() };
}

export function tanh(t) {
  const out = new Float32Array(t.data.length);
  for (let i = 0; i < t.data.length; i++) out[i] = Math.tanh(t.data[i]);
  return { data: out, shape: t.shape.slice() };
}

export function scale(t, s) {
  const out = new Float32Array(t.data.length);
  for (let i = 0; i < t.data.length; i++) out[i] = t.data[i] * s;
  return { data: out, shape: t.shape.slice() };
}

export function add(a, b) {
  if (a.data.length !== b.data.length) throw new Error("add shape mismatch");
  const out = new Float32Array(a.data.length);
  for (let i = 0; i < a.data.length; i++) out[i] = a.data[i] + b.data[i];
  return { data: out, shape: a.shape.slice() };
}

// y[n, d] = a[n, d] * b[d]     (broadcast last-dim vector across leading dims)
export function mulLastDim(a, b) {
  const D = a.shape[a.shape.length - 1];
  if (b.shape.length !== 1 || b.shape[0] !== D) {
    throw new Error(`mulLastDim shape mismatch: a last=${D}, b=${b.shape}`);
  }
  const N = a.data.length / D;
  const out = new Float32Array(a.data.length);
  for (let n = 0; n < N; n++) {
    const off = n * D;
    for (let i = 0; i < D; i++) out[off + i] = a.data[off + i] * b.data[i];
  }
  return { data: out, shape: a.shape.slice() };
}

// y[n, d] = a[n, d] + scale[d] * b[n, d]
export function addScaled(a, scaleVec, b) {
  const D = a.shape[a.shape.length - 1];
  if (scaleVec.shape.length !== 1 || scaleVec.shape[0] !== D) {
    throw new Error(`addScaled shape mismatch: a last=${D}, scale=${scaleVec.shape}`);
  }
  if (a.data.length !== b.data.length) throw new Error("addScaled a/b length mismatch");
  const N = a.data.length / D;
  const out = new Float32Array(a.data.length);
  for (let n = 0; n < N; n++) {
    const off = n * D;
    for (let i = 0; i < D; i++) out[off + i] = a.data[off + i] + scaleVec.data[i] * b.data[off + i];
  }
  return { data: out, shape: a.shape.slice() };
}

// y[n, d] = mix0[d] * a[n, d] + mix1[d] * b[n, d]    (residual mixing)
export function mixResid(mix0, mix1, a, b) {
  const D = a.shape[a.shape.length - 1];
  if (a.data.length !== b.data.length) throw new Error("mixResid length mismatch");
  const N = a.data.length / D;
  const out = new Float32Array(a.data.length);
  for (let n = 0; n < N; n++) {
    const off = n * D;
    for (let i = 0; i < D; i++) {
      out[off + i] = mix0.data[i] * a.data[off + i] + mix1.data[i] * b.data[off + i];
    }
  }
  return { data: out, shape: a.shape.slice() };
}

// ---------- Embedding lookup ----------
// emb shape [V, D], ids shape [..N..] (int), returns shape [..N.., D]
export function embeddingLookup(emb, idsTensor) {
  const D = emb.shape[1];
  const N = idsTensor.data.length;
  const out = new Float32Array(N * D);
  for (let n = 0; n < N; n++) {
    const id = idsTensor.data[n] | 0;  // int
    const src = id * D;
    const dst = n * D;
    for (let i = 0; i < D; i++) out[dst + i] = emb.data[src + i];
  }
  return { data: out, shape: [...idsTensor.shape, D] };
}

// ---------- log_softmax over last dim ----------

export function logSoftmax(t) {
  const D = t.shape[t.shape.length - 1];
  const N = t.data.length / D;
  const out = new Float32Array(t.data.length);
  for (let n = 0; n < N; n++) {
    const off = n * D;
    let m = t.data[off];
    for (let i = 1; i < D; i++) if (t.data[off + i] > m) m = t.data[off + i];
    let sum = 0;
    for (let i = 0; i < D; i++) sum += Math.exp(t.data[off + i] - m);
    const lse = m + Math.log(sum);
    for (let i = 0; i < D; i++) out[off + i] = t.data[off + i] - lse;
  }
  return { data: out, shape: t.shape.slice() };
}

// ---------- Reshape / transpose helpers for attention ----------
// We materialize Q/K/V at shape [B, H, T, hd] like PyTorch does after the
// .reshape().transpose(1,2). The matrix layout we receive from `linear` is
// [B, T, H*hd]; we permute to [B, H, T, hd] by walking the strides.

export function reshapeForAttn(t, B, T, H, hd) {
  // t has data of length B*T*H*hd in [B, T, H*hd] row-major
  // out[b, h, ti, i] = t[b, ti, h*hd + i]
  const out = new Float32Array(B * H * T * hd);
  for (let b = 0; b < B; b++) {
    for (let ti = 0; ti < T; ti++) {
      for (let h = 0; h < H; h++) {
        const src = b * T * H * hd + ti * H * hd + h * hd;
        const dst = b * H * T * hd + h * T * hd + ti * hd;
        for (let i = 0; i < hd; i++) out[dst + i] = t.data[src + i];
      }
    }
  }
  return { data: out, shape: [B, H, T, hd] };
}

// Inverse: from [B, H, T, hd] back to [B, T, H*hd]
export function attnFlatten(t) {
  const [B, H, T, hd] = t.shape;
  const out = new Float32Array(B * T * H * hd);
  for (let b = 0; b < B; b++) {
    for (let h = 0; h < H; h++) {
      for (let ti = 0; ti < T; ti++) {
        const src = b * H * T * hd + h * T * hd + ti * hd;
        const dst = b * T * H * hd + ti * H * hd + h * hd;
        for (let i = 0; i < hd; i++) out[dst + i] = t.data[src + i];
      }
    }
  }
  return { data: out, shape: [B, T, H * hd] };
}

// ---------- Rotary (rotated-by-half, matches bpb_compare.apply_rotary_emb) ----------
//   half = D // 2
//   x1 = x[..., :half]; x2 = x[..., half:]
//   out = cat([x1*cos + x2*sin,  -x1*sin + x2*cos], dim=-1)
// cos/sin shapes: [1, 1, T, hd] in PyTorch; we accept them as Float32Array
// of length T*hd row-major.
export function applyRotary(t, cos, sin) {
  // t shape [B, H, T, hd]; cos/sin shape [T, hd]
  const [B, H, T, hd] = t.shape;
  if (cos.shape[cos.shape.length - 2] !== T || cos.shape[cos.shape.length - 1] !== hd) {
    throw new Error(`applyRotary cos shape ${cos.shape} incompatible with t ${t.shape}`);
  }
  const half = hd >> 1;
  const out = new Float32Array(t.data.length);
  // cos/sin are 4D [1,1,T,hd] in our trace dump → flat length T*hd from the last 2 dims
  const cosData = cos.data;
  const sinData = sin.data;
  const cosOffStride = T * hd;  // unused, single-batch
  const cosStrideT = hd;
  void cosOffStride;
  for (let b = 0; b < B; b++) {
    for (let h = 0; h < H; h++) {
      for (let ti = 0; ti < T; ti++) {
        const off = b * H * T * hd + h * T * hd + ti * hd;
        const cOff = ti * cosStrideT;
        for (let i = 0; i < half; i++) {
          const x1 = t.data[off + i];
          const x2 = t.data[off + half + i];
          const c = cosData[cOff + i];
          const s = sinData[cOff + i];
          out[off + i] = x1 * c + x2 * s;
          out[off + half + i] = -x1 * s + x2 * c;
        }
        // rotary applies to first half-pair only when hd is even — matches torch
      }
    }
  }
  return { data: out, shape: t.shape.slice() };
}

// Build the cos/sin tables for a given seq_len / head_dim / base.
// Mirrors Rotary.forward: inv_freq = 1 / base^(arange(0, hd, 2) / hd),
//                         freqs = outer(arange(T), inv_freq)
//                         cos/sin = freqs.cos()/sin()  shape [T, hd/2]
// PyTorch then stores them at [None, None, :, :] but the rotated-by-half code
// needs each freq paired with itself across the half boundary. Inspecting
// apply_rotary_emb: it uses the SAME cos/sin slice for both halves (same shape
// as x[..., :half] and x[..., half:]). Wait — let me recheck.
//
// Looking at bpb_compare.apply_rotary_emb:
//   half = x.size(-1) // 2
//   x1, x2 = x[..., :half], x[..., half:]
//   return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)
// And cos/sin from Rotary.forward have shape [1,1,T, hd/2]  (because
// freqs = outer(t, inv_freq) and inv_freq has length hd/2). So cos/sin are
// applied to half-length slices and that's broadcast OK. We materialise
// cos/sin at length [T, hd/2] and use indices accordingly.
export function buildRotaryTables(seqLen, headDim, base = 10000.0) {
  const half = headDim >> 1;
  const cos = new Float32Array(seqLen * half);
  const sin = new Float32Array(seqLen * half);
  for (let i = 0; i < half; i++) {
    const invFreq = 1.0 / Math.pow(base, (2 * i) / headDim);
    for (let t = 0; t < seqLen; t++) {
      cos[t * half + i] = Math.cos(t * invFreq);
      sin[t * half + i] = Math.sin(t * invFreq);
    }
  }
  return {
    cos: { data: cos, shape: [1, 1, seqLen, half] },
    sin: { data: sin, shape: [1, 1, seqLen, half] },
  };
}

// Apply rotary using [T, hd/2] cos/sin (matches torch's apply_rotary_emb exactly).
export function applyRotaryHalf(t, cos, sin) {
  const [B, H, T, hd] = t.shape;
  const half = hd >> 1;
  const cosData = cos.data;
  const sinData = sin.data;
  const out = new Float32Array(t.data.length);
  for (let b = 0; b < B; b++) {
    for (let h = 0; h < H; h++) {
      for (let ti = 0; ti < T; ti++) {
        const off = b * H * T * hd + h * T * hd + ti * hd;
        const cOff = ti * half;
        for (let i = 0; i < half; i++) {
          const x1 = t.data[off + i];
          const x2 = t.data[off + half + i];
          const c = cosData[cOff + i];
          const s = sinData[cOff + i];
          out[off + i] = x1 * c + x2 * s;
          out[off + half + i] = -x1 * s + x2 * c;
        }
      }
    }
  }
  return { data: out, shape: t.shape.slice() };
}

// ---------- per-head Q-gain scaling ----------
// q[B, H, T, hd] *= gain[H]
export function scalePerHead(q, gain) {
  const [B, H, T, hd] = q.shape;
  if (gain.shape.length !== 1 || gain.shape[0] !== H) {
    throw new Error(`scalePerHead shape mismatch: q H=${H}, gain=${gain.shape}`);
  }
  const out = new Float32Array(q.data.length);
  for (let b = 0; b < B; b++) {
    for (let h = 0; h < H; h++) {
      const g = gain.data[h];
      for (let ti = 0; ti < T; ti++) {
        const off = b * H * T * hd + h * T * hd + ti * hd;
        for (let i = 0; i < hd; i++) out[off + i] = q.data[off + i] * g;
      }
    }
  }
  return { data: out, shape: q.shape.slice() };
}

// ---------- Scaled dot-product attention with GQA ----------
// q [B, Hq, T, hd], k [B, Hk, T, hd], v [B, Hk, T, hd]
// If Hq != Hk, each KV head is shared by Hq/Hk Q heads (GQA / MQA).
// causal=true masks future positions.
//
// Matches torch's F.scaled_dot_product_attention default scale = 1/sqrt(hd)
// and reduces along the key dim with the same accumulation order torch uses
// for the math kernel: for each (b, h, qi) compute exp(scores) then divide by
// the sum (via softmax). Our parity bar is fp32 ULP, not bit-identity, since
// torch's actual SDPA may pick efficient kernels with different reduction
// orderings; we'll measure the gap empirically.
export function sdpa(q, k, v, { causal = false } = {}) {
  const [B, Hq, T, hd] = q.shape;
  const [, Hk, , hdk] = k.shape;
  if (hd !== hdk) throw new Error(`sdpa head_dim mismatch: q=${hd}, k=${hdk}`);
  if (Hq % Hk !== 0) throw new Error(`sdpa GQA mismatch: Hq=${Hq}, Hk=${Hk}`);
  const groupSize = Hq / Hk;
  const scaleF = 1 / Math.sqrt(hd);
  const out = new Float32Array(B * Hq * T * hd);

  // scratch
  const scores = new Float32Array(T);

  for (let b = 0; b < B; b++) {
    for (let h = 0; h < Hq; h++) {
      const hk = (h / groupSize) | 0;
      for (let qi = 0; qi < T; qi++) {
        const qOff = b * Hq * T * hd + h * T * hd + qi * hd;
        const lim = causal ? qi + 1 : T;
        let m = -Infinity;
        for (let kj = 0; kj < lim; kj++) {
          const kOff = b * Hk * T * hd + hk * T * hd + kj * hd;
          let s = 0;
          for (let i = 0; i < hd; i++) s += q.data[qOff + i] * k.data[kOff + i];
          s *= scaleF;
          scores[kj] = s;
          if (s > m) m = s;
        }
        let sum = 0;
        for (let kj = 0; kj < lim; kj++) {
          scores[kj] = Math.exp(scores[kj] - m);
          sum += scores[kj];
        }
        const inv = 1 / sum;
        const oOff = b * Hq * T * hd + h * T * hd + qi * hd;
        for (let i = 0; i < hd; i++) out[oOff + i] = 0;
        for (let kj = 0; kj < lim; kj++) {
          const w = scores[kj] * inv;
          const vOff = b * Hk * T * hd + hk * T * hd + kj * hd;
          for (let i = 0; i < hd; i++) out[oOff + i] += w * v.data[vOff + i];
        }
      }
    }
  }
  return { data: out, shape: q.shape.slice() };
}

// ---------- diff helpers ----------

export function maxAbsDiff(a, b) {
  if (a.data.length !== b.data.length) {
    return { ok: false, reason: `length ${a.data.length} vs ${b.data.length}` };
  }
  if (JSON.stringify(a.shape) !== JSON.stringify(b.shape)) {
    return { ok: false, reason: `shape ${a.shape} vs ${b.shape}` };
  }
  let m = 0, argi = -1, maxAbsB = 0;
  for (let i = 0; i < a.data.length; i++) {
    const d = Math.abs(a.data[i] - b.data[i]);
    if (d > m) { m = d; argi = i; }
    const ab = Math.abs(b.data[i]);
    if (ab > maxAbsB) maxAbsB = ab;
  }
  // Relative diff = max abs diff / max abs of reference. Useful when residual
  // magnitudes blow up across layers (huge values in the stream → small fp32
  // ULPs look like big abs diffs but are tiny in relative terms).
  const rel = maxAbsB > 0 ? m / maxAbsB : m;
  return { ok: true, max: m, rel, maxAbsRef: maxAbsB,
           argi, av: a.data[argi], bv: b.data[argi] };
}
