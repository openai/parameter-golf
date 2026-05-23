// Minimal safetensors reader + dequantizer.
//
// This is the JS counterpart of
// visualization/scripts/pack_safetensors.py + reconstruct_onnx_data.py.
// Both must produce a byte-identical `model.onnx.data` from the same
// `model.safetensors.gz` + `onnx_data_manifest.json`. The parity is asserted
// by SHA-256 in scripts/node_parity_safetensors.js (Node) and in the demo
// at load time (browser).
//
// Format reference: https://github.com/huggingface/safetensors#format
//   u64 LE header_len | utf-8 header_json | concatenated raw tensor bytes
//
// Dtypes we handle (everything we ship today): F32, F16, BF16, I8, F8_E4M3.
// fp8_e4m3fn → fp32 is exact (every fp8 byte has an exact fp32 image), done
// via a 256-entry LUT we build once. bf16 → fp32 is a left-shift by 16. fp16
// → fp32 uses a 65536-entry LUT (faster than bit-shuffle in JS).

const F32 = "F32";
const F16 = "F16";
const BF16 = "BF16";
const I8 = "I8";
const F8_E4M3 = "F8_E4M3";  // safetensors writes this for torch.float8_e4m3fn

// ---------- LUTs ----------

let _fp8_lut = null;
function fp8E4m3fnToFp32Lut() {
  if (_fp8_lut) return _fp8_lut;
  // e4m3fn: 1 sign | 4 exp (bias 7) | 3 mantissa.
  // NaN at 0x7F and 0xFF (sign + exp=0xF + mantissa=0x7).
  // No infinities (the "fn" variant). Max representable = 448.
  const lut = new Float32Array(256);
  for (let b = 0; b < 256; b++) {
    const sign = b >> 7;
    const exp = (b >> 3) & 0xF;
    const mant = b & 0x7;
    let v;
    if (exp === 0xF && mant === 0x7) {
      v = NaN;
    } else if (exp === 0) {
      // Subnormal: (-1)^s * 2^-6 * (mant/8)
      v = (mant / 8) * Math.pow(2, -6);
      if (sign) v = -v;
    } else {
      // Normal: (-1)^s * 2^(exp - 7) * (1 + mant/8)
      v = (1 + mant / 8) * Math.pow(2, exp - 7);
      if (sign) v = -v;
    }
    lut[b] = v;
  }
  // Negative zero round-trips: 0x80 produces -0 because (mant/8)=0 and we
  // negate. Confirmed by Object.is(lut[0x80], -0).
  _fp8_lut = lut;
  return lut;
}

let _fp16_lut = null;
function fp16ToFp32Lut() {
  if (_fp16_lut) return _fp16_lut;
  const lut = new Float32Array(65536);
  const scratchU32 = new Uint32Array(1);
  const scratchF32 = new Float32Array(scratchU32.buffer);
  for (let h = 0; h < 65536; h++) {
    const sign = (h >> 15) & 1;
    const exp = (h >> 10) & 0x1F;
    const frac = h & 0x3FF;
    let bits;
    if (exp === 0) {
      if (frac === 0) {
        bits = sign << 31;  // ±0
      } else {
        // Subnormal half → normal float
        let e = -14;
        let m = frac;
        while ((m & 0x400) === 0) {
          m <<= 1;
          e--;
        }
        m &= 0x3FF;
        bits = (sign << 31) | ((e + 127) << 23) | (m << 13);
      }
    } else if (exp === 0x1F) {
      // Inf or NaN
      bits = (sign << 31) | (0xFF << 23) | (frac << 13);
    } else {
      bits = (sign << 31) | ((exp - 15 + 127) << 23) | (frac << 13);
    }
    scratchU32[0] = bits >>> 0;
    lut[h] = scratchF32[0];
  }
  _fp16_lut = lut;
  return lut;
}

// ---------- safetensors parsing ----------

export function parseSafetensors(buf) {
  // buf: ArrayBuffer | Uint8Array
  const u8 = buf instanceof Uint8Array ? buf : new Uint8Array(buf);
  const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
  // Header length is u64 little-endian; we only support files <2 GB so the
  // high u32 must be zero.
  const hi = dv.getUint32(4, true);
  if (hi !== 0) throw new Error("safetensors header > 4 GiB not supported");
  const headerLen = dv.getUint32(0, true);
  const headerBytes = u8.subarray(8, 8 + headerLen);
  const header = JSON.parse(new TextDecoder("utf-8").decode(headerBytes));
  const dataStart = 8 + headerLen;
  const metadata = header.__metadata__ || {};

  const tensors = {};
  for (const name of Object.keys(header)) {
    if (name === "__metadata__") continue;
    const t = header[name];
    const [start, end] = t.data_offsets;
    const slice = u8.subarray(dataStart + start, dataStart + end);
    tensors[name] = { dtype: t.dtype, shape: t.shape, bytes: slice };
  }
  return { metadata, tensors };
}

// ---------- dequant ----------

function flatLen(shape) {
  let n = 1;
  for (const d of shape) n *= d;
  return n;
}

function castI8ToF32(bytes) {
  const i8 = new Int8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const out = new Float32Array(i8.length);
  for (let i = 0; i < i8.length; i++) out[i] = i8[i];
  return out;
}

function castF16BytesToF32(bytes) {
  const lut = fp16ToFp32Lut();
  const u16 = new Uint16Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 2);
  const out = new Float32Array(u16.length);
  for (let i = 0; i < u16.length; i++) out[i] = lut[u16[i]];
  return out;
}

function castF8E4M3FnBytesToF32(bytes) {
  const lut = fp8E4m3fnToFp32Lut();
  const out = new Float32Array(bytes.length);
  for (let i = 0; i < bytes.length; i++) out[i] = lut[bytes[i]];
  return out;
}

function castF32BytesToF32(bytes) {
  // safetensors stores fp32 as little-endian raw bytes; safe to view directly
  // on little-endian hosts (all modern web platforms). Copy so the result is
  // independent of the input buffer's lifetime.
  const f32 = new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
  return new Float32Array(f32);  // copy
}

function bf16RoundTrip(f32) {
  // PyTorch's `.bfloat16().float()` uses round-to-nearest-even on conversion
  // to bf16, then a left-shift by 16 to come back to fp32. Reference:
  //   c10/util/BFloat16-inl.h::round_to_nearest_even
  // NaN inputs become the canonical bfloat16 NaN (0x7FC0 → 0x7FC00000).
  const u32 = new Uint32Array(f32.buffer, f32.byteOffset, f32.length);
  for (let i = 0; i < u32.length; i++) {
    let bits = u32[i];
    if (((bits & 0x7F800000) === 0x7F800000) && ((bits & 0x007FFFFF) !== 0)) {
      bits = 0x7FC00000;  // canonical NaN
    } else {
      const lsb = (bits >>> 16) & 1;
      const roundingBias = 0x7FFF + lsb;
      bits = (bits + roundingBias) & 0xFFFF0000;
    }
    u32[i] = bits >>> 0;
  }
  return f32;
}

/**
 * Walks the safetensors file and returns a map
 *   { base → { data: Float32Array, shape: number[] } }
 * that is byte-identical to the dequantized state-dict the Python pipeline
 * produces (pack_safetensors.dequantize_safetensors).
 *
 * For the AR int8+scale scheme the shape comes from the .q tensor; for fp8
 * MDLM it's also .q; for plain fp32 passthrough it's .p.
 */
export function dequantSafetensors({ metadata, tensors }) {
  // Group entries by base name (split on the trailing .q/.s/.p).
  const bases = new Map();
  for (const k of Object.keys(tensors)) {
    const dot = k.lastIndexOf(".");
    if (dot < 0) throw new Error(`unexpected tensor key: ${k}`);
    const base = k.slice(0, dot);
    const role = k.slice(dot + 1);
    let group = bases.get(base);
    if (!group) {
      group = {};
      bases.set(base, group);
    }
    group[role] = tensors[k];
  }

  const out = {};
  for (const [base, parts] of bases) {
    if (parts.p) {
      if (parts.p.dtype !== F32) {
        throw new Error(`expected F32 for ${base}.p, got ${parts.p.dtype}`);
      }
      out[base] = { data: castF32BytesToF32(parts.p.bytes), shape: parts.p.shape };
    } else if (parts.q && parts.s) {
      if (parts.q.dtype !== I8) throw new Error(`expected I8 for ${base}.q, got ${parts.q.dtype}`);
      if (parts.s.dtype !== F16) throw new Error(`expected F16 for ${base}.s, got ${parts.s.dtype}`);
      const q = castI8ToF32(parts.q.bytes);
      const s = castF16BytesToF32(parts.s.bytes);
      const shape = parts.q.shape;
      const rows = shape[0];
      const colsPerRow = q.length / rows;
      if (s.length !== rows) throw new Error(`scale rows mismatch for ${base}`);
      for (let r = 0; r < rows; r++) {
        const sr = s[r];
        const off = r * colsPerRow;
        for (let c = 0; c < colsPerRow; c++) q[off + c] = q[off + c] * sr;
      }
      const target = metadata[`target:${base}`];
      if (target === "bfloat16") {
        bf16RoundTrip(q);
      } else if (target && target !== "float32") {
        throw new Error(`unknown target dtype: ${target} for ${base}`);
      }
      out[base] = { data: q, shape };
    } else if (parts.q) {
      if (parts.q.dtype !== F8_E4M3) {
        throw new Error(`expected F8_E4M3 for ${base}.q, got ${parts.q.dtype}`);
      }
      out[base] = { data: castF8E4M3FnBytesToF32(parts.q.bytes), shape: parts.q.shape };
    } else {
      throw new Error(`bad parts for ${base}: ${Object.keys(parts).join(",")}`);
    }
  }
  return out;
}
