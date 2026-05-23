// MDLM forward pass in pure JS, mirroring traced_diffusion_forward in
// visualization/scripts/generate_goldens.py. Each line is paired with the
// equivalent Python op so the reader can confirm correctness.
//
// Capture format matches the goldens trace: a Map keyed by tensor name like
// "enc00/q_after_rotary" so the parity page can diff side-by-side.

import {
  add, addScaled, applyRotaryHalf, attnFlatten, buildRotaryTables,
  embeddingLookup, linear, logSoftmax, mixResid, mulLastDim, reluSquared,
  reshapeForAttn, rmsNorm, scale, scalePerHead, sdpa, tanh, makeTensor, FP32_EPS,
} from "./ops.js";

// Helper: turn an array of token IDs (and an array of bool/int mask values)
// into the masked_ids tensor (shape [1, n], dtype float32 for storage but values
// are all integers).
export function buildMaskedIds(tokenIds, mask, maskId) {
  const n = tokenIds.length;
  const data = new Float32Array(n);
  for (let i = 0; i < n; i++) data[i] = mask[i] ? maskId : tokenIds[i];
  return { data, shape: [1, n] };
}

// Forward pass over a single (1, n) batch. `mask` is a Uint8Array of length n.
// `cap` (optional) is a function (name, tensor) => void to capture intermediates.
// Returns the final logits tensor [1, n, vocab_size].
export function mdlmForward(weights, arch, tokenIds, mask, cap = () => {}) {
  const {
    num_layers, model_dim, vocab_size, num_heads, num_kv_heads,
    mlp_mult, logit_softcap, rope_base, mask_id,
  } = arch;
  const headDim = model_dim / num_heads;
  const kvDim = num_kv_heads * headDim;
  const numEnc = Math.floor(num_layers / 2);
  const numDec = num_layers - numEnc;
  const T = tokenIds.length;
  const B = 1;

  // --- Input embedding & initial RMSNorm ---
  const maskedIds = buildMaskedIds(tokenIds, mask, mask_id);
  cap("masked_ids", maskedIds);

  const tokEmb = weights["tok_emb.weight"];               // [vocab_size+1, D] (or [vocab_size, D] for AR)
  let x = embeddingLookup(tokEmb, maskedIds);             // [1, T, D]
  cap("tok_emb_out", x);

  x = rmsNorm(x, FP32_EPS);
  cap("init_rmsnorm_out", x);
  const x0 = x;
  cap("x0", x0);

  // Pre-build rotary tables once for this seq_len.
  const { cos, sin } = buildRotaryTables(T, headDim, rope_base);

  function runBlock(blockIdx, xIn, x0In, prefix) {
    // ---- residual mix ----
    const residMix = weights[`blocks.${blockIdx}.resid_mix`]; // [2, D]
    const mix0 = { data: residMix.data.subarray(0, model_dim), shape: [model_dim] };
    const mix1 = { data: residMix.data.subarray(model_dim, 2 * model_dim), shape: [model_dim] };
    let xMixed = mixResid(mix0, mix1, xIn, x0In);
    cap(`${prefix}/resid_mix_out`, xMixed);

    // ---- attention norm ----
    const xAttnNorm = rmsNorm(xMixed, FP32_EPS);
    cap(`${prefix}/attn_norm_out`, xAttnNorm);

    // ---- Q/K/V projections ----
    const Wq = weights[`blocks.${blockIdx}.attn.c_q.weight`]; // [D, D]
    const Wk = weights[`blocks.${blockIdx}.attn.c_k.weight`]; // [kvDim, D]
    const Wv = weights[`blocks.${blockIdx}.attn.c_v.weight`]; // [kvDim, D]
    const qLin = linear(xAttnNorm, Wq);                       // [1, T, D]
    const kLin = linear(xAttnNorm, Wk);                       // [1, T, kvDim]
    const vLin = linear(xAttnNorm, Wv);                       // [1, T, kvDim]

    let q = reshapeForAttn(qLin, B, T, num_heads, headDim);     // [1, Hq, T, hd]
    let k = reshapeForAttn(kLin, B, T, num_kv_heads, headDim);  // [1, Hk, T, hd]
    let v = reshapeForAttn(vLin, B, T, num_kv_heads, headDim);  // [1, Hk, T, hd]
    cap(`${prefix}/q_proj`, q);
    cap(`${prefix}/k_proj`, k);
    cap(`${prefix}/v_proj`, v);

    // ---- Q/K rmsnorm (over head_dim) ----
    q = rmsNorm(q, FP32_EPS);
    k = rmsNorm(k, FP32_EPS);
    cap(`${prefix}/q_after_rmsnorm`, q);
    cap(`${prefix}/k_after_rmsnorm`, k);

    // ---- Rotary ----
    cap(`${prefix}/rope_cos`, cos);
    cap(`${prefix}/rope_sin`, sin);
    q = applyRotaryHalf(q, cos, sin);
    k = applyRotaryHalf(k, cos, sin);
    cap(`${prefix}/q_after_rotary`, q);
    cap(`${prefix}/k_after_rotary`, k);

    // ---- Per-head Q gain ----
    const qGain = weights[`blocks.${blockIdx}.attn.q_gain`];   // [Hq]
    q = scalePerHead(q, qGain);
    cap(`${prefix}/q_after_gain`, q);

    // ---- SDPA ----
    const attnOut = sdpa(q, k, v, { causal: false });           // [1, Hq, T, hd]
    cap(`${prefix}/attn_sdpa_out`, attnOut);
    const attnFlat = attnFlatten(attnOut);                       // [1, T, D]
    cap(`${prefix}/attn_sdpa_flat`, attnFlat);

    const Wo = weights[`blocks.${blockIdx}.attn.proj.weight`]; // [D, D]
    const attnProj = linear(attnFlat, Wo);                      // [1, T, D]
    cap(`${prefix}/attn_proj_out`, attnProj);

    // ---- post-attn residual ----
    const attnScale = weights[`blocks.${blockIdx}.attn_scale`]; // [D]
    let xAttn = addScaled(xMixed, attnScale, attnProj);
    cap(`${prefix}/post_attn_residual`, xAttn);

    // ---- MLP ----
    const xMlpNorm = rmsNorm(xAttn, FP32_EPS);
    cap(`${prefix}/mlp_norm_out`, xMlpNorm);
    const Wfc = weights[`blocks.${blockIdx}.mlp.fc.weight`];   // [mlp_mult*D, D]
    const Wpr = weights[`blocks.${blockIdx}.mlp.proj.weight`]; // [D, mlp_mult*D]
    const mlpFc = linear(xMlpNorm, Wfc);
    cap(`${prefix}/mlp_fc_out`, mlpFc);
    const mlpAct = reluSquared(mlpFc);
    cap(`${prefix}/mlp_relu_sq`, mlpAct);
    const mlpProj = linear(mlpAct, Wpr);
    cap(`${prefix}/mlp_proj_out`, mlpProj);

    const mlpScale = weights[`blocks.${blockIdx}.mlp_scale`]; // [D]
    const xOut = addScaled(xAttn, mlpScale, mlpProj);
    cap(`${prefix}/block_out`, xOut);
    return xOut;
  }

  const skips = [];
  for (let i = 0; i < numEnc; i++) {
    x = runBlock(i, x, x0, `enc${String(i).padStart(2, "0")}`);
    skips.push(x);
    cap(`enc${String(i).padStart(2, "0")}/skip_pushed`, x);
  }

  for (let i = 0; i < numDec; i++) {
    if (skips.length > 0) {
      const sk = skips.pop();
      cap(`dec${String(i).padStart(2, "0")}/skip_popped`, sk);
      const skipW = {
        data: weights["skip_weights"].data.subarray(i * model_dim, (i + 1) * model_dim),
        shape: [model_dim],
      };
      x = addScaled(x, skipW, sk);
      cap(`dec${String(i).padStart(2, "0")}/post_skip_add`, x);
    }
    x = runBlock(numEnc + i, x, x0, `dec${String(i).padStart(2, "0")}`);
  }

  // Final norm + LM head + softcap
  x = rmsNorm(x, FP32_EPS);
  cap("final_norm_out", x);

  // For MDLM, the LM head uses tok_emb.weight[:vocab_size] — drop the extra
  // mask-id row before doing the linear.
  let lmHead = tokEmb;
  if (tokEmb.shape[0] === vocab_size + 1) {
    const D = tokEmb.shape[1];
    lmHead = {
      data: tokEmb.data.subarray(0, vocab_size * D),
      shape: [vocab_size, D],
    };
  }
  const logitsPre = linear(x, lmHead);
  cap("logits_pre_softcap", logitsPre);

  // softcap * tanh(logits / softcap)
  const logitsScaled = scale(logitsPre, 1 / logit_softcap);
  const logitsTanh = tanh(logitsScaled);
  const logits = scale(logitsTanh, logit_softcap);
  cap("logits", logits);

  return logits;
}
