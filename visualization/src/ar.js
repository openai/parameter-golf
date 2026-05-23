// AR (causal) forward pass in pure JS, mirroring traced_ar_forward in
// generate_goldens.py. Same Block / SelfAttention / MLP plumbing as MDLM,
// but: causal=true, no mask-id substitution, LM head uses the full tok_emb.

import {
  addScaled, applyRotaryHalf, attnFlatten, buildRotaryTables,
  embeddingLookup, linear, mixResid, reluSquared, reshapeForAttn, rmsNorm,
  scale, scalePerHead, sdpa, tanh, FP32_EPS,
} from "./ops.js";

export function arForward(weights, arch, tokenIds, cap = () => {}) {
  const {
    num_layers, model_dim, vocab_size, num_heads, num_kv_heads,
    mlp_mult, logit_softcap, rope_base,
  } = arch;
  const headDim = model_dim / num_heads;
  const numEnc = Math.floor(num_layers / 2);
  const numDec = num_layers - numEnc;
  const T = tokenIds.length;
  const B = 1;

  // Encode token IDs as a [1, T] float32 tensor (storage type for embeddingLookup).
  const idsData = new Float32Array(T);
  for (let i = 0; i < T; i++) idsData[i] = tokenIds[i];
  const idsTensor = { data: idsData, shape: [1, T] };
  cap("input_ids", idsTensor);

  const tokEmb = weights["tok_emb.weight"];      // [vocab_size, D]
  let x = embeddingLookup(tokEmb, idsTensor);    // [1, T, D]
  cap("tok_emb_out", x);

  x = rmsNorm(x, FP32_EPS);
  cap("init_rmsnorm_out", x);
  const x0 = x;
  cap("x0", x0);

  const { cos, sin } = buildRotaryTables(T, headDim, rope_base);

  function runBlock(blockIdx, xIn, x0In, prefix) {
    const residMix = weights[`blocks.${blockIdx}.resid_mix`];
    const mix0 = { data: residMix.data.subarray(0, model_dim), shape: [model_dim] };
    const mix1 = { data: residMix.data.subarray(model_dim, 2 * model_dim), shape: [model_dim] };
    let xMixed = mixResid(mix0, mix1, xIn, x0In);
    cap(`${prefix}/resid_mix_out`, xMixed);

    const xAttnNorm = rmsNorm(xMixed, FP32_EPS);
    cap(`${prefix}/attn_norm_out`, xAttnNorm);

    const Wq = weights[`blocks.${blockIdx}.attn.c_q.weight`];
    const Wk = weights[`blocks.${blockIdx}.attn.c_k.weight`];
    const Wv = weights[`blocks.${blockIdx}.attn.c_v.weight`];
    const qLin = linear(xAttnNorm, Wq);
    const kLin = linear(xAttnNorm, Wk);
    const vLin = linear(xAttnNorm, Wv);

    let q = reshapeForAttn(qLin, B, T, num_heads, headDim);
    let k = reshapeForAttn(kLin, B, T, num_kv_heads, headDim);
    let v = reshapeForAttn(vLin, B, T, num_kv_heads, headDim);
    cap(`${prefix}/q_proj`, q);
    cap(`${prefix}/k_proj`, k);
    cap(`${prefix}/v_proj`, v);

    q = rmsNorm(q, FP32_EPS);
    k = rmsNorm(k, FP32_EPS);
    cap(`${prefix}/q_after_rmsnorm`, q);
    cap(`${prefix}/k_after_rmsnorm`, k);

    cap(`${prefix}/rope_cos`, cos);
    cap(`${prefix}/rope_sin`, sin);
    q = applyRotaryHalf(q, cos, sin);
    k = applyRotaryHalf(k, cos, sin);
    cap(`${prefix}/q_after_rotary`, q);
    cap(`${prefix}/k_after_rotary`, k);

    const qGain = weights[`blocks.${blockIdx}.attn.q_gain`];
    q = scalePerHead(q, qGain);
    cap(`${prefix}/q_after_gain`, q);

    // CAUSAL attention here (the only real diff from MDLM).
    const attnOut = sdpa(q, k, v, { causal: true });
    cap(`${prefix}/attn_sdpa_out`, attnOut);
    const attnFlat = attnFlatten(attnOut);
    cap(`${prefix}/attn_sdpa_flat`, attnFlat);

    const Wo = weights[`blocks.${blockIdx}.attn.proj.weight`];
    const attnProj = linear(attnFlat, Wo);
    cap(`${prefix}/attn_proj_out`, attnProj);

    const attnScale = weights[`blocks.${blockIdx}.attn_scale`];
    let xAttn = addScaled(xMixed, attnScale, attnProj);
    cap(`${prefix}/post_attn_residual`, xAttn);

    const xMlpNorm = rmsNorm(xAttn, FP32_EPS);
    cap(`${prefix}/mlp_norm_out`, xMlpNorm);
    const Wfc = weights[`blocks.${blockIdx}.mlp.fc.weight`];
    const Wpr = weights[`blocks.${blockIdx}.mlp.proj.weight`];
    const mlpFc = linear(xMlpNorm, Wfc);
    cap(`${prefix}/mlp_fc_out`, mlpFc);
    const mlpAct = reluSquared(mlpFc);
    cap(`${prefix}/mlp_relu_sq`, mlpAct);
    const mlpProj = linear(mlpAct, Wpr);
    cap(`${prefix}/mlp_proj_out`, mlpProj);

    const mlpScale = weights[`blocks.${blockIdx}.mlp_scale`];
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

  x = rmsNorm(x, FP32_EPS);
  cap("final_norm_out", x);

  // AR uses the FULL tok_emb.weight (no mask-id row to drop).
  const logitsPre = linear(x, tokEmb);
  cap("logits_pre_softcap", logitsPre);
  const logitsScaled = scale(logitsPre, 1 / logit_softcap);
  const logitsTanh = tanh(logitsScaled);
  const logits = scale(logitsTanh, logit_softcap);
  cap("logits", logits);
  return logits;
}
