use pg_model::config::TrainConfig;
/// Sliding window evaluation + legal score-first TTT.
///
/// CRITICAL: Scoring windows are INSIDE the TTT chunk loop.
/// For each chunk:
///   Phase 1: SCORE this chunk with stride-64 sliding windows (no gradients)
///   Phase 2: TRAIN on already-scored chunk (SGD, 3 epochs)
/// Last chunk: scored but never trained on.
use pg_model::{ForwardBuffer, GptModel};

/// Compute NLL with softcap applied to logits: cap * tanh(logit / cap).
/// This matches the training loss which applies softcap inside cross_entropy_forward.
#[inline]
fn nll_with_softcap(logits: &[f32], target: usize, softcap: f32) -> f32 {
    let cap = |l: f32| {
        if softcap > 0.0 {
            softcap * (l / softcap).tanh()
        } else {
            l
        }
    };
    let target_logit = cap(logits[target]);
    let max_logit = logits
        .iter()
        .map(|&l| cap(l))
        .fold(f32::NEG_INFINITY, f32::max);
    let sum_exp: f32 = logits.iter().map(|&l| (cap(l) - max_logit).exp()).sum();
    let log_sum_exp = max_logit + sum_exp.ln();
    log_sum_exp - target_logit
}

/// Sliding window BPB evaluation (no TTT).
/// Returns (mean_loss, bpb).
pub fn eval_sliding(
    model: &GptModel,
    val_tokens: &[u32],
    base_bytes: &[f32], // per-token byte counts for BPB
    stride: usize,
    seq_len: usize,
) -> (f64, f64) {
    let total_tokens = val_tokens.len() - 1;
    let mut loss_sum = 0.0f64;
    let mut token_count = 0u64;
    let mut byte_count = 0.0f64;

    // Generate window starts
    let window_starts: Vec<usize> = (0..total_tokens)
        .step_by(stride)
        .filter(|&ws| {
            let end = (ws + seq_len).min(total_tokens);
            end - ws >= 1
        })
        .collect();

    let mut buf = ForwardBuffer::new(&model.config, seq_len);

    for &ws in &window_starts {
        let end = (ws + seq_len).min(total_tokens);
        let wlen = end - ws;

        // Build input/target
        let input = &val_tokens[ws..end];
        let target = &val_tokens[ws + 1..end + 1];

        // Resize buffer for this window (no reallocation)
        buf.resize_tokens(wlen);

        model.forward(input, &mut buf);

        // Compute per-token NLL with softcap
        let vocab = model.config.vocab_size;
        let softcap = model.config.logit_softcap;
        for t in 0..wlen {
            let logits = &buf.logits[t * vocab..(t + 1) * vocab];
            let tgt = target[t] as usize;
            let nll = nll_with_softcap(logits, tgt, softcap);

            // Only score the "new" tokens (stride region)
            let s = if ws == 0 {
                0
            } else {
                wlen.saturating_sub(stride)
            };
            if t >= s {
                loss_sum += nll as f64;
                token_count += 1;
                let tok_idx = ws + t;
                if tok_idx < base_bytes.len() {
                    byte_count += base_bytes[tok_idx] as f64;
                } else {
                    byte_count += 1.0; // fallback
                }
            }
        }
    }

    let val_loss = if token_count > 0 {
        loss_sum / token_count as f64
    } else {
        0.0
    };
    let bits_per_token = val_loss / 2.0f64.ln();
    let tokens_per_byte = if byte_count > 0.0 {
        token_count as f64 / byte_count
    } else {
        1.0
    };
    let bpb = bits_per_token * tokens_per_byte;

    (val_loss, bpb)
}

/// TTT chunk info for managing the score-first training loop.
pub struct TttChunk {
    pub chunk_start: usize,
    pub chunk_end: usize,
    pub windows: Vec<usize>, // window start positions assigned to this chunk
}

/// Assign windows to chunks for TTT.
pub fn build_ttt_chunks(
    total_tokens: usize,
    ttt_chunk_tokens: usize,
    stride: usize,
    seq_len: usize,
) -> Vec<TttChunk> {
    let window_starts: Vec<usize> = (0..total_tokens)
        .step_by(stride)
        .filter(|&ws| {
            let end = (ws + seq_len).min(total_tokens);
            end - ws >= stride || ws == 0
        })
        .collect();

    let num_chunks = (total_tokens + ttt_chunk_tokens - 1) / ttt_chunk_tokens;
    let mut chunks: Vec<TttChunk> = (0..num_chunks)
        .map(|ci| TttChunk {
            chunk_start: ci * ttt_chunk_tokens,
            chunk_end: ((ci + 1) * ttt_chunk_tokens).min(total_tokens),
            windows: Vec::new(),
        })
        .collect();

    // Assign each window to a chunk based on its scored region
    for &ws in &window_starts {
        let end = (ws + seq_len).min(total_tokens);
        let wlen = end - ws;
        let s = if ws == 0 { 0 } else { (wlen - stride).max(0) };
        let scored_start = ws + s;
        let ci = (scored_start / ttt_chunk_tokens).min(num_chunks - 1);
        chunks[ci].windows.push(ws);
    }

    chunks
}

/// Score a single chunk's windows (Phase 1 of TTT).
/// Returns (loss_sum, token_count, byte_count) for this chunk.
pub fn score_chunk(
    model: &GptModel,
    val_tokens: &[u32],
    base_bytes: &[f32],
    chunk: &TttChunk,
    stride: usize,
    seq_len: usize,
) -> (f64, u64, f64) {
    let total_tokens = val_tokens.len() - 1;
    let mut loss_sum = 0.0f64;
    let mut token_count = 0u64;
    let mut byte_count = 0.0f64;
    let mut buf = ForwardBuffer::new(&model.config, seq_len);

    for &ws in &chunk.windows {
        let end = (ws + seq_len).min(total_tokens);
        let wlen = end - ws;

        let input = &val_tokens[ws..end];
        let target = &val_tokens[ws + 1..end + 1];

        buf.resize_tokens(wlen);
        model.forward(input, &mut buf);

        let vocab = model.config.vocab_size;
        let softcap = model.config.logit_softcap;
        let s = if ws == 0 { 0 } else { (wlen - stride).max(0) };

        for t in s..wlen {
            let logits = &buf.logits[t * vocab..(t + 1) * vocab];
            let tgt = target[t] as usize;
            let nll = nll_with_softcap(logits, tgt, softcap);

            loss_sum += nll as f64;
            token_count += 1;
            let tok_idx = ws + t;
            byte_count += if tok_idx < base_bytes.len() {
                base_bytes[tok_idx] as f64
            } else {
                1.0
            };
        }
    }

    (loss_sum, token_count, byte_count)
}

/// Full TTT evaluation loop.
/// Score-first: score chunk, then train on it. Last chunk scored but not trained.
/// Returns (val_loss, bpb).
///
/// Note: Training phase requires backward pass (not yet implemented for CPU).
/// This function currently only implements the scoring phase.
pub fn eval_ttt_scoring_only(
    model: &GptModel,
    val_tokens: &[u32],
    base_bytes: &[f32],
    config: &TrainConfig,
) -> (f64, f64) {
    let stride = config.eval_stride;
    let seq_len = model.config.train_seq_len;
    let total_tokens = val_tokens.len() - 1;

    let chunks = build_ttt_chunks(total_tokens, config.ttt_chunk_tokens, stride, seq_len);

    let mut total_loss = 0.0f64;
    let mut total_tokens_scored = 0u64;
    let mut total_bytes = 0.0f64;

    for (ci, chunk) in chunks.iter().enumerate() {
        // Phase 1: Score
        let (loss, tokens, bytes) =
            score_chunk(model, val_tokens, base_bytes, chunk, stride, seq_len);
        total_loss += loss;
        total_tokens_scored += tokens;
        total_bytes += bytes;

        // Phase 2: Train (requires backward pass — placeholder)
        let is_last = ci == chunks.len() - 1;
        if !is_last && config.ttt_epochs > 0 {
            // TODO: TTT training phase (requires manual backward pass)
            // For now, scoring-only mode
        }
    }

    let val_loss = if total_tokens_scored > 0 {
        total_loss / total_tokens_scored as f64
    } else {
        0.0
    };
    let bits_per_token = val_loss / 2.0f64.ln();
    let tokens_per_byte = if total_bytes > 0.0 {
        total_tokens_scored as f64 / total_bytes
    } else {
        1.0
    };

    (val_loss, bits_per_token * tokens_per_byte)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_ttt_chunks() {
        let chunks = build_ttt_chunks(1000, 200, 64, 128);
        assert_eq!(chunks.len(), 5); // 1000 / 200 = 5 chunks

        // All windows should be assigned
        let total_windows: usize = chunks.iter().map(|c| c.windows.len()).sum();
        assert!(total_windows > 0);

        // Chunks should cover the token range
        assert_eq!(chunks[0].chunk_start, 0);
        assert_eq!(chunks[4].chunk_end, 1000);
    }

    #[test]
    fn test_eval_sliding_basic() {
        // Tiny model + token sequence
        let config = pg_model::config::ModelConfig {
            vocab_size: 16,
            num_layers: 1,
            model_dim: 8,
            num_heads: 2,
            num_kv_heads: 2,
            head_dim: 4,
            mlp_mult: 2.0,
            mlp_dim: 16,
            rope_base: 10000.0,
            rope_dims: 2,
            xsa_last_n: 0,
            logit_softcap: 30.0,
            qk_gain_init: 1.0,
            recurrence_enabled: false,
            recurrence_start_layer: 0,
            recurrence_repeat_layers: 0,
            parallel_residual: false,
            attn_out_gate_enabled: false,
            attn_out_gate_width: 24,
            sparse_attn_gate_enabled: false,
            sparse_attn_gate_width: 12,
            sparse_attn_gate_scale: 1.0,
            smear_gate_boundary_token_id: Some(1),
            vrl_enabled: false,
            ve_enabled: false,
            ve_dim: 4,
            ve_layers: vec![],
            bigram_vocab_size: 4,
            bigram_dim: 4,
            ln_scale: false,
            tie_embeddings: true,
            tied_embed_init_std: 0.005,
            train_seq_len: 8,
            eval_seq_len: 8,
        };

        let model = GptModel::new(config);
        let tokens: Vec<u32> = (0..20).map(|i| (i % 16) as u32).collect();
        let base_bytes = vec![1.0f32; 20];

        let (loss, bpb) = eval_sliding(&model, &tokens, &base_bytes, 4, 8);

        assert!(loss.is_finite(), "loss should be finite: {}", loss);
        assert!(loss > 0.0, "loss should be positive: {}", loss);
        assert!(bpb.is_finite(), "bpb should be finite: {}", bpb);
        eprintln!("Sliding eval: loss={:.4}, bpb={:.4}", loss, bpb);
    }
}
