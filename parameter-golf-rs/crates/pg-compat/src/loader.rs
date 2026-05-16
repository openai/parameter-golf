/// Load PyTorch state dict (safetensors format) into GptModel.
///
/// Key mapping from PyTorch state dict:
///   qo_bank → model.qo_bank [2*n, d, d]
///   kv_bank → model.kv_bank [2*n, kv, d]
///   mlp_up_bank → model.mlp_up_bank [n, mlp, d]
///   mlp_down_bank → model.mlp_down_bank [n, d, mlp]
///   tok_emb.weight → model.tok_emb [vocab, d]
///   bigram.embed.weight → model.bigram_embed [bigram_vocab, bigram_dim]
///   bigram.proj.weight → model.bigram_proj [d, bigram_dim]
///   bigram.scale → model.bigram_scale (scalar)
///   smear.gate → model.smear_gate [d]
///   skip_weights → model.skip_weights [num_skip, d]
///   blocks.{i}.attn_scale → model.blocks[i].attn_scale [d]
///   blocks.{i}.mlp_scale → model.blocks[i].mlp_scale [d]
///   blocks.{i}.resid_mix → model.blocks[i].resid_mix [2, d]
///   blocks.{i}.attn.q_gain → model.blocks[i].q_gain [num_heads]
///   ve_shared.embed.weight → model.ve_embed [vocab, ve_dim]
///   ve_shared.proj.weight → model.ve_proj [kv_dim, ve_dim]
///   ve_shared.scale → model.ve_scale (scalar)
///   ve_layer_scales.{i} → model.ve_layer_scales[i] (scalar)
use std::path::Path;

use crate::safetensors::SafeTensorsFile;

/// Errors during weight loading.
#[derive(Debug)]
pub enum LoadError {
    Io(String),
    MissingTensor(String),
    ShapeMismatch {
        name: String,
        expected: usize,
        got: usize,
    },
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoadError::Io(msg) => write!(f, "IO error: {}", msg),
            LoadError::MissingTensor(name) => write!(f, "missing tensor: {}", name),
            LoadError::ShapeMismatch {
                name,
                expected,
                got,
            } => {
                write!(
                    f,
                    "shape mismatch for '{}': expected {} elements, got {}",
                    name, expected, got
                )
            }
        }
    }
}

/// Load weights from a safetensors file into a GptModel.
/// The model must already be constructed with the correct config.
pub fn load_safetensors(
    path: &Path,
    model: &mut pg_model::GptModel,
) -> Result<Vec<String>, LoadError> {
    let st = SafeTensorsFile::load(path).map_err(LoadError::Io)?;
    load_from_safetensors(&st, model)
}

/// Load weights from an already-parsed SafeTensorsFile.
pub fn load_from_safetensors(
    st: &SafeTensorsFile,
    model: &mut pg_model::GptModel,
) -> Result<Vec<String>, LoadError> {
    let mut loaded = Vec::new();
    let mut warnings = Vec::new();
    let c = &model.config;
    let n = c.num_layers;

    // Helper: load a tensor, checking size
    macro_rules! load_tensor {
        ($name:expr, $dest:expr) => {
            match st.get_tensor_f32($name) {
                Ok(data) => {
                    if data.len() != $dest.len() {
                        return Err(LoadError::ShapeMismatch {
                            name: $name.to_string(),
                            expected: $dest.len(),
                            got: data.len(),
                        });
                    }
                    $dest.copy_from_slice(&data);
                    loaded.push($name.to_string());
                }
                Err(_) => {
                    warnings.push(format!("missing: {}", $name));
                }
            }
        };
    }

    // Helper: load scalar
    macro_rules! load_scalar {
        ($name:expr, $dest:expr) => {
            match st.get_tensor_f32($name) {
                Ok(data) => {
                    if data.len() == 1 {
                        $dest = data[0];
                        loaded.push($name.to_string());
                    } else {
                        warnings.push(format!(
                            "{}: expected scalar, got {} elements",
                            $name,
                            data.len()
                        ));
                    }
                }
                Err(_) => {
                    warnings.push(format!("missing: {}", $name));
                }
            }
        };
    }

    // Banks
    load_tensor!("qo_bank", model.qo_bank);
    load_tensor!("kv_bank", model.kv_bank);
    load_tensor!("mlp_up_bank", model.mlp_up_bank);
    load_tensor!("mlp_down_bank", model.mlp_down_bank);

    // Token embedding
    load_tensor!("tok_emb.weight", model.tok_emb);

    // Bigram
    load_tensor!("bigram.embed.weight", model.bigram_embed);
    load_tensor!("bigram.proj.weight", model.bigram_proj);
    load_scalar!("bigram.scale", model.bigram_scale);

    // SmearGate
    load_tensor!("smear.gate", model.smear_gate);

    // Skip weights
    load_tensor!("skip_weights", model.skip_weights);

    // Per-block parameters
    for i in 0..n {
        let prefix = format!("blocks.{}", i);

        load_tensor!(
            &format!("{}.attn_scale", prefix),
            model.blocks[i].attn_scale
        );
        load_tensor!(&format!("{}.mlp_scale", prefix), model.blocks[i].mlp_scale);
        load_tensor!(&format!("{}.resid_mix", prefix), model.blocks[i].resid_mix);
        load_tensor!(&format!("{}.attn.q_gain", prefix), model.blocks[i].q_gain);
    }

    // Value Embedding
    if c.ve_enabled {
        load_tensor!("ve_shared.embed.weight", model.ve_embed);
        load_tensor!("ve_shared.proj.weight", model.ve_proj);
        load_scalar!("ve_shared.scale", model.ve_scale);

        for (idx, _layer) in c.ve_layers.iter().enumerate() {
            let name = format!("ve_layer_scales.{}", idx);
            match st.get_tensor_f32(&name) {
                Ok(data) => {
                    if data.len() == 1 {
                        model.ve_layer_scales[idx] = data[0];
                        loaded.push(name);
                    }
                }
                Err(_) => {
                    warnings.push(format!("missing: {}", name));
                }
            }
        }
    }

    if !warnings.is_empty() {
        eprintln!("Weight loading warnings:");
        for w in &warnings {
            eprintln!("  {}", w);
        }
    }

    eprintln!("Loaded {} tensors from safetensors", loaded.len());
    Ok(loaded)
}

/// List expected tensor names for a given config.
pub fn expected_tensor_names(config: &pg_model::config::ModelConfig) -> Vec<String> {
    let n = config.num_layers;
    let mut names = vec![
        "qo_bank".into(),
        "kv_bank".into(),
        "mlp_up_bank".into(),
        "mlp_down_bank".into(),
        "tok_emb.weight".into(),
        "bigram.embed.weight".into(),
        "bigram.proj.weight".into(),
        "bigram.scale".into(),
        "smear.gate".into(),
        "skip_weights".into(),
    ];

    for i in 0..n {
        names.push(format!("blocks.{}.attn_scale", i));
        names.push(format!("blocks.{}.mlp_scale", i));
        names.push(format!("blocks.{}.resid_mix", i));
        names.push(format!("blocks.{}.attn.q_gain", i));
    }

    if config.ve_enabled {
        names.push("ve_shared.embed.weight".into());
        names.push("ve_shared.proj.weight".into());
        names.push("ve_shared.scale".into());
        for (idx, _) in config.ve_layers.iter().enumerate() {
            names.push(format!("ve_layer_scales.{}", idx));
        }
    }

    names
}

/// Dump a `GptModel` into the safetensors layout that `load_safetensors`
/// expects. This is the inverse of `load_from_safetensors` and gives us a
/// hermetic round-trip without needing a real PyTorch checkpoint on disk.
///
/// Used by the round-trip test below and by the per-step equivalence harness
/// to write the "ground truth" weight file that PyTorch will then re-import.
pub fn dump_model_safetensors(model: &pg_model::GptModel) -> Vec<u8> {
    use crate::writer::{OutTensor, f32_tensor, write_safetensors};

    let c = &model.config;
    let n = c.num_layers;
    let d = c.model_dim;
    let kv = c.kv_dim();
    let mlp = c.mlp_dim;

    // Scalar storage — owned slices kept alive for the duration of the call
    let bigram_scale_buf = [model.bigram_scale];
    let ve_scale_buf = [model.ve_scale];
    let ve_layer_scale_bufs: Vec<[f32; 1]> = model.ve_layer_scales.iter().map(|&v| [v]).collect();

    let block_attn_scales: Vec<&[f32]> = model
        .blocks
        .iter()
        .map(|b| b.attn_scale.as_slice())
        .collect();
    let block_mlp_scales: Vec<&[f32]> = model
        .blocks
        .iter()
        .map(|b| b.mlp_scale.as_slice())
        .collect();
    let block_resid_mix: Vec<&[f32]> = model
        .blocks
        .iter()
        .map(|b| b.resid_mix.as_slice())
        .collect();
    let block_q_gain: Vec<&[f32]> = model.blocks.iter().map(|b| b.q_gain.as_slice()).collect();

    // Pre-format the per-block names so we can hand string slices to OutTensor
    let block_names: Vec<[String; 4]> = (0..n)
        .map(|i| {
            [
                format!("blocks.{}.attn_scale", i),
                format!("blocks.{}.mlp_scale", i),
                format!("blocks.{}.resid_mix", i),
                format!("blocks.{}.attn.q_gain", i),
            ]
        })
        .collect();

    let ve_layer_names: Vec<String> = (0..model.ve_layer_scales.len())
        .map(|i| format!("ve_layer_scales.{}", i))
        .collect();

    let mut tensors: Vec<OutTensor> = Vec::new();

    tensors.push(f32_tensor("qo_bank", vec![2 * n, d, d], &model.qo_bank));
    tensors.push(f32_tensor("kv_bank", vec![2 * n, kv, d], &model.kv_bank));
    tensors.push(f32_tensor(
        "mlp_up_bank",
        vec![n, mlp, d],
        &model.mlp_up_bank,
    ));
    tensors.push(f32_tensor(
        "mlp_down_bank",
        vec![n, d, mlp],
        &model.mlp_down_bank,
    ));

    tensors.push(f32_tensor(
        "tok_emb.weight",
        vec![c.vocab_size, d],
        &model.tok_emb,
    ));

    tensors.push(f32_tensor(
        "bigram.embed.weight",
        vec![c.bigram_vocab_size, c.bigram_dim],
        &model.bigram_embed,
    ));
    tensors.push(f32_tensor(
        "bigram.proj.weight",
        vec![d, c.bigram_dim],
        &model.bigram_proj,
    ));
    tensors.push(f32_tensor("bigram.scale", vec![1], &bigram_scale_buf));

    tensors.push(f32_tensor("smear.gate", vec![d], &model.smear_gate));
    tensors.push(f32_tensor(
        "skip_weights",
        vec![c.num_skip_weights(), d],
        &model.skip_weights,
    ));

    for i in 0..n {
        tensors.push(f32_tensor(
            &block_names[i][0],
            vec![d],
            block_attn_scales[i],
        ));
        tensors.push(f32_tensor(&block_names[i][1], vec![d], block_mlp_scales[i]));
        tensors.push(f32_tensor(
            &block_names[i][2],
            vec![2, d],
            block_resid_mix[i],
        ));
        tensors.push(f32_tensor(
            &block_names[i][3],
            vec![c.num_heads],
            block_q_gain[i],
        ));
    }

    if c.ve_enabled {
        tensors.push(f32_tensor(
            "ve_shared.embed.weight",
            vec![c.vocab_size, c.ve_dim],
            &model.ve_embed,
        ));
        tensors.push(f32_tensor(
            "ve_shared.proj.weight",
            vec![kv, c.ve_dim],
            &model.ve_proj,
        ));
        tensors.push(f32_tensor("ve_shared.scale", vec![1], &ve_scale_buf));
        for (i, slice) in ve_layer_scale_bufs.iter().enumerate() {
            tensors.push(f32_tensor(&ve_layer_names[i], vec![1], slice));
        }
    }

    write_safetensors(&tensors)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::safetensors::SafeTensorsFile;

    #[test]
    fn test_expected_tensor_names() {
        let config = pg_model::config::ModelConfig::sota();
        let names = expected_tensor_names(&config);

        // Should include banks + per-block + VE
        assert!(names.contains(&"qo_bank".to_string()));
        assert!(names.contains(&"blocks.0.attn_scale".to_string()));
        assert!(names.contains(&"blocks.10.attn.q_gain".to_string()));
        assert!(names.contains(&"ve_shared.embed.weight".to_string()));

        eprintln!("Expected {} tensors", names.len());
    }

    /// Tiny config matching what we use for fast tests across the workspace.
    fn tiny_config() -> pg_model::config::ModelConfig {
        pg_model::config::ModelConfig {
            vocab_size: 16,
            num_layers: 2,
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
        }
    }

    /// Fill all parameters with deterministic non-zero values so a no-op load
    /// would be detectable.
    fn fill_deterministic(model: &mut pg_model::GptModel) {
        let mut rng_state: u32 = 0xDEADBEEF;
        let mut next = || {
            // xorshift32 → uniform in [-0.1, 0.1]
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 17;
            rng_state ^= rng_state << 5;
            (rng_state as f32 / u32::MAX as f32 - 0.5) * 0.2
        };
        for v in model.tok_emb.iter_mut() {
            *v = next();
        }
        for v in model.qo_bank.iter_mut() {
            *v = next();
        }
        for v in model.kv_bank.iter_mut() {
            *v = next();
        }
        for v in model.mlp_up_bank.iter_mut() {
            *v = next();
        }
        for v in model.mlp_down_bank.iter_mut() {
            *v = next();
        }
        for v in model.bigram_embed.iter_mut() {
            *v = next();
        }
        for v in model.bigram_proj.iter_mut() {
            *v = next();
        }
        model.bigram_scale = next();
        for v in model.smear_gate.iter_mut() {
            *v = next();
        }
        for v in model.skip_weights.iter_mut() {
            *v = next();
        }
        for b in model.blocks.iter_mut() {
            for v in b.attn_scale.iter_mut() {
                *v = next();
            }
            for v in b.mlp_scale.iter_mut() {
                *v = next();
            }
            for v in b.resid_mix.iter_mut() {
                *v = next();
            }
            for v in b.q_gain.iter_mut() {
                *v = next();
            }
        }
    }

    #[test]
    fn test_dump_load_round_trip_bytes_match() {
        let mut model = pg_model::GptModel::new(tiny_config());
        fill_deterministic(&mut model);

        // Dump → in-memory safetensors blob
        let bytes = dump_model_safetensors(&model);
        let st = SafeTensorsFile::from_bytes(&bytes).unwrap();

        // Load into a fresh model
        let mut loaded = pg_model::GptModel::new(tiny_config());
        load_from_safetensors(&st, &mut loaded).unwrap();

        // Every parameter buffer must match exactly
        assert_eq!(loaded.tok_emb, model.tok_emb);
        assert_eq!(loaded.qo_bank, model.qo_bank);
        assert_eq!(loaded.kv_bank, model.kv_bank);
        assert_eq!(loaded.mlp_up_bank, model.mlp_up_bank);
        assert_eq!(loaded.mlp_down_bank, model.mlp_down_bank);
        assert_eq!(loaded.bigram_embed, model.bigram_embed);
        assert_eq!(loaded.bigram_proj, model.bigram_proj);
        assert_eq!(loaded.bigram_scale, model.bigram_scale);
        assert_eq!(loaded.smear_gate, model.smear_gate);
        assert_eq!(loaded.skip_weights, model.skip_weights);
        for i in 0..loaded.blocks.len() {
            assert_eq!(loaded.blocks[i].attn_scale, model.blocks[i].attn_scale);
            assert_eq!(loaded.blocks[i].mlp_scale, model.blocks[i].mlp_scale);
            assert_eq!(loaded.blocks[i].resid_mix, model.blocks[i].resid_mix);
            assert_eq!(loaded.blocks[i].q_gain, model.blocks[i].q_gain);
        }
    }

    #[test]
    fn test_dump_load_round_trip_forward_matches() {
        // Build, dump, load, then verify both models produce identical logits
        // on the same input.
        let cfg = tiny_config();
        let mut model = pg_model::GptModel::new(cfg.clone());
        fill_deterministic(&mut model);

        let bytes = dump_model_safetensors(&model);
        let st = SafeTensorsFile::from_bytes(&bytes).unwrap();
        let mut loaded = pg_model::GptModel::new(cfg.clone());
        load_from_safetensors(&st, &mut loaded).unwrap();

        let input: Vec<u32> = (0..cfg.train_seq_len as u32)
            .map(|i| i % cfg.vocab_size as u32)
            .collect();

        let mut buf_a = pg_model::ForwardBuffer::new(&cfg, cfg.train_seq_len);
        let mut buf_b = pg_model::ForwardBuffer::new(&cfg, cfg.train_seq_len);
        model.forward(&input, &mut buf_a);
        loaded.forward(&input, &mut buf_b);

        assert_eq!(buf_a.logits.len(), buf_b.logits.len());
        let max_diff = buf_a
            .logits
            .iter()
            .zip(buf_b.logits.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 1e-5,
            "Round-trip forward parity failed: max abs diff = {}",
            max_diff
        );
    }
}
