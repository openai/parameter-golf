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
    ShapeMismatch { name: String, expected: usize, got: usize },
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoadError::Io(msg) => write!(f, "IO error: {}", msg),
            LoadError::MissingTensor(name) => write!(f, "missing tensor: {}", name),
            LoadError::ShapeMismatch { name, expected, got } => {
                write!(f, "shape mismatch for '{}': expected {} elements, got {}", name, expected, got)
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
                        warnings.push(format!("{}: expected scalar, got {} elements", $name, data.len()));
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

        load_tensor!(&format!("{}.attn_scale", prefix), model.blocks[i].attn_scale);
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

#[cfg(test)]
mod tests {
    use super::*;

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
        for n in &names {
            eprintln!("  {}", n);
        }
    }
}
