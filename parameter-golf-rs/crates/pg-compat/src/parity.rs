/// Per-Step Equivalence Validation harness (TDD §4).
///
/// Goal: prove that a single Rust training step produces the same loss and
/// gradients (within tolerance) as a single PyTorch training step from the
/// same weights and the same batch.
///
/// This module is the *Rust side*. The PyTorch side is a small standalone
/// script (`scripts/dump_pytorch_step.py`, generated separately) that:
///   1. Loads the SOTA checkpoint
///   2. Runs a forward + backward on a deterministic batch (seed=42)
///   3. Writes a `pytorch_step_dump.bin` in the format defined below
///
/// The Rust side calls `dump_rust_step` to write `rust_step_dump.bin`, then
/// `compare_dumps` to assert the two are equivalent within tolerance.
///
/// ### Dump format
///
/// We deliberately use safetensors for the dump because both sides already
/// have a safetensors implementation:
///   - `pytorch_step_dump.bin` is just `safetensors.save_file({...})` from torch
///   - `rust_step_dump.bin` is `pg_compat::writer::write_safetensors(&[...])`
///
/// The expected key set:
///   - `loss`         — shape [1], scalar
///   - `input_ids`    — shape [seq], i64 → we store as f32 for portability
///   - `targets`      — shape [seq], same
///   - `grad.<name>`  — one entry per gradient buffer
///
/// ### Tolerances
///
/// - `loss_atol = 1e-3`
/// - `grad_max_rel_diff = 0.02` (2% relative error in max-norm sense)
/// - `grad_atol_floor   = 1e-6` (don't divide by tiny denominators)
use std::collections::BTreeMap;
use std::path::Path;

use crate::safetensors::SafeTensorsFile;
use crate::writer::{OutTensor, f32_tensor, write_safetensors};

/// A single named gradient (or activation) from one stack.
pub struct NamedTensor<'a> {
    pub name: &'a str,
    pub shape: Vec<usize>,
    pub data: &'a [f32],
}

/// Write a Rust step dump to disk.
///
/// `loss` is the scalar loss.
/// `grads` are the named gradient tensors. Use the same naming convention as
/// the PyTorch dump (e.g. `grad.qo_bank`, `grad.tok_emb`, ...).
pub fn dump_rust_step(path: &Path, loss: f32, grads: &[NamedTensor]) -> std::io::Result<()> {
    let loss_buf = [loss];

    // Build the safetensors entries
    let mut tensors: Vec<OutTensor> = Vec::with_capacity(grads.len() + 1);
    tensors.push(f32_tensor("loss", vec![1], &loss_buf));
    for g in grads {
        tensors.push(f32_tensor(g.name, g.shape.clone(), g.data));
    }

    let bytes = write_safetensors(&tensors);
    std::fs::write(path, bytes)
}

/// A comparison report.
#[derive(Debug, Clone)]
pub struct ParityReport {
    pub loss_diff: f32,
    pub grad_diffs: Vec<GradDiff>,
    pub passed: bool,
}

#[derive(Debug, Clone)]
pub struct GradDiff {
    pub name: String,
    pub max_abs_diff: f32,
    pub max_rel_diff: f32,
    pub passed: bool,
    pub note: Option<String>,
}

/// Tolerances for the comparison.
#[derive(Debug, Clone)]
pub struct ParityTolerances {
    pub loss_atol: f32,
    pub grad_max_rel: f32,
    pub grad_atol_floor: f32,
}

impl Default for ParityTolerances {
    fn default() -> Self {
        Self {
            loss_atol: 1e-3,
            grad_max_rel: 0.02,
            grad_atol_floor: 1e-6,
        }
    }
}

/// Compare two safetensors dumps (Rust vs PyTorch) and return a report.
///
/// Both files must contain the same `loss` key and the same set of `grad.*`
/// keys. Extra keys are tolerated. Missing keys produce a failed `GradDiff`
/// with a `note`.
pub fn compare_dumps(
    rust_path: &Path,
    torch_path: &Path,
    tol: &ParityTolerances,
) -> Result<ParityReport, String> {
    let rust = SafeTensorsFile::load(rust_path).map_err(|e| format!("loading rust dump: {}", e))?;
    let torch =
        SafeTensorsFile::load(torch_path).map_err(|e| format!("loading torch dump: {}", e))?;

    // Compare loss
    let rust_loss = rust
        .get_tensor_f32("loss")
        .map_err(|e| format!("rust dump missing loss: {}", e))?;
    let torch_loss = torch
        .get_tensor_f32("loss")
        .map_err(|e| format!("torch dump missing loss: {}", e))?;
    if rust_loss.is_empty() || torch_loss.is_empty() {
        return Err("loss tensor empty in one of the dumps".into());
    }
    let loss_diff = (rust_loss[0] - torch_loss[0]).abs();
    let loss_passed = loss_diff <= tol.loss_atol;

    // Compare grads. We walk torch's keys (the ground truth set) and look for
    // each one in rust.
    let mut grad_diffs: Vec<GradDiff> = Vec::new();
    let mut all_passed = loss_passed;

    let torch_grad_keys: BTreeMap<&str, &str> = torch
        .tensors
        .keys()
        .filter(|k| k.starts_with("grad."))
        .map(|k| (k.as_str(), k.as_str()))
        .collect();

    for (key, _) in &torch_grad_keys {
        match (rust.get_tensor_f32(key), torch.get_tensor_f32(key)) {
            (Ok(rg), Ok(tg)) => {
                if rg.len() != tg.len() {
                    grad_diffs.push(GradDiff {
                        name: key.to_string(),
                        max_abs_diff: f32::INFINITY,
                        max_rel_diff: f32::INFINITY,
                        passed: false,
                        note: Some(format!(
                            "shape mismatch: rust={} torch={}",
                            rg.len(),
                            tg.len()
                        )),
                    });
                    all_passed = false;
                    continue;
                }
                let mut max_abs = 0.0f32;
                let mut torch_max = 0.0f32;
                for (a, b) in rg.iter().zip(tg.iter()) {
                    max_abs = max_abs.max((a - b).abs());
                    torch_max = torch_max.max(b.abs());
                }
                let denom = torch_max.max(tol.grad_atol_floor);
                let max_rel = max_abs / denom;
                let passed = max_rel <= tol.grad_max_rel;
                if !passed {
                    all_passed = false;
                }
                grad_diffs.push(GradDiff {
                    name: key.to_string(),
                    max_abs_diff: max_abs,
                    max_rel_diff: max_rel,
                    passed,
                    note: None,
                });
            }
            (Err(_), _) => {
                grad_diffs.push(GradDiff {
                    name: key.to_string(),
                    max_abs_diff: f32::INFINITY,
                    max_rel_diff: f32::INFINITY,
                    passed: false,
                    note: Some("missing in rust dump".to_string()),
                });
                all_passed = false;
            }
            _ => {}
        }
    }

    Ok(ParityReport {
        loss_diff,
        grad_diffs,
        passed: all_passed,
    })
}

/// Pretty-print a parity report.
pub fn format_report(report: &ParityReport) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "ParityReport: passed={} loss_diff={:.3e}\n",
        report.passed, report.loss_diff
    ));
    out.push_str(&format!(
        "{:<48} {:>14} {:>14} {:>6}\n",
        "tensor", "max_abs", "max_rel", "ok"
    ));
    for d in &report.grad_diffs {
        let note = d.note.as_deref().unwrap_or("");
        out.push_str(&format!(
            "{:<48} {:>14.3e} {:>14.3e} {:>6} {}\n",
            d.name,
            d.max_abs_diff,
            d.max_rel_diff,
            if d.passed { "yes" } else { "NO" },
            note
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_path(name: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("pg_parity_{}.safetensors", name));
        p
    }

    #[test]
    fn test_dump_and_compare_identical() {
        let g_a: Vec<f32> = vec![0.1, -0.2, 0.3, -0.4];
        let g_b: Vec<f32> = vec![1.0, 2.0, 3.0];

        let grads = vec![
            NamedTensor {
                name: "grad.qo_bank",
                shape: vec![2, 2],
                data: &g_a,
            },
            NamedTensor {
                name: "grad.tok_emb",
                shape: vec![3],
                data: &g_b,
            },
        ];

        let path_a = tmp_path("ident_a");
        let path_b = tmp_path("ident_b");
        dump_rust_step(&path_a, 1.234, &grads).unwrap();
        dump_rust_step(&path_b, 1.234, &grads).unwrap();

        let tol = ParityTolerances::default();
        let report = compare_dumps(&path_a, &path_b, &tol).unwrap();
        assert!(report.passed, "{}", format_report(&report));
        assert!(report.loss_diff < 1e-6);
        assert_eq!(report.grad_diffs.len(), 2);
        for d in &report.grad_diffs {
            assert!(d.passed);
            assert!(d.max_abs_diff < 1e-6);
        }
    }

    #[test]
    fn test_compare_detects_mismatch() {
        let g_ref: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let g_bad: Vec<f32> = vec![1.0, 2.0, 3.0, 4.5]; // 12.5% relative error in max
        let grads_ref = vec![NamedTensor {
            name: "grad.x",
            shape: vec![4],
            data: &g_ref,
        }];
        let grads_bad = vec![NamedTensor {
            name: "grad.x",
            shape: vec![4],
            data: &g_bad,
        }];

        let p_ref = tmp_path("ref");
        let p_bad = tmp_path("bad");
        dump_rust_step(&p_ref, 1.0, &grads_ref).unwrap();
        dump_rust_step(&p_bad, 1.0, &grads_bad).unwrap();

        let tol = ParityTolerances {
            grad_max_rel: 0.02,
            ..Default::default()
        };
        let report = compare_dumps(&p_bad, &p_ref, &tol).unwrap();
        assert!(!report.passed);
        assert!(report.grad_diffs[0].max_rel_diff > 0.1);
    }

    #[test]
    fn test_compare_loss_tolerance() {
        let grads: Vec<NamedTensor> = Vec::new();
        let p1 = tmp_path("loss_a");
        let p2 = tmp_path("loss_b");
        dump_rust_step(&p1, 1.000, &grads).unwrap();
        dump_rust_step(&p2, 1.0005, &grads).unwrap();

        let tol = ParityTolerances::default();
        let r = compare_dumps(&p1, &p2, &tol).unwrap();
        assert!(r.passed);
        assert!(r.loss_diff < tol.loss_atol);
    }
}
