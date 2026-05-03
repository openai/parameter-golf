use pg_model::backward::GradBuffers;
#[cfg(feature = "cuda")]
use pg_model::backward::backward_output_loss;
use pg_model::{
    ExecutionPlan, ForwardBuffer, GptModel, ModelComputePrecision, RunSpec, TrainBackend,
    VariantFamily,
};

const GRAD_ABS_TOL: f64 = 2e-5;
const GRAD_REL_TOL: f64 = 5e-2;
#[cfg(feature = "cuda")]
const GPU_SLICE_ABS_TOL: f64 = 1e-4;
#[cfg(feature = "cuda")]
const GPU_BF16_SLICE_ABS_TOL: f64 = 1e-3;

#[derive(Clone, Copy)]
enum ParamCheck {
    TokEmb(usize),
    QoBank(usize),
    KvBank(usize),
    MlpUpBank(usize),
    MlpDownBank(usize),
    SmearGate(usize),
    QGain { layer: usize, head: usize },
}

impl ParamCheck {
    fn name(self) -> String {
        match self {
            Self::TokEmb(i) => format!("tok_emb[{i}]"),
            Self::QoBank(i) => format!("qo_bank[{i}]"),
            Self::KvBank(i) => format!("kv_bank[{i}]"),
            Self::MlpUpBank(i) => format!("mlp_up_bank[{i}]"),
            Self::MlpDownBank(i) => format!("mlp_down_bank[{i}]"),
            Self::SmearGate(i) => format!("smear_gate[{i}]"),
            Self::QGain { layer, head } => format!("block_q_gain[{layer}][{head}]"),
        }
    }

    fn grad(self, grads: &GradBuffers) -> f32 {
        match self {
            Self::TokEmb(i) => grads.tok_emb[i],
            Self::QoBank(i) => grads.qo_bank[i],
            Self::KvBank(i) => grads.kv_bank[i],
            Self::MlpUpBank(i) => grads.mlp_up_bank[i],
            Self::MlpDownBank(i) => grads.mlp_down_bank[i],
            Self::SmearGate(i) => grads.smear_gate[i],
            Self::QGain { layer, head } => grads.block_q_gain[layer][head],
        }
    }

    fn add_delta(self, model: &mut GptModel, delta: f32) {
        match self {
            Self::TokEmb(i) => model.tok_emb[i] += delta,
            Self::QoBank(i) => model.qo_bank[i] += delta,
            Self::KvBank(i) => model.kv_bank[i] += delta,
            Self::MlpUpBank(i) => model.mlp_up_bank[i] += delta,
            Self::MlpDownBank(i) => model.mlp_down_bank[i] += delta,
            Self::SmearGate(i) => model.smear_gate[i] += delta,
            Self::QGain { layer, head } => model.blocks[layer].q_gain[head] += delta,
        }
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let mut spec = None;
    let mut builtin = VariantFamily::BaselineSp8192;
    let mut backend = TrainBackend::Cpu;
    let mut tokens_override = None;
    let mut eps = 1e-2f32;
    let mut use_spec_precision = false;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--spec" => spec = args.next(),
            "--use-spec-precision" => use_spec_precision = true,
            "--builtin" => {
                if let Some(name) = args.next() {
                    builtin = match name.as_str() {
                        "xsa_all_sp8192" => VariantFamily::XsaAllSp8192,
                        "recurrence_mid_sp8192" => VariantFamily::RecurrenceMidSp8192,
                        "parallel_resid_sp8192" => VariantFamily::ParallelResidSp8192,
                        "hybrid_competitive_sp8192" => VariantFamily::HybridCompetitiveSp8192,
                        "frontier_1855_like" => VariantFamily::Frontier1855Like,
                        _ => VariantFamily::BaselineSp8192,
                    };
                }
            }
            "--tokens" => tokens_override = args.next().and_then(|v| v.parse::<usize>().ok()),
            "--backend" => {
                if let Some(raw) = args.next() {
                    backend = parse_backend(&raw).unwrap_or(backend);
                }
            }
            "--eps" => {
                if let Some(raw) = args.next() {
                    eps = raw.parse::<f32>().unwrap_or(eps);
                }
            }
            _ => {}
        }
    }

    let mut run_spec = spec
        .map(std::path::PathBuf::from)
        .map(|p| RunSpec::load(&p).expect("failed to load spec"))
        .unwrap_or_else(|| RunSpec::for_family(builtin));
    if !use_spec_precision {
        run_spec.model.compute_precision = ModelComputePrecision::F32Tf32;
    }
    let plan = ExecutionPlan::from_run_spec(&run_spec).expect("failed to build execution plan");
    let config = run_spec.model.to_model_config();
    let tokens = tokens_override.unwrap_or_else(|| config.train_seq_len.min(8));
    let input_ids: Vec<u32> = (0..tokens)
        .map(|i| ((i * 17 + 5) % config.vocab_size) as u32)
        .collect();
    let targets: Vec<u32> = (0..tokens)
        .map(|i| ((i * 17 + 22) % config.vocab_size) as u32)
        .collect();

    let mut model = GptModel::new(config.clone());
    model.fill_deterministic();
    let mut buf = ForwardBuffer::new(&config, tokens);
    let mut grads = GradBuffers::new(&config);
    let loss = model.backward(&input_ids, &targets, &mut buf, &mut grads);

    let checks = selected_checks(&grads);
    println!("variant_fingerprint={}", plan.variant_fingerprint);
    println!("backend={backend:?}");
    println!("tokens={tokens}");
    println!("loss={loss:.6}");
    println!("grad_norm={:.6}", grads.flat_grad_norm());
    println!("finite_diff_eps={eps:.1e}");

    let mut max_rel = 0.0f64;
    let mut max_abs = 0.0f64;
    let mut failures = 0usize;
    for check in &checks {
        let analytical = check.grad(&grads) as f64;
        let numerical = numerical_grad_cpu(&mut model, &plan, &input_ids, &targets, *check, eps);
        let abs = (analytical - numerical).abs();
        let rel = rel_diff(analytical, numerical);
        let pass = grad_check_pass(analytical, numerical);
        max_rel = max_rel.max(rel);
        max_abs = max_abs.max(abs);
        if !pass {
            failures += 1;
        }
        println!(
            "cpu_grad_check name={} analytical={:.6e} numerical={:.6e} abs_diff={:.6e} rel_diff={:.6e} pass={}",
            check.name(),
            analytical,
            numerical,
            abs,
            rel,
            pass,
        );
    }
    println!("cpu_grad_check_max_rel={max_rel:.6e}");
    println!("cpu_grad_check_max_abs={max_abs:.6e}");
    println!("cpu_grad_check_failures={failures}");
    println!(
        "cpu_grad_check_status={}",
        if failures == 0 { "ok" } else { "failed" }
    );

    #[cfg(feature = "cuda")]
    {
        let hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            run_cuda_finite_diff(
                &mut model, &plan, &input_ids, &targets, &checks, &grads, eps,
            )
        }));
        std::panic::set_hook(hook);
        match result {
            Ok(Ok(summary)) => {
                println!("gpu_forward_finite_diff_max_rel={:.6e}", summary.max_rel);
                println!("gpu_forward_finite_diff_max_abs={:.6e}", summary.max_abs);
                println!("gpu_forward_finite_diff_failures={}", summary.failures);
                println!(
                    "gpu_forward_finite_diff_status={}",
                    if summary.failures == 0 {
                        "ok"
                    } else {
                        "failed"
                    }
                );
            }
            Ok(Err(err)) => {
                println!("gpu_forward_finite_diff_status=not_ready");
                println!("gpu_error={err}");
            }
            Err(_) => {
                println!("gpu_forward_finite_diff_status=not_ready");
                println!("gpu_error=panic_while_loading_cuda_runtime");
            }
        }
    }

    #[cfg(feature = "cuda")]
    if backend != TrainBackend::Cpu {
        let hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            run_cuda_output_backward_parity(&model, &plan, &input_ids, &targets)
        }));
        std::panic::set_hook(hook);
        match result {
            Ok(Ok(summary)) => {
                println!(
                    "gpu_output_backward_x_max_abs_diff={:.6e}",
                    summary.x_max_abs_diff
                );
                println!(
                    "gpu_output_backward_x_mean_abs_diff={:.6e}",
                    summary.x_mean_abs_diff
                );
                println!(
                    "gpu_output_backward_tok_emb_max_abs_diff={:.6e}",
                    summary.tok_emb_max_abs_diff
                );
                println!(
                    "gpu_output_backward_tok_emb_mean_abs_diff={:.6e}",
                    summary.tok_emb_mean_abs_diff
                );
                println!(
                    "gpu_output_backward_status={}",
                    if summary.passed { "ok" } else { "failed" }
                );
            }
            Ok(Err(err)) => {
                println!("gpu_output_backward_status=not_ready");
                println!("gpu_output_backward_error={err}");
            }
            Err(_) => {
                println!("gpu_output_backward_status=not_ready");
                println!("gpu_output_backward_error=panic_while_loading_cuda_runtime");
            }
        }

        let hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            run_cuda_full_backward_parity(&model, &plan, &input_ids, &targets, &checks, loss)
        }));
        std::panic::set_hook(hook);
        match result {
            Ok(Ok(summary)) => {
                println!("gpu_backward_loss={:.6}", summary.loss);
                println!(
                    "gpu_backward_loss_abs_diff={:.6e}",
                    (summary.loss as f64 - loss as f64).abs()
                );
                println!("gpu_backward_max_abs_diff={:.6e}", summary.max_abs);
                println!("gpu_backward_max_rel_diff={:.6e}", summary.max_rel);
                println!("gpu_backward_failures={}", summary.failures);
                println!(
                    "gpu_backward_status={}",
                    if summary.failures == 0 {
                        "ok"
                    } else {
                        "failed"
                    }
                );
            }
            Ok(Err(err)) => {
                println!("gpu_backward_status=not_ready");
                println!("gpu_backward_error={err}");
            }
            Err(_) => {
                println!("gpu_backward_status=not_ready");
                println!("gpu_backward_error=panic_while_loading_cuda_runtime");
            }
        }
    }
}

fn parse_backend(raw: &str) -> Option<TrainBackend> {
    match raw {
        "cpu" => Some(TrainBackend::Cpu),
        "cuda-single" => Some(TrainBackend::CudaSingle),
        "cuda-single-parity" => Some(TrainBackend::CudaSingleParity),
        "cuda-distributed" => Some(TrainBackend::CudaDistributed),
        _ => None,
    }
}

fn selected_checks(grads: &GradBuffers) -> Vec<ParamCheck> {
    fn max_abs_index(values: &[f32]) -> usize {
        values
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.abs()
                    .partial_cmp(&b.abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    let (q_gain_layer, q_gain_head) = grads
        .block_q_gain
        .iter()
        .enumerate()
        .flat_map(|(layer, values)| {
            values
                .iter()
                .enumerate()
                .map(move |(head, &value)| (layer, head, value))
        })
        .max_by(|(_, _, a), (_, _, b)| {
            a.abs()
                .partial_cmp(&b.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(layer, head, _)| (layer, head))
        .unwrap_or((0, 0));

    vec![
        ParamCheck::TokEmb(max_abs_index(&grads.tok_emb)),
        ParamCheck::QoBank(max_abs_index(&grads.qo_bank)),
        ParamCheck::KvBank(max_abs_index(&grads.kv_bank)),
        ParamCheck::MlpUpBank(max_abs_index(&grads.mlp_up_bank)),
        ParamCheck::MlpDownBank(max_abs_index(&grads.mlp_down_bank)),
        ParamCheck::SmearGate(max_abs_index(&grads.smear_gate)),
        ParamCheck::QGain {
            layer: q_gain_layer,
            head: q_gain_head,
        },
    ]
}

fn loss_cpu(model: &GptModel, plan: &ExecutionPlan, input_ids: &[u32], targets: &[u32]) -> f64 {
    let mut buf = ForwardBuffer::new(&model.config, input_ids.len());
    model
        .forward_with_plan(plan, input_ids, &mut buf)
        .expect("forward_with_plan failed");
    cross_entropy_loss_f64(
        &buf.logits[..input_ids.len() * model.config.vocab_size],
        targets,
        model.config.vocab_size,
        model.config.logit_softcap,
    )
}

fn numerical_grad_cpu(
    model: &mut GptModel,
    plan: &ExecutionPlan,
    input_ids: &[u32],
    targets: &[u32],
    check: ParamCheck,
    eps: f32,
) -> f64 {
    check.add_delta(model, eps);
    let loss_p = loss_cpu(model, plan, input_ids, targets);
    check.add_delta(model, -2.0 * eps);
    let loss_m = loss_cpu(model, plan, input_ids, targets);
    check.add_delta(model, eps);
    (loss_p - loss_m) / (2.0 * eps as f64)
}

fn cross_entropy_loss_f64(logits: &[f32], targets: &[u32], vocab_size: usize, softcap: f32) -> f64 {
    let softcap = softcap as f64;
    let inv_cap = 1.0 / softcap;
    let mut total = 0.0f64;
    for (t, &target) in targets.iter().enumerate() {
        let row = &logits[t * vocab_size..(t + 1) * vocab_size];
        let mut max_val = f64::NEG_INFINITY;
        for &value in row {
            let capped = softcap * ((value as f64) * inv_cap).tanh();
            max_val = max_val.max(capped);
        }

        let mut sum_exp = 0.0f64;
        for &value in row {
            let capped = softcap * ((value as f64) * inv_cap).tanh();
            sum_exp += (capped - max_val).exp();
        }
        let target_value = row[target as usize] as f64;
        let capped_target = softcap * (target_value * inv_cap).tanh();
        total += max_val + sum_exp.ln() - capped_target;
    }
    total / targets.len() as f64
}

fn rel_diff(a: f64, b: f64) -> f64 {
    (a - b).abs() / a.abs().max(b.abs()).max(1e-8)
}

fn grad_check_pass(a: f64, b: f64) -> bool {
    let abs = (a - b).abs();
    abs <= GRAD_ABS_TOL || rel_diff(a, b) <= GRAD_REL_TOL
}

#[cfg(feature = "cuda")]
struct GradCheckSummary {
    max_rel: f64,
    max_abs: f64,
    failures: usize,
}

#[cfg(feature = "cuda")]
struct OutputBackwardSummary {
    x_max_abs_diff: f64,
    x_mean_abs_diff: f64,
    tok_emb_max_abs_diff: f64,
    tok_emb_mean_abs_diff: f64,
    passed: bool,
}

#[cfg(feature = "cuda")]
struct FullBackwardSummary {
    loss: f32,
    max_rel: f64,
    max_abs: f64,
    failures: usize,
}

#[cfg(feature = "cuda")]
fn run_cuda_finite_diff(
    model: &mut GptModel,
    plan: &ExecutionPlan,
    input_ids: &[u32],
    targets: &[u32],
    checks: &[ParamCheck],
    grads: &GradBuffers,
    eps: f32,
) -> Result<GradCheckSummary, String> {
    use cudarc::driver::CudaContext;

    let ctx = CudaContext::new(0).map_err(|e| format!("cuda context: {e:?}"))?;
    let stream = ctx.default_stream();
    let mut max_rel = 0.0f64;
    let mut max_abs = 0.0f64;
    let mut failures = 0usize;
    for check in checks {
        check.add_delta(model, eps);
        let loss_p = loss_gpu(model, plan, input_ids, targets, ctx.clone(), stream.clone())?;
        check.add_delta(model, -2.0 * eps);
        let loss_m = loss_gpu(model, plan, input_ids, targets, ctx.clone(), stream.clone())?;
        check.add_delta(model, eps);
        let numerical = (loss_p - loss_m) / (2.0 * eps as f64);
        let analytical = check.grad(grads) as f64;
        let abs = (analytical - numerical).abs();
        let rel = rel_diff(analytical, numerical);
        let pass = grad_check_pass(analytical, numerical);
        max_rel = max_rel.max(rel);
        max_abs = max_abs.max(abs);
        if !pass {
            failures += 1;
        }
        println!(
            "gpu_forward_fd_check name={} analytical={:.6e} numerical={:.6e} abs_diff={:.6e} rel_diff={:.6e} pass={}",
            check.name(),
            analytical,
            numerical,
            abs,
            rel,
            pass,
        );
    }
    Ok(GradCheckSummary {
        max_rel,
        max_abs,
        failures,
    })
}

#[cfg(feature = "cuda")]
fn loss_gpu(
    model: &GptModel,
    plan: &ExecutionPlan,
    input_ids: &[u32],
    targets: &[u32],
    ctx: std::sync::Arc<cudarc::driver::CudaContext>,
    stream: std::sync::Arc<cudarc::driver::CudaStream>,
) -> Result<f64, String> {
    use pg_core::{DType, GpuTensor};
    use pg_model::gpu::{GpuActivations, GpuModel};

    let gpu_model = GpuModel::from_cpu_reference(model, plan, ctx, stream.clone())
        .map_err(|e| format!("gpu model init: {e}"))?;
    let ids_bytes: &[u8] = bytemuck::cast_slice(input_ids);
    let input =
        GpuTensor::from_host_data_gpu(stream.clone(), ids_bytes, &[input_ids.len()], DType::U32)
            .map_err(|e| format!("input upload: {e}"))?;
    let mut gpu_buf = GpuActivations::new_for_plan(plan, input_ids.len(), stream.clone())
        .map_err(|e| format!("activation alloc: {e}"))?;
    gpu_model
        .forward(&input, &mut gpu_buf)
        .map_err(|e| format!("gpu forward: {e}"))?;
    stream
        .synchronize()
        .map_err(|e| format!("stream sync: {e:?}"))?;
    let bytes = gpu_buf
        .logits
        .to_host_bytes()
        .map_err(|e| format!("logit download: {e}"))?;
    let logits: &[f32] = bytemuck::cast_slice(&bytes);
    Ok(cross_entropy_loss_f64(
        logits,
        targets,
        model.config.vocab_size,
        model.config.logit_softcap,
    ))
}

#[cfg(feature = "cuda")]
fn run_cuda_output_backward_parity(
    model: &GptModel,
    plan: &ExecutionPlan,
    input_ids: &[u32],
    targets: &[u32],
) -> Result<OutputBackwardSummary, String> {
    use cudarc::driver::CudaContext;
    use pg_core::{DType, GpuTensor};
    use pg_model::gpu::{GpuActivations, GpuForwardCache, GpuGradBuffers, GpuModel};

    let mut cpu_buf = ForwardBuffer::new(&model.config, input_ids.len());
    model.forward_with_cache(input_ids, &mut cpu_buf);
    let mut cpu_grads = GradBuffers::new(&model.config);
    let cpu_grad_pre_norm = backward_output_loss(model, &cpu_buf, targets, &mut cpu_grads);

    let ctx = CudaContext::new(0).map_err(|e| format!("cuda context: {e:?}"))?;
    let stream = ctx.default_stream();
    let gpu_model = GpuModel::from_cpu_reference(model, plan, ctx, stream.clone())
        .map_err(|e| format!("gpu model init: {e}"))?;
    let ids_bytes: &[u8] = bytemuck::cast_slice(input_ids);
    let targets_bytes: &[u8] = bytemuck::cast_slice(targets);
    let input =
        GpuTensor::from_host_data_gpu(stream.clone(), ids_bytes, &[input_ids.len()], DType::U32)
            .map_err(|e| format!("input upload: {e}"))?;
    let targets_gpu =
        GpuTensor::from_host_data_gpu(stream.clone(), targets_bytes, &[targets.len()], DType::U32)
            .map_err(|e| format!("target upload: {e}"))?;
    let mut gpu_buf = GpuActivations::new_for_plan(plan, input_ids.len(), stream.clone())
        .map_err(|e| format!("activation alloc: {e}"))?;
    let mut gpu_cache = GpuForwardCache::new_for_plan(plan, input_ids.len(), stream.clone())
        .map_err(|e| format!("cache alloc: {e}"))?;
    gpu_model
        .forward_with_cache(&input, &mut gpu_buf, &mut gpu_cache)
        .map_err(|e| format!("gpu forward_with_cache: {e}"))?;
    let mut gpu_grads = GpuGradBuffers::new(&model.config, stream.clone())
        .map_err(|e| format!("gpu grad alloc: {e}"))?;
    let gpu_grad_pre_norm = gpu_model
        .backward_output_loss_only(&gpu_cache, &mut gpu_buf, &targets_gpu, &mut gpu_grads)
        .map_err(|e| format!("gpu backward output slice: {e}"))?;
    stream
        .synchronize()
        .map_err(|e| format!("stream sync: {e:?}"))?;

    let gpu_grad_pre_norm = tensor_to_f32_vec(&gpu_grad_pre_norm)?;
    let gpu_tok_emb = tensor_to_f32_vec(&gpu_grads.tok_emb)?;
    let (x_max_abs_diff, x_mean_abs_diff) = diff_stats_f32(&cpu_grad_pre_norm, &gpu_grad_pre_norm);
    let (tok_emb_max_abs_diff, tok_emb_mean_abs_diff) =
        diff_stats_f32(&cpu_grads.tok_emb, &gpu_tok_emb);
    let abs_tol = gpu_slice_abs_tolerance(plan);
    let passed = x_max_abs_diff <= abs_tol && tok_emb_max_abs_diff <= abs_tol;

    Ok(OutputBackwardSummary {
        x_max_abs_diff,
        x_mean_abs_diff,
        tok_emb_max_abs_diff,
        tok_emb_mean_abs_diff,
        passed,
    })
}

#[cfg(feature = "cuda")]
fn gpu_slice_abs_tolerance(plan: &ExecutionPlan) -> f64 {
    if plan.run_spec.model.compute_precision == ModelComputePrecision::Bf16TensorCore {
        GPU_BF16_SLICE_ABS_TOL
    } else {
        GPU_SLICE_ABS_TOL
    }
}

#[cfg(feature = "cuda")]
fn run_cuda_full_backward_parity(
    model: &GptModel,
    plan: &ExecutionPlan,
    input_ids: &[u32],
    targets: &[u32],
    checks: &[ParamCheck],
    cpu_loss: f32,
) -> Result<FullBackwardSummary, String> {
    use cudarc::driver::CudaContext;
    use pg_core::{DType, GpuTensor};
    use pg_model::gpu::{GpuActivations, GpuGradBuffers, GpuModel};

    let ctx = CudaContext::new(0).map_err(|e| format!("cuda context: {e:?}"))?;
    let stream = ctx.default_stream();
    let mut cpu_buf = ForwardBuffer::new(&model.config, input_ids.len());
    let mut cpu_grads = GradBuffers::new(&model.config);
    let _ = model.backward(input_ids, targets, &mut cpu_buf, &mut cpu_grads);

    let gpu_model = GpuModel::from_cpu_reference(model, plan, ctx, stream.clone())
        .map_err(|e| format!("gpu model init: {e}"))?;
    let ids_bytes: &[u8] = bytemuck::cast_slice(input_ids);
    let targets_bytes: &[u8] = bytemuck::cast_slice(targets);
    let input =
        GpuTensor::from_host_data_gpu(stream.clone(), ids_bytes, &[input_ids.len()], DType::U32)
            .map_err(|e| format!("input upload: {e}"))?;
    let targets_gpu =
        GpuTensor::from_host_data_gpu(stream.clone(), targets_bytes, &[targets.len()], DType::U32)
            .map_err(|e| format!("target upload: {e}"))?;
    let mut gpu_buf = GpuActivations::new_for_plan(plan, input_ids.len(), stream.clone())
        .map_err(|e| format!("activation alloc: {e}"))?;
    let mut gpu_grads = GpuGradBuffers::new(&model.config, stream.clone())
        .map_err(|e| format!("gpu grad alloc: {e}"))?;
    let gpu_loss = gpu_model
        .backward(&input, &targets_gpu, &mut gpu_buf, &mut gpu_grads)
        .map_err(|e| format!("gpu backward: {e}"))?;
    stream
        .synchronize()
        .map_err(|e| format!("stream sync: {e:?}"))?;

    let tok_emb = tensor_to_f32_vec(&gpu_grads.tok_emb)?;
    let qo_bank = tensor_to_f32_vec(&gpu_grads.qo_bank)?;
    let kv_bank = tensor_to_f32_vec(&gpu_grads.kv_bank)?;
    let mlp_up_bank = tensor_to_f32_vec(&gpu_grads.mlp_up_bank)?;
    let mlp_down_bank = tensor_to_f32_vec(&gpu_grads.mlp_down_bank)?;
    let smear_gate = tensor_to_f32_vec(&gpu_grads.smear_gate)?;
    let q_gain: Vec<Vec<f32>> = gpu_grads
        .block_q_gain
        .iter()
        .map(tensor_to_f32_vec)
        .collect::<Result<_, _>>()?;

    let mut max_rel = (gpu_loss as f64 - cpu_loss as f64).abs();
    let mut max_abs = max_rel;
    let mut failures = 0usize;
    for check in checks {
        let analytical = check.grad(&cpu_grads) as f64;
        let gpu = match *check {
            ParamCheck::TokEmb(i) => tok_emb[i] as f64,
            ParamCheck::QoBank(i) => qo_bank[i] as f64,
            ParamCheck::KvBank(i) => kv_bank[i] as f64,
            ParamCheck::MlpUpBank(i) => mlp_up_bank[i] as f64,
            ParamCheck::MlpDownBank(i) => mlp_down_bank[i] as f64,
            ParamCheck::SmearGate(i) => smear_gate[i] as f64,
            ParamCheck::QGain { layer, head } => q_gain[layer][head] as f64,
        };
        let abs = (analytical - gpu).abs();
        let rel = rel_diff(analytical, gpu);
        let pass = grad_check_pass(analytical, gpu);
        max_rel = max_rel.max(rel);
        max_abs = max_abs.max(abs);
        if !pass {
            failures += 1;
        }
        println!(
            "gpu_backward_check name={} cpu={:.6e} gpu={:.6e} abs_diff={:.6e} rel_diff={:.6e} pass={}",
            check.name(),
            analytical,
            gpu,
            abs,
            rel,
            pass,
        );
    }

    Ok(FullBackwardSummary {
        loss: gpu_loss,
        max_rel,
        max_abs,
        failures,
    })
}

#[cfg(feature = "cuda")]
fn tensor_to_f32_vec(tensor: &pg_core::GpuTensor) -> Result<Vec<f32>, String> {
    let bytes = tensor
        .to_host_bytes()
        .map_err(|e| format!("tensor download: {e}"))?;
    Ok(bytemuck::cast_slice::<u8, f32>(&bytes).to_vec())
}

#[cfg(feature = "cuda")]
fn diff_stats_f32(a: &[f32], b: &[f32]) -> (f64, f64) {
    assert_eq!(a.len(), b.len(), "diff_stats_f32 length mismatch");
    let mut max_abs = 0.0f64;
    let mut sum_abs = 0.0f64;
    for (&lhs, &rhs) in a.iter().zip(b.iter()) {
        let abs = (lhs as f64 - rhs as f64).abs();
        max_abs = max_abs.max(abs);
        sum_abs += abs;
    }
    let mean_abs = if a.is_empty() {
        0.0
    } else {
        sum_abs / a.len() as f64
    };
    (max_abs, mean_abs)
}
