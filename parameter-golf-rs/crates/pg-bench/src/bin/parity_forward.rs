use pg_model::{
    ExecutionPlan, ForwardBuffer, GptModel, ModelComputePrecision, RunSpec, TrainBackend,
    VariantFamily,
};

fn main() {
    let mut args = std::env::args().skip(1);
    let mut spec = None;
    let mut builtin = VariantFamily::BaselineSp8192;
    let mut backend = TrainBackend::Cpu;
    let mut use_spec_precision = false;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--spec" => spec = args.next(),
            "--use-spec-precision" => use_spec_precision = true,
            "--backend" => {
                if let Some(raw) = args.next() {
                    backend = parse_backend(&raw).unwrap_or(backend);
                }
            }
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
    let mut model = GptModel::new(config.clone());
    model.fill_deterministic();

    let tokens = config.train_seq_len.min(16);
    let input_ids: Vec<u32> = (0..tokens)
        .map(|i| ((i * 17 + 5) % config.vocab_size) as u32)
        .collect();
    let mut cpu_buf = ForwardBuffer::new(&config, tokens);
    model
        .forward_with_plan(&plan, &input_ids, &mut cpu_buf)
        .expect("cpu reference failed");

    println!("variant_fingerprint={}", plan.variant_fingerprint);
    println!("backend={backend:?}");
    println!("tokens={tokens}");
    println!("vocab_size={}", config.vocab_size);

    #[cfg(feature = "cuda")]
    {
        match run_cuda_parity(&plan, &model, &input_ids, &cpu_buf.logits) {
            Ok((max_abs, mean_abs)) => {
                println!("gpu_forward_max_abs_diff={max_abs:.6}");
                println!("gpu_forward_mean_abs_diff={mean_abs:.6}");
                println!(
                    "status={}",
                    if max_abs < 1e-3 {
                        "parity_ok"
                    } else {
                        "parity_failed"
                    }
                );
            }
            Err(err) => {
                println!("status=gpu_path_not_ready");
                println!("gpu_error={err}");
            }
        }
    }

    #[cfg(not(feature = "cuda"))]
    {
        let max_abs = cpu_buf
            .logits
            .iter()
            .fold(0.0f32, |acc, v| acc.max(v.abs()));
        println!("cpu_forward_max_abs_logit={max_abs:.6}");
        println!("status=cpu_reference_only");
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

#[cfg(feature = "cuda")]
fn run_cuda_parity(
    plan: &ExecutionPlan,
    model: &GptModel,
    input_ids: &[u32],
    cpu_logits: &[f32],
) -> Result<(f32, f32), String> {
    use cudarc::driver::CudaContext;
    use pg_core::{DType, GpuTensor};
    use pg_model::gpu::{GpuActivations, GpuModel};

    let ctx = CudaContext::new(0).map_err(|e| format!("cuda context: {e:?}"))?;
    let stream = ctx.default_stream();
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

    let gpu_logits_bytes = gpu_buf
        .logits
        .to_host_bytes()
        .map_err(|e| format!("logit download: {e}"))?;
    let gpu_logits: &[f32] = bytemuck::cast_slice(&gpu_logits_bytes);
    if gpu_logits.len() != cpu_logits.len() {
        return Err(format!(
            "logit length mismatch: cpu={} gpu={}",
            cpu_logits.len(),
            gpu_logits.len()
        ));
    }

    let mut max_abs = 0.0f32;
    let mut sum_abs = 0.0f32;
    for (cpu, gpu) in cpu_logits.iter().zip(gpu_logits.iter()) {
        let diff = (cpu - gpu).abs();
        max_abs = max_abs.max(diff);
        sum_abs += diff;
    }
    Ok((max_abs, sum_abs / cpu_logits.len() as f32))
}
