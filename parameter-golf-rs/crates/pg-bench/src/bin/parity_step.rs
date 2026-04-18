use pg_model::{ExecutionPlan, ForwardBuffer, GptModel, RunSpec, VariantFamily};
use pg_model::backward::GradBuffers;

fn main() {
    let mut args = std::env::args().skip(1);
    let mut spec = None;
    let mut builtin = VariantFamily::BaselineSp8192;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--spec" => spec = args.next(),
            "--builtin" => {
                if let Some(name) = args.next() {
                    builtin = match name.as_str() {
                        "xsa_all_sp8192" => VariantFamily::XsaAllSp8192,
                        "recurrence_mid_sp8192" => VariantFamily::RecurrenceMidSp8192,
                        "parallel_resid_sp8192" => VariantFamily::ParallelResidSp8192,
                        "hybrid_competitive_sp8192" => VariantFamily::HybridCompetitiveSp8192,
                        _ => VariantFamily::BaselineSp8192,
                    };
                }
            }
            _ => {}
        }
    }

    let run_spec = spec
        .map(std::path::PathBuf::from)
        .map(|p| RunSpec::load(&p).expect("failed to load spec"))
        .unwrap_or_else(|| RunSpec::for_family(builtin));
    let plan = ExecutionPlan::from_run_spec(&run_spec).expect("failed to build execution plan");
    let config = run_spec.model.to_model_config();
    let mut model = GptModel::new(config.clone());
    let mut buf = ForwardBuffer::new(&config, config.train_seq_len.min(64));
    let mut grads = GradBuffers::new(&config);
    let input_ids: Vec<u32> = (0..buf.tokens).map(|i| (i % config.vocab_size) as u32).collect();
    let targets: Vec<u32> = (1..=buf.tokens).map(|i| (i % config.vocab_size) as u32).collect();
    let loss = model.backward(&input_ids, &targets, &mut buf, &mut grads);

    println!("variant_fingerprint={}", plan.variant_fingerprint);
    println!("tokens={}", buf.tokens);
    println!("loss={loss:.6}");
    println!("grad_norm={:.6}", grads.flat_grad_norm());
    println!("status=cpu_step_reference_ok");
}
