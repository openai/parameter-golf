/// Preliminary experiments — CPU-only results for Week 1 blog/paper.
///
/// 1. Quantization scheme sweep: ~20 configs, report compressed size + MSE
/// 2. Prune-then-quantize ordering A/B test: validate Progressive Intensity Hypothesis
/// 3. Architecture const-generic sweep: param counts + estimated sizes
use pg_model::arch::*;
use pg_model::config::ModelConfig;
use pg_quant::prune::*;
use pg_quant::scheme::*;

use std::time::Instant;

fn main() {
    println!("============================================================");
    println!("  PARAMETER GOLF — PRELIMINARY EXPERIMENTS (CPU)");
    println!("============================================================\n");

    experiment_1_quant_sweep();
    experiment_2_prune_ordering();
    experiment_3_arch_sweep();
}

// ─── Experiment 1: Quantization Scheme Sweep ────────────────────────────────

fn experiment_1_quant_sweep() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  Experiment 1: Quantization Scheme Sweep");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  Evaluating ~20 quantization configs on synthetic model weights.");
    println!("  Metrics: compressed artifact size (bytes) + reconstruction MSE.\n");

    let config = ModelConfig::sota();
    let n = config.num_layers;
    let d = config.model_dim;
    let kv = config.kv_dim();
    let mlp = config.mlp_dim;
    let vocab = config.vocab_size;

    // Element counts matching the SOTA model
    let attn_qkvo_elems = 2 * n * d * d + 2 * n * kv * d;
    let mlp_up_elems = n * mlp * d;
    let mlp_down_elems = n * d * mlp;
    let embed_elems = vocab * d;

    // Generate synthetic weights with realistic distribution
    let total_elems = attn_qkvo_elems + mlp_up_elems + mlp_down_elems + embed_elems;
    let synthetic_weights = generate_realistic_weights(total_elems);

    // Split into groups
    let (attn_w, rest) = synthetic_weights.split_at(attn_qkvo_elems);
    let (_mlp_up_w, rest) = rest.split_at(mlp_up_elems);
    let (_mlp_down_w, _embed_w) = rest.split_at(mlp_down_elems);

    // Define schemes to sweep
    let schemes: Vec<(&str, Scheme)> = vec![
        (
            "SOTA baseline (int6 attn / int5 MLP / int8 embed)",
            Scheme::sota_baseline(),
        ),
        (
            "Inverted (int5 attn / int6 MLP / int8 embed)",
            Scheme::inverted_split(),
        ),
        (
            "Aggressive (int4 attn / int5 MLP / int6 embed)",
            Scheme::aggressive(),
        ),
        ("Uniform int4", Scheme::uniform(Bits::B4)),
        ("Uniform int5", Scheme::uniform(Bits::B5)),
        ("Uniform int6", Scheme::uniform(Bits::B6)),
        ("Uniform int7", Scheme::uniform(Bits::B7)),
        ("Uniform int8", Scheme::uniform(Bits::B8)),
        (
            "int7 attn / int5 MLP",
            make_scheme(Bits::B7, Bits::B5, Bits::B8),
        ),
        (
            "int6 attn / int4 MLP",
            make_scheme(Bits::B6, Bits::B4, Bits::B8),
        ),
        (
            "int5 attn / int4 MLP / int6 embed",
            make_scheme(Bits::B5, Bits::B4, Bits::B6),
        ),
        (
            "int4 attn / int4 MLP / int4 embed",
            make_scheme(Bits::B4, Bits::B4, Bits::B4),
        ),
        (
            "int8 attn / int4 MLP",
            make_scheme(Bits::B8, Bits::B4, Bits::B8),
        ),
        (
            "int6 attn / int6 MLP / int6 embed",
            make_scheme(Bits::B6, Bits::B6, Bits::B6),
        ),
        (
            "int7 attn / int6 MLP / int8 embed",
            make_scheme(Bits::B7, Bits::B6, Bits::B8),
        ),
        (
            "int5 attn / int5 MLP / int6 embed",
            make_scheme(Bits::B5, Bits::B5, Bits::B6),
        ),
    ];

    let budget = 16_000_000usize;

    println!(
        "{:<50} {:>12} {:>12} {:>10} {:>12}",
        "Scheme", "Raw (bytes)", "Est zstd", "Fits 16MB?", "Attn MSE"
    );
    println!("{}", "-".repeat(100));

    let start = Instant::now();

    for (name, scheme) in &schemes {
        // Estimate compressed size
        let est = scheme.estimate_size(
            attn_qkvo_elems,
            mlp_up_elems,
            mlp_down_elems,
            embed_elems,
            0.65,
        );

        // Actually quantize the attention weights and measure MSE
        // Use a representative sub-matrix for MSE
        let test_rows = 32;
        let test_cols = 512;
        let test_w = &attn_w[..test_rows * test_cols];
        let packed = quantize_with(test_w, test_rows, test_cols, &scheme.attn_q);
        let recon = packed.dequantize();
        let mse: f64 = test_w
            .iter()
            .zip(recon.iter())
            .map(|(&a, &b)| ((a - b) as f64).powi(2))
            .sum::<f64>()
            / (test_rows * test_cols) as f64;

        let fits = if est <= budget { "✅ YES" } else { "❌ NO" };

        println!(
            "{:<50} {:>12} {:>12} {:>10} {:>12.6e}",
            name,
            packed.raw_bytes() * (attn_qkvo_elems / (test_rows * test_cols)),
            est,
            fits,
            mse
        );
    }

    let elapsed = start.elapsed();
    println!("\nSweep completed in {:.2}s\n", elapsed.as_secs_f64());
}

// ─── Experiment 2: Prune-then-Quantize A/B Ordering ─────────────────────────

fn experiment_2_prune_ordering() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  Experiment 2: Prune-then-Quantize Ordering (A/B Test)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  Progressive Intensity Hypothesis: prune (weak) before quantize (strong).");
    println!("  Path A: prune → quantize | Path B: quantize → prune\n");

    let matrix_sizes = vec![
        ("Small (64×128)", 64, 128),
        ("Medium (256×512)", 256, 512),
        ("Large (512×512)", 512, 512),
        ("GEMM-sized (512×1536)", 512, 1536),
    ];

    let keep_ratios = vec![0.95, 0.90, 0.85, 0.75, 0.50];
    let bit_configs = vec![("int4", Bits::B4), ("int5", Bits::B5), ("int6", Bits::B6)];

    let start = Instant::now();

    for (size_name, rows, cols) in &matrix_sizes {
        println!("  Matrix: {} ({}×{})", size_name, rows, cols);
        println!(
            "  {:<12} {:<8} {:>14} {:>14} {:>10}",
            "Keep ratio", "Bits", "MSE(A:P→Q)", "MSE(B:Q→P)", "A wins?"
        );
        println!("  {}", "-".repeat(65));

        let weights = generate_realistic_weights(rows * cols);

        for keep_ratio in &keep_ratios {
            for (bit_name, bits) in &bit_configs {
                let prune_cfg = PruneConfig {
                    strategy: PruneStrategy::TopKPerRow {
                        keep_ratio: *keep_ratio,
                    },
                    rescale_after_prune: true,
                };
                let quant_cfg = GroupConfig::new(*bits, Block::PerRow);

                let (mse_a, mse_b) =
                    ordering_ab_test(&weights, *rows, *cols, &prune_cfg, &quant_cfg);
                let winner = if mse_a <= mse_b + 1e-12 { "✅" } else { "❌" };

                println!(
                    "  {:<12} {:<8} {:>14.6e} {:>14.6e} {:>10}",
                    format!("{:.0}%", keep_ratio * 100.0),
                    bit_name,
                    mse_a,
                    mse_b,
                    winner
                );
            }
        }
        println!();
    }

    // 2:4 structured sparsity test
    println!("  === 2:4 Structured Sparsity (H100 native) ===");
    println!(
        "  {:<20} {:<8} {:>14} {:>14} {:>10}",
        "Matrix", "Bits", "MSE(A:P→Q)", "MSE(B:Q→P)", "A wins?"
    );
    println!("  {}", "-".repeat(70));

    for (size_name, rows, cols) in &matrix_sizes {
        let weights = generate_realistic_weights(rows * cols);
        for (bit_name, bits) in &bit_configs {
            let prune_cfg = PruneConfig {
                strategy: PruneStrategy::TwoToFour,
                rescale_after_prune: true,
            };
            let quant_cfg = GroupConfig::new(*bits, Block::PerRow);
            let (mse_a, mse_b) = ordering_ab_test(&weights, *rows, *cols, &prune_cfg, &quant_cfg);
            let winner = if mse_a <= mse_b + 1e-12 { "✅" } else { "❌" };
            println!(
                "  {:<20} {:<8} {:>14.6e} {:>14.6e} {:>10}",
                size_name, bit_name, mse_a, mse_b, winner
            );
        }
    }

    let elapsed = start.elapsed();
    println!(
        "\nOrdering test completed in {:.2}s\n",
        elapsed.as_secs_f64()
    );
}

// ─── Experiment 3: Architecture Sweep ───────────────────────────────────────

fn experiment_3_arch_sweep() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  Experiment 3: Const-Generic Architecture Sweep");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  10 architecture variants via Rust const generics.");
    println!("  Reporting: param counts, estimated compressed sizes, memory.\n");

    let budget = 16_000_000usize;

    println!(
        "{:<25} {:>8} {:>8} {:>6} {:>6} {:>5} {:>5} {:>10} {:>12} {:>12} {:>10}",
        "Variant",
        "D",
        "Layers",
        "Heads",
        "KV",
        "MLP×",
        "RoPE",
        "Params",
        "int5/6 size",
        "int4/5 size",
        "GPU mem"
    );
    println!("{}", "-".repeat(130));

    print_arch::<BaselineArch>("Baseline", budget);
    print_arch::<WideArch>("Wide", budget);
    print_arch::<WideNarrowMlpArch>("Wide-NarrowMLP", budget);
    print_arch::<DeepArch>("Deep (13L)", budget);
    print_arch::<DeepNarrowArch>("Deep-Narrow (16L)", budget);
    print_arch::<FullXsaArch>("Full-XSA", budget);
    print_arch::<MoreRopeArch>("More-RoPE", budget);
    print_arch::<HighGqaArch>("High-GQA", budget);
    print_arch::<LongCtxArch>("Long-Context", budget);
    print_arch::<AggressiveQuantArch>("Aggressive-Q (d640)", budget);

    println!();
    println!("  Legend: int5/6 = SOTA baseline quant, int4/5 = aggressive quant");
    println!("  Budget: 16,000,000 bytes (16 MB)");
    println!();
}

fn print_arch<A: ArchTrait>(name: &str, budget: usize) {
    let cfg = A::config();
    let n = cfg.num_layers;
    let d = cfg.model_dim;
    let kv = cfg.kv_dim();
    let mlp = cfg.mlp_dim;
    let vocab = cfg.vocab_size;

    // Count params
    let params = 2 * n * d * d       // qo_bank
        + 2 * n * kv * d             // kv_bank
        + n * mlp * d                // mlp_up
        + n * d * mlp                // mlp_down
        + vocab * d                  // tok_emb
        + cfg.bigram_vocab_size * cfg.bigram_dim // bigram_embed
        + d * cfg.bigram_dim         // bigram_proj
        + d                          // smear_gate
        + n * d * 2                  // per-layer scalars (approx)
        + n * cfg.num_heads; // q_gains

    // Compressed size estimates
    let attn = 2 * n * d * d + 2 * n * kv * d;
    let mlp_up = n * mlp * d;
    let mlp_down = n * d * mlp;
    let embed = vocab * d;

    let baseline_scheme = Scheme::sota_baseline();
    let aggressive_scheme = Scheme::aggressive();

    let size_baseline = baseline_scheme.estimate_size(attn, mlp_up, mlp_down, embed, 0.65);
    let size_aggr = aggressive_scheme.estimate_size(attn, mlp_up, mlp_down, embed, 0.65);

    let mem = pg_model::gpu::estimate_memory(&cfg, cfg.train_seq_len);
    let mem_gb = mem as f64 / (1024.0 * 1024.0 * 1024.0);

    let fit_b = if size_baseline <= budget {
        "✅"
    } else {
        "❌"
    };
    let fit_a = if size_aggr <= budget { "✅" } else { "❌" };

    println!(
        "{:<25} {:>8} {:>8} {:>6} {:>6} {:>5} {:>5} {:>10} {:>10} {} {:>10} {} {:>8.1} GB",
        name,
        d,
        n,
        cfg.num_heads,
        cfg.num_kv_heads,
        cfg.mlp_mult as usize,
        cfg.rope_dims,
        format_num(params),
        format_bytes(size_baseline),
        fit_b,
        format_bytes(size_aggr),
        fit_a,
        mem_gb
    );
}

// ─── Helpers ────────────────────────────────────────────────────────────────

fn make_scheme(attn_bits: Bits, mlp_bits: Bits, embed_bits: Bits) -> Scheme {
    let attn = GroupConfig::new(attn_bits, Block::PerRow);
    let mlp = GroupConfig::new(mlp_bits, Block::PerRow);
    let embed = GroupConfig::new(embed_bits, Block::PerRow);
    Scheme {
        attn_q: attn.clone(),
        attn_k: attn.clone(),
        attn_v: attn.clone(),
        attn_o: attn,
        mlp_up: mlp.clone(),
        mlp_down: mlp,
        embed,
    }
}

fn generate_realistic_weights(n: usize) -> Vec<f32> {
    // Simulate Xavier-initialized weights with a long tail
    let mut s = 0xDEADBEEFu32;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            let u = (s as f64) / (u32::MAX as f64);
            // Box-Muller approximation for normal distribution
            let v = ((u - 0.5) * 2.0).clamp(-0.999, 0.999);
            (v * 0.02) as f32 // Xavier scale for d=512
        })
        .collect()
}

fn format_num(n: usize) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1e6)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1e3)
    } else {
        format!("{}", n)
    }
}

fn format_bytes(n: usize) -> String {
    if n >= 1_000_000 {
        format!("{:.2} MB", n as f64 / 1e6)
    } else if n >= 1_000 {
        format!("{:.1} KB", n as f64 / 1e3)
    } else {
        format!("{} B", n)
    }
}
