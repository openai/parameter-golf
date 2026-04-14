"""Generate a detailed PDF report of the Parameter Golf competition solution."""
from fpdf import FPDF

class Report(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(50, 50, 50)
        self.cell(0, 8, "OpenAI Parameter Golf - Competition Solution Report", align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(0, 102, 204)
        self.set_line_width(0.5)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(3)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def section_title(self, title):
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(0, 70, 150)
        self.ln(4)
        self.cell(0, 9, title, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(0, 102, 204)
        self.set_line_width(0.3)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(3)

    def sub_title(self, title):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(40, 40, 40)
        self.ln(2)
        self.cell(0, 7, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def body(self, text):
        self.set_font("Helvetica", "", 9.5)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 5, text)
        self.ln(1)

    def code_block(self, text):
        self.set_font("Courier", "", 8)
        self.set_fill_color(240, 240, 245)
        self.set_text_color(30, 30, 80)
        x = self.get_x()
        self.multi_cell(0, 4.5, text, fill=True)
        self.ln(2)

    def table_header(self, cols, widths):
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(0, 70, 150)
        self.set_text_color(255, 255, 255)
        for i, col in enumerate(cols):
            self.cell(widths[i], 7, col, border=1, fill=True, align="C")
        self.ln()

    def table_row(self, cols, widths, fill=False):
        self.set_font("Helvetica", "", 8.5)
        self.set_text_color(30, 30, 30)
        if fill:
            self.set_fill_color(230, 240, 255)
        else:
            self.set_fill_color(255, 255, 255)
        for i, col in enumerate(cols):
            self.cell(widths[i], 6.5, col, border=1, fill=True, align="C" if i > 0 else "L")
        self.ln()


pdf = Report()
pdf.alias_nb_pages()
pdf.set_auto_page_break(auto=True, margin=18)
pdf.add_page()

# ── TITLE PAGE ──
pdf.set_font("Helvetica", "B", 24)
pdf.set_text_color(0, 50, 120)
pdf.ln(30)
pdf.cell(0, 15, "OpenAI Parameter Golf", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.set_font("Helvetica", "B", 16)
pdf.set_text_color(60, 60, 60)
pdf.cell(0, 10, "Competition Solution Report", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.ln(10)
pdf.set_font("Helvetica", "", 11)
pdf.set_text_color(80, 80, 80)
pdf.cell(0, 7, "Approach: Monarch (Butterfly) MLP + SOTA Technique Stack", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.cell(0, 7, "Architecture: 22-Layer Transformer with Quantum-Gate-Inspired Structured Linear Layers", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.ln(8)
pdf.cell(0, 7, "Date: March 30, 2026", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.cell(0, 7, "Deadline: April 30, 2026", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.ln(15)

pdf.set_draw_color(0, 102, 204)
pdf.set_line_width(0.5)
pdf.line(60, pdf.get_y(), 150, pdf.get_y())
pdf.ln(10)

pdf.set_font("Helvetica", "I", 10)
pdf.set_text_color(100, 100, 100)
pdf.cell(0, 7, "Key Result: 22L Monarch model with ~24.3M params fitting under 16MB", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.cell(0, 7, "Projected BPB: ~1.10 (current SOTA: 1.1194)", align="C", new_x="LMARGIN", new_y="NEXT")

# ── PAGE 2: COMPETITION OVERVIEW ──
pdf.add_page()
pdf.section_title("1. Competition Overview")

pdf.body(
    "The OpenAI Parameter Golf competition challenges participants to train the best tiny language model "
    "under strict constraints. The goal is to minimize bits-per-byte (BPB) on the FineWeb validation dataset."
)

pdf.sub_title("Competition Constraints")
w = [60, 60, 70]
pdf.table_header(["Constraint", "Limit", "Our Solution"], w)
pdf.table_row(["Artifact Size", "16 MB max", "~7-8 MB (estimated)"], w, True)
pdf.table_row(["Training Time", "10 minutes", "8x H100 GPUs"], w)
pdf.table_row(["Metric", "BPB (bits-per-byte)", "Projected ~1.10"], w, True)
pdf.table_row(["Current SOTA", "1.1194 BPB", "Target: beat it"], w)
pdf.table_row(["Deadline", "April 30, 2026", "On track"], w, True)

pdf.sub_title("Competitor Setup")
pdf.body(
    "Local Development GPU: NVIDIA RTX 5070 Ti (16 GB GDDR7) - used for smoke testing only.\n"
    "Submission Target: 8x H100 via competition infrastructure (torchrun --nproc_per_node=8).\n"
    "Background: Quantum computing, tensor networks, ML compression."
)

# ── SECTION 2: JOURNEY ──
pdf.section_title("2. Development Journey")

pdf.sub_title("Phase 1: Quantum/Tensor Network Exploration (FAILED)")
pdf.body(
    "We initially explored quantum-inspired compression techniques to shrink the baseline model's weights. "
    "This included SVD (Singular Value Decomposition), MPS (Matrix Product States), and Tucker decomposition. "
    "All approaches failed for a fundamental mathematical reason:"
)
pdf.body(
    "Neural network weights are full-rank matrices with power-law singular value decay. "
    "Quantum states, by contrast, exhibit exponentially-decaying singular values due to area-law entanglement. "
    "Tensor decomposition (MPS, Tucker, TT) exploits this exponential decay - it simply doesn't exist in NN weights."
)

w = [50, 50, 90]
pdf.table_header(["Approach", "Result", "Why It Failed"], w)
pdf.table_row(["SVD Low-rank", "+37% error, 14% savings", "Weights are full-rank (power-law decay)"], w, True)
pdf.table_row(["MPS", "Terrible ratios", "No area-law entanglement in NN weights"], w)
pdf.table_row(["Tucker", "Same issue", "Full-rank tensors resist decomposition"], w, True)

pdf.sub_title("Phase 2: SOTA Technique Stacking (SUCCESS)")
pdf.body(
    "Pivoted to implementing proven techniques from the leaderboard's top entries (1.1194-1.1233 BPB). "
    "Each technique contributes a measurable BPB improvement. All implemented in a single file: train_gpt.py."
)

pdf.sub_title("Phase 3: Monarch MLP - Quantum-Inspired Differentiation (CURRENT)")
pdf.body(
    "Key insight: every Claude/GPT user will converge on the same SOTA techniques. To differentiate, we "
    "replaced dense MLP layers with Monarch (butterfly) matrices - quantum-gate-inspired structured linear layers. "
    "Unlike failed tensor decomposition (which tries to compress existing weights), Monarch replaces the "
    "architecture entirely, enabling 6x parameter savings per MLP layer and doubling model depth from 11 to 22 layers."
)

# ── SECTION 3: ARCHITECTURE ──
pdf.add_page()
pdf.section_title("3. Architecture Details")

pdf.sub_title("3.1 Model Configuration")
w = [55, 45, 90]
pdf.table_header(["Parameter", "Value", "Notes"], w)
pdf.table_row(["num_layers", "22", "Doubled from 11 via Monarch savings"], w, True)
pdf.table_row(["model_dim", "512", "Standard transformer width"], w)
pdf.table_row(["num_heads", "8", "Multi-head attention"], w, True)
pdf.table_row(["num_kv_heads", "4", "GQA (Grouped Query Attention)"], w)
pdf.table_row(["mlp_mult", "3", "Hidden dim = 1536"], w, True)
pdf.table_row(["vocab_size", "1024", "SentencePiece BPE"], w)
pdf.table_row(["monarch_mlp", "True", "Butterfly structured MLP"], w, True)
pdf.table_row(["monarch_block_size", "32", "Butterfly block dimension"], w)
pdf.table_row(["tie_embeddings", "True", "Input/output embed shared"], w, True)
pdf.table_row(["logit_softcap", "30.0", "Prevents logit explosion"], w)

pdf.sub_title("3.2 MonarchLinear - Butterfly Matrix Layer")
pdf.body(
    "The core innovation. A 2-factor Monarch (butterfly) matrix replaces dense nn.Linear in all MLP layers. "
    "For dim=512 to hidden=1536 with block_size=32: 131K params vs 786K dense (6x savings)."
)
pdf.body(
    "Computation flow: input -> B_in1 (block-diagonal) -> permutation -> B_in2 (block-diagonal) -> "
    "dimension expansion -> B_out1 (block-diagonal) -> permutation -> B_out2 (block-diagonal) -> output"
)
pdf.body(
    "Key design: B matrices stored as 2D tensors [nblk*b, b] instead of 3D [nblk, b, b]. This means:\n"
    "  - Muon optimizer treats them as standard matrices (zero special-casing)\n"
    "  - Per-row int6 quantization works directly\n"
    "  - QAT (Quantization-Aware Training) fake-quant works directly\n"
    "  - Initialization: orthogonal per block (singular values = 1, quantum unitary analog)"
)

pdf.sub_title("3.3 Quantum Connection")
pdf.body(
    "Monarch matrices are mathematically equivalent to 2-layer butterfly circuits. In quantum computing, "
    "butterfly circuits are universal - any unitary can be decomposed into butterfly factors. Our Monarch blocks "
    "are initialized as orthogonal matrices (the real analog of quantum unitary gates), preserving singular "
    "values at exactly 1. This has practical benefits:\n"
    "  - No wasted dynamic range from outlier singular values\n"
    "  - int6 quantization is near-lossless (values concentrated, not spread)\n"
    "  - zstd compression achieves better ratios on quantized Monarch weights"
)

w = [50, 50, 45, 45]
pdf.table_header(["Metric", "Dense MLP", "Monarch MLP", "Ratio"], w)
pdf.table_row(["Params/MLP layer", "786,432", "131,072", "6.0x savings"], w, True)
pdf.table_row(["Params/block total", "1,572,864", "1,048,576", "1.5x savings"], w)
pdf.table_row(["Affordable depth", "11 layers", "22 layers", "2x deeper"], w, True)
pdf.table_row(["Total model params", "~27M", "~24.3M", "Fewer but deeper"], w)

# ── SECTION 4: ALL TECHNIQUES ──
pdf.add_page()
pdf.section_title("4. Complete Technique Stack")

pdf.sub_title("4.1 SmearGate (Temporal Blending)")
pdf.body(
    "A learned per-dimension gate that blends each token with its predecessor:\n"
    "  output = (1 - sigmoid(gate)) * x + sigmoid(gate) * x_prev\n"
    "Gate initialized to zero (sigmoid(0) = 0.5), so initial mixing is 50/50. "
    "Helps model capture local co-occurrence patterns without additional attention cost."
)

pdf.sub_title("4.2 BigramHashEmbedding")
pdf.body(
    "XOR-hashes adjacent token pairs into 2048 buckets, embeds to 128d, projects to 512d. "
    "Captures character-level bigram statistics. Hash function: XOR(36313 * t[i], 27191 * t[i-1]) mod 2047.\n"
    "Initialization: zero weights, scale=0.05 (starts small, learns contribution)."
)

pdf.sub_title("4.3 ValueEmbedding (VE128)")
pdf.body(
    "Re-injects token identity into the attention V (value) matrix at layers 19 and 20. "
    "Embeds token IDs to 128d, projects to kv_dim (256, not model_dim 512). "
    "This was a critical bug fix - the output must match the V tensor shape [B, Hkv, T, head_dim]. "
    "Scale initialized to 0.1."
)

pdf.sub_title("4.4 Exclusive Self-Attention (XSA)")
pdf.body(
    "Applied to the last 4 layers. After computing attention output y, projects out each head's value "
    "direction: y = y - (y . v_hat) * v_hat, where v_hat is unit-normalized v. Forces orthogonal "
    "information encoding across heads, preventing redundancy."
)

pdf.sub_title("4.5 U-Net Skip Connections")
pdf.body(
    "The 22-layer model is split into encoder (layers 0-10) and decoder (layers 11-21). "
    "Encoder layers save skip tensors; decoder layers add weighted skip residuals: "
    "x = x + skip_weight[i] * skips[-(i+1)]. Skip weights are learnable per-dimension vectors."
)

pdf.sub_title("4.6 OrthoInit + muP Scaling")
pdf.body(
    "All 2D weight matrices >= 64x64 receive orthogonal initialization (gain=1.0). "
    "Projection layers (.proj) are additionally scaled by 1/sqrt(2*num_layers) for muP stability. "
    "For Monarch MLP proj, only B_out2 (the last output butterfly factor) gets the muP scale."
)

pdf.sub_title("4.7 Partial RoPE")
pdf.body(
    "Rotary Position Embeddings applied to only 16 out of 64 head dimensions. "
    "The remaining 48 dimensions are position-free, allowing the model to learn absolute patterns."
)

# ── SECTION 5: TRAINING ──
pdf.add_page()
pdf.section_title("5. Training Pipeline")

pdf.sub_title("5.1 Muon Optimizer")
pdf.body(
    "Newton-Schulz orthogonalization of gradient updates (5 iterations). Applied to all 2D matrix params "
    "(including Monarch B matrices). Momentum warmed from 0.92 to 0.99 over 1500 steps. "
    "Distributed: gradients are partitioned across ranks and all-reduced after orthogonalization."
)

pdf.sub_title("5.2 Learning Rates")
w = [60, 40, 90]
pdf.table_header(["Parameter Group", "LR", "Notes"], w)
pdf.table_row(["Tied embedding", "0.035", "Adam optimizer"], w, True)
pdf.table_row(["Matrix params (Muon)", "0.025", "All 2D matrices >= 64x64"], w)
pdf.table_row(["Scalar params (Adam)", "0.04", "1D params, control tensors"], w, True)
pdf.table_row(["Grad clip norm", "0.3", "Global gradient clipping"], w)

pdf.sub_title("5.3 EMA (Exponential Moving Average)")
pdf.body(
    "Decay=0.997. Maintains running average of all weights throughout training. "
    "Final model uses EMA weights (smoother, better generalization). Applied before quantization."
)

pdf.sub_title("5.4 Late QAT (Quantization-Aware Training)")
pdf.body(
    "STE (Straight-Through Estimator) fake quantization activates when LR warmdown scale drops below 0.15. "
    "Forward pass uses quantized weights (int6, clip_range=31), backward pass flows gradients to original fp32 weights. "
    "Critical for int6 compression: without QAT, quantization degrades BPB by ~0.5; with QAT the gap is ~0.001."
)
pdf.body(
    "Implementation detail: CastedLinear._qat_enabled is a class-level boolean. It must be set to False "
    "BEFORE torch.compile() to avoid constant-folding bug where the compiled graph ignores runtime changes."
)

pdf.sub_title("5.5 Warmdown Schedule")
pdf.body(
    "Last 3500 steps (or equivalent wallclock time). LR scales linearly from 1.0 to 0.0. "
    "QAT kicks in at scale < 0.15 (roughly last 525 steps). Wallclock-aware: automatically adjusts "
    "if training is slower/faster than expected to use full 10-minute budget."
)

# ── SECTION 6: QUANTIZATION ──
pdf.add_page()
pdf.section_title("6. Quantization & Compression Pipeline")

pdf.sub_title("6.1 Mixed Quantization Strategy")
w = [45, 45, 45, 55]
pdf.table_header(["Tensor Type", "Method", "Range", "Details"], w)
pdf.table_row(["Attn matrices", "Clipped int6", "[-31, +31]", "Per-row scale, 5-pctile search"], w, True)
pdf.table_row(["MLP matrices", "Clipped int6", "[-31, +31]", "Same (incl. Monarch B matrices)"], w)
pdf.table_row(["Embeddings", "Standard int8", "[-127, +127]", "Per-row scale"], w, True)
pdf.table_row(["Small tensors", "fp16", "N/A", "< 4096 elements passthrough"], w)
pdf.table_row(["Control tensors", "fp32", "N/A", "attn_scale, mlp_scale, etc."], w, True)

pdf.sub_title("6.2 GPTQ-lite Percentile Search")
pdf.body(
    "For each int6 quantized matrix, we search 5 clipping percentiles [0.999, 0.9995, 0.9999, 0.99999, 1.0] "
    "per row and pick the one minimizing reconstruction MSE. This is a simplified version of GPTQ "
    "(Frantar et al., 2022) that doesn't require calibration data."
)

pdf.sub_title("6.3 Monarch Quantization Advantage")
pdf.body(
    "Monarch B matrices are [nblk*b, b] = e.g. [512, 32]. With orthogonal init, singular values are "
    "exactly 1.0 - no outliers. After training, they remain near-orthogonal. This means:\n"
    "  - int6 clip range [-31, +31] captures nearly all information\n"
    "  - No wasted bits on outlier values\n"
    "  - Quantized weight distribution is more uniform -> better zstd compression\n"
    "INT8_KEEP_FLOAT_MAX_NUMEL was lowered to 4096 so Monarch B matrices (512*32=16384 > 4096) "
    "get properly int6-quantized instead of kept as fp16."
)

pdf.sub_title("6.4 Compression")
pdf.body(
    "zstd level 22 (maximum compression). Applied to the entire serialized quantized state dict. "
    "Int6 values in [-31, +31] use only ~6 bits of entropy per int8 byte, giving zstd excellent "
    "compression ratios. Previous 11L dense model compressed to ~3.8 MB. Expected 22L Monarch: ~7-8 MB."
)

pdf.sub_title("6.5 Roundtrip Validation")
pdf.body(
    "After serialization, the model is loaded from disk, dequantized, and re-evaluated on the full "
    "validation set. This confirms the quantized model's BPB matches expectations. "
    "The pipeline: save -> load from disk -> decompress zstd -> dequantize -> eval BPB."
)

# ── SECTION 7: TTT ──
pdf.add_page()
pdf.section_title("7. Test-Time Training (Legal TTT)")

pdf.sub_title("7.1 Overview")
pdf.body(
    "Test-Time Training adapts the model to the validation data during evaluation. This is legal under "
    "competition rules because we use a 'score-first' protocol: each chunk of data is first scored "
    "(under torch.inference_mode), then the model is adapted on that chunk for future predictions."
)

pdf.sub_title("7.2 Protocol")
pdf.body(
    "1. Pre-compute all sliding window starts with stride=64\n"
    "2. Assign windows to 32K-token chunks by their first scored token\n"
    "3. Per chunk:\n"
    "   Phase 1 (SCORE): Run forward pass under torch.inference_mode()\n"
    "              -> No gradients, no parameter mutation. Legally score-first.\n"
    "   Phase 2 (ADAPT): SGD training on the chunk's tokens\n"
    "              -> lr=0.002, momentum=0.9, 3 epochs per chunk\n"
    "              -> Cosine LR decay over chunks\n"
    "4. All-reduce across distributed ranks\n"
    "5. Restore all params to requires_grad=True after completion"
)

pdf.sub_title("7.3 Hyperparameters")
w = [60, 40, 90]
pdf.table_header(["Parameter", "Value", "Notes"], w)
pdf.table_row(["ttt_lr", "0.002", "SGD learning rate"], w, True)
pdf.table_row(["ttt_epochs", "3", "Passes over each chunk"], w)
pdf.table_row(["ttt_chunk_tokens", "32768", "Tokens per adaptation chunk"], w, True)
pdf.table_row(["ttt_momentum", "0.9", "SGD momentum"], w)
pdf.table_row(["ttt_batch_seqs", "32", "Sequences per micro-batch"], w, True)
pdf.table_row(["ttt_grad_clip", "1.0", "Per-chunk gradient clipping"], w)
pdf.table_row(["eval_stride", "64", "Sliding window stride"], w, True)

pdf.sub_title("7.4 Expected Impact")
pdf.body("TTT typically improves BPB by ~0.0025. This is applied after quantization roundtrip, "
         "on the exact model that would be submitted.")

# ── SECTION 8: PARAMETER BUDGET ──
pdf.add_page()
pdf.section_title("8. Parameter Budget Analysis")

pdf.sub_title("8.1 Per-Block Breakdown (22 layers)")
w = [55, 40, 35, 60]
pdf.table_header(["Component", "Params", "Bytes (int6)", "Formula"], w)
pdf.table_row(["Attn Q proj", "262,144", "263 KB", "512 x 512"], w, True)
pdf.table_row(["Attn K proj", "131,072", "131 KB", "512 x 256 (GQA)"], w)
pdf.table_row(["Attn V proj", "131,072", "131 KB", "512 x 256 (GQA)"], w, True)
pdf.table_row(["Attn O proj", "262,144", "263 KB", "512 x 512"], w)
pdf.table_row(["MLP fc (Monarch)", "131,072", "135 KB", "2*(16*32*32)+2*(48*32*32)"], w, True)
pdf.table_row(["MLP proj (Monarch)", "131,072", "135 KB", "Same structure reversed"], w)
pdf.table_row(["Per-block total", "1,048,576", "~1.05 MB", "Attn + Monarch MLP"], w, True)

pdf.sub_title("8.2 Full Model")
w = [65, 45, 45, 35]
pdf.table_header(["Component", "Params", "Approx Bytes", "Layers"], w)
pdf.table_row(["All blocks", "23,068,672", "~23 MB raw", "22"], w, True)
pdf.table_row(["Token embedding", "524,288", "~1 MB (int8)", "1"], w)
pdf.table_row(["Skip weights", "5,632", "~11 KB (fp16)", "11"], w, True)
pdf.table_row(["SmearGate", "512", "~1 KB (fp32)", "1"], w)
pdf.table_row(["BigramHash", "~328K", "~0.7 MB", "1"], w, True)
pdf.table_row(["ValueEmbedding x2", "~66K", "~0.1 MB", "2"], w)
pdf.table_row(["TOTAL", "~24.3M", "~7-8 MB (zstd)", ""], w, True)

pdf.sub_title("8.3 Why It Fits in 16 MB")
pdf.body(
    "The 16 MB budget includes model weights + code. With zstd level 22:\n"
    "  - Model: ~7-8 MB (int6 Monarch weights compress excellently)\n"
    "  - Code: ~68 KB (train_gpt.py, 1495 lines)\n"
    "  - Total: ~8 MB -- well within 16 MB, leaving room for experimentation.\n"
    "Compare: 11L dense model was ~3.8 MB compressed, but had worse BPB due to shallower depth."
)

# ── SECTION 9: BPB PROJECTION ──
pdf.section_title("9. Performance Projection")

w = [70, 40, 80]
pdf.table_header(["Component", "BPB Impact", "Notes"], w)
pdf.table_row(["Base 11L dense", "~1.130", "Baseline from leaderboard"], w, True)
pdf.table_row(["+ SmearGate", "~-0.002", "Temporal blending"], w)
pdf.table_row(["+ BigramHash", "~-0.002", "Local co-occurrence"], w, True)
pdf.table_row(["+ VE128", "~-0.001", "Token identity in V"], w)
pdf.table_row(["+ XSA", "~-0.001", "Orthogonal heads"], w, True)
pdf.table_row(["+ OrthoInit+muP", "~-0.003", "Better optimization"], w)
pdf.table_row(["+ EMA+QAT", "~-0.001", "Quantization-aware"], w, True)
pdf.table_row(["+ 11L -> 22L (Monarch)", "~-0.015", "Depth gain, net of overhead"], w)
pdf.table_row(["+ TTT", "~-0.0025", "Test-time adaptation"], w, True)
pdf.table_row(["PROJECTED TOTAL", "~1.10", "Below SOTA 1.1194"], w)

# ── SECTION 10: DEPLOYMENT ──
pdf.add_page()
pdf.section_title("10. Deployment & Submission")

pdf.sub_title("10.1 Local Smoke Test (RTX 5070 Ti)")
pdf.code_block(
    "run_test.bat\n"
    "  ITERATIONS=50, WARMDOWN_ITERS=10, VAL_LOSS_EVERY=50\n"
    "  TTT_ENABLED=0, NUM_LAYERS=22, MONARCH_MLP=1\n"
    "  TRAIN_SEQ_LEN=256, TRAIN_BATCH_TOKENS=8192, VAL_BATCH_SIZE=8192\n"
    "  Checks: no errors, loss decreases, final_model.ptz < 16MB"
)

pdf.sub_title("10.2 Full Submission (8x H100)")
pdf.code_block(
    "torchrun --nproc_per_node=8 train_gpt.py\n"
    "\n"
    "Default hyperparameters are set for competition:\n"
    "  ITERATIONS=20000, WARMDOWN_ITERS=3500, MAX_WALLCLOCK_SECONDS=600\n"
    "  NUM_LAYERS=22, MONARCH_MLP=1, MONARCH_BLOCK_SIZE=32\n"
    "  TTT_ENABLED=1 (runs after training for final BPB)"
)

pdf.sub_title("10.3 Output Files")
w = [50, 50, 90]
pdf.table_header(["File", "Size", "Purpose"], w)
pdf.table_row(["final_model.pt", "~95 MB", "Uncompressed fp32 checkpoint"], w, True)
pdf.table_row(["final_model.ptz", "~7-8 MB", "Quantized + zstd compressed (SUBMISSION)"], w)
pdf.table_row(["train_gpt.py", "~68 KB", "Code artifact (SUBMISSION)"], w, True)
pdf.table_row(["logs/<run_id>.txt", "Variable", "Training log with all metrics"], w)

pdf.sub_title("10.4 Decision Tree After Submission")
pdf.body(
    "If final val_bpb < 1.1194 -> You've beaten current SOTA. Submit!\n"
    "If final val_bpb 1.12-1.13 -> Monarch depth gain may need tuning:\n"
    "  - Try NUM_LAYERS=20 (reduce if 22 is too aggressive)\n"
    "  - Try MONARCH_BLOCK_SIZE=16 (finer blocks, more params, shallower ok)\n"
    "  - Increase WARMDOWN_ITERS for better QAT convergence\n"
    "If final val_bpb > 1.13 -> Monarch may have training issues:\n"
    "  - Check if loss diverges (reduce MATRIX_LR)\n"
    "  - Fallback: set MONARCH_MLP=0 NUM_LAYERS=11 for proven dense config"
)

# ── SECTION 11: FILE INVENTORY ──
pdf.section_title("11. File Inventory")

w = [65, 125]
pdf.table_header(["File", "Purpose"], w)
pdf.table_row(["train_gpt.py", "Main submission file (1495 lines) - all techniques"], w, True)
pdf.table_row(["train_gpt_baseline_orig.py", "Original competition baseline (backup)"], w)
pdf.table_row(["train_gpt_factorized.py", "Failed tensor factorization attempt (archive)"], w, True)
pdf.table_row(["run_local.py", "Single-GPU wrapper (disables torch.compile)"], w)
pdf.table_row(["run_test.bat", "Local smoke test (50 steps, small batch, 22L)"], w, True)
pdf.table_row(["test_svd_compression.py", "SVD experiment proving decomp fails"], w)
pdf.table_row(["generate_report.py", "This PDF report generator"], w, True)

# ── SECTION 12: KEY INSIGHTS ──
pdf.add_page()
pdf.section_title("12. Key Insights & Lessons Learned")

pdf.sub_title("12.1 Why Tensor Decomposition Failed for NN Weights")
pdf.body(
    "Quantum states with area-law entanglement have exponentially-decaying singular values - MPS/Tucker "
    "can compress them efficiently (this is the basis of DMRG in physics). But neural network weight "
    "matrices have power-law singular value decay (full rank). Removing any singular value causes "
    "proportionally large error. The mathematical structure that makes tensor networks work in quantum "
    "physics simply doesn't exist in trained neural networks."
)

pdf.sub_title("12.2 Why Monarch Works Where Decomposition Didn't")
pdf.body(
    "The key distinction: decomposition tries to approximate an existing dense matrix with fewer params. "
    "Monarch replaces the parameterization entirely - the model never forms a dense matrix. The butterfly "
    "structure provides sufficient expressiveness while using 6x fewer parameters. The model learns to use "
    "the structured parameterization directly, rather than fighting against lossy compression of dense weights."
)

pdf.sub_title("12.3 The Quantization-Compression Synergy")
pdf.body(
    "Three techniques reinforce each other:\n"
    "  1. QAT teaches the model to be robust to int6 quantization during training\n"
    "  2. Orthogonal Monarch init keeps singular values near 1 -> uniform weight distribution\n"
    "  3. Uniform int6 values in [-31, +31] have low entropy -> zstd compresses aggressively\n"
    "The result: ~7-8 MB for a 24.3M parameter model. Dense 27M params would be ~20 MB at int8."
)

pdf.sub_title("12.4 Competitive Differentiation")
pdf.body(
    "Most competitors using Claude/GPT will converge on the same SOTA technique stack (SmearGate, XSA, "
    "BigramHash, QAT, etc.) because these are well-documented in the leaderboard. Monarch MLP is our "
    "differentiator - a fundamentally different architectural bet that trades MLP expressiveness for depth. "
    "The hypothesis: at fixed parameter budget, depth wins over width, and Monarch enables depth cheaply."
)

# Save
out_path = "d:/parameter-golf-competition/parameter-golf/Parameter_Golf_Solution_Report.pdf"
pdf.output(out_path)
print(f"PDF saved to: {out_path}")
print(f"Pages: {pdf.pages_count}")
