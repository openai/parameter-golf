"""QAT v3 — atomic patch, fixes lessons from v1 + v2.

LESSONS FROM TONIGHT:
- v1: Fake-quanted only matrices, but used wrong scale formula (per-row max
  instead of GPTQ's clip_sigmas * row_std / qmax). Result: didn't reduce
  quant penalty meaningfully (0.0144 → still ~0.0144).
- v2: Fixed scale formula AND added tok_emb fake-quant. Result: quant penalty
  actually dropped 48% (0.0147 → 0.0077), BUT pre-quant cost was +0.023 BPB
  due to fake-quanting tok_emb. Net negative.

v3 design:
- Matrices: fake-quant at int6 with GPTQ-matched scale (v2's formula)
- tok_emb: NO fake-quant (too sensitive — let it train at full precision)
- Warmup: enable QAT only after step 1500 (~30% of training). Model finds
  a strong baseline FIRST, then learns quant-robustness on top.
- bigram.embed: NO fake-quant (small, low-impact)

Expected:
- Pre-quant cost: ~+0.001 BPB (only matrix fake-quant overhead)
- Quant penalty reduction: similar to v2 on matrices (~50% of matrix-quant penalty)
- Net: pre-quant ~1.086, quant ~1.092, sliding ~1.075, TTT ~1.073 → record by 0.005-0.010 BPB

USAGE:
  python3 patch_qat_v3.py
  # then: QAT_ENABLED=1 QAT_WARMUP_STEPS=1500 torchrun ...

Idempotent: safe to run multiple times. Detects existing v1/v2 state and migrates.
"""
import os
import sys
import re

FILE = "/workspace/parameter-golf/code/train_gpt_stacked_v2_fixed.py"


def main():
    if not os.path.exists(FILE):
        print(f"ERROR: file not found: {FILE}")
        sys.exit(1)
    src = open(FILE).read()
    initial_size = len(src)

    # ============ PHASE A: Clean slate (remove any v1/v2 QAT) ============
    print("=== Phase A: clean any prior QAT state ===")

    # Remove v2 helpers (if present)
    v2_helpers_re = re.compile(
        r"\ndef _fake_quantize_per_row_gptq\(W, bits, clip_sigmas\):\n.+?QAT_EMBED_CLIP = float\(os\.environ\.get\(\"QAT_EMBED_CLIP\", \"20\.0\"\)\)\n",
        re.DOTALL,
    )
    if v2_helpers_re.search(src):
        src = v2_helpers_re.sub("\n", src, count=1)
        print("  Removed v2 QAT helpers")

    # Remove v1 helpers (if present)
    v1_helpers_re = re.compile(
        r"\ndef _fake_quantize_int6\(W\):\n.+?QAT_ENABLED=bool\(int\(os\.environ\.get\(\"QAT_ENABLED\",\"0\"\)\)\)\n",
        re.DOTALL,
    )
    if v1_helpers_re.search(src):
        src = v1_helpers_re.sub("\n", src, count=1)
        print("  Removed v1 QAT helpers")

    # Revert v2 CastedLinear forward (multi-line) to original
    v2_castedlinear = (
        "def forward(self,x):\n"
        "\t\tw_raw=self.weight\n"
        "\t\tif QAT_ENABLED and self.training and getattr(self,\"_qat_active\",False):\n"
        "\t\t\tw_raw=_fake_quantize_per_row_gptq(w_raw, QAT_MATRIX_BITS, QAT_MATRIX_CLIP)\n"
        "\t\tw=w_raw.to(x.dtype)\n"
        "\t\tbias=self.bias.to(x.dtype) if self.bias is not None else None\n"
        "\t\treturn F.linear(x,w,bias)"
    )
    original_castedlinear = "def forward(self,x):w=self.weight.to(x.dtype);bias=self.bias.to(x.dtype)if self.bias is not None else None;return F.linear(x,w,bias)"
    if v2_castedlinear in src:
        src = src.replace(v2_castedlinear, original_castedlinear, 1)
        print("  Reverted v2 CastedLinear.forward")

    # Revert v1 CastedLinear forward (if present)
    v1_castedlinear = (
        "def forward(self,x):\n"
        "\t\tw_raw=self.weight\n"
        "\t\tif QAT_ENABLED and self.training and getattr(self,\"_qat_active\",False):w_raw=_fake_quantize_int6(w_raw)\n"
        "\t\tw=w_raw.to(x.dtype)\n"
        "\t\tbias=self.bias.to(x.dtype)if self.bias is not None else None\n"
        "\t\treturn F.linear(x,w,bias)"
    )
    if v1_castedlinear in src:
        src = src.replace(v1_castedlinear, original_castedlinear, 1)
        print("  Reverted v1 CastedLinear.forward")

    # Revert v2 tok_emb wrapping (multi-line) back to original one-liner
    v2_tokemb_block = (
        "_qat_w_emb=None\n"
        "\t\tif QAT_ENABLED and self.training:\n"
        "\t\t\t_qat_w_emb=_fake_quantize_per_row_gptq(self.tok_emb.weight, QAT_EMBED_BITS, QAT_EMBED_CLIP)\n"
        "\t\t\tx=F.embedding(input_ids, _qat_w_emb)\n"
        "\t\telse:\n"
        "\t\t\tx=self.tok_emb(input_ids)"
    )
    original_tokemb = "x=self.tok_emb(input_ids)"
    if v2_tokemb_block in src:
        src = src.replace(v2_tokemb_block, original_tokemb, 1)
        print("  Reverted v2 tok_emb wrapping")

    # Revert v2 tied output wrapping
    v2_tied_block = (
        "if self.tie_embeddings:\n"
        "\t\t\tif _qat_w_emb is not None:\n"
        "\t\t\t\tlogits_proj=F.linear(x, _qat_w_emb)\n"
        "\t\t\telse:\n"
        "\t\t\t\tlogits_proj=F.linear(x, self.tok_emb.weight)"
    )
    original_tied = "if self.tie_embeddings:logits_proj=F.linear(x,self.tok_emb.weight)"
    if v2_tied_block in src:
        src = src.replace(v2_tied_block, original_tied, 1)
        print("  Reverted v2 tied output")

    # Save cleaned state
    open(FILE, "w").write(src)
    print(f"  Saved cleaned state ({len(src):,} bytes vs initial {initial_size:,})")

    # ============ PHASE B: Apply v3 (matrices only + warmup) ============
    print("\n=== Phase B: apply v3 ===")

    # Add v3 helpers (insert before RMSNorm)
    v3_helpers = """
# ============ QAT v3: matrices only + GPTQ-matched scale + warmup ============
# Step counter is updated by the train loop via _qat_set_step(step)
_QAT_CURRENT_STEP = 0


def _qat_set_step(step):
    global _QAT_CURRENT_STEP
    _QAT_CURRENT_STEP = step


def _qat_active_now():
    return QAT_ENABLED and _QAT_CURRENT_STEP >= QAT_WARMUP_STEPS


def _fake_quantize_matrix_gptq(W):
    \"\"\"GPTQ-matched fake-quant for matrix weights at QAT_MATRIX_BITS.\"\"\"
    qmax = 2**(QAT_MATRIX_BITS - 1) - 1
    row_std = W.float().std(dim=1, keepdim=True)
    s = (QAT_MATRIX_CLIP * row_std / qmax).clamp_min(1e-10)
    W_q = torch.round(W / s).clamp(-qmax, qmax)
    W_dq = W_q * s
    # Straight-through estimator
    return W_dq.detach() + W - W.detach()


QAT_ENABLED = bool(int(os.environ.get("QAT_ENABLED", "0")))
QAT_MATRIX_BITS = int(os.environ.get("QAT_MATRIX_BITS", "6"))
QAT_MATRIX_CLIP = float(os.environ.get("QAT_MATRIX_CLIP", "12.85"))
QAT_WARMUP_STEPS = int(os.environ.get("QAT_WARMUP_STEPS", "0"))  # default 0 = always-on (no train-loop changes needed)

"""
    if "_fake_quantize_matrix_gptq" not in src:
        if "class RMSNorm" not in src:
            print("  ERROR: RMSNorm anchor not found")
            sys.exit(1)
        src = src.replace("class RMSNorm", v3_helpers + "class RMSNorm", 1)
        print("  Added v3 helpers (with warmup support)")

    # Modify CastedLinear.forward
    new_castedlinear = (
        "def forward(self,x):\n"
        "\t\tw_raw=self.weight\n"
        "\t\tif _qat_active_now() and self.training and getattr(self,'_qat_active',False):\n"
        "\t\t\tw_raw=_fake_quantize_matrix_gptq(w_raw)\n"
        "\t\tw=w_raw.to(x.dtype)\n"
        "\t\tbias=self.bias.to(x.dtype) if self.bias is not None else None\n"
        "\t\treturn F.linear(x,w,bias)"
    )
    if original_castedlinear in src and "_fake_quantize_matrix_gptq" not in [
        line.strip() for line in src.split("\n") if "def forward(self,x):" in line[:60]
    ]:
        src = src.replace(original_castedlinear, new_castedlinear, 1)
        print("  Wrapped CastedLinear.forward with v3")

    # Save before doing the optional marker step
    open(FILE, "w").write(src)

    # Verify _qat_active markers exist (carried over from v1/v2)
    if "lyr._qat_active=True" in src:
        print("  Block matrix _qat_active markers already present (from v1/v2)")
    else:
        # Need to add marker block in GPT.__init__
        print("  WARNING: _qat_active markers missing — would need to add manually")
        print("    Anchor candidates:")
        if "self._init_weights()" in src:
            anchor = "self.skip_gates=nn.Parameter(torch.zeros(self.num_skip_weights,h.model_dim,dtype=torch.float32))if h.skip_gates_enabled else None"
            if anchor in src:
                marker = """
\t\tif QAT_ENABLED:
\t\t\tfor block in self.blocks:
\t\t\t\tfor lyr in (block.attn.c_q, block.attn.c_k, block.attn.c_v, block.attn.proj, block.mlp.fc, block.mlp.proj):
\t\t\t\t\tlyr._qat_active=True"""
                src = src.replace(anchor, anchor + marker, 1)
                print("    Added _qat_active markers")

    open(FILE, "w").write(src)

    # ============ PHASE C: Verify syntax ============
    import py_compile
    py_compile.compile(FILE, doraise=True)

    # ============ PHASE D: Sanity checks ============
    print("\n=== Phase C: verify ===")
    final = open(FILE).read()
    checks = [
        ("def _fake_quantize_matrix_gptq(W):", True, "v3 helper"),
        ("_QAT_CURRENT_STEP = 0", True, "v3 step counter"),
        ("def _qat_set_step(step):", True, "v3 setter"),
        ("def _qat_active_now():", True, "v3 active check"),
        ("QAT_WARMUP_STEPS = int", True, "v3 warmup env"),
        ("_fake_quantize_matrix_gptq(w_raw)", True, "v3 in CastedLinear"),
        ("lyr._qat_active=True", True, "matrix markers"),
        # Things that should be ABSENT:
        ("_fake_quantize_int6", False, "v1 helper (gone)"),
        ("_fake_quantize_per_row_gptq", False, "v2 helper (gone)"),
        ("F.embedding(input_ids, _qat_w_emb)", False, "v2 tok_emb wrap (gone)"),
    ]
    all_ok = True
    for marker, expected_present, desc in checks:
        actual = marker in final
        ok = actual == expected_present
        sym = "OK " if ok else "BAD"
        print(f"  [{sym}] {desc}: {'present' if actual else 'absent'}")
        if not ok:
            all_ok = False

    if not all_ok:
        print("\nFAILED — review above")
        sys.exit(1)

    print(f"\nDONE. File: {FILE} ({len(final):,} bytes)")
    print("Train loop must call: _qat_set_step(step) at the top of each step")


if __name__ == "__main__":
    main()
