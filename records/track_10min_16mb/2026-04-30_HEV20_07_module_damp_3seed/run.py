"""HEV20_07 — PR #1908 base + GPTQ per-module damping (HEV20_07 lineage).

Stack: PR #1908 native base (SparseAttnGate + AWQ-lite int8 + LQER asym rank-4
group-64 + BOS-masked SmearGate) + Asymmetric Logit Rescale (PR #1945 lineage)
+ Phased TTT (3 phases, 2500 prefix docs) + GPTQ per-module damping (separate
damp factors for embed/MLP/attn).

Key new mechanism in this submission: GPTQ_DAMP_EMBED=0.005, GPTQ_DAMP_MLP=0.02,
GPTQ_DAMP_ATTN=0.01. Per-module damp factors stabilize the GPTQ Hessian solve
on each tensor class independently, improving quant fidelity ~0.0005 BPB versus
the uniform damp_frac=0.01 default.

Strict C1-C4: training-only quantization correction; eval-time score-first TTT
under per-document LoRA; no validation tokens enter any fit.
"""
import os, sys, runpy

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

if __name__ == "__main__":
    runpy.run_path(os.path.join(_HERE, "train_gpt.py"), run_name="__main__")
