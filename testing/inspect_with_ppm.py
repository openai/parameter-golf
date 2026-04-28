"""Forward-pass + PPM-D byte-mixture inspection.

Loads a post-EMA pre-quant final_model.pt, runs forward on N val tokens,
applies the PPM-D byte mixture (extracted from PR #1857), and outputs a
markdown report comparing NN-only vs NN+PPM bpb.

Run on a 1xH100 pod. Usage:
    python3 testing/inspect_with_ppm.py --ckpt <path> --train_gpt <path>
"""
import argparse, ctypes, importlib.util, math, os, struct, subprocess, sys, tempfile, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import sentencepiece as spm

PPM_C_SRC_FILE = Path(__file__).parent / "ppm_scorer.c"

# ------------------------- helpers -------------------------

def load_train_gpt_module(path):
    spec = importlib.util.spec_from_file_location("tg", path)
    m = importlib.util.module_from_spec(spec)
    sys.modules["tg"] = m
    spec.loader.exec_module(m)
    return m

def seed_env_from_train_log(log_path):
    """Read 'Hyperparameters:' block from a train.log and set values as env vars
    BEFORE importing train_gpt.py. This makes the constructed model match the
    saved checkpoint's architecture exactly."""
    import re
    txt = Path(log_path).read_text()
    # Find the hparam block
    in_block = False
    n_set = 0
    for line in txt.split("\n"):
        if line.strip().startswith("Hyperparameters:"):
            in_block = True
            continue
        if in_block:
            if not line.startswith("  "):  # block ended
                break
            m = re.match(r"  (\w+): (.+)$", line)
            if not m: continue
            key, val = m.group(1).upper(), m.group(2).strip()
            # Convert booleans to 0/1 (env vars are strings)
            if val == "True": val = "1"
            elif val == "False": val = "0"
            elif val == "None": val = ""
            # Convert "4.0" -> "4" to match int() parsers (logger formats ints as floats)
            elif re.match(r"^-?\d+\.0$", val): val = val[:-2]
            os.environ[key] = val
            n_set += 1
    print(f"[hparams] seeded {n_set} env vars from {log_path}")

def build_token_bytes_lut(sp, vocab_size):
    sz = max(int(sp.vocab_size()), vocab_size)
    bytestrs = [b""] * sz
    has_space = np.zeros(sz, dtype=np.uint8)
    is_boundary = np.zeros(sz, dtype=np.uint8)
    for tid in range(int(sp.vocab_size())):
        if sp.is_control(tid) or sp.is_unknown(tid) or sp.is_unused(tid):
            is_boundary[tid] = 1
            continue
        piece = sp.id_to_piece(tid)
        if sp.is_byte(tid):
            bytestrs[tid] = bytes([int(piece[3:-1], 16)])
            continue
        if piece.startswith("▁"):
            has_space[tid] = 1
            piece = piece[1:]
        bytestrs[tid] = piece.encode("utf-8")
    flat = b"".join(bytestrs)
    lens = np.array([len(b) for b in bytestrs], dtype=np.int32)
    offs = np.zeros(sz, dtype=np.int32)
    offs[1:] = np.cumsum(lens[:-1])
    return np.frombuffer(flat, dtype=np.uint8).copy(), offs, lens, has_space, is_boundary

def compile_ppm_lib(c_src_path, antihijack=False):
    so_name = "ppm_scorer_antihijack.so" if antihijack else "ppm_scorer.so"
    so_path = Path(tempfile.gettempdir()) / so_name
    cmd = ["gcc", "-O3", "-march=native", "-fopenmp", "-shared", "-fPIC",
           "-o", str(so_path), str(c_src_path), "-lm"]
    print(f"[compile] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    lib = ctypes.CDLL(str(so_path))
    if antihijack:
        # Anti-hijack version: extra c_double arg for nn_skip_thr
        lib.ppm_score_omp.argtypes = [
            ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_double), ctypes.c_int64,
            ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_uint8), ctypes.c_int, ctypes.c_int,
            ctypes.c_double, ctypes.c_double, ctypes.c_double,
            ctypes.c_double,  # nn_skip_thr (anti-hijack)
            ctypes.c_uint32, ctypes.c_int64, ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
        ]
    else:
        lib.ppm_score_omp.argtypes = [
            ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_double), ctypes.c_int64,
            ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_uint8), ctypes.c_int, ctypes.c_int,
            ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_uint32,
            ctypes.c_int64, ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
        ]
    lib.ppm_score_omp.restype = ctypes.c_int
    return lib

def categorize(piece):
    p = piece[1:] if piece.startswith("▁") else piece
    if not p:
        return "empty"
    if any(s in p for s in ["http", "://", "www.", ".com", ".org", ".net", ".io", "@"]):
        return "URL"
    s = p.replace(".", "").replace(",", "").replace("-", "")
    if s and s.isdigit():
        return "NUMERIC"
    if any(c in p for c in ["{", "}", "[", "]", "()", ";", "==", "!=", "->", "::"]):
        return "CODE"
    if all(c in "0123456789abcdef" for c in p) and 4 <= len(p) <= 64:
        return "HEX"
    if "/" in p and all(c in "/_-" or c.isalnum() for c in p):
        return "PATH"
    return "PROSE"

# ------------------------- main -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--train_gpt", required=True)
    ap.add_argument("--val_tok", default="/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/fineweb_val_000000.bin")
    ap.add_argument("--val_bytes", default="/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/fineweb_val_bytes_000000.bin")
    ap.add_argument("--tokenizer", default="")
    ap.add_argument("--tokens", type=int, default=8192)
    ap.add_argument("--out", default="/tmp/ppm_inspect.md")
    ap.add_argument("--ppm_order", type=int, default=4)
    ap.add_argument("--lambda_hi", type=float, default=0.9)
    ap.add_argument("--lambda_lo", type=float, default=0.05)
    ap.add_argument("--ppm_threshold", type=float, default=0.9)
    ap.add_argument("--ppm_nn_skip_thr_nats", type=float, default=0.0, help="anti-hijack: suppress gate when NN per-byte logp > -this. 0 disables.")
    ap.add_argument("--ppm_c_src", default="", help="override path to PPM C source (default = ppm_scorer.c next to this file)")
    ap.add_argument("--ppm_omp_threads", type=int, default=8)
    ap.add_argument("--ppm_chunk_tokens", type=int, default=4194304)
    ap.add_argument("--ppm_log_cache", type=int, default=1048576)
    ap.add_argument("--train_log", default="", help="train.log to read hparams from (for env-var seeding)")
    args = ap.parse_args()

    # Seed env vars from train.log BEFORE importing train_gpt
    if args.train_log:
        seed_env_from_train_log(args.train_log)
    else:
        # Default: try to find a train.log next to the checkpoint
        cand = Path(args.ckpt).parent / "train.log"
        if cand.exists():
            seed_env_from_train_log(cand)

    device = torch.device("cuda")

    # ---- tokenizer ----
    tk_path = args.tokenizer
    if not tk_path:
        for p in [
            "/workspace/parameter-golf/data/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model",
            str(Path(args.train_gpt).parent / "tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model"),
        ]:
            if Path(p).exists():
                tk_path = p
                break
    sp = spm.SentencePieceProcessor()
    sp.load(tk_path)
    vocab = sp.vocab_size()
    print(f"[tk] vocab={vocab}")

    # ---- val tokens ----
    # Format: 1024-byte header (256 int32s) + uint16 tokens
    raw = Path(args.val_tok).read_bytes()
    HEADER = 1024
    payload = raw[HEADER:]
    n_total = len(payload) // 2
    all_tokens = list(struct.unpack(f"<{n_total}H", payload))
    n = min(args.tokens, n_total)
    tokens = all_tokens[:n]
    print(f"[val] {n}/{n_total} tokens loaded (max id={max(tokens)})")
    assert max(tokens) < 8192, f"token id out of range: {max(tokens)} (expected < 8192)"

    # ---- val_bytes sidecar (authoritative bytes-per-token) ----
    val_bytes_per_tok_full = None
    if Path(args.val_bytes).exists():
        bts_raw = Path(args.val_bytes).read_bytes()
        # Sidecar is uint16 per token (NOT int32). Total ~151M bytes for full val.
        bts_payload = bts_raw[HEADER:]
        n_bts = len(bts_payload) // 2
        val_bytes_per_tok_full = np.frombuffer(bts_payload, dtype=np.uint16)[:n].astype(np.int64)
        print(f"[val] sidecar bytes loaded: {n_bts} entries, sum first {n} = {int(val_bytes_per_tok_full.sum()):,} bytes")
    else:
        print(f"[val] WARNING: sidecar {args.val_bytes} not found; will fall back to piece-encoding")

    # ---- model ----
    print(f"[model] importing {args.train_gpt}")
    mod = load_train_gpt_module(Path(args.train_gpt))
    H = mod.Hyperparameters()

    use_quantized = args.ckpt.endswith(".ptz") or args.ckpt.endswith(".int6.ptz")
    if use_quantized:
        # POST-QUANT path: use train_gpt.deserialize() which reads .int6.ptz
        # (and the SpinQuant template from .pt) to reconstruct the quantized
        # eval model. This matches what eval_val(diagnostic quantized) sees.
        print(f"[model] POST-QUANT path: deserialize() from {args.ckpt}")
        H.quantized_model_path = args.ckpt
        # deserialize also reads model_path (final_model.pt) for SpinQuant template
        pt_path = args.ckpt.replace(".int6.ptz", ".pt").replace(".ptz", ".pt")
        H.model_path = pt_path
        print(f"[model] template .pt path: {pt_path}")
        # cwd matters for deserialize since it uses relative paths
        import os as _os
        _orig_cwd = _os.getcwd()
        _os.chdir(str(Path(args.ckpt).parent))
        try:
            model = mod.deserialize(H, device)
        finally:
            _os.chdir(_orig_cwd)
        print(f"[model] deserialize OK")
    else:
        print(f"[model] PRE-QUANT path: constructing {mod.GPT.__name__}")
        try:
            model = mod.GPT(H).to(device).bfloat16()
            print("[model] constructed via GPT(h) signature")
        except TypeError as e:
            print(f"[model] GPT(h) failed ({e}); trying kwargs")
            gpt_kwargs = dict(
                vocab_size=H.vocab_size, num_layers=H.num_layers, model_dim=H.model_dim,
                num_heads=H.num_heads, num_kv_heads=H.num_kv_heads, mlp_mult=H.mlp_mult,
                tie_embeddings=H.tie_embeddings, tied_embed_init_std=H.tied_embed_init_std,
                logit_softcap=H.logit_softcap, rope_base=H.rope_base, qk_gain_init=H.qk_gain_init,
                recur_layers=H.recur_layers, recur_start_step=H.recur_start_step,
                parallel_start_layer=H.parallel_start_layer, rope_dims=H.rope_dims,
            )
            model = mod.GPT(**gpt_kwargs).to(device).bfloat16()
        # Mirror train_gpt.py post-construction precision tweaks
        if hasattr(mod, "CastedLinear"):
            for m in model.modules():
                if isinstance(m, mod.CastedLinear):
                    m.float()
        for fn in ["restore_low_dim_params_to_fp32", "restore_fp32_params"]:
            if hasattr(mod, fn):
                getattr(mod, fn)(model)
                break
        print(f"[model] loading state from {args.ckpt}")
        state = torch.load(args.ckpt, map_location=device, weights_only=False)
        if isinstance(state, dict):
            if "state_dict" in state: state = state["state_dict"]
            elif "model" in state: state = state["model"]
        try:
            model.load_state_dict(state, strict=True)
            print("[model] strict load OK")
        except Exception as e:
            missing, unexpected = model.load_state_dict(state, strict=False)
            print(f"[model] non-strict load: missing={len(missing)} unexpected={len(unexpected)}")
            if len(missing) <= 5: print(f"  missing: {missing}")
            if len(unexpected) <= 5: print(f"  unexpected: {unexpected}")
    model.eval()
    # CRITICAL: enable loop layers (depth recurrence). Without this the model
    # runs in non-looped mode (much higher val_bpb, doesn't match training).
    if H.num_loops > 0 and hasattr(model, "looping_active"):
        model.looping_active = True
        print("[model] looping_active=True (depth recurrence enabled)")

    # ---- forward (chunked, MATCHES official eval_val: seq_len=2048 with varlen attn) ----
    # Official pipeline = torch.compile(forward_logits) + torch.autocast(bf16)
    seq_len = 2048
    n_pred = n - 1
    nll_nats = np.zeros(n_pred, dtype=np.float64)
    TOP_K_CACHE = int(os.environ.get("TOP_K_CACHE", "50"))
    top_ids = np.zeros((n_pred, TOP_K_CACHE), dtype=np.int32)
    top_logp = np.zeros((n_pred, TOP_K_CACHE), dtype=np.float32)

    # Optional: online TRUE full-Σ proper-margin accumulation
    PROPER_MARGIN_ONLINE = int(os.environ.get("PROPER_MARGIN_ONLINE", "0"))
    if PROPER_MARGIN_ONLINE:
        print("[pm] PROPER_MARGIN_ONLINE=1 — building per-prefix tid lists")
        bytestrs_pm = [b""] * vocab
        has_space_pm = [False] * vocab
        is_boundary_pm = [False] * vocab
        for tid in range(vocab):
            if sp.is_control(tid) or sp.is_unknown(tid) or sp.is_unused(tid):
                is_boundary_pm[tid] = True; continue
            piece = sp.id_to_piece(tid)
            if sp.is_byte(tid):
                bytestrs_pm[tid] = bytes([int(piece[3:-1], 16)]); continue
            if piece.startswith("▁"):
                has_space_pm[tid] = True; piece = piece[1:]
            bytestrs_pm[tid] = piece.encode("utf-8")
        pc_w_list, pc_n_list = {}, {}        # prefix -> tids that START WITH prefix
        pc_w_exact, pc_n_exact = {}, {}      # prefix -> tids whose bytes EQUAL prefix exactly
        for use_space, tgt_starts, tgt_exact in [(True, pc_w_list, pc_w_exact), (False, pc_n_list, pc_n_exact)]:
            tmp_starts = {}; tmp_exact = {}
            for tid in range(vocab):
                if is_boundary_pm[tid]: continue
                inc = has_space_pm[tid] and use_space
                bs = (b" " if inc else b"") + bytestrs_pm[tid]
                if not bs: continue
                for jj in range(1, len(bs) + 1):
                    tmp_starts.setdefault(bs[:jj], []).append(tid)
                tmp_exact.setdefault(bs, []).append(tid)  # full bytes only
            for k_, v_ in tmp_starts.items():
                tgt_starts[k_] = torch.tensor(v_, dtype=torch.long, device=device)
            for k_, v_ in tmp_exact.items():
                tgt_exact[k_] = torch.tensor(v_, dtype=torch.long, device=device)
        pm_byte_total_nats = 0.0
        pm_bytes_total = 0
        pm_token_total_nats = 0.0
        # Buffers for proper-margin per-byte mix experiment
        pm_byte_stream_list = []  # list of uint8 (the actual byte at each position)
        pm_nn_logp_list = []      # list of float (proper-margin log-prob at each position)
        print(f"[pm] tables ready (with-space={len(pc_w_list)}, no-space={len(pc_n_list)})")
    print(f"[fwd] running on {n} tokens in chunks of {seq_len} with varlen attention (BOS-aware)")

    # Find _build_cu_seqlens helper from the train_gpt module
    build_cu = getattr(mod, "_build_cu_seqlens", None)
    if build_cu is None:
        print("[fwd] WARNING: _build_cu_seqlens not found; falling back to no-mask attention")
    else:
        print("[fwd] using mod._build_cu_seqlens (BOS-aware varlen attention)")

    # torch.compile the forward (dynamic=True so the last short chunk doesn't recompile-bomb).
    try:
        forward_compiled = torch.compile(model.forward_logits, dynamic=True, fullgraph=False)
        print("[fwd] torch.compile(forward_logits) enabled (dynamic=True)")
    except Exception as e:
        forward_compiled = model.forward_logits
        print(f"[fwd] torch.compile failed ({e}); falling back to eager")

    BOS_ID_VAL = getattr(mod, "BOS_ID", None) or 1
    t0 = time.time()
    pos = 0
    # We chunk so that each chunk has exactly seq_len input tokens (and seq_len target tokens shifted by 1).
    # Mirrors official eval: x = local[:-1], y = local[1:], chunks span seq_len each.
    while pos < n_pred:
        end = min(pos + seq_len + 1, n)
        if end - pos < 2:
            break
        chunk_tokens = tokens[pos:end]
        x_t = torch.tensor(chunk_tokens[:-1], dtype=torch.long, device=device)
        cu_seqlens, max_seqlen = None, 0
        if build_cu is not None:
            bos_pos = (x_t == BOS_ID_VAL).nonzero(as_tuple=True)[0].tolist()
            cu_seqlens, max_seqlen = build_cu(bos_pos, x_t.numel(), x_t.device, seq_len, 64)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            if cu_seqlens is not None:
                logits = forward_compiled(x_t[None], cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
            else:
                logits = forward_compiled(x_t[None])
            log_probs = F.log_softmax(logits.float(), dim=-1)
            tgt = torch.tensor(chunk_tokens[1:], dtype=torch.long, device=device)
            nll_chunk = -log_probs[0].gather(1, tgt.unsqueeze(-1)).squeeze(-1)
            n_chunk = nll_chunk.shape[0]
            nll_nats[pos:pos + n_chunk] = nll_chunk.cpu().numpy()
            tk = log_probs[0].topk(TOP_K_CACHE, dim=-1)
            top_ids[pos:pos + n_chunk] = tk.indices.cpu().numpy()
            top_logp[pos:pos + n_chunk] = tk.values.cpu().numpy()

            if PROPER_MARGIN_ONLINE:
                probs_chunk = log_probs[0].exp()  # (n_chunk, vocab)
                for j_in_chunk in range(n_chunk):
                    i_global = pos + j_in_chunk
                    tid_g = int(chunk_tokens[j_in_chunk + 1])
                    pid_g = int(chunk_tokens[j_in_chunk])
                    if tid_g < 0 or tid_g >= vocab or is_boundary_pm[tid_g]:
                        continue
                    inc_sp = has_space_pm[tid_g] and (pid_g < 0 or not is_boundary_pm[pid_g])
                    full_b = (b" " if inc_sp else b"") + bytestrs_pm[tid_g]
                    if not full_b:
                        continue
                    n_b_g = int(val_bytes_per_tok_full[i_global + 1]) if val_bytes_per_tok_full is not None else len(full_b)
                    use_sp_ctx = (pid_g < 0 or is_boundary_pm[pid_g])
                    # pc_w_list: has-space tokens get leading space (matches "prev NOT boundary")
                    # pc_n_list: no leading space ever (matches "prev IS boundary")
                    pc_g = pc_n_list if use_sp_ctx else pc_w_list
                    pc_g_exact = pc_n_exact if use_sp_ctx else pc_w_exact
                    pj = probs_chunk[j_in_chunk]
                    prefix_b = b""
                    prefix_mass = 1.0
                    nn_byte_t = 0.0
                    n_full = len(full_b)
                    for byte_idx_, bt in enumerate(full_b):
                        np_b = prefix_b + bytes([bt])
                        is_last_byte = (byte_idx_ == n_full - 1)
                        if is_last_byte:
                            # FAITHFUL TERMINATION: at the last byte of the canonical span,
                            # use only tokens whose bytes EXACTLY equal the full span bytes.
                            # This is what makes byte-level NLL = token-level NLL when no
                            # same-byte alternates exist (proper bit-conservation).
                            tids_t = pc_g_exact.get(np_b)
                        else:
                            tids_t = pc_g.get(np_b)
                        if tids_t is None:
                            ext_m = 1e-30
                        else:
                            ext_m = float(pj[tids_t].sum().item())
                        if ext_m < 1e-30: ext_m = 1e-30
                        per_byte_logp = math.log(ext_m / max(prefix_mass, 1e-30))
                        nn_byte_t += -per_byte_logp
                        pm_byte_stream_list.append(bt)
                        pm_nn_logp_list.append(per_byte_logp)
                        prefix_mass = ext_m
                        prefix_b = np_b
                    pm_byte_total_nats += nn_byte_t
                    pm_bytes_total += n_b_g
                    pm_token_total_nats += float(nll_chunk[j_in_chunk].item())
        pos += seq_len
        if pos == seq_len or pos % (seq_len * 256) == 0:
            elapsed = time.time() - t0
            print(f"[fwd] pos={pos}/{n_pred} ({100*pos/n_pred:.1f}%) elapsed={elapsed:.1f}s")
    print(f"[fwd] done in {time.time()-t0:.1f}s; mean NLL/tok = {nll_nats.mean():.4f} nats")

    if PROPER_MARGIN_ONLINE:
        LOG2_ = math.log(2.0)
        tok_bpb_pm = pm_token_total_nats / max(pm_bytes_total, 1) / LOG2_
        byte_bpb_pm = pm_byte_total_nats / max(pm_bytes_total, 1) / LOG2_
        print(f"[pm] === FULL-Σ PROPER MARGIN ===")
        print(f"[pm] bytes={pm_bytes_total:,}")
        print(f"[pm] token-level NN BPB:                {tok_bpb_pm:.5f}")
        print(f"[pm] byte-level NN BPB (proper, full):  {byte_bpb_pm:.5f}")
        print(f"[pm] diff (should be 0 by chain rule):  {tok_bpb_pm - byte_bpb_pm:+.5f}")


    # ---- build PPM args ----
    print(f"[ppm] building byte LUT (vocab={vocab})")
    flat, offs, lens, has_space, is_boundary = build_token_bytes_lut(sp, vocab)
    target_ids = np.array(tokens[1:n], dtype=np.int64)         # next-token at each pos
    prev_ids = np.array(tokens[0:n-1], dtype=np.int64)         # previous token
    print(f"[ppm] target shape={target_ids.shape} flat={len(flat)}B")

    # Per-token NN log-prob = -nll_nats; passed to scorer as nll (positive)
    # The scorer expects nll in nats, will divide by n_bytes internally.

    out = np.zeros(6, dtype=np.float64)
    use_antihijack = args.ppm_nn_skip_thr_nats > 0
    c_src_path = args.ppm_c_src if args.ppm_c_src else str(PPM_C_SRC_FILE)
    if use_antihijack and not args.ppm_c_src:
        # default to anti-hijack source on the pod
        for cand in ["/workspace/their_ppm_antihijack.c", str(PPM_C_SRC_FILE.parent / "ppm_scorer_antihijack.c")]:
            if Path(cand).exists():
                c_src_path = cand
                break
    print(f"[ppm] using C source: {c_src_path}, antihijack={use_antihijack}")
    lib = compile_ppm_lib(c_src_path, antihijack=use_antihijack)

    print(f"[ppm] order={args.ppm_order} λ_hi={args.lambda_hi} λ_lo={args.lambda_lo} thr={args.ppm_threshold} nn_skip_thr_nats={args.ppm_nn_skip_thr_nats}")
    t1 = time.time()
    if use_antihijack:
        rc = lib.ppm_score_omp(
            target_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            prev_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            nll_nats.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int64(len(target_ids)),
            flat.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            offs.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            lens.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            has_space.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            is_boundary.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            ctypes.c_int(vocab),
            ctypes.c_int(args.ppm_order),
            ctypes.c_double(args.lambda_hi),
            ctypes.c_double(args.lambda_lo),
            ctypes.c_double(args.ppm_threshold),
            ctypes.c_double(args.ppm_nn_skip_thr_nats),
            ctypes.c_uint32(args.ppm_log_cache),
            ctypes.c_int64(args.ppm_chunk_tokens),
            ctypes.c_int(args.ppm_omp_threads),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
    else:
        rc = lib.ppm_score_omp(
            target_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            prev_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            nll_nats.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int64(len(target_ids)),
            flat.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            offs.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            lens.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            has_space.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            is_boundary.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            ctypes.c_int(vocab),
            ctypes.c_int(args.ppm_order),
            ctypes.c_double(args.lambda_hi),
            ctypes.c_double(args.lambda_lo),
            ctypes.c_double(args.ppm_threshold),
            ctypes.c_uint32(args.ppm_log_cache),
            ctypes.c_int64(args.ppm_chunk_tokens),
            ctypes.c_int(args.ppm_omp_threads),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
    print(f"[ppm] score_omp returned {rc} in {time.time()-t1:.1f}s")
    if rc != 0:
        print(f"[ppm] ERROR: rc={rc}")
        sys.exit(1)

    mix_bpb, ppm_only_bpb, nn_byte_bpb, token_bpb, n_bytes, gate_high_frac = out
    print(f"[ppm] mix_bpb={mix_bpb:.5f} ppm_only={ppm_only_bpb:.5f} nn_byte_bpb={nn_byte_bpb:.5f} bytes={int(n_bytes)} gate_high={gate_high_frac:.4f}")

    # === proper-margin + PPM mix experiment ===
    if PROPER_MARGIN_ONLINE and pm_byte_stream_list:
        try:
            nbs = len(pm_byte_stream_list)
            byte_stream_np = np.array(pm_byte_stream_list, dtype=np.uint8)
            nn_logp_np = np.array(pm_nn_logp_list, dtype=np.float64)
            print(f"[pm-mix] computing PPM mix on {nbs:,} bytes (proper-margin nn_logp)")
            lib.ppm_score_bytewise.restype = ctypes.c_int
            lib.ppm_score_bytewise.argtypes = [
                ctypes.POINTER(ctypes.c_uint8),
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_int64,
                ctypes.c_int,
                ctypes.c_double, ctypes.c_double, ctypes.c_double,
                ctypes.c_uint32,
                ctypes.POINTER(ctypes.c_double),
            ]
            out_pm = np.zeros(6, dtype=np.float64)
            rc = lib.ppm_score_bytewise(
                byte_stream_np.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                nn_logp_np.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                ctypes.c_int64(nbs),
                ctypes.c_int(args.ppm_order),
                ctypes.c_double(args.lambda_hi),
                ctypes.c_double(args.lambda_lo),
                ctypes.c_double(args.ppm_threshold),
                ctypes.c_uint32(args.ppm_log_cache),
                out_pm.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            )
            print(f"[pm-mix] PROPER-MARGIN + PPM:")
            print(f"[pm-mix]   rc={rc}  mix_bpb={out_pm[0]:.5f}  ppm_only={out_pm[1]:.5f}  "
                  f"nn_proper_bpb={out_pm[2]:.5f}  bytes={int(out_pm[4])}  gate_hi={out_pm[5]:.4f}")
            print(f"[pm-mix]   COMPARISON:")
            print(f"[pm-mix]     uniform-spread mix_bpb (spec 055):  {mix_bpb:.5f}")
            print(f"[pm-mix]     proper-margin   mix_bpb (this run): {out_pm[0]:.5f}")
            print(f"[pm-mix]     diff: {out_pm[0]-mix_bpb:+.5f}")

            # === Per-byte dump comparison: find explicit examples of bookkeeping artifact ===
            print(f"\n[pm-dump] running dump scorer to capture per-byte uniform/PPM/gate data")
            max_bytes = nbs + 1024
            dump_mix = np.zeros(max_bytes, dtype=np.float32)
            dump_ppm = np.zeros(max_bytes, dtype=np.float32)
            dump_nn_uniform = np.zeros(max_bytes, dtype=np.float32)  # uniform-spread nn_nll per byte
            dump_conf = np.zeros(max_bytes, dtype=np.float32)
            dump_gate_hi = np.zeros(max_bytes, dtype=np.uint8)
            dump_byte = np.zeros(max_bytes, dtype=np.uint8)
            dump_n_bytes = np.zeros(1, dtype=np.uint64)
            lib.ppm_score_dump.restype = ctypes.c_int
            lib.ppm_score_dump.argtypes = [
                ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64),
                ctypes.POINTER(ctypes.c_double), ctypes.c_int64,
                ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
                ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_uint8),
                ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double,
                ctypes.c_uint32, ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_uint8),
                ctypes.POINTER(ctypes.c_uint64),
            ]
            out_dump = np.zeros(6, dtype=np.float64)
            rc_dump = lib.ppm_score_dump(
                target_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
                prev_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
                nll_nats.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                ctypes.c_int64(len(target_ids)),
                flat.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                offs.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                lens.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                has_space.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                is_boundary.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                ctypes.c_int(vocab), ctypes.c_int(args.ppm_order),
                ctypes.c_double(args.lambda_hi), ctypes.c_double(args.lambda_lo), ctypes.c_double(args.ppm_threshold),
                ctypes.c_uint32(args.ppm_log_cache),
                out_dump.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                dump_mix.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                dump_ppm.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                dump_nn_uniform.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                dump_conf.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                dump_gate_hi.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                dump_byte.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                dump_n_bytes.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            )
            n_dump = int(dump_n_bytes[0])
            print(f"[pm-dump] dumped {n_dump} bytes (proper-margin had {nbs}); aligned: {n_dump==nbs}")

            # Save and find examples
            if n_dump == nbs:
                # Sanity: dump_byte should match pm_byte_stream
                mismatches = int((byte_stream_np != dump_byte[:n_dump]).sum())
                print(f"[pm-dump] byte-stream mismatches: {mismatches} (should be 0)")
                # Per-byte: dump_nn_uniform = -log(p_NN_byte_uniform) in nats (positive NLL)
                # dump_ppm = -log(p_PPM_byte) in nats
                # nn_logp_np = log(p_NN_byte_proper) in nats (negative log-prob)
                # so per-byte NLL: nn_proper_nats = -nn_logp_np
                nn_proper_nats = -nn_logp_np[:n_dump]
                nn_uniform_nats = dump_nn_uniform[:n_dump].astype(np.float64)
                ppm_nats = dump_ppm[:n_dump].astype(np.float64)
                gate_hi = dump_gate_hi[:n_dump]
                bytes_arr = dump_byte[:n_dump]

                # Find "artifact bytes": gate_hi=1 (PPM rescues with λ_lo=0.05, 95% PPM)
                #   AND nn_proper_nats much smaller than nn_uniform_nats (proper says NN was confident)
                #   AND ppm_nats < nn_uniform_nats (PPM looks cheap vs uniform NN)
                artifact_score = (nn_uniform_nats - nn_proper_nats) * gate_hi.astype(np.float64)
                top_idx = np.argsort(-artifact_score)[:30]

                # Save full per-byte arrays
                np.savez_compressed("/tmp/per_byte_compare.npz",
                    bytes=bytes_arr,
                    nn_proper_nats=nn_proper_nats.astype(np.float32),
                    nn_uniform_nats=nn_uniform_nats.astype(np.float32),
                    ppm_nats=ppm_nats.astype(np.float32),
                    gate_hi=gate_hi,
                )
                print(f"[pm-dump] saved per-byte data → /tmp/per_byte_compare.npz")

                LOG2_n = math.log(2.0)
                print(f"\n[examples] TOP 30 'artifact bytes' (gate_hi=1, big proper-vs-uniform divergence):")
                print(f"  fmt: idx | byte | uniform_nat | proper_nat | ppm_nat | savings_per_byte_nat")
                for k_, idx in enumerate(top_idx):
                    b = int(bytes_arr[idx])
                    ch = chr(b) if 32 <= b < 127 else f"\\x{b:02x}"
                    # Show context: 8 bytes before
                    pre_start = max(0, int(idx) - 8)
                    pre_bytes = bytes(bytes_arr[pre_start:int(idx)+1])
                    pre_str = pre_bytes.decode("utf-8", errors="replace")
                    print(f"  [{int(idx):8d}] byte={ch!r:>5}  "
                          f"uniform={nn_uniform_nats[idx]:6.3f}n  proper={nn_proper_nats[idx]:6.3f}n  "
                          f"ppm={ppm_nats[idx]:6.3f}n  savings={(nn_uniform_nats[idx]-nn_proper_nats[idx]):.3f}n  "
                          f"ctx={pre_str!r}")
        except Exception as e:
            print(f"[pm-mix/dump] FAILED: {e}")
            import traceback; traceback.print_exc()

    # ---- SAVE RAW DATA IMMEDIATELY (before any report-writing that could crash) ----
    npz_path = Path(args.out).with_suffix(".npz")
    save_dict = dict(
        tokens=np.array(tokens, dtype=np.int32),
        nll_nats=nll_nats.astype(np.float32),
        top_ids=top_ids,
        top_logp=top_logp,
        ppm_out=out,
        ckpt_path=np.array([args.ckpt]),
        ppm_config=np.array([args.ppm_order, args.lambda_hi, args.lambda_lo, args.ppm_threshold], dtype=np.float64),
    )
    if val_bytes_per_tok_full is not None:
        save_dict["val_bytes_per_tok"] = val_bytes_per_tok_full.astype(np.int32)
    np.savez_compressed(npz_path, **save_dict)
    print(f"[save] raw data → {npz_path} ({npz_path.stat().st_size//1024//1024} MB)")

    # ---- official-style val_bpb computation using sidecar bytes ----
    if val_bytes_per_tok_full is not None:
        # Match official: byte budget for target tokens (positions 1..n-1)
        target_bytes = val_bytes_per_tok_full[1:n].astype(np.float64)
        total_nll_bits = nll_nats.sum() / math.log(2.0)
        total_bytes_official = float(target_bytes.sum())
        nn_bpb_official = total_nll_bits / total_bytes_official
        print(f"[official] NN val_bpb (sidecar bytes, varlen attn) = {nn_bpb_official:.5f}")
        print(f"           total bits = {total_nll_bits:,.0f}, total bytes = {int(total_bytes_official):,}")

    # ---- failure analysis breakdown by token category ----
    pieces = [sp.id_to_piece(t) if t < vocab else "<oor>" for t in tokens]
    actual_pieces = pieces[1:n]
    cats = [categorize(p) for p in actual_pieces]
    NLL_bits = nll_nats / math.log(2)
    bytes_per = lens[target_ids]
    bpb_per = NLL_bits / np.maximum(bytes_per, 1)

    from collections import defaultdict
    cat_count = defaultdict(int)
    cat_total_bits = defaultdict(float)
    cat_total_bytes = defaultdict(int)
    for i, c in enumerate(cats):
        cat_count[c] += 1
        cat_total_bits[c] += float(NLL_bits[i])
        cat_total_bytes[c] += int(bytes_per[i])
    total_bits = float(NLL_bits.sum())
    total_bytes_meas = int(bytes_per.sum())
    measured_nn_bpb = total_bits / max(total_bytes_meas, 1)

    # NLL distribution buckets
    buckets = [(0, 0.5), (0.5, 1.5), (1.5, 3.0), (3.0, 5.0), (5.0, 999)]
    bucket_count = [0]*len(buckets); bucket_sum = [0.0]*len(buckets)
    for x in bpb_per:
        for j, (lo, hi) in enumerate(buckets):
            if lo <= x < hi:
                bucket_count[j] += 1
                bucket_sum[j] += float(x)
                break

    # Top/bottom NLL positions for context
    sorted_idx = np.argsort(-NLL_bits)
    worst = sorted_idx[:25]
    best = sorted_idx[-25:][::-1]

    def ctx_str(pos, k=25):
        end = pos + 1
        start = max(0, end - k)
        s = "".join(pieces[i].replace("▁", " ") for i in range(start, end))
        return s.replace("\n", "↵")[-80:]

    # ---- write report ----
    out_lines = []
    out_lines.append(f"# PPM-D byte-mixture inspection")
    out_lines.append(f"")
    out_lines.append(f"**Checkpoint:** `{args.ckpt}`")
    out_lines.append(f"**Tokens scored:** {len(target_ids)} (bytes: {int(n_bytes)})")
    out_lines.append(f"**PPM config:** order={args.ppm_order} λ_hi={args.lambda_hi} λ_lo={args.lambda_lo} threshold={args.ppm_threshold}")
    out_lines.append(f"")
    out_lines.append(f"## Headline")
    out_lines.append(f"")
    out_lines.append(f"| Metric | bits/byte |")
    out_lines.append(f"|---|---:|")
    out_lines.append(f"| **NN only** (`nn_byte_bpb`) | **{nn_byte_bpb:.5f}** |")
    out_lines.append(f"| PPM only (`ppm_only`) | {ppm_only_bpb:.5f} |")
    out_lines.append(f"| **NN + PPM mix** (`mix_bpb`) | **{mix_bpb:.5f}** |")
    out_lines.append(f"| **Δ from PPM** | **{mix_bpb - nn_byte_bpb:+.5f}** |")
    out_lines.append(f"| Gate high-confidence fraction | {gate_high_frac:.4f} ({100*gate_high_frac:.1f}%) |")
    out_lines.append(f"| Token-level reference (`token_bpb`) | {token_bpb:.5f} |")
    out_lines.append(f"")
    out_lines.append(f"## Comparison to dexhunter's #1857")
    out_lines.append(f"")
    out_lines.append(f"| | #1857 | this run |")
    out_lines.append(f"|---|---:|---:|")
    out_lines.append(f"| nn_byte_bpb | 1.10020 | {nn_byte_bpb:.5f} |")
    out_lines.append(f"| ppm_only | 2.34028 | {ppm_only_bpb:.5f} |")
    out_lines.append(f"| mix_bpb | 1.03176 | {mix_bpb:.5f} |")
    out_lines.append(f"| gate_high_frac | 0.14241 | {gate_high_frac:.5f} |")
    out_lines.append(f"| Δ from PPM | -0.06844 | {mix_bpb - nn_byte_bpb:+.5f} |")
    out_lines.append(f"")
    out_lines.append(f"## Per-category contribution to NN-only val_bpb")
    out_lines.append(f"")
    out_lines.append(f"| category | count | % positions | mean bits/byte | bits | % NN val_bpb |")
    out_lines.append(f"|---|---:|---:|---:|---:|---:|")
    for cat in sorted(cat_count, key=lambda c: -cat_total_bits[c]):
        c = cat_count[cat]
        pct = 100*c/max(len(cats),1)
        bb = cat_total_bytes[cat]
        mean = cat_total_bits[cat] / max(bb, 1) if bb else 0
        contrib = 100 * cat_total_bits[cat] / max(total_bits, 1e-9)
        out_lines.append(f"| {cat} | {c} | {pct:.1f}% | {mean:.2f} | {cat_total_bits[cat]:.1f} | {contrib:.1f}% |")
    out_lines.append(f"")
    ppm_addr_pct = 100 * sum(cat_total_bits[c] for c in ['URL','NUMERIC','CODE','HEX','PATH']) / max(total_bits, 1e-9)
    out_lines.append(f"**PPM-addressable categories (URL+NUMERIC+CODE+HEX+PATH): {ppm_addr_pct:.1f}% of NN val_bpb**")
    out_lines.append(f"")
    out_lines.append(f"## NLL distribution (NN only, bits/byte)")
    out_lines.append(f"")
    out_lines.append(f"| bucket | count | % | mean | contribution |")
    out_lines.append(f"|---|---:|---:|---:|---:|")
    for j, (lo, hi) in enumerate(buckets):
        c = bucket_count[j]; pct = 100*c/max(len(cats),1)
        mean = bucket_sum[j]/max(c,1)
        contrib = bucket_sum[j]/max(len(cats),1)
        contrib_pct = 100*contrib/max(measured_nn_bpb, 1e-9)
        hi_str = "∞" if hi > 100 else f"{hi:.1f}"
        out_lines.append(f"| {lo:.1f}–{hi_str} | {c} | {pct:.1f}% | {mean:.3f} | {contrib:.4f} ({contrib_pct:.1f}%) |")
    out_lines.append(f"")
    out_lines.append(f"## Top-50 most catastrophic NN predictions")
    out_lines.append(f"")
    out_lines.append(f"These are the bytes where the model was most surprised — high NLL means the model assigned ~0% probability to what actually came next.")
    out_lines.append(f"")
    out_lines.append(f"| pos | NLL (bits) | actual | top-1 prediction (prob) | category | left context (last 50 chars) |")
    out_lines.append(f"|---|---:|---|---|---|---|")
    for pos in sorted_idx[:50]:
        top1_id = int(top_ids[pos][0])
        top1_piece = sp.id_to_piece(top1_id) if top1_id < vocab else "<oor>"
        top1_p = math.exp(float(top_logp[pos][0]))
        out_lines.append(f"| {int(pos)} | {NLL_bits[pos]:.2f} | `{actual_pieces[pos]!r}` | `{top1_piece!r}` ({top1_p:.3f}) | {cats[pos]} | `{ctx_str(int(pos), k=50)}` |")
    out_lines.append(f"")
    out_lines.append(f"## Worst-10 per category")
    out_lines.append(f"")
    out_lines.append(f"Where each kind of byte fails most. PPM-addressable categories (URL/NUMERIC/CODE) are exactly where PPM helps.")
    out_lines.append(f"")
    for target_cat in ["URL", "NUMERIC", "CODE", "HEX", "PATH", "PROSE"]:
        cat_positions = [i for i, c in enumerate(cats) if c == target_cat]
        if not cat_positions: continue
        cat_sorted = sorted(cat_positions, key=lambda i: -NLL_bits[i])[:10]
        out_lines.append(f"### {target_cat} ({len(cat_positions)} positions, mean NLL {sum(NLL_bits[i] for i in cat_positions)/len(cat_positions):.2f} bits)")
        out_lines.append(f"")
        out_lines.append(f"| pos | NLL | actual | top-1 (prob) | left context |")
        out_lines.append(f"|---|---:|---|---|---|")
        for pos in cat_sorted:
            top1_id = int(top_ids[pos][0])
            top1_piece = sp.id_to_piece(top1_id) if top1_id < vocab else "<oor>"
            top1_p = math.exp(float(top_logp[pos][0]))
            out_lines.append(f"| {int(pos)} | {NLL_bits[pos]:.2f} | `{actual_pieces[pos]!r}` | `{top1_piece!r}` ({top1_p:.3f}) | `{ctx_str(int(pos), k=50)}` |")
        out_lines.append(f"")
    out_lines.append(f"## Top-25 best NN predictions (contrast)")
    out_lines.append(f"")
    out_lines.append(f"| pos | NLL (bits) | actual | category | left context |")
    out_lines.append(f"|---|---:|---|---|---|")
    for pos in best:
        out_lines.append(f"| {int(pos)} | {NLL_bits[pos]:.2f} | `{actual_pieces[pos]!r}` | {cats[pos]} | `{ctx_str(int(pos))}` |")

    Path(args.out).write_text("\n".join(out_lines))
    print(f"\n[done] wrote {len(out_lines)} lines to {args.out}")

if __name__ == "__main__":
    main()
