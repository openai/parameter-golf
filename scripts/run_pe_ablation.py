#!/usr/bin/env python3
"""Launch the PE ablation suite in one bounded RunPod HTTP-bootstrap session.

The launcher intentionally uploads only the AGENTS.md allowlist bundle:
``train_gpt.py``, ``cached_challenge_fineweb.py``, ``tokenizer_specs.json``,
and ``requirements.txt``.  Variant selection is passed through environment
variables so the pod never needs non-allowlisted ``results/pe_ablation_*.py``
helper scripts.

Dry run locally before any paid launch:
    python scripts/run_pe_ablation.py --dry-run
"""

import argparse
import ast
import base64
import csv
from dataclasses import dataclass
import hashlib
import io
import lzma
import sys
import tarfile
from pathlib import Path

# Add scripts/ to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from runpod_http_rehearsal import (
    REPO_ROOT, build_boot_command,
    wait_http_proxy, download_file, H100_COST_PER_GPU_HR,
    build_launcher_state, write_launcher_state, record_launcher_exception,
    terminate_pod_with_launcher_state,
    wait_startup_readiness_and_maybe_download_status,
)
from pe_ablation_train_template import get_template_metadata, get_template_source
from runpod_safe import RUNTIME_WAIT_SECONDS, balance, create_pod, terminate_and_wait, wait_runtime

PER_RUN_WALLCLOCK_SECONDS = 600
RETRIEVAL_BUFFER_SECONDS = 300
DEFAULT_GPU_COUNT = 8
DEFAULT_SMOKE_GPU_COUNT = 1
DEFAULT_MAX_MINUTES = 120
DEFAULT_SMOKE_MAX_MINUTES = 8
CSV_FIELDS = (
    "run_id",
    "seed",
    "ns_steps",
    "coeff_type",
    "gram_ns",
    "steps_completed",
    "train_seconds",
    "optimizer_seconds",
    "optimizer_pct",
    "sliding_bpb",
    "ttt_bpb",
    "artifact_bytes",
)
CSV_HEADER = ",".join(CSV_FIELDS)
BUNDLE_ALLOWED_ARCNAMES = (
    "train_gpt.py",
    "cached_challenge_fineweb.py",
    "tokenizer_specs.json",
    "requirements.txt",
)
TEMPLATE_TRAIN_SCRIPT = REPO_ROOT / "scripts" / "pe_ablation_train_template.py"
ABLATED_MODEL_ARTIFACT = "final_model.int6.ptz"
ARTIFACT_CAP_BYTES = 16_000_000
HISTORICAL_MODEL_BYTE_REFERENCES = (15_973_220, 15_974_228)
CONSERVATIVE_WRAPPER_BYTE_CAP = 26_000
SMOKE_RUN_ID = "smoke_exact_payload"


def _replace_once(text, old, new):
    if old not in text:
        raise RuntimeError(f"Expected source fragment not found: {old[:80]!r}")
    return text.replace(old, new, 1)


def _replace_between(text, start, end, replacement):
    try:
        start_idx = text.index(start)
        end_idx = text.index(end, start_idx)
    except ValueError as exc:
        raise RuntimeError(f"Expected source range not found between {start!r} and {end!r}") from exc
    return text[:start_idx] + replacement + text[end_idx:]


def build_ablation_train_source():
    """Generate the allowlisted on-pod train_gpt.py from reviewed #1809-style code.

    The reviewed local template provider is ``scripts/pe_ablation_train_template.py``;
    the archive entry is always named exactly ``train_gpt.py``.  The generated
    source keeps the SP8192 recurrence, parallel-residual, GPTQ-int6, eval, and
    TTT semantics while making the R0/R1/R2 Newton-Schulz choice
    environment-configurable.
    """
    source = get_template_source(REPO_ROOT)
    source = "# Generated allowlisted train_gpt.py for #1809/#1493 PE ablation.\n" + source
    source = source.replace(
        'for cat in sorted(categories):log(f"  {cat}: {", ".join(sorted(categories[cat]))}")',
        'for cat in sorted(categories):log(f"  {cat}: {\', \'.join(sorted(categories[cat]))}")',
    )
    source = source.replace(
        'log(f"train_shards: {len(list(Path(h.datasets_dir).resolve().glob("fineweb_train_*.bin")))}");',
        'log(f"train_shards: {len(list(Path(h.datasets_dir).resolve().glob(\'fineweb_train_*.bin\')))}");',
    )
    source = _replace_once(
        source,
        "code_bytes=len(code.encode('utf-8'))",
        "code_bytes=os.path.getsize(__file__)",
    )

    ns_block = '''# Environment-configurable Newton-Schulz coefficients for PE ablation.
_FIXED_NS_COEFFS=(3.4445,-4.775,2.0315)
_POLAR_NS_COEFFS=((4.0848,-6.8946,2.927),(3.9505,-6.3029,2.6377),(3.7418,-5.5913,2.3037),(2.8769,-3.1427,1.2046),(2.8366,-3.0525,1.2012))
MUON_NS_COEFF_TYPE=os.environ.get('MUON_NS_COEFF_TYPE','fixed').strip().lower()
MUON_NS_GRAM=bool(int(os.environ.get('MUON_NS_GRAM','1')))
MUON_NS_IS_POLAR=MUON_NS_COEFF_TYPE in ('pe','polar','polar_express')
PE_ABLATION_CSV_FIELDS=('run_id','seed','ns_steps','coeff_type','gram_ns','steps_completed','train_seconds','optimizer_seconds','optimizer_pct','sliding_bpb','ttt_bpb','artifact_bytes')
PE_ABLATION_CSV_HEADER=','.join(PE_ABLATION_CSV_FIELDS)
def _format_bpb_for_csv(value):return 'N/A' if value is None else f"{value:.8f}"
def format_pe_ablation_csv(run_id,seed,ns_steps,coeff_type,gram_ns,steps_completed,train_seconds,optimizer_seconds,optimizer_pct,sliding_bpb,ttt_bpb,artifact_bytes):
	return ','.join((str(run_id),str(seed),str(ns_steps),str(coeff_type),'yes' if gram_ns else 'no',str(steps_completed),f"{train_seconds:.3f}",f"{optimizer_seconds:.3f}",f"{optimizer_pct:.3f}",_format_bpb_for_csv(sliding_bpb),_format_bpb_for_csv(ttt_bpb),str(artifact_bytes)))
def _selected_ns_coeffs(steps):
	if MUON_NS_IS_POLAR:
		if steps<1 or steps>len(_POLAR_NS_COEFFS):raise ValueError(f"Polar Express NS supports 1-{len(_POLAR_NS_COEFFS)} steps, got {steps}")
		return _POLAR_NS_COEFFS[-steps:]
	if MUON_NS_COEFF_TYPE!='fixed':raise ValueError(f"Unsupported MUON_NS_COEFF_TYPE={MUON_NS_COEFF_TYPE!r}; expected fixed or pe")
	return (_FIXED_NS_COEFFS,)*steps
@torch.compile
def _ns_standard(G,steps=4,eps=1e-07):
	"""Newton-Schulz with environment-selected coefficients. For square-ish matrices."""
	coeffs=_selected_ns_coeffs(steps);X=G.bfloat16();X/=X.norm()+eps;transposed=G.size(0)>G.size(1)
	if transposed:X=X.T
	for i in range(steps):
		a,b,c=coeffs[i];A=X@X.T
		if MUON_NS_IS_POLAR and i==0:s=torch.rsqrt(A.abs().sum(dim=-1).clamp(min=eps));X=X*s.unsqueeze(-1);A=A*s.unsqueeze(-1)*s.unsqueeze(-2)
		B=b*A+c*A@A;X=a*X+B@X
	return X.T if transposed else X
@torch.compile
def _ns_gram(G,steps=4,eps=1e-07):
	"""Gram-Newton-Schulz for rectangular matrices (aspect >= 1.5)."""
	coeffs=_selected_ns_coeffs(steps);X=G.bfloat16();X/=X.norm()+eps;transposed=G.size(0)>G.size(1)
	if transposed:X=X.T
	n=X.size(0);I_n=torch.eye(n,device=X.device,dtype=X.dtype);R=X@X.T;Q=I_n.clone()
	for i in range(steps):
		a,b,c=coeffs[i]
		if i==2:X=Q@X;R=X@X.T;Q=I_n.clone()
		if MUON_NS_IS_POLAR and i==0:s=torch.rsqrt(R.abs().sum(dim=-1).clamp(min=eps));X=X*s.unsqueeze(-1);R=R*s.unsqueeze(-1)*s.unsqueeze(-2)
		Z=b*R+c*(R@R)
		if i==0 or i==2:Q=a*I_n+Z
		else:Q=a*Q+Z@Q
		is_last=i==steps-1;next_restart=i+1==2
		if not is_last and not next_restart:RZ=Z@R+a*R;R=Z@RZ+a*RZ
	X=Q@X
	return X.T if transposed else X
def zeropower_via_newtonschulz5(G,steps=4,eps=1e-07):
	"""Dispatch: Gram-NS for rectangular matrices, standard NS for square-ish."""
	n,m=G.size(0),G.size(1)
	if MUON_NS_GRAM and max(n,m)/max(min(n,m),1)>=1.5:return _ns_gram(G,steps,eps)
	return _ns_standard(G,steps,eps)
'''
    if "# Polar Express per-iteration optimal minimax coefficients" in source:
        source = _replace_between(source, "# Polar Express per-iteration optimal minimax coefficients", "class Muon(torch.optim.Optimizer):", ns_block)
    else:
        source = _replace_between(source, "@torch.compile\ndef zeropower_via_newtonschulz5", "class Muon(torch.optim.Optimizer):", ns_block)
    source = _replace_once(
        source,
        "optimizers.step();return train_loss",
        "torch.cuda.synchronize();_opt_t0=time.perf_counter();optimizers.step();torch.cuda.synchronize();_opt_elapsed_ms=1e3*(time.perf_counter()-_opt_t0);return train_loss,_opt_elapsed_ms",
    )
    source = _replace_once(
        source,
        "ema_state={name:t.detach().float().clone()for(name,t)in base_model.state_dict().items()};ema_decay=h.ema_decay;training_time_ms=.0;stop_after_step=None;torch.cuda.synchronize();t0=time.perf_counter();step=0",
        "ema_state={name:t.detach().float().clone()for(name,t)in base_model.state_dict().items()};ema_decay=h.ema_decay;training_time_ms=.0;stop_after_step=None;_opt_time_accum=.0;_opt_step_count=0;torch.cuda.synchronize();t0=time.perf_counter();step=0",
    )
    source = _replace_once(
        source,
        "train_loss=step_fn(step,scale)",
        "train_loss,_step_opt_ms=step_fn(step,scale);_opt_time_accum+=_step_opt_ms;_opt_step_count+=1",
    )
    source = _replace_once(
        source,
        "if should_log_train:tok_per_sec=step*h.train_batch_tokens/(approx_training_time_ms/1e3);log(f\"{step}/{h.iterations} train_loss: {train_loss.item():.4f} train_time: {approx_training_time_ms/60000:.1f}m tok/s: {tok_per_sec:.0f}\")",
        "if should_log_train:tok_per_sec=step*h.train_batch_tokens/(approx_training_time_ms/1e3);avg_opt_ms=_opt_time_accum/_opt_step_count if _opt_step_count>0 else 0;avg_step_ms=approx_training_time_ms/step if step>0 else 1;opt_pct=100.*avg_opt_ms/avg_step_ms if avg_step_ms>0 else 0;log(f\"{step}/{h.iterations} train_loss: {train_loss.item():.4f} train_time: {approx_training_time_ms/60000:.1f}m tok/s: {tok_per_sec:.0f} optimizer_ms_per_step: {avg_opt_ms:.1f}ms ({opt_pct:.1f}% of step)\")",
    )
    source = _replace_once(
        source,
        "return base_model,compiled_model",
        "return base_model,compiled_model,step,training_time_ms/1e3,_opt_time_accum/1e3,_final_opt_pct",
    )
    source = _replace_once(
        source,
        "log(f\"peak memory allocated: {torch.cuda.max_memory_allocated()//1024//1024} MiB reserved: {torch.cuda.max_memory_reserved()//1024//1024} MiB\");log('ema:applying EMA weights');current_state=base_model.state_dict();avg_state={name:t.to(dtype=current_state[name].dtype)for(name,t)in ema_state.items()};base_model.load_state_dict(avg_state,strict=True);return base_model,compiled_model,step,training_time_ms/1e3,_opt_time_accum/1e3,_final_opt_pct",
        "log(f\"peak memory allocated: {torch.cuda.max_memory_allocated()//1024//1024} MiB reserved: {torch.cuda.max_memory_reserved()//1024//1024} MiB\");log('ema:applying EMA weights');current_state=base_model.state_dict();avg_state={name:t.to(dtype=current_state[name].dtype)for(name,t)in ema_state.items()};base_model.load_state_dict(avg_state,strict=True);_final_opt_pct=100.*_opt_time_accum/training_time_ms if training_time_ms>0 else 0;return base_model,compiled_model,step,training_time_ms/1e3,_opt_time_accum/1e3,_final_opt_pct",
    )
    source = _replace_once(
        source,
        "base_model,compiled_model=train_model(h,device,val_data);",
        "base_model,compiled_model,_steps_completed,_train_seconds,_optimizer_seconds,_optimizer_pct=train_model(h,device,val_data);",
    )
    source = _replace_once(
        source,
        "serialize(h,base_model,Path(__file__).read_text(encoding='utf-8'))",
        "_artifact_bytes,_=serialize(h,base_model,None)",
    )
    csv_block = """\tif h.is_main_process:
		log('PE_ABLATION_CSV: '+format_pe_ablation_csv(h.run_id,h.seed,h.muon_backend_steps,MUON_NS_COEFF_TYPE,MUON_NS_GRAM,_steps_completed,_train_seconds,_optimizer_seconds,_optimizer_pct,_sliding_bpb if isinstance(_sliding_bpb,float) else None,_ttt_bpb if isinstance(_ttt_bpb,float) else None,_artifact_bytes))
"""
    source = _replace_between(
        source,
        "\tif h.sliding_window_enabled:timed_eval('quantized_sliding_window'",
        "\tif h.etlb_enabled and h.sliding_window_enabled:",
        "\t_sliding_bpb='N/A'\n\tif h.sliding_window_enabled:_,_sliding_bpb=timed_eval('quantized_sliding_window',eval_val_sliding,h,device,val_data,eval_model)\n\t_ttt_bpb='N/A'\n\tif h.ttt_enabled and h.sliding_window_enabled:\n\t\tdel eval_model,compiled_model;torch._dynamo.reset();torch.cuda.empty_cache();ttt_model=deserialize(h,device)\n\t\tif h.num_loops>0:ttt_model.looping_active=True\n\t\t_,_ttt_bpb=timed_eval('quantized_ttt',eval_val_ttt,h,device,val_data,ttt_model);del ttt_model\n" + csv_block,
    )
    return source


def _sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def build_ablation_train_wrapper(source=None):
    """Return the compact allowlisted ``train_gpt.py`` submitted to RunPod.

    The full generated source remains the audited provenance payload.  The
    on-pod file is a deterministic stdlib-only wrapper that expands that source
    in memory and executes it with the wrapper's globals.  Keeping ``__file__``
    untouched means ``Path(__file__).read_text()`` and any future
    ``os.path.getsize(__file__)`` calls in the generated code account for the
    actual submitted wrapper bytes, not the decompressed source bytes.
    """
    if source is None:
        source = build_ablation_train_source()
    payload = base64.b85encode(lzma.compress(source.encode("utf-8"), preset=9))
    return "\n".join([
        "#!/usr/bin/env python3",
        "# Compact deterministic wrapper for generated #1809-style PE ablation source.",
        "import base64 as _B,lzma as _L",
        f"_PAYLOAD={payload!r}",
        "_SOURCE=_L.decompress(_B.b85decode(_PAYLOAD)).decode('utf-8')",
        "exec(compile(_SOURCE,__file__,'exec'),globals())",
        "",
    ])


def decode_ablation_train_wrapper(wrapper_text):
    """Decode a wrapper produced by :func:`build_ablation_train_wrapper`."""
    tree = ast.parse(wrapper_text)
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "_PAYLOAD" for target in node.targets):
            continue
        payload = ast.literal_eval(node.value)
        if isinstance(payload, str):
            payload = payload.encode("ascii")
        return lzma.decompress(base64.b85decode(payload)).decode("utf-8")
    raise RuntimeError("Wrapper payload assignment not found")


def ablation_source_provenance():
    """Return dry-run provenance for the reviewed source and generated payload."""
    template_meta = get_template_metadata(REPO_ROOT)
    source_text = get_template_source(REPO_ROOT)
    generated = build_ablation_train_source()
    wrapper = build_ablation_train_wrapper(generated)
    wrapper_bytes_raw = wrapper.encode("utf-8")
    wrapper_bytes = len(wrapper_bytes_raw)
    historical_model_bytes = list(HISTORICAL_MODEL_BYTE_REFERENCES)
    if historical_model_bytes:
        estimated_cap_headroom_min = ARTIFACT_CAP_BYTES - wrapper_bytes - max(historical_model_bytes)
        estimated_cap_headroom_max = ARTIFACT_CAP_BYTES - wrapper_bytes - min(historical_model_bytes)
    else:
        estimated_cap_headroom_min = None
        estimated_cap_headroom_max = None
    return {
        "payload_description": template_meta["payload_description"],
        "exact_upstream_1809": template_meta["exact_upstream_1809"],
        "template_path": template_meta["template_path"],
        "template_sha256": template_meta["template_sha256"],
        "embedded_source_sha256": template_meta["embedded_source_sha256"],
        "embedded_source_bytes": template_meta["embedded_source_bytes"],
        "normalized_template_sha256": _sha256_text(source_text),
        "generated_payload_sha256": _sha256_text(generated),
        "full_generated_source_sha256": _sha256_text(generated),
        "generated_line_count": len(generated.splitlines()),
        "generated_payload_bytes": len(generated.encode("utf-8")),
        "wrapper_sha256": _sha256_bytes(wrapper_bytes_raw),
        "wrapper_bytes": wrapper_bytes,
        "artifact_byte_accounting": f"wrapper_bytes+{ABLATED_MODEL_ARTIFACT}_bytes",
        "historical_model_bytes": historical_model_bytes,
        "estimated_cap_headroom_min": estimated_cap_headroom_min,
        "estimated_cap_headroom_max": estimated_cap_headroom_max,
        "bundle_entries": list(BUNDLE_ALLOWED_ARCNAMES),
    }


def _add_bytes_to_tar(tf, arcname, data):
    info = tarfile.TarInfo(arcname)
    info.size = len(data)
    info.mode = 0o644
    info.mtime = 0
    tf.addfile(info, io.BytesIO(data))


def dry_run_audit_text():
    provenance = ablation_source_provenance()
    historical_model_bytes = provenance["historical_model_bytes"]
    historical_model_text = ",".join(str(value) for value in historical_model_bytes) if historical_model_bytes else "none"
    return "\n".join([
        f"payload_description={provenance['payload_description']}",
        f"exact_upstream_1809={str(provenance['exact_upstream_1809']).lower()}",
        "recipe_transparency=derived from reviewed embedded template source; #1809-style GPTQ/int6/Brotli SP8192 ablation payload; not exact upstream #1809",
        f"template_path={provenance['template_path']}",
        f"template_sha256={provenance['template_sha256']}",
        f"embedded_source_sha256={provenance['embedded_source_sha256']}",
        f"embedded_source_bytes={provenance['embedded_source_bytes']}",
        f"normalized_template_sha256={provenance['normalized_template_sha256']}",
        f"generated_payload_sha256={provenance['generated_payload_sha256']}",
        f"full_generated_source_sha256={provenance['full_generated_source_sha256']}",
        f"generated_line_count={provenance['generated_line_count']}",
        f"generated_payload_bytes={provenance['generated_payload_bytes']}",
        f"wrapper_sha256={provenance['wrapper_sha256']}",
        f"wrapper_bytes={provenance['wrapper_bytes']}",
        f"artifact_byte_accounting={provenance['artifact_byte_accounting']}",
        f"historical_model_bytes={historical_model_text}",
        f"estimated_cap_headroom_min={provenance['estimated_cap_headroom_min']}",
        f"estimated_cap_headroom_max={provenance['estimated_cap_headroom_max']}",
        "bundle_entries=" + ",".join(provenance["bundle_entries"]),
        "DATA_PATH_INFO=informational_only; template derives datasets_dir from DATA_DIR and VOCAB_SIZE",
        "TOKENIZER_PATH_INFO=informational_only; template derives tokenizer_path from DATA_DIR and VOCAB_SIZE",
        "allowlist bundle entries:",
        f"  archive=train_gpt.py source=compact_wrapper generated_payload_sha256={provenance['generated_payload_sha256']} wrapper_sha256={provenance['wrapper_sha256']} wrapper_bytes={provenance['wrapper_bytes']}",
        "  archive=cached_challenge_fineweb.py source=data/cached_challenge_fineweb.py",
        "  archive=tokenizer_specs.json source=data/tokenizer_specs.json",
        "  archive=requirements.txt source=requirements.txt",
        f"artifact={ABLATED_MODEL_ARTIFACT}",
        f"csv_header={CSV_HEADER}",
        "variants=R0_fixed5_Gram:fixed:5:gram,R1_fixed4_Gram:fixed:4:gram,R2_PE4_Gram:pe_last4:4:gram",
        "on_pod_results_helpers=none",
    ])


def extract_pe_ablation_csv_rows(log_text):
    """Return parseable PE_ABLATION_CSV rows with the exact requested field count."""
    rows = []
    for line in log_text.splitlines():
        if "PE_ABLATION_CSV:" not in line:
            continue
        row = line.split("PE_ABLATION_CSV:", 1)[1].strip()
        parsed = next(csv.reader([row]))
        if len(parsed) == len(CSV_FIELDS):
            rows.append(row)
    return rows


def fallback_csv_row(run_id, seed, ns_steps, coeff_type, gram_ns, artifact_bytes):
    """Build an explicit fallback row without pretending missing metrics are present."""
    return ",".join([
        str(run_id),
        str(seed),
        str(ns_steps),
        str(coeff_type),
        "yes" if gram_ns else "no",
        "MISSING",
        "MISSING",
        "MISSING",
        "MISSING",
        "MISSING",
        "MISSING",
        str(artifact_bytes),
    ])


@dataclass(frozen=True)
class AblationRun:
    run_id: str
    ns_steps: int
    coeff_type: str


VARIANTS = (
    AblationRun("R0_fixed5_Gram", 5, "fixed"),
    AblationRun("R1_fixed4_Gram", 4, "fixed"),
    AblationRun("R2_PE4_Gram", 4, "pe"),
)


def build_run_artifacts(runs):
    return {
        run.run_id: [
            "train_stdout.txt",
            "exit_code.txt",
            "run_meta.env",
            f"{run.run_id}.txt",
            "final_model.pt",
            ABLATED_MODEL_ARTIFACT,
        ]
        for run in runs
    }


RUN_ARTIFACTS = build_run_artifacts(VARIANTS)


SMOKE_ARTIFACTS = {
    SMOKE_RUN_ID: [
        "smoke_stdout.txt",
        "train_stdout.txt",
        "exit_code.txt",
        "run_meta.env",
    ]
}


GLOBAL_ARTIFACTS = [
    "status.txt",
    "pgolf_exit_code.txt",
    "pgolf_stdout.txt",
    "overall_exit_code.txt",
    "pe_ablation_results.csv",
    "run_status.csv",
]


def select_variants(variant_text):
    """Return selected ablation variants from a comma-separated run-id list."""
    if not variant_text:
        return VARIANTS
    lookup = {run.run_id: run for run in VARIANTS}
    selected = []
    for raw in variant_text.split(","):
        run_id = raw.strip()
        if not run_id:
            continue
        if run_id not in lookup:
            valid = ", ".join(sorted(lookup))
            raise SystemExit(f"Unknown variant {run_id!r}; expected one of: {valid}")
        selected.append(lookup[run_id])
    if not selected:
        raise SystemExit("--variants did not select any runs")
    return tuple(selected)


def build_ablation_bundle_b64():
    """Return an AGENTS.md allowlist-only bundle for the ablation run.

    The uploaded tar members are exactly the allowlisted file names.  The
    training script member is generated into the archive as ``train_gpt.py``;
    no ``results/pe_ablation_*.py`` path or helper script exists on-pod.
    """
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        generated_source = build_ablation_train_source()
        _add_bytes_to_tar(tf, "train_gpt.py", build_ablation_train_wrapper(generated_source).encode("utf-8"))
        for rel_path in (
            Path("data/cached_challenge_fineweb.py"),
            Path("data/tokenizer_specs.json"),
            Path("requirements.txt"),
        ):
            tf.add(str(REPO_ROOT / rel_path), arcname=rel_path.name)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def build_ablation_cmd(gpus=DEFAULT_GPU_COUNT, runs=None):
    """Build the on-pod command that downloads data once and runs selected variants."""
    runs = tuple(runs or VARIANTS)
    run_lines = "\n".join(f"run_one {run.run_id} {run.ns_steps} {run.coeff_type}" for run in runs)
    clean_lines = "\n".join(
        f"# /root/runs/{run.run_id} -> /root/rehearsal_out/{run.run_id}" for run in runs
    )
    cmd = r'''
set -uo pipefail
cd /root/rehearsal_src
mkdir -p /root/rehearsal_out /root/runs
ARTIFACT_CAP_BYTES=16000000

echo "=== Installing requirements ==="
pip install -q --break-system-packages -r requirements.txt 2>&1 || { setup_ec=$?; echo "ERROR: requirements install failed with exit_code=${setup_ec}"; exit "${setup_ec}"; }

echo "=== Downloading SP8192 data once ==="
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 cached_challenge_fineweb.py --variant sp8192 2>&1 || { setup_ec=$?; echo "ERROR: SP8192 data download failed with exit_code=${setup_ec}"; exit "${setup_ec}"; }
echo "=== Data download complete ==="

printf 'run_id,seed,ns_steps,coeff_type,gram_ns,steps_completed,train_seconds,optimizer_seconds,optimizer_pct,sliding_bpb,ttt_bpb,artifact_bytes\n' > /root/rehearsal_out/pe_ablation_results.csv
printf 'run_id,exit_code,start_utc,end_utc,artifact_bytes,log_bytes,status,note\n' > /root/rehearsal_out/run_status.csv
overall_exit_code=0

run_one() {
    run_id="$1"
    ns_steps="$2"
    coeff_type="$3"
    run_dir="/root/runs/${run_id}"
    out_dir="/root/rehearsal_out/${run_id}"

    echo ""
    echo "========================================="
    echo "=== ${run_id}: ${coeff_type}, ${ns_steps} NS steps ==="
    echo "========================================="

    rm -rf "${run_dir}"
    mkdir -p "${run_dir}" "${out_dir}"
    cp /root/rehearsal_src/train_gpt.py "${run_dir}/train_gpt.py"
    ln -sfn /root/rehearsal_src/datasets "${run_dir}/datasets"
    ln -sfn /root/rehearsal_src/tokenizers "${run_dir}/tokenizers"

    cat > "${out_dir}/run_meta.env" <<EOF
RUN_ID=${run_id}
SEED=42
MAX_WALLCLOCK_SECONDS=600
VOCAB_SIZE=8192
DATA_DIR=${run_dir}/
QK_GAIN_INIT=5.25
TTT_ENABLED=1
TTT_LR=0.005
TTT_EPOCHS=3
SLIDING_WINDOW_ENABLED=1
EVAL_STRIDE=64
MIN_LR=0.1
GPTQ_RESERVE_SECONDS=0.5
MUON_BACKEND_STEPS=${ns_steps}
MUON_NS_COEFF_TYPE=${coeff_type}
MUON_NS_GRAM=1
MATRIX_BITS=6
EMBED_BITS=8
COMPRESSOR=brotli
DATA_PATH_INFO=${run_dir}/datasets/fineweb10B_sp8192
TOKENIZER_PATH_INFO=${run_dir}/tokenizers/fineweb_8192_bpe.model
DATA_PATH_INFO_NOTE=informational_only_template_derives_paths_from_DATA_DIR_and_VOCAB_SIZE
TOKENIZER_PATH_INFO_NOTE=informational_only_template_derives_paths_from_DATA_DIR_and_VOCAB_SIZE
EOF

    start_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    (
        cd "${run_dir}"
        export RUN_ID="${run_id}"
        export SEED=42
        export MAX_WALLCLOCK_SECONDS=600
        export VOCAB_SIZE=8192
        export DATA_DIR="${run_dir}/"
        export QK_GAIN_INIT=5.25
        export TTT_ENABLED=1
        export TTT_LR=0.005
        export TTT_EPOCHS=3
        export SLIDING_WINDOW_ENABLED=1
        export EVAL_STRIDE=64
        export MIN_LR=0.1
        export GPTQ_RESERVE_SECONDS=0.5
        export MUON_BACKEND_STEPS="${ns_steps}"
        export MUON_NS_COEFF_TYPE="${coeff_type}"
        export MUON_NS_GRAM=1
        export MATRIX_BITS=6
        export EMBED_BITS=8
        export COMPRESSOR=brotli
        torchrun --standalone --nproc_per_node=__GPU_COUNT__ train_gpt.py
    ) > "${out_dir}/train_stdout.txt" 2>&1
    ec=$?
    end_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf '%s\n' "${ec}" > "${out_dir}/exit_code.txt"

    cp -f "${run_dir}/logs/${run_id}.txt" "${out_dir}/${run_id}.txt" 2>/dev/null || true
    cp -f "${run_dir}/final_model.pt" "${out_dir}/final_model.pt" 2>/dev/null || true
    cp -f "${run_dir}/final_model.int6.ptz" "${out_dir}/final_model.int6.ptz" 2>/dev/null || true

    model_artifact_bytes=0
    if [ -s "${out_dir}/final_model.int6.ptz" ]; then
        model_artifact_bytes=$(wc -c < "${out_dir}/final_model.int6.ptz")
    fi
    code_bytes=$(wc -c < "${run_dir}/train_gpt.py" 2>/dev/null || printf '0')
    artifact_bytes=$((model_artifact_bytes + code_bytes))
    log_bytes=0
    if [ -s "${out_dir}/${run_id}.txt" ]; then
        log_bytes=$(wc -c < "${out_dir}/${run_id}.txt")
    elif [ -s "${out_dir}/train_stdout.txt" ]; then
        log_bytes=$(wc -c < "${out_dir}/train_stdout.txt")
    fi

    run_status="OK"
    run_note="usable_evidence_candidate"
    if [ "${artifact_bytes}" -ge "${ARTIFACT_CAP_BYTES}" ]; then
        run_status="CAP_VIOLATION"
        run_note="artifact_bytes>=16000000; not usable as record evidence"
        echo "CAP_VIOLATION: ${run_id} artifact_bytes=${artifact_bytes} artifact_bytes>=16000000; not usable as record evidence"
        if [ "${overall_exit_code}" -eq 0 ]; then
            overall_exit_code=70
        fi
    elif [ "${ec}" -ne 0 ]; then
        run_status="FAILED"
        run_note="exit_code=${ec}"
    fi

    csv_line="$(grep -h 'PE_ABLATION_CSV:' "${out_dir}/${run_id}.txt" "${out_dir}/train_stdout.txt" 2>/dev/null | tail -n 1 | sed 's/.*PE_ABLATION_CSV: //')"
    csv_fields=0
    if [ -n "${csv_line}" ]; then
        csv_fields="$(printf '%s\n' "${csv_line}" | awk -F, '{print NF; exit}')"
    fi
    if [ -n "${csv_line}" ] && [ "${csv_fields}" = "12" ]; then
        printf '%s\n' "${csv_line}" >> /root/rehearsal_out/pe_ablation_results.csv
    else
        if [ -n "${csv_line}" ]; then
            echo "WARNING: ignoring malformed PE_ABLATION_CSV for ${run_id} with ${csv_fields} fields"
        fi
        printf '%s,42,%s,%s,yes,MISSING,MISSING,MISSING,MISSING,MISSING,MISSING,%s\n' "${run_id}" "${ns_steps}" "${coeff_type}" "${artifact_bytes}" >> /root/rehearsal_out/pe_ablation_results.csv
    fi
    printf '%s,%s,%s,%s,%s,%s,%s,%s\n' "${run_id}" "${ec}" "${start_utc}" "${end_utc}" "${artifact_bytes}" "${log_bytes}" "${run_status}" "${run_note}" >> /root/rehearsal_out/run_status.csv

    if [ "${ec}" -ne 0 ] && [ "${overall_exit_code}" -eq 0 ]; then
        overall_exit_code="${ec}"
    fi
    echo "=== ${run_id} complete with exit_code=${ec} ==="
}

# Clean run directories and artifact directories for dry-run audit:
__CLEAN_DIR_LINES__
__RUN_LINES__

printf '%s\n' "${overall_exit_code}" > /root/rehearsal_out/overall_exit_code.txt
echo ""
echo "=== PE ablation run_status.csv ==="
cat /root/rehearsal_out/run_status.csv
echo ""
echo "=== PE ablation pe_ablation_results.csv ==="
cat /root/rehearsal_out/pe_ablation_results.csv
exit "${overall_exit_code}"
'''
    return (
        cmd.replace("__GPU_COUNT__", str(gpus))
        .replace("__CLEAN_DIR_LINES__", clean_lines)
        .replace("__RUN_LINES__", run_lines)
    )


def build_smoke_cmd():
    """Build an exact-bundle retrieval smoke that compiles/decodes without training."""
    return r'''
set -uo pipefail
cd /root/rehearsal_src
mkdir -p /root/rehearsal_out /root/runs/smoke_exact_payload /root/rehearsal_out/smoke_exact_payload

printf 'run_id,seed,ns_steps,coeff_type,gram_ns,steps_completed,train_seconds,optimizer_seconds,optimizer_pct,sliding_bpb,ttt_bpb,artifact_bytes\n' > /root/rehearsal_out/pe_ablation_results.csv
printf 'run_id,exit_code,start_utc,end_utc,artifact_bytes,log_bytes,status,note\n' > /root/rehearsal_out/run_status.csv

run_id="smoke_exact_payload"
run_dir="/root/runs/${run_id}"
out_dir="/root/rehearsal_out/${run_id}"
cp /root/rehearsal_src/train_gpt.py "${run_dir}/train_gpt.py"
cat > "${out_dir}/run_meta.env" <<EOF
RUN_ID=${run_id}
SMOKE_MODE=compile_decode_no_training
SMOKE_LIMITATION=retrieval smoke only; no training or validation data download
PAYLOAD_KIND=#1809-style GPTQ/int6/Brotli SP8192 ablation payload
EOF

start_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
(
python3 - <<'PY'
from pathlib import Path
import ast
import base64
import hashlib
import lzma

wrapper_path = Path('/root/rehearsal_src/train_gpt.py')
wrapper = wrapper_path.read_text(encoding='utf-8')
compile(wrapper, 'train_gpt.py', 'exec')
payload = None
tree = ast.parse(wrapper)
for node in tree.body:
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == '_PAYLOAD':
                payload = ast.literal_eval(node.value)
                break
    if payload is not None:
        break
if payload is None:
    raise RuntimeError('_PAYLOAD assignment not found in wrapper')
if isinstance(payload, str):
    payload = payload.encode('ascii')
source = lzma.decompress(base64.b85decode(payload)).decode('utf-8')
compile(source, 'decoded_train_gpt.py', 'exec')
print('SMOKE_MODE=compile_decode_no_training')
print('SMOKE_LIMITATION=retrieval smoke only; no training or validation data download')
print('wrapper_bytes={}'.format(len(wrapper.encode('utf-8'))))
print('wrapper_sha256={}'.format(hashlib.sha256(wrapper.encode('utf-8')).hexdigest()))
print('decoded_source_bytes={}'.format(len(source.encode('utf-8'))))
print('decoded_source_sha256={}'.format(hashlib.sha256(source.encode('utf-8')).hexdigest()))
PY
) > "${out_dir}/smoke_stdout.txt" 2>&1
ec=$?
end_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf '%s\n' "${ec}" > "${out_dir}/exit_code.txt"
cp -f "${out_dir}/smoke_stdout.txt" "${out_dir}/train_stdout.txt" 2>/dev/null || true
artifact_bytes=$(wc -c < "${run_dir}/train_gpt.py" 2>/dev/null || printf '0')
log_bytes=$(wc -c < "${out_dir}/smoke_stdout.txt" 2>/dev/null || printf '0')
status="OK"
note="retrieval smoke only; no training or validation data download"
if [ "${ec}" -ne 0 ]; then
    status="FAILED"
    note="compile/decode failed; retrieval smoke only; no training or validation data download"
fi
printf '%s,%s,%s,%s,%s,%s,%s,%s\n' "${run_id}" "${ec}" "${start_utc}" "${end_utc}" "${artifact_bytes}" "${log_bytes}" "${status}" "${note}" >> /root/rehearsal_out/run_status.csv
printf '%s\n' "${ec}" > /root/rehearsal_out/overall_exit_code.txt
echo "=== PE ablation exact-payload smoke ==="
cat "${out_dir}/smoke_stdout.txt" 2>/dev/null || true
echo "=== PE ablation run_status.csv ==="
cat /root/rehearsal_out/run_status.csv
exit "${ec}"
'''


def main():
    parser = argparse.ArgumentParser(description="PE ablation: R0/R1/R2 on a bounded RunPod HTTP-bootstrap pod")
    parser.add_argument(
        "--max-minutes",
        type=int,
        default=None,
        help=(
            f"Total pod job time budget before retrieval buffer "
            f"(default: {DEFAULT_MAX_MINUTES} min production, {DEFAULT_SMOKE_MAX_MINUTES} min smoke)"
        ),
    )
    parser.add_argument(
        "--gpus",
        type=int,
        choices=[1, 2, 4, 8],
        default=None,
        help=f"H100 GPU count (default: {DEFAULT_GPU_COUNT} production, {DEFAULT_SMOKE_GPU_COUNT} smoke)",
    )
    parser.add_argument(
        "--variants",
        default=None,
        help="Comma-separated production variants to run (default: all R0/R1/R2); ignored by --smoke",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run exact-bundle compile/decode retrieval smoke only; no training or validation data download",
    )
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--dry-run", action="store_true", help="Print cmd and exit")
    parser.add_argument("--docker-image", default=None)
    parser.add_argument(
        "--runtime-timeout-sec",
        type=int,
        default=RUNTIME_WAIT_SECONDS,
        help=f"Seconds to wait for RunPod runtime startup (default: {RUNTIME_WAIT_SECONDS})",
    )
    args = parser.parse_args()

    max_minutes = args.max_minutes
    if max_minutes is None:
        max_minutes = DEFAULT_SMOKE_MAX_MINUTES if args.smoke else DEFAULT_MAX_MINUTES
    gpu_count = args.gpus
    if gpu_count is None:
        gpu_count = DEFAULT_SMOKE_GPU_COUNT if args.smoke else DEFAULT_GPU_COUNT
    selected_runs = select_variants(args.variants) if not args.smoke else ()
    user_cmd = build_smoke_cmd() if args.smoke else build_ablation_cmd(gpus=gpu_count, runs=selected_runs)
    run_artifacts = SMOKE_ARTIFACTS if args.smoke else build_run_artifacts(selected_runs)
    pod_name = "pgolf-pe-ablation-smoke" if args.smoke else "pgolf-pe-ablation"

    if args.dry_run:
        print("=== DRY RUN: allowlist bundle audit ===")
        print(dry_run_audit_text())
        if args.smoke:
            print("=== DRY RUN: smoke mode ===")
            print(
                "Exact-bundle retrieval smoke only: compiles wrapper, decodes payload, "
                "writes retrieval artifacts; no training or validation data download."
            )
        else:
            print("=== DRY RUN: selected variants ===")
            print(",".join(run.run_id for run in selected_runs))
        print(f"gpus={gpu_count}")
        print(f"max_minutes={max_minutes}")
        print(f"runtime_timeout_sec={args.runtime_timeout_sec}")
        print("=== DRY RUN: on-pod command ===")
        print(user_cmd)
        print("=== DRY RUN: HTTP bootstrap command ===")
        print(build_boot_command(user_cmd))
        return

    pod_id = None
    out_base = None
    launcher_state = None
    original_exc = None
    try:
        bal, _ = balance()
        cost_est = gpu_count * H100_COST_PER_GPU_HR * max_minutes / 60.0
        print(
            f"Balance: ${bal:.2f}  Est cost: ${cost_est:.2f}  "
            f"({gpu_count} GPUs, {max_minutes} min + {RETRIEVAL_BUFFER_SECONDS}s retrieval buffer)"
        )
        if bal < cost_est * 1.5:
            raise SystemExit(f"Insufficient balance (need >= 1.5× est = ${cost_est * 1.5:.2f})")

        bundle_b64 = build_ablation_bundle_b64()
        print(f"Bundle: {len(bundle_b64)} chars base64 (allowlist only)")

        docker_args = build_boot_command(user_cmd)
        hard_deadline_sec = max_minutes * 60 + RETRIEVAL_BUFFER_SECONDS
        pod = create_pod(
            name=pod_name,
            gpus=gpu_count,
            max_minutes=max_minutes,
            docker_args=docker_args,
            extra_env={"PGOLF_BUNDLE_B64": bundle_b64, "PGOLF_MAX_MINUTES": str(max_minutes)},
            ports="30000/http",
            start_ssh=False,
            deadline_sec=hard_deadline_sec,
            image=args.docker_image,
        )
        pod_id = pod["id"]

        default_prefix = "pe_ablation_smoke" if args.smoke else "pe_ablation"
        out_base = Path(args.results_dir) if args.results_dir else REPO_ROOT / "results" / f"{default_prefix}_{pod_id}"
        launcher_state = build_launcher_state(
            launcher="run_pe_ablation",
            pod_id=pod_id,
            pod_name=pod_name,
            gpus=gpu_count,
            max_minutes=max_minutes,
            results_dir=out_base,
            hard_deadline_sec=hard_deadline_sec,
            bundle_b64=bundle_b64,
            command=user_cmd,
            docker_args=docker_args,
            docker_image=args.docker_image,
            runtime_timeout_sec=args.runtime_timeout_sec,
        )
        write_launcher_state(out_base, launcher_state)
        print(f"Pod: {pod_id}  ${pod.get('costPerHr', '?')}/hr")

        rt = wait_runtime(pod_id, timeout=args.runtime_timeout_sec)
        print(f"Pod RUNNING (uptime={rt['uptimeInSeconds']}s)")

        wait_startup_readiness_and_maybe_download_status(
            pod_id,
            30000,
            out_base,
            wait_func=wait_http_proxy,
            download_func=download_file,
        )
        wait_http_proxy(pod_id, 30000, timeout=hard_deadline_sec)
        print("RunPod HTTP endpoint ready, downloading available artifacts...")

        for name in GLOBAL_ARTIFACTS:
            path = download_file(pod_id, 30000, name, out_base, optional=True)
            if path:
                print(f"  {path.relative_to(out_base)} ({path.stat().st_size})")
            else:
                print(f"  {name} (not found)")

        for run_id, artifacts in run_artifacts.items():
            for artifact in artifacts:
                remote = f"{run_id}/{artifact}"
                path = download_file(pod_id, 30000, remote, out_base, optional=True)
                if path:
                    print(f"  {path.relative_to(out_base)} ({path.stat().st_size})")
                else:
                    print(f"  {remote} (not found)")
    except BaseException as exc:
        original_exc = exc
        if pod_id is not None and out_base is not None and launcher_state is not None:
            try:
                record_launcher_exception(out_base, launcher_state, exc)
            except BaseException as state_exc:
                print(
                    f"WARNING: failed to record launcher exception for pod {pod_id}: "
                    f"{state_exc.__class__.__name__}",
                    file=sys.stderr,
                )
        raise
    finally:
        if pod_id is not None and out_base is not None and launcher_state is not None:
            print(f"Terminating pod {pod_id}...")
            try:
                terminate_pod_with_launcher_state(
                    out_base,
                    launcher_state,
                    pod_id,
                    terminate_and_wait,
                    original_exc=original_exc,
                )
            except BaseException as cleanup_exc:
                if original_exc is None:
                    raise
                print(
                    f"WARNING: failed during cleanup bookkeeping for pod {pod_id} after "
                    f"{original_exc.__class__.__name__}: {cleanup_exc.__class__.__name__}",
                    file=sys.stderr,
                )

    csv_path = out_base / "pe_ablation_results.csv"
    if csv_path.exists():
        print("\n=== PE ABLATION RESULTS ===")
        print(csv_path.read_text(encoding="utf-8"))

    status_path = out_base / "run_status.csv"
    if status_path.exists():
        print("\n=== PER-RUN STATUS ===")
        print(status_path.read_text(encoding="utf-8"))

    print(f"\nArtifacts saved to: {out_base}")


if __name__ == "__main__":
    main()
