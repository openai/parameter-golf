#!/usr/bin/env python3
"""Patch PR #2018 with q-aware dynamic n-gram tilt.

This is an eval-time-only, score-before-update refinement of the PR #2018
closed-form token tilt. The existing code picks a prefix-only n-gram hint h and
fixed boost beta from empirical prefix confidence r. This patch also uses the
model's own prefix probability q=p_model(h) before the target token is scored:

    beta_dyn = clamp(logit(r) - logit(q), 0, beta_cap)

and applies the normalized tilt only when the prefix-estimated expected gain is
positive. It does not use the realized target to choose the hint or beta.
"""

from __future__ import annotations

import sys
from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise SystemExit(f"missing patch anchor: {label}")
    return text.replace(old, new, 1)


def patch_tilt(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if "qaware_dynamic" in text:
        print(f"{path}: q-aware ngram patch already present")
        return

    text = replace_once(
        text,
        "    hint_ids = np.zeros(total, dtype=np.int64)\n"
        "    boost = np.zeros(total, dtype=np.float32)\n"
        "    base_boost_per_expert = np.array([token_boost, within_boost, word_boost], dtype=np.float32)\n"
        "    hint_per_expert = np.stack([\n"
        "        token_top_tok.astype(np.int64),\n"
        "        within_top_tok.astype(np.int64),\n"
        "        word_top_tok.astype(np.int64),\n"
        "    ], axis=1)\n\n"
        "    rows = np.arange(total)\n"
        "    hint_ids[any_gate] = hint_per_expert[rows[any_gate], best_idx[any_gate]]\n"
        "    boost[any_gate] = base_boost_per_expert[best_idx[any_gate]]\n",
        "    hint_ids = np.zeros(total, dtype=np.int64)\n"
        "    boost = np.zeros(total, dtype=np.float32)\n"
        "    hint_conf = np.zeros(total, dtype=np.float32)\n"
        "    base_boost_per_expert = np.array([token_boost, within_boost, word_boost], dtype=np.float32)\n"
        "    hint_per_expert = np.stack([\n"
        "        token_top_tok.astype(np.int64),\n"
        "        within_top_tok.astype(np.int64),\n"
        "        word_top_tok.astype(np.int64),\n"
        "    ], axis=1)\n"
        "    conf_per_expert = np.stack([\n"
        "        token_top_prob.astype(np.float32),\n"
        "        within_top_prob.astype(np.float32),\n"
        "        word_top_prob.astype(np.float32),\n"
        "    ], axis=1)\n\n"
        "    rows = np.arange(total)\n"
        "    hint_ids[any_gate] = hint_per_expert[rows[any_gate], best_idx[any_gate]]\n"
        "    boost[any_gate] = base_boost_per_expert[best_idx[any_gate]]\n"
        "    hint_conf[any_gate] = conf_per_expert[rows[any_gate], best_idx[any_gate]]\n",
        "hint confidence selection",
    )

    text = replace_once(
        text,
        '            "boost":      np.zeros(0, dtype=np.float32),\n'
        '            "sp":         sp,\n',
        '            "boost":      np.zeros(0, dtype=np.float32),\n'
        '            "hint_conf":  np.zeros(0, dtype=np.float32),\n'
        '            "sp":         sp,\n',
        "empty return hint_conf",
    )
    text = replace_once(
        text,
        '        "boost":      boost,\n'
        '        "sp":         sp,\n'
        '        "starts_new_word_lut": starts_new_word_lut,\n',
        '        "boost":      boost,\n'
        '        "hint_conf":  hint_conf,\n'
        '        "sp":         sp,\n'
        '        "starts_new_word_lut": starts_new_word_lut,\n',
        "main return hint_conf",
    )

    text = replace_once(
        text,
        "def apply_tilt_to_ptl_torch(\n"
        "    ptl: torch.Tensor,\n"
        "    log_q_hint: torch.Tensor,\n"
        "    target_ids: torch.Tensor,\n"
        "    hint_ids: torch.Tensor,\n"
        "    gate_mask: torch.Tensor,\n"
        "    boost: torch.Tensor,\n"
        "):\n",
        "def apply_tilt_to_ptl_torch(\n"
        "    ptl: torch.Tensor,\n"
        "    log_q_hint: torch.Tensor,\n"
        "    target_ids: torch.Tensor,\n"
        "    hint_ids: torch.Tensor,\n"
        "    gate_mask: torch.Tensor,\n"
        "    boost: torch.Tensor,\n"
        "    hint_conf: torch.Tensor | None = None,\n"
        "    qaware_dynamic: bool = False,\n"
        "    gain_floor: float = 0.0,\n"
        "):\n",
        "slow signature",
    )
    text = replace_once(
        text,
        "    boost64 = boost.to(torch.float64)\n"
        "    q = log_q_hint.to(torch.float64).clamp_(max=0.0).exp()\n"
        "    is_hit = (target_ids == hint_ids).to(torch.float64)\n"
        "    log_Z = torch.log1p(q * (torch.expm1(boost64)))\n"
        "    ptl_tilted = ptl.to(torch.float64) - boost64 * is_hit + log_Z\n"
        "    return torch.where(gate_mask, ptl_tilted, ptl.to(torch.float64)).to(ptl.dtype)\n",
        "    boost64 = boost.to(torch.float64)\n"
        "    q = log_q_hint.to(torch.float64).clamp_(max=0.0).exp()\n"
        "    if qaware_dynamic and hint_conf is not None:\n"
        "        r = hint_conf.to(torch.float64).clamp(1e-4, 1.0 - 1e-4)\n"
        "        qc = q.clamp(1e-6, 1.0 - 1e-6)\n"
        "        beta_dyn = (torch.logit(r) - torch.logit(qc)).clamp(min=0.0)\n"
        "        boost64 = torch.minimum(boost64, beta_dyn)\n"
        "    is_hit = (target_ids == hint_ids).to(torch.float64)\n"
        "    log_Z = torch.log1p(q * (torch.expm1(boost64)))\n"
        "    if qaware_dynamic and hint_conf is not None:\n"
        "        r = hint_conf.to(torch.float64).clamp(0.0, 1.0)\n"
        "        gate_mask = gate_mask & ((r * boost64 - log_Z) > float(gain_floor)) & (boost64 > 1e-7)\n"
        "    ptl_tilted = ptl.to(torch.float64) - boost64 * is_hit + log_Z\n"
        "    return torch.where(gate_mask, ptl_tilted, ptl.to(torch.float64)).to(ptl.dtype)\n",
        "slow qaware body",
    )

    text = replace_once(
        text,
        "def apply_tilt_to_ptl_torch_fast(\n"
        "    ptl: torch.Tensor,\n"
        "    log_q_hint: torch.Tensor,\n"
        "    target_ids: torch.Tensor,\n"
        "    hint_ids: torch.Tensor,\n"
        "    gate_mask: torch.Tensor,\n"
        "    boost: torch.Tensor,\n"
        "):\n",
        "def apply_tilt_to_ptl_torch_fast(\n"
        "    ptl: torch.Tensor,\n"
        "    log_q_hint: torch.Tensor,\n"
        "    target_ids: torch.Tensor,\n"
        "    hint_ids: torch.Tensor,\n"
        "    gate_mask: torch.Tensor,\n"
        "    boost: torch.Tensor,\n"
        "    hint_conf: torch.Tensor | None = None,\n"
        "    qaware_dynamic: bool = False,\n"
        "    gain_floor: float = 0.0,\n"
        "):\n",
        "fast signature",
    )
    text = replace_once(
        text,
        "    boost32 = boost.to(torch.float32)\n"
        "    q = log_q_hint.to(torch.float32).clamp_(max=0.0).exp()\n"
        "    is_hit = (target_ids == hint_ids).to(torch.float32)\n"
        "    log_Z = torch.log1p(q * (torch.expm1(boost32)))\n"
        "    ptl_f32 = ptl.to(torch.float32)\n"
        "    ptl_tilted = ptl_f32 - boost32 * is_hit + log_Z\n"
        "    return torch.where(gate_mask, ptl_tilted, ptl_f32).to(ptl.dtype)\n",
        "    boost32 = boost.to(torch.float32)\n"
        "    q = log_q_hint.to(torch.float32).clamp_(max=0.0).exp()\n"
        "    if qaware_dynamic and hint_conf is not None:\n"
        "        r = hint_conf.to(torch.float32).clamp(1e-4, 1.0 - 1e-4)\n"
        "        qc = q.clamp(1e-6, 1.0 - 1e-6)\n"
        "        beta_dyn = (torch.logit(r) - torch.logit(qc)).clamp(min=0.0)\n"
        "        boost32 = torch.minimum(boost32, beta_dyn)\n"
        "    is_hit = (target_ids == hint_ids).to(torch.float32)\n"
        "    log_Z = torch.log1p(q * (torch.expm1(boost32)))\n"
        "    if qaware_dynamic and hint_conf is not None:\n"
        "        r = hint_conf.to(torch.float32).clamp(0.0, 1.0)\n"
        "        gate_mask = gate_mask & ((r * boost32 - log_Z) > float(gain_floor)) & (boost32 > 1e-7)\n"
        "    ptl_f32 = ptl.to(torch.float32)\n"
        "    ptl_tilted = ptl_f32 - boost32 * is_hit + log_Z\n"
        "    return torch.where(gate_mask, ptl_tilted, ptl_f32).to(ptl.dtype)\n",
        "fast qaware body",
    )

    path.write_text(text, encoding="utf-8")
    print(f"patched {path}")


def patch_train(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if "ngram_qaware_dynamic" in text:
        print(f"{path}: q-aware train patch already present")
        return

    if '    agree_add_boost = float(os.environ.get("AGREE_ADD_BOOST", "0.500"))\n' in text:
        text = text.replace(
            '    agree_add_boost = float(os.environ.get("AGREE_ADD_BOOST", "0.500"))\n',
            '    agree_add_boost = float(os.environ.get("AGREE_ADD_BOOST", "0.500"))\n'
            '    ngram_qaware_dynamic = bool(int(os.environ.get("NGRAM_QAWARE_DYNAMIC", "0")))\n'
            '    ngram_qaware_gain_floor = float(os.environ.get("NGRAM_QAWARE_GAIN_FLOOR", "0.0"))\n',
            1,
        )
    elif '    agree_add_boost = float(os.environ.get("AGREE_ADD_BOOST", "0.0"))\n' in text:
        text = text.replace(
            '    agree_add_boost = float(os.environ.get("AGREE_ADD_BOOST", "0.0"))\n',
            '    agree_add_boost = float(os.environ.get("AGREE_ADD_BOOST", "0.0"))\n'
            '    ngram_qaware_dynamic = bool(int(os.environ.get("NGRAM_QAWARE_DYNAMIC", "0")))\n'
            '    ngram_qaware_gain_floor = float(os.environ.get("NGRAM_QAWARE_GAIN_FLOOR", "0.0"))\n',
            1,
        )
    else:
        raise SystemExit("missing patch anchor: Hyperparameters ngram qaware")
    text = replace_once(
        text,
        '    hint_global = torch.from_numpy(hints_pkg["hint_ids"].astype("int64"))\n'
        '    gate_global = torch.from_numpy(hints_pkg["gate_mask"])\n'
        '    boost_global = torch.from_numpy(hints_pkg["boost"].astype("float32"))\n',
        '    hint_global = torch.from_numpy(hints_pkg["hint_ids"].astype("int64"))\n'
        '    gate_global = torch.from_numpy(hints_pkg["gate_mask"])\n'
        '    boost_global = torch.from_numpy(hints_pkg["boost"].astype("float32"))\n'
        '    conf_global = torch.from_numpy(hints_pkg.get("hint_conf", np.zeros_like(hints_pkg["boost"])).astype("float32"))\n',
        "outside conf global",
    )
    text = replace_once(
        text,
        "    return (hint_global, gate_global, boost_global)\n",
        "    return (hint_global, gate_global, boost_global, conf_global)\n",
        "outside tuple return",
    )
    text = replace_once(
        text,
        "    ngram_boost_global = None\n",
        "    ngram_boost_global = None\n"
        "    ngram_conf_global = None\n",
        "conf global init",
    )
    text = replace_once(
        text,
        "        ngram_hint_global, ngram_gate_global, ngram_boost_global = precomputed_hints\n",
        "        if len(precomputed_hints) == 3:\n"
        "            ngram_hint_global, ngram_gate_global, ngram_boost_global = precomputed_hints\n"
        "            ngram_conf_global = torch.zeros_like(ngram_boost_global, dtype=torch.float32)\n"
        "        else:\n"
        "            ngram_hint_global, ngram_gate_global, ngram_boost_global, ngram_conf_global = precomputed_hints\n",
        "precomputed tuple unpack",
    )
    text = replace_once(
        text,
        '        ngram_hint_global = torch.from_numpy(hints_pkg["hint_ids"].astype("int64"))\n'
        '        ngram_gate_global = torch.from_numpy(hints_pkg["gate_mask"])\n'
        '        ngram_boost_global = torch.from_numpy(hints_pkg["boost"].astype("float32"))\n',
        '        ngram_hint_global = torch.from_numpy(hints_pkg["hint_ids"].astype("int64"))\n'
        '        ngram_gate_global = torch.from_numpy(hints_pkg["gate_mask"])\n'
        '        ngram_boost_global = torch.from_numpy(hints_pkg["boost"].astype("float32"))\n'
        '        ngram_conf_global = torch.from_numpy(hints_pkg.get("hint_conf", np.zeros_like(hints_pkg["boost"])).astype("float32"))\n',
        "inline conf global",
    )
    text = replace_once(
        text,
        "            boost_gpu = None\n",
        "            boost_gpu = None\n"
        "            conf_gpu = None\n",
        "conf gpu init",
    )
    text = replace_once(
        text,
        "                boost_gpu = ngram_boost_global[hint_idx_cpu].to(\n"
        "                    device=device, dtype=torch.float32, non_blocking=True\n"
        "                )\n"
        "                hint_ids_gpu = torch.where(valid, hint_ids_gpu, torch.zeros_like(hint_ids_gpu))\n"
        "                gate_mask_gpu = gate_mask_gpu & valid\n",
        "                boost_gpu = ngram_boost_global[hint_idx_cpu].to(\n"
        "                    device=device, dtype=torch.float32, non_blocking=True\n"
        "                )\n"
        "                conf_gpu = ngram_conf_global[hint_idx_cpu].to(\n"
        "                    device=device, dtype=torch.float32, non_blocking=True\n"
        "                )\n"
        "                hint_ids_gpu = torch.where(valid, hint_ids_gpu, torch.zeros_like(hint_ids_gpu))\n"
        "                gate_mask_gpu = gate_mask_gpu & valid\n"
        "                conf_gpu = torch.where(valid, conf_gpu, torch.zeros_like(conf_gpu))\n",
        "gather conf gpu",
    )
    text = replace_once(
        text,
        "                    boost=boost_gpu,\n"
        "                )\n",
        "                    boost=boost_gpu,\n"
        "                    hint_conf=conf_gpu,\n"
        "                    qaware_dynamic=h.ngram_qaware_dynamic,\n"
        "                    gain_floor=h.ngram_qaware_gain_floor,\n"
        "                )\n",
        "apply qaware args",
    )

    path.write_text(text, encoding="utf-8")
    print(f"patched {path}")


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: patch_qaware_ngram_2018.py <record_dir_or_train_gpt.py>", file=sys.stderr)
        return 2
    target = Path(argv[1])
    record_dir = target.parent if target.name == "train_gpt.py" else target
    train = record_dir / "train_gpt.py"
    tilt = record_dir / "online_ngram_tilt.py"
    if not train.exists() or not tilt.exists():
        print(f"missing train_gpt.py or online_ngram_tilt.py under {record_dir}", file=sys.stderr)
        return 2
    patch_tilt(tilt)
    patch_train(train)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
