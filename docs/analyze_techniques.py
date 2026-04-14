#!/usr/bin/env python3
"""
Analyze all local train_gpt.py files to extract key techniques.
Outputs a markdown table mapping techniques to submissions.
"""

import os
import re
from pathlib import Path
from collections import defaultdict

# Define technique patterns to look for
TECHNIQUE_PATTERNS = {
    # Quantization
    "GPTQ": r"gptq|Gptq|GPTQ|calibration.*quantiz",
    "QAT": r"qat_enabled|QAT|quantization.*aware|straight.through",
    "Int6": r"int6|INT6|int.6|nf6|fp6",
    "Int8": r"int8|INT8|int.8",
    "Int4": r"int4|INT4|int.4|nf4",
    "NF": r"nf\d|normalized.*float",

    # Architecture
    "XSA": r"xsa|cross.seq|cross_seq|XSA",
    "BigramHash": r"bigram.*hash|bigram_vocab|BIGRAM",
    "TrigramHash": r"trigram.*hash|TRIGRAM|trigram_enabled",
    "RoPE": r"rope|rotary.*pos|ROPE|rope_dims|rope_base",
    "PartialRoPE": r"partial.*rope|rope_dims.*[0-9]",
    "TTT": r"ttt|test.time.*train|pre.eval|lora.*ttt|test.*time.*training",
    "LoRA": r"lora|low.rank.*adapt",
    "Depth Recurrence": r"depth.*recur|recurrent.*depth|depth_recurrent",
    "GDN": r"gdn|gated.*delta|deltanet",
    "Mamba": r"mamba|state.*space|ssm",
    "U-Net": r"u.net|unet|U-Net",

    # Optimization
    "Muon": r"muon|Muon|newton.schulz|orthogonal",
    "Adam": r"adam|Adam|adamw|AdamW",
    "SGD": r"sgd|SGD",
    "AdamW": r"adamw|AdamW",

    # Regularization & Training
    "SWA": r"swa|stochastic.*weight|weight.*averaging|SWA",
    "EMA": r"ema|exponential.*moving|EMA",
    "Weight Decay": r"weight.*decay|wd|adam_wd|muon_wd",
    "Gradient Clipping": r"grad.*clip|gradient.*clip",

    # Attention & Layers
    "Flash Attention": r"flash.*attn|flash_attn",
    "Sliding Window": r"sliding.*window|window.*eval|eval_stride",
    "GQA": r"gqa|grouped.*query.*attn|num_kv_heads",
    "Multi-Head": r"num_heads|attention.*heads",
    "SmearGate": r"smear.*gate|SmearGate",
    "LeakyReLU": r"leaky.*relu|LeakyReLU",

    # Embedding & Tokenization
    "Tied Embeddings": r"tie.*embed|tied_embed",
    "Vocab Optimization": r"vocab_size|custom.*vocab|sp\d+|BPE.*vocab",
    "FP16 Embed": r"fp16.*embed|embed.*float16",
    "OrthoInit": r"ortho.*init|orthogonal.*init",

    # Eval & Metrics
    "Prediction Mixing": r"prediction.*mix|pred.*mix|ensemble.*eval",
    "Logit Softcap": r"logit.*softcap",

    # Compression
    "LZMA": r"lzma|LZMA",
    "Zstd": r"zstd|zstandard",

    # Other
    "Cosine Warmdown": r"cosine.*warm|warmdown|cooldown",
    "NTK": r"ntk|neural.*token.*kernel",
    "Warmup": r"warmup_steps|warmup",
}

def extract_config_value(content, key):
    """Extract a config value from train_gpt.py content."""
    pattern = rf'"{key}"\s*,\s*["\']([^"\']+)["\']'
    match = re.search(pattern, content)
    if match:
        return match.group(1)

    pattern = rf'{key}\s*=\s*([^\n,]+)'
    match = re.search(pattern, content)
    if match:
        val = match.group(1).strip()
        # Clean up
        val = re.sub(r'\s*#.*$', '', val)
        return val.strip().strip("'\"")

    return None

def detect_techniques(content):
    """Detect techniques used in a train_gpt.py file."""
    detected = []
    for technique, pattern in TECHNIQUE_PATTERNS.items():
        if re.search(pattern, content, re.IGNORECASE):
            detected.append(technique)
    return sorted(list(set(detected)))

def get_bpb_from_submission(folder_path):
    """Extract BPB from submission.json or folder name."""
    # Try folder name first (e.g., "2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072")
    folder_name = os.path.basename(folder_path)

    # Try to find BPB in the name
    match = re.search(r'(\d+\.\d+)', folder_name)
    if match:
        return match.group(1)

    # Try submission.json
    try:
        import json
        submission_file = os.path.join(folder_path, "submission.json")
        if os.path.exists(submission_file):
            with open(submission_file) as f:
                data = json.load(f)
                if "val_loss" in data:
                    # Convert val_loss to BPB (val_loss is typically ln(2) * BPB)
                    import math
                    bpb = data["val_loss"] / math.log(2)
                    return f"{bpb:.4f}"
    except:
        pass

    return None

def analyze_all_submissions():
    """Analyze all local submissions."""
    base_path = Path("/Users/sanathbs/03_Dev_Lab/projects/Personal/golf-parameter/parameter-golf/records")

    results = {}
    technique_to_submissions = defaultdict(list)

    # Analyze record submissions
    track_dir = base_path / "track_10min_16mb"
    if track_dir.exists():
        for submission_dir in sorted(track_dir.iterdir()):
            if not submission_dir.is_dir():
                continue

            train_file = submission_dir / "train_gpt.py"
            if not train_file.exists():
                continue

            folder_name = submission_dir.name

            # Read train_gpt.py
            with open(train_file) as f:
                content = f.read()

            techniques = detect_techniques(content)
            bpb = get_bpb_from_submission(str(submission_dir))

            results[folder_name] = {
                "type": "record",
                "techniques": techniques,
                "bpb": bpb,
                "date": folder_name.split("_")[0] if "_" in folder_name else "unknown",
            }

            for tech in techniques:
                technique_to_submissions[tech].append(folder_name)

    # Analyze non-record submissions
    nonrec_dir = base_path / "track_non_record_16mb"
    if nonrec_dir.exists():
        for submission_dir in sorted(nonrec_dir.iterdir()):
            if not submission_dir.is_dir():
                continue

            train_file = submission_dir / "train_gpt.py"
            if not train_file.exists():
                continue

            folder_name = submission_dir.name

            with open(train_file) as f:
                content = f.read()

            techniques = detect_techniques(content)
            bpb = get_bpb_from_submission(str(submission_dir))

            results[folder_name] = {
                "type": "non-record",
                "techniques": techniques,
                "bpb": bpb,
                "date": folder_name.split("_")[0] if "_" in folder_name else "unknown",
            }

            for tech in techniques:
                technique_to_submissions[tech].append(folder_name)

    return results, technique_to_submissions

def generate_markdown(results, technique_to_submissions):
    """Generate markdown output."""
    md = []
    md.append("# Parameter Golf Local Submissions Analysis\n")
    md.append(f"**Total submissions analyzed**: {len(results)} (21 record + 3 non-record)\n")
    md.append("**Generated**: 2026-04-14 (automated technique extraction)\n\n")
    md.append("---\n\n")

    # Technique frequency
    md.append("## Techniques by Frequency\n\n")
    sorted_techniques = sorted(technique_to_submissions.items(),
                              key=lambda x: len(x[1]), reverse=True)

    for technique, submissions in sorted_techniques:
        count = len(submissions)
        pct = (count / len(results)) * 100
        md.append(f"- **{technique}** — {count} submissions ({pct:.0f}%)\n")

    md.append("\n---\n\n")

    # Technique to submissions mapping
    md.append("## Techniques → Submissions Mapping\n\n")

    for technique, submissions in sorted_techniques:
        md.append(f"### {technique}\n\n")

        # Sort submissions by date (reverse = newest first)
        sorted_subs = sorted(submissions, reverse=True)

        for sub in sorted_subs:
            result = results[sub]
            bpb = result.get("bpb", "N/A")
            sub_type = result.get("type", "unknown")

            # Shorten the folder name
            short_name = sub.replace("2026-", "")

            md.append(f"- `{short_name}` ({sub_type}, BPB: {bpb})\n")

        md.append("\n")

    md.append("\n---\n\n")

    # Record submissions (sorted by date desc)
    md.append("## Record Submissions (10min_16mb) — Chronological\n\n")
    md.append("| Date | Submission | BPB | Techniques |\n")
    md.append("|------|-----------|-----|------------|\n")

    records = [(k, v) for k, v in results.items() if v["type"] == "record"]
    records.sort(reverse=True)  # Newest first

    for folder_name, data in records:
        date = data.get("date", "unknown")
        short_name = folder_name.replace("2026-", "")
        bpb = data.get("bpb", "N/A")
        techniques = ", ".join(data.get("techniques", [])[:5])  # First 5
        if len(data.get("techniques", [])) > 5:
            techniques += f" +{len(data['techniques']) - 5}"

        md.append(f"| {date} | `{short_name}` | {bpb} | {techniques} |\n")

    return "".join(md)

if __name__ == "__main__":
    print("Analyzing local Parameter Golf submissions...\n")
    results, technique_to_submissions = analyze_all_submissions()

    markdown = generate_markdown(results, technique_to_submissions)

    # Write to file
    output_file = Path("/Users/sanathbs/03_Dev_Lab/projects/Personal/golf-parameter/parameter-golf/docs/local-techniques-analysis.md")
    output_file.write_text(markdown)

    print(f"✅ Analysis complete!")
    print(f"📊 Found {len(technique_to_submissions)} unique techniques")
    print(f"📝 Output written to: {output_file}")
    print("\nTop 10 techniques by adoption:")
    for i, (tech, subs) in enumerate(sorted(technique_to_submissions.items(), key=lambda x: len(x[1]), reverse=True)[:10], 1):
        print(f"   {i}. {tech}: {len(subs)} submissions")
