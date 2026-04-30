#!/usr/bin/env python3
"""
Extract techniques from top 10 GitHub PRs.
Combines with local submission analysis to create comprehensive mapping.
"""

import os
import re
from pathlib import Path
from collections import defaultdict

def extract_pr_info(md_file):
    """Extract title, description, and code from PR markdown."""
    with open(md_file) as f:
        content = f.read()

    # Extract PR title (usually the first H1 or in metadata)
    title_match = re.search(r'#\s+([^\n]+)', content)
    title = title_match.group(1) if title_match else "Unknown"

    # Extract score if mentioned
    score_match = re.search(r'(\d+\.\d{4})\s*(?:BPB|bpb)', content)
    bpb = score_match.group(1) if score_match else None

    # Look for code blocks
    code_blocks = re.findall(r'```(?:python|py)?\n(.*?)\n```', content, re.DOTALL)

    return {
        "title": title,
        "bpb": bpb,
        "code_blocks": code_blocks,
        "content": content
    }

def detect_techniques_from_content(content):
    """Detect techniques from PR content and code."""
    techniques = set()

    technique_keywords = {
        "GPTQ": [r"gptq", r"quantiz.*calibr", r"hessian.*quant"],
        "QAT": [r"qat", r"quantization.*aware", r"straight.through"],
        "Int6": [r"int6", r"int.6", r"nf6"],
        "Int8": [r"int8", r"int.8"],
        "BigramHash": [r"bigram.*hash", r"bigram_vocab"],
        "XSA": [r"cross.seq|cross_seq", r"xsa", r"cross.*attention"],
        "TTT": [r"test.time.*train", r"lora.*ttt", r"pre.eval"],
        "GDN": [r"gated.*delta", r"deltanet", r"gdn"],
        "LoRA": [r"lora", r"low.rank"],
        "Depth Recurrence": [r"depth.*recur", r"recurrent.*depth"],
        "Flash Attention": [r"flash.*attn", r"flash_attn"],
        "SWA": [r"stochastic.*weight", r"weight.*averaging", r"swa"],
        "EMA": [r"exponential.*moving", r"ema"],
        "Muon": [r"muon", r"newton.schulz", r"orthogonal"],
        "SmearGate": [r"smear.*gate"],
        "Cosine Warmdown": [r"cosine.*warm", r"warmdown"],
        "Sliding Window": [r"sliding.*window"],
        "Diffusion": [r"diffusion", r"masked.*diffusion"],
        "JEPA": [r"jepa", r"joint.*embedding"],
        "U-Net": [r"u.net|unet"],
    }

    for technique, patterns in technique_keywords.items():
        for pattern in patterns:
            if re.search(pattern, content, re.IGNORECASE):
                techniques.add(technique)
                break

    return sorted(list(techniques))

def main():
    pr_dir = Path(".firecrawl")
    pr_files = sorted(pr_dir.glob("pr-*.md"))

    results = {}
    technique_to_prs = defaultdict(list)

    print("Analyzing top 10 GitHub PRs...\n")

    for pr_file in pr_files:
        pr_num = pr_file.stem.split("-")[1]

        info = extract_pr_info(str(pr_file))
        techniques = detect_techniques_from_content(info["content"])

        results[f"PR #{pr_num}"] = {
            "title": info["title"],
            "bpb": info["bpb"],
            "techniques": techniques,
        }

        for tech in techniques:
            technique_to_prs[tech].append(f"PR #{pr_num}")

        print(f"✅ PR #{pr_num}: {len(techniques)} techniques found")

    print(f"\n📊 Total techniques found across 10 PRs: {len(technique_to_prs)}\n")

    # Generate markdown
    md = []
    md.append("# Top 10 GitHub PRs — Technique Analysis\n\n")
    md.append("**Analyzed PRs**: Top 10 non-record and experimental submissions\n")
    md.append("**Generated**: 2026-04-14\n\n")
    md.append("---\n\n")

    md.append("## Top 10 Submissions by PR\n\n")

    for pr, data in sorted(results.items()):
        bpb = data["bpb"] or "N/A"
        techniques = ", ".join(data["techniques"]) if data["techniques"] else "N/A"

        md.append(f"### {pr}\n\n")
        md.append(f"**Title**: {data['title']}\n\n")
        md.append(f"**Score**: {bpb}\n\n")
        md.append(f"**Techniques**: {techniques}\n\n")

    md.append("\n---\n\n")

    # Techniques in top PRs
    md.append("## Techniques in Top 10 PRs (by frequency)\n\n")

    sorted_techniques = sorted(technique_to_prs.items(),
                              key=lambda x: len(x[1]), reverse=True)

    for tech, prs in sorted_techniques:
        md.append(f"- **{tech}** — {', '.join(prs)}\n")

    output_file = Path("docs/top-prs-analysis.md")
    output_file.write_text("\n".join(md))

    print(f"✅ Analysis complete!")
    print(f"📝 Output written to: {output_file}")

if __name__ == "__main__":
    main()
