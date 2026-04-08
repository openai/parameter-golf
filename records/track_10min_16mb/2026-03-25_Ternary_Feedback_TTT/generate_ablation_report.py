import re
import sys
import os

def parse_logs(log_path):
    if not os.path.exists(log_path):
        return "Log file not found."

    with open(log_path, 'r') as f:
        content = f.read()

    # Split by configurations
    configs = re.split(r'==========================================================================', content)
    results = []
    
    current_config = None
    current_seed = None
    
    for section in configs:
        name_match = re.search(r'RUNNING: ([\d\.\w]+) \(Seed: (\d+)\)', section)
        if name_match:
            current_config = name_match.group(1)
            current_seed = name_match.group(2)
            continue
            
        if current_config:
            # Extract metrics
            params = re.search(r'model_params:(\d+)', section)
            final_bpb = re.search(r'final_sliding val_loss:[\d\.]+ val_bpb:([\d\.]+)', section)
            
            p_val = params.group(1) if params else "N/A"
            bpb_val = final_bpb.group(1) if final_bpb else "N/A"
            
            if bpb_val != "N/A":
                results.append({
                    "Config": current_config,
                    "Seed": current_seed,
                    "Params": p_val,
                    "BPB": float(bpb_val)
                })
            current_config = None

    if not results:
        return "No completed runs found in logs."

    # Aggregate by config
    agg = {}
    for r in results:
        cfg = r["Config"]
        if cfg not in agg:
            agg[cfg] = {"seeds": [], "params": r["Params"]}
        agg[cfg]["seeds"].append(r["BPB"])

    report = "# 🏆 Ultimate Ablation Study Report\n\n"
    report += "| Configuration | Params | Mean BPB | Seeds | StdDev |\n"
    report += "| :--- | :--- | :--- | :--- | :--- |\n"
    
    import numpy as np
    
    for cfg in sorted(agg.keys()):
        seeds = agg[cfg]["seeds"]
        params = agg[cfg]["params"]
        mean_bpb = np.mean(seeds)
        std_bpb = np.std(seeds) if len(seeds) > 1 else 0.0
        seed_str = ", ".join([f"{s:.4f}" for s in seeds])
        
        report += f"| **{cfg}** | {int(params):,} | **{mean_bpb:.4f}** | {seed_str} | {std_bpb:.4f} |\n"
        
    return report

if __name__ == "__main__":
    log_file = "ultimate_ablation.log"
    if len(sys.argv) > 1:
        log_file = sys.argv[1]
    
    print(parse_logs(log_file))
