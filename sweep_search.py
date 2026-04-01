import os
import itertools
import subprocess
import time

# Ultra-fast micro-sweep configuration
# We only have ~1 hour of compute ($25)
# We test just enough to see the convergence trajectory (approx 120 seconds per run)
MATRIX_LRS = [0.025, 0.035]
MLP_MULTS = [3.5, 4.0]
KV_LATENT_DIMS = [64, 128]

# Short run parameters constraint
MICRO_ITERATIONS = 1500  # Abbreviate from 20000 to 1500 to just check the trajectory slope
MICRO_WALLCLOCK = 120.0  # Force timeout at 2 minutes

def run_micro_sweep():
    print("🚨 EMERGENCY COMPUTE PIVOT: Starting Micro-Sweeps")
    print(f"Time Budget: {MICRO_WALLCLOCK}s per config")
    print("=========================================================")
    
    configs = list(itertools.product(MATRIX_LRS, MLP_MULTS, KV_LATENT_DIMS))
    total_runs = len(configs)
    current_run = 1
    
    for lr, mult, ldim in configs:
        run_id = f"micro_lr{lr}_mlp{mult}_ldim{ldim}"
        print(f"\n[{current_run}/{total_runs}] Starting micro-run: {run_id}")
        
        env = os.environ.copy()
        env["MATRIX_LR"] = str(lr)
        env["MLP_MULT"] = str(mult)
        env["KV_LATENT_DIM"] = str(ldim)
        env["RUN_ID"] = run_id
        
        # Override scale parameters to make runs cheap and short
        env["ITERATIONS"] = str(MICRO_ITERATIONS)
        env["MAX_WALLCLOCK_SECONDS"] = str(MICRO_WALLCLOCK)
        # We don't spend time on quantization loops during micro sweeps
        env["QAT_ENABLED"] = "0"
        env["VAL_LOSS_EVERY"] = "100" 
        env["TRAIN_LOG_EVERY"] = "10" 
        env["SAVE_CHECKPOINT"] = "0"        
        cmd = [
            "torchrun", 
            "--standalone", 
            "--nproc_per_node=8", 
            "train_antigravity.py"
        ]
        
        start_time = time.time()
        try:
            subprocess.run(cmd, env=env, check=True)
            print(f"✅ Micro-Run {run_id} completed successfully in {time.time() - start_time:.1f}s")
        except subprocess.CalledProcessError as e:
            print(f"❌ Micro-Run {run_id} stopped or failed with code {e.returncode}")
            
        current_run += 1
        
    print("\n✅ Micro-sweep phase complete. Review the validation loss trajectories in the latest records/ folders to select your two finalists.")

if __name__ == "__main__":
    run_micro_sweep()
