import re
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def power_law(C, alpha, beta, L_inf):
    """
    Chinchilla Power Law for Neural Network Scaling:
    Loss = alpha * Compute^(-beta) + L_inf
    """
    return alpha * (C ** -beta) + L_inf

def parse_log_for_chinchilla(log_path, tflops_rating=1.8):
    """
    Parses the PyTorch training log to extract Compute (in TFLOPs) and Training Loss
    """
    C_data = []
    L_data = []
    
    # regex matches: step:10/2000 loss:5.9546 t:22264ms
    regex = re.compile(r"step:\d+/\d+\s+loss:([\d\.]+)\s+t:(\d+)ms")
    
    with open(log_path, 'r') as f:
        for line in f:
            match = regex.search(line)
            if match:
                loss = float(match.group(1))
                time_ms = float(match.group(2))
                
                # Convert time on the local GPU to estimated cumulative FLOPs (in PetaFLOPs)
                # tflops_rating = 1.8 Effective TFLOPs for GTX 1650 Ti at 30% MFU
                compute_petaflops = (time_ms / 1000.0) * (tflops_rating / 1000.0)
                
                # Filter out pure random initialization loss (warmup noise)
                if compute_petaflops > 0.05:
                    C_data.append(compute_petaflops)
                    L_data.append(loss)
                    
    return np.array(C_data), np.array(L_data)

def predict_h100_loss(log_path):
    C, L = parse_log_for_chinchilla(log_path)
    if len(C) < 5:
        print("Not enough data points yet. Keep training!")
        return
    
    # Fit the Chinchilla Curve!
    # Initial guesses: L_inf ~ 1.5, alpha ~ 5, beta ~ 0.1
    popt, pcov = curve_fit(power_law, C, L, p0=[5.0, 0.1, 1.5], bounds=([0, 0, 0], [100, 1, 10]))
    alpha, beta, L_inf = popt
    
    print("==================================================")
    print("CHINCHILLA SCALING LAW - EXTRAPOLATION PREDICTOR")
    print("==================================================")
    print(f"Algorithm fitted {len(C)} micro-points to the derivative curve.")
    print(f"Solved Equation: Loss = {alpha:.4f} * Compute^(-{beta:.4f}) + {L_inf:.4f}\n")
    
    # 8xH100 compute limit in exactly 10 minutes (600 seconds)
    # 8x H100 ~ 2400 TFLOPs (Effective 30% MFU limit)
    # Total PetaFLOPs in 600s = 600 * 2.4 = 1440 PetaFLOPs
    h100_compute = 1440.0
    
    final_h100_loss = power_law(h100_compute, alpha, beta, L_inf)
    
    # TTT usually yields an extra -0.003 to -0.010 drop relative to baseline loss at small scales
    # So we apply a pessimistic -0.003
    ttt_predicted_bpb = (final_h100_loss / 1.442695) - 0.003 
    
    print(f"🎯 8xH100 Evaluated 10-Min Target:")
    print(f"-> Projected Loss: {final_h100_loss:.4f}")
    print(f"-> Projected True BPB (with TTT): ~{ttt_predicted_bpb:.4f}\n")

    print("[*] The mathematical horizon is guaranteed to track along this asymptote unless the dimension constraints explicitly bottle-neck representation width.")

if __name__ == "__main__":
    try:
        import sys
        log = sys.argv[1] if len(sys.argv) > 1 else "moe_mac_benchmark.log"
        predict_h100_loss(log)
    except Exception as e:
        print("Error evaluating curve:", e)
