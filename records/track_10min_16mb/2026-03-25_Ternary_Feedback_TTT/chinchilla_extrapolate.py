import numpy as np
import argparse

def chinchilla_optimal_loss(N, D):
    """
    Compute theoretical minimum loss from Chinchilla scaling laws (Hoffmann et al. 2022).
    L(N, D) = E + A / N^alpha + B / D^beta
    """
    E = 1.69
    A = 406.4
    B = 410.7
    alpha = 0.34
    beta = 0.28
    return E + A / (N ** alpha) + B / (D ** beta)

def project(params_1060ti, tokens_per_sec_1060ti, target_params=10e9, h100_multiplier=7900/6.0):
    """
    Extrapolates token throughput and expected BPB.
    Assuming 1060Ti is ~6 TFLOPS and 8x H100 is ~7900 TFLOPS (dense).
    """
    time_seconds = [600, 1800, 3600] # 10m, 30m, 60m
    
    print("=" * 60)
    print("   CHINCHILLA EXTRAPOLATION: 1060Ti -> 8x H100")
    print("=" * 60)
    print(f"Base Profile: {params_1060ti/1e6:.1f}M params at {tokens_per_sec_1060ti:,.0f} tok/s on 1060Ti")
    print(f"Target Scale: {target_params/1e9:.1f}B params on 8x H100 (~{h100_multiplier:.0f}x compute)\n")
    
    for t in time_seconds:
        # Base 1060ti 
        D_1060 = tokens_per_sec_1060ti * t
        L_1060 = chinchilla_optimal_loss(params_1060ti, D_1060)
        
        # Extrapolate to H100s for a 10B model
        param_ratio = target_params / params_1060ti
        tokens_per_sec_h100 = tokens_per_sec_1060ti * (h100_multiplier / param_ratio)
        
        D_h100 = tokens_per_sec_h100 * t
        L_h100 = chinchilla_optimal_loss(target_params, D_h100)
        
        print(f"Time: {t//60} mins")
        print(f"  [1060Ti - {params_1060ti/1e6:.0f}M]")
        print(f"    Tokens Seen: {D_1060/1e6:,.1f} M")
        print(f"    Expected Loss: {L_1060:.3f} | BPB => {L_1060 / np.log(2):.3f}")
        print(f"  [8xH100 - {target_params/1e9:.0f}B]")
        print(f"    Tokens Seen: {D_h100/1e9:,.2f} B")
        print(f"    Expected Loss: {L_h100:.3f} | BPB => {L_h100 / np.log(2):.3f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", type=float, default=300e6, help="Parameters of the benchmarked model")
    parser.add_argument("--tok_sec", type=float, default=25000, help="Tokens/sec measured on 1060ti")
    args = parser.parse_args()
    project(args.params, args.tok_sec)
