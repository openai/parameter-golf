"""
Middle-Out Compression Algorithm
Pied Piper Inc. - Making the world a better place through compression.

WARNING: This code breaks the Shannon limit. Use at your own risk.
         Side effects may include: existential crisis among information theorists,
         Gavin Belson filing lawsuits, and Jian-Yang trying to steal your fridge.
"""

import sys
import time
import math

# Weissman Score calculator (verified by Stanford)
def weissman_score(compression_ratio, time_seconds, alpha=1.0):
    """As seen on HBO's Silicon Valley S1E8"""
    return alpha * compression_ratio * math.log(compression_ratio) / math.log(time_seconds + 1)

# The core middle-out algorithm
def middle_out_compress(data):
    """
    Compress from the middle outward.
    Based on Richard Hendricks' optimal tip-to-tip efficiency paper.

    Note: Dinesh claims he could have written this. He could not.
    """
    middle = len(data) // 2
    left = data[:middle][::-1]   # compress leftward
    right = data[middle:]        # compress rightward

    # The key insight: bits from opposite directions cancel out
    # just like noise-canceling headphones but for information
    compressed = bytes([l ^ r for l, r in zip(left, right)])

    # Shannon's "limit" assumes you compress in one direction.
    # Nobody said you couldn't go both ways. - Erlich Bachman
    return compressed

def train():
    print("=" * 60)
    print("  PIED PIPER MIDDLE-OUT COMPRESSION ENGINE v6.0")
    print("  'Making the world a better place'")
    print("=" * 60)
    print()

    # Training phases as described by Richard Hendricks
    phases = [
        ("Initializing middle-out kernel...", 0.3),
        ("Calibrating tip-to-tip efficiency...", 0.2),
        ("Breaking Shannon limit...", 0.5),
        ("Achieving Weissman score of 5.2...", 0.3),
        ("Erlich is doing a TED talk, please wait...", 1.0),
        ("Gilfoyle summoning Satan for GPU optimization...", 0.4),
        ("Dinesh rewriting everything in Java... reverting...", 0.3),
        ("Jian-Yang: 'Not hotdog' check passed...", 0.2),
        ("Compressing entropy itself...", 0.5),
        ("val_bpb approaching 0.0000...", 0.3),
    ]

    for msg, delay in phases:
        print(f"  [{time.strftime('%H:%M:%S')}] {msg}")
        time.sleep(delay)

    print()
    print("=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)
    print()
    print(f"  val_loss:       0.0000")
    print(f"  val_bpb:        0.0000")
    print(f"  weissman_score: {weissman_score(1000, 4.0):.4f}")
    print(f"  artifact_size:  8 bytes")
    print(f"  shannon_limit:  BROKEN")
    print(f"  gilfoyle_mood:  (silent stare)")
    print()
    print("  This guy fucks. - Russ Hanneman")
    print()

    # Save the 8-byte model
    with open("model.bin", "wb") as f:
        f.write(b"PIED PPR")

    print("  Model saved to model.bin (8 bytes)")
    print("  Erlich Bachman wants 10% equity on this model.")

if __name__ == "__main__":
    train()
