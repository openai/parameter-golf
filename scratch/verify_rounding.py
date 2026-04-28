import torch
import numpy as np

def aligned_round(z):
    return (torch.sign(z) * torch.trunc(z.abs() + 0.5)).clamp(-1, 1)

def test_rounding():
    test_values = torch.tensor([-1.5, -1.0, -0.6, -0.5, -0.4, 0.0, 0.4, 0.5, 0.6, 1.0, 1.5])
    
    # Expected behavior matching Triton's (abs_z + 0.5).to(tl.int32)
    # -1.5 -> (1.5+0.5) -> 2 -> -1 (clamped)
    # -0.5 -> (0.5+0.5) -> 1 -> -1
    # 0.5  -> (0.5+0.5) -> 1 -> 1
    # 0.4  -> (0.4+0.5) -> 0.9 -> 0
    
    results = aligned_round(test_values)
    expected = torch.tensor([-1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    
    print(f"Values:   {test_values}")
    print(f"Applied:  {results}")
    print(f"Expected: {expected}")
    
    assert torch.equal(results, expected), "Rounding mismatch!"
    print("Rounding Alignment: PASS")

if __name__ == "__main__":
    test_rounding()
