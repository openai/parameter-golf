import torch
import triton_kernels
from triton_kernels import triton_engram_hash_gather

def test_engram_dispatch():
    print("Testing Engram Dispatch Logic...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, T = 2, 8
    num_buckets = 1024
    head_dim = 16
    
    # Primes for 4 tables
    primes = torch.tensor([92821, 131071, 174763, 216091], dtype=torch.long, device=device)
    ids = torch.randint(0, 1000, (B, T), device=device)
    
    # Mock tables
    class MockTable:
        def __init__(self):
            self.weight = torch.randn(num_buckets, head_dim, device=device)
            
    # Case 1: 2x2 (standard)
    print("Testing 2x2 dispatch...")
    tables_2x2 = [MockTable() for _ in range(4)]
    out_2x2 = triton_engram_hash_gather(ids, tables_2x2, primes, 2, 2, head_dim, num_buckets)
    assert out_2x2.shape == (B, T, 64)
    print("2x2: SUCCESS")
    
    # Case 2: 1x4 (Bigram only, 4 heads) -> Should use kernel_2x2 via num_total==4
    print("Testing 1x4 dispatch...")
    tables_1x4 = [MockTable() for _ in range(4)]
    out_1x4 = triton_engram_hash_gather(ids, tables_1x4, primes, 1, 4, head_dim, num_buckets)
    assert out_1x4.shape == (B, T, 64)
    print("1x4: SUCCESS")
    
    # Case 3: 4x1 (4 orders, 1 head)
    print("Testing 4x1 dispatch...")
    tables_4x1 = [MockTable() for _ in range(4)]
    out_4x1 = triton_engram_hash_gather(ids, tables_4x1, primes, 4, 1, head_dim, num_buckets)
    assert out_4x1.shape == (B, T, 64)
    print("4x1: SUCCESS")
    
    # Case 4: 3x3 (optimized path)
    print("Testing 3x3 dispatch...")
    primes_9 = torch.tensor([92821, 131071, 174763, 216091, 262147, 314159, 393241, 462841, 524287], dtype=torch.long, device=device)
    tables_3x3 = [MockTable() for _ in range(9)]
    out_3x3 = triton_engram_hash_gather(ids, tables_3x3, primes_9, 3, 3, head_dim, num_buckets)
    assert out_3x3.shape == (B, T, 9 * head_dim)
    print("3x3: SUCCESS")

if __name__ == "__main__":
    test_engram_dispatch()
