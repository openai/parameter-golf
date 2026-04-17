from __future__ import annotations
import triton_kernels

def main():
    print("Test DCE")
    # Use something from triton_kernels
    x = triton_kernels.HAS_TRITON
    print(f"HAS_TRITON: {x}")

if __name__ == "__main__":
    main()
