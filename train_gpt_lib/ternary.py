import torch
from torch import Tensor
import lzma
import struct
import math

def ternary_ste(w: Tensor, group_size: int = 128) -> Tensor:
    """
    Implements per-group absmean STE for BitNet b1.58.
    Quantizes weights to {-1, 0, +1} with scaling by groups.
    """
    orig_shape = w.shape
    w_flat = w.view(-1, group_size)
    
    # Per-group scale: mean(abs(w))
    scale = w_flat.abs().mean(dim=-1, keepdim=True)
    
    # Quantize: round(w / scale) clamped to [-1, 1]
    # We use a small epsilon to avoid division by zero
    w_q = (w_flat / (scale + 1e-8)).round().clamp(-1, 1)
    
    # Dequantize for forward pass
    w_dq = (w_q * scale).view(orig_shape)
    
    # Straight-Through Estimator: return w_dq but with gradients of w
    return w + (w_dq - w).detach()

def pack_trits(x: Tensor) -> Tensor:
    """
    Base-3 encoding: 5 trits per byte (3^5 = 243 <= 256).
    Input x should be in {-1, 0, 1}.
    """
    # Shift {-1, 0, 1} to {0, 1, 2}
    x = x.to(torch.int64) + 1
    
    num_elements = x.numel()
    padding = (5 - (num_elements % 5)) % 5
    if padding > 0:
        x = torch.cat([x.flatten(), torch.zeros(padding, dtype=torch.int64, device=x.device)])
    
    x = x.view(-1, 5)
    # Base-3 packing: x0 + 3*x1 + 9*x2 + 27*x3 + 81*x4
    powers = torch.tensor([1, 3, 9, 27, 81], dtype=torch.int64, device=x.device)
    packed = (x * powers).sum(dim=-1).to(torch.uint8)
    return packed

def unpack_trits(packed: Tensor, num_elements: int) -> Tensor:
    """
    Base-3 decoding.
    """
    packed_long = packed.to(torch.int64)
    trits = []
    curr = packed_long
    for i in range(5):
        trits.append(curr % 3)
        curr //= 3
    
    x = torch.stack(trits, dim=-1).view(-1)[:num_elements]
    # Shift {0, 1, 2} back to {-1, 0, 1}
    return x.to(torch.float32) - 1

def serialize_ternary_lzma(state_dict: dict[str, Tensor], fp8_tensors: set[str] | None = None) -> bytes:
    """
    Serializes state_dict using Ternary + LZMA-9 for matrices.
    Small tensors and scales are stored in fp16 or fp8.
    """
    if fp8_tensors is None:
        fp8_tensors = set()
        
    output = io.BytesIO()
    # Header: number of tensors
    output.write(struct.pack("<I", len(state_dict)))
    
    for name, t in state_dict.items():
        name_bytes = name.encode("utf-8")
        output.write(struct.pack("<I", len(name_bytes)))
        output.write(name_bytes)
        
        t_cpu = t.detach().cpu()
        shape = t_cpu.shape
        output.write(struct.pack("<I", len(shape)))
        for s in shape:
            output.write(struct.pack("<I", s))
            
        # Decision: Ternary or FP16/FP8
        # Only 2D matrices with enough elements are ternarized
        if t_cpu.ndim == 2 and t_cpu.numel() > 1024:
            # Ternary path
            output.write(b"\x01") # Type: Ternary
            
            # Per-group scales (group_size=128)
            group_size = 128
            w_flat = t_cpu.view(-1, group_size)
            scales = w_flat.abs().mean(dim=-1)
            
            # Quantize
            w_q = (w_flat / (scales.view(-1, 1) + 1e-8)).round().clamp(-1, 1)
            
            # Pack trits
            packed_trits = pack_trits(w_q.flatten())
            
            # Store scales in fp16 or fp8
            if name in fp8_tensors:
                output.write(b"\x08") # Scale type: FP8 (e4m3)
                # Note: torch.float8_e4m3fn might not be available on all systems, 
                # but we'll assume it is for this setup.
                scales_fp8 = scales.to(torch.float8_e4m3fn)
                output.write(scales_fp8.numpy().tobytes())
            else:
                output.write(b"\x10") # Scale type: FP16
                output.write(scales.to(torch.float16).numpy().tobytes())
                
            # LZMA compress packed trits
            compressed = lzma.compress(packed_trits.numpy().tobytes(), preset=9)
            output.write(struct.pack("<I", len(compressed)))
            output.write(compressed)
        else:
            # FP16 path
            output.write(b"\x00") # Type: FP16
            output.write(t_cpu.to(torch.float16).numpy().tobytes())
            
    return output.getvalue()

def deserialize_ternary_lzma(blob: bytes) -> dict[str, Tensor]:
    """
    Deserializes state_dict.
    """
    input_stream = io.BytesIO(blob)
    num_tensors = struct.unpack("<I", input_stream.read(4))[0]
    state_dict = {}
    
    for _ in range(num_tensors):
        name_len = struct.unpack("<I", input_stream.read(4))[0]
        name = input_stream.read(name_len).decode("utf-8")
        
        num_dims = struct.unpack("<I", input_stream.read(4))[0]
        shape = []
        for _ in range(num_dims):
            shape.append(struct.unpack("<I", input_stream.read(4))[0])
        numel = 1
        for s in shape:
            numel *= s
            
        tensor_type = input_stream.read(1)
        if tensor_type == b"\x01": # Ternary
            scale_type = input_stream.read(1)
            group_size = 128
            num_groups = numel // group_size
            
            if scale_type == b"\x08": # FP8
                scales_raw = input_stream.read(num_groups)
                scales = torch.from_numpy(np.frombuffer(scales_raw, dtype=np.uint8)).to(torch.float8_e4m3fn).to(torch.float32)
            else: # FP16
                scales_raw = input_stream.read(num_groups * 2)
                scales = torch.from_numpy(np.frombuffer(scales_raw, dtype=np.float16)).to(torch.float32)
                
            compressed_len = struct.unpack("<I", input_stream.read(4))[0]
            compressed = input_stream.read(compressed_len)
            packed_trits_raw = lzma.decompress(compressed)
            
            # Calculate expected packed length: ceil(numel / 5)
            packed_len = (numel + 4) // 5
            packed_trits = torch.from_numpy(np.frombuffer(packed_trits_raw, dtype=np.uint8))
            
            trits = unpack_trits(packed_trits, numel).view(-1, group_size)
            w = (trits * scales.view(-1, 1)).view(shape)
            state_dict[name] = w
        else: # FP16
            data_raw = input_stream.read(numel * 2)
            state_dict[name] = torch.from_numpy(np.frombuffer(data_raw, dtype=np.float16)).to(torch.float32).view(shape)
            
    return state_dict

import io
import numpy as np
