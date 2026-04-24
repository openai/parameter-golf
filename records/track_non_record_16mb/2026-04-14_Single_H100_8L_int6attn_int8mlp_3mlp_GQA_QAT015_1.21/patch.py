import os

with open("train_gpt.py", "r") as f:
    content = f.read()

# Add UNQUANTIZED_NAME_PATTERNS
int8_pattern = """INT8_QAT_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "INT8_QAT_NAME_PATTERNS",
        "attn.c_q.weight,attn.c_k.weight,attn.c_v.weight,attn.proj.weight,mlp.fc.weight,mlp.proj.weight",
    ).split(",")
    if pattern
)"""
unquant_pattern = """
UNQUANTIZED_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "UNQUANTIZED_NAME_PATTERNS",
        "blocks.0.attn.c_v.weight,blocks.1.attn.c_v.weight,blocks.2.attn.c_v.weight,blocks.6.attn.c_v.weight,blocks.7.attn.c_v.weight",
    ).split(",")
    if pattern
)"""
if "UNQUANTIZED_NAME_PATTERNS =" not in content:
    content = content.replace(int8_pattern, int8_pattern + unquant_pattern)

# Update QAT bits logic
qat_logic_old = """            if any(pattern in param_name for pattern in INT6_NAME_PATTERNS):
                module._qat_bits = 6
            elif any(pattern in param_name for pattern in INT8_QAT_NAME_PATTERNS):
                module._qat_bits = 8"""
qat_logic_new = """            if any(pattern in param_name for pattern in UNQUANTIZED_NAME_PATTERNS):
                module._qat_bits = 0
            elif any(pattern in param_name for pattern in INT6_NAME_PATTERNS):
                module._qat_bits = 6
            elif any(pattern in param_name for pattern in INT8_QAT_NAME_PATTERNS):
                module._qat_bits = 8"""
if "UNQUANTIZED_NAME_PATTERNS):" not in content:
    content = content.replace(qat_logic_old, qat_logic_new)

# Update quantize_state_dict_int8
quantize_logic_old = """        if not t.is_floating_point():
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = t
            stats["int8_payload_bytes"] += tensor_nbytes(t)
            continue"""
quantize_logic_new = """        if not t.is_floating_point() or any(p in name for p in UNQUANTIZED_NAME_PATTERNS) or any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS):
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = t
            stats["int8_payload_bytes"] += tensor_nbytes(t)
            continue"""
if "any(p in name for p in UNQUANTIZED_NAME_PATTERNS)" not in content:
    content = content.replace(quantize_logic_old, quantize_logic_new)

with open("train_gpt.py", "w") as f:
    f.write(content)
