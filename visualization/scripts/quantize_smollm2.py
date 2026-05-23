"""Adapted from lsb/sidechat's model_work/quantize.py for the SmolLM2 family.

Pipeline:

  fp32 → bake Transpose(embed) into initializer
       → MatMulNBits weight-only quantize (block_size=64, symmetric,
         accuracy_level=4, QOperator)
       → rewrite the GatherBlockQuantized int4 weight as uint4 with
         zero_point=8 so its packed bytes match MatMulNBits.B exactly
       → save with external data, both initializers pointing at the
         same byte range (the embedding is on disk once)

We skip the fp32→fp16 conversion that the 360M variant uses: SmolLM2's
layernorm has explicit `Cast(x, fp32)` reductions, and the public
`onnxconverter_common.float16.convert_float_to_float16` mishandles those
(leaving mixed precision around the Cast that ORT then refuses). The
output is therefore "q4" — int4 weights, fp32 activations — not "q4f16".
For transformers.js this means `dtype: "q4"` rather than `"q4f16"`. The
weights are still 4-bit; "no doubles" means fp64 is absent, and it is.

Outputs `model_q4.onnx` + `model_q4.onnx_data{,_1,_2,_3}` in --out-dir,
splitting external data across exactly 4 chunk files (per the sidechat
convention; transformers.js can stream them).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto
from onnxconverter_common import float16
from onnxruntime.quantization import matmul_nbits_quantizer, quant_utils


# Skip onnxconverter_common's topological-sort step — it OOMs on big graphs
# and isn't needed; ORT does its own at load.
float16.sort_topology = lambda g: None


def bake_transpose_into_initializer(model):
    """Constant-fold any Transpose(initializer) into a new initializer with
    the transpose baked in. Removes the Transpose node. Required so that
    MatMulNBits can quantize the LM-head weight that's only ever produced
    transposed in the original graph."""
    inits = {i.name: i for i in model.graph.initializer}
    new_inits, to_remove = [], []
    for node in list(model.graph.node):
        if node.op_type != "Transpose" or len(node.input) != 1 or node.input[0] not in inits:
            continue
        arr = onnx.numpy_helper.to_array(inits[node.input[0]])
        perm = next((a.ints for a in node.attribute if a.name == "perm"),
                    list(reversed(range(arr.ndim))))
        new_inits.append(onnx.numpy_helper.from_array(
            np.transpose(arr, list(perm)).copy(), name=node.output[0]))
        to_remove.append(node)
    for n in to_remove:
        model.graph.node.remove(n)
    model.graph.initializer.extend(new_inits)
    return len(to_remove)


def make_uint4_tensor(name, shape, packed_bytes):
    """ONNX helper doesn't directly support uint4 from numpy, so build the
    TensorProto by hand."""
    t = TensorProto()
    t.name = name
    t.data_type = TensorProto.UINT4
    t.dims.extend(shape)
    t.raw_data = bytes(packed_bytes)
    return t


def summarize(path):
    m = onnx.load(str(path), load_external_data=False)
    return f"file={os.path.getsize(path) / 1e6:.1f}MB, nodes={len(m.graph.node)}, init={len(m.graph.initializer)}"


class ChunkWriter:
    """Writes blobs into a sequence of chunk files of bounded size. Each
    chunk's filename is `<base>` for index 0, `<base>_<i>` for i > 0
    (transformers.js / ORT-Web's external-data convention)."""

    def __init__(self, out_dir: Path, base_name: str, cap: int):
        self.out_dir = out_dir
        self.base_name = base_name
        self.cap = cap
        self.chunks: list = []  # [name, file_handle, size_so_far]

    def _open_new(self):
        idx = len(self.chunks)
        name = self.base_name if idx == 0 else f"{self.base_name}_{idx}"
        path = self.out_dir / name
        self.chunks.append([name, open(path, "wb"), 0])

    def _ensure_room(self, need):
        assert need <= self.cap, f"single blob ({need} B) exceeds chunk cap ({self.cap} B)"
        if not self.chunks or self.chunks[-1][2] + need > self.cap:
            self._open_new()
        return self.chunks[-1]

    def write(self, data):
        entry = self._ensure_room(len(data))
        name = entry[0]
        offset = entry[2]
        entry[1].write(data)
        entry[2] += len(data)
        return name, offset, len(data)

    def close(self):
        sizes = []
        for name, fh, size in self.chunks:
            fh.close()
            sizes.append((name, size))
        return sizes


def _find_embed_and_lm_head_weight_names(model_fp16, vocab: int, hidden: int):
    """Find the initializer names for (embedding, lm-head) weights.

    The 360M graph from the sidechat repo has a Transpose(embed) node before
    the LM head's MatMul, so `bake_transpose_into_initializer` produces a
    friendly `model.embed_tokens.weight_transposed` initializer. The 135M
    graph from optimum's exporter doesn't — weight tying is materialised as
    a *separate* transposed initializer (`onnx::MatMul_<num>` of shape
    [hidden, vocab]). Find both by walking the graph rather than by name.
    """
    init_by_name = {i.name: i for i in model_fp16.graph.initializer}
    # Embedding: a Gather node whose first input is an [vocab, hidden] init.
    embed_w = None
    for n in model_fp16.graph.node:
        if n.op_type != "Gather" or len(n.input) < 2:
            continue
        cand = init_by_name.get(n.input[0])
        if cand is not None and list(cand.dims) == [vocab, hidden]:
            embed_w = cand.name
            break
    if embed_w is None:
        raise RuntimeError(f"no Gather with [{vocab}, {hidden}] initializer found")

    # LM head: a MatMul whose output is `logits` (or feeds the model output)
    # and whose right-hand input is an [hidden, vocab] init.
    output_names = {o.name for o in model_fp16.graph.output}
    lm_head_w = None
    for n in model_fp16.graph.node:
        if n.op_type != "MatMul" or len(n.input) < 2 or len(n.output) < 1:
            continue
        if n.output[0] not in output_names:
            continue
        cand = init_by_name.get(n.input[1])
        if cand is not None and list(cand.dims) == [hidden, vocab]:
            lm_head_w = cand.name
            break
    if lm_head_w is None:
        raise RuntimeError(
            f"no MatMul producing graph output with [{hidden}, {vocab}] initializer found"
        )
    return embed_w, lm_head_w


def quantize(*, src: Path, out_dir: Path, vocab: int, hidden: int,
             block_size: int = 64, target_n_chunks: int = 4,
             chunk_hard_cap: int = 50_000_000):
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / "model_q4.onnx"
    base_name = "model_q4.onnx_data"

    n_blocks = hidden // block_size
    if hidden % block_size != 0:
        raise ValueError(f"hidden={hidden} not divisible by block_size={block_size}")

    print(f"loading {src} (≈{os.path.getsize(src) / 1e9:.2f} GB)…")
    model_fp32 = onnx.load(str(src))

    print("bake Transpose (if any) → quantize…")
    bake_transpose_into_initializer(model_fp32)

    # Find the names BEFORE quantization (the quantizer will append `_Q4` /
    # `_scales` suffixes to whatever it picks up).
    embed_name, lm_head_name = _find_embed_and_lm_head_weight_names(
        model_fp32, vocab=vocab, hidden=hidden,
    )
    print(f"  embedding initializer: {embed_name!r}  shape=[{vocab}, {hidden}]")
    print(f"  lm-head  initializer:  {lm_head_name!r}  shape=[{hidden}, {vocab}]")

    quantizer = matmul_nbits_quantizer.MatMulNBitsQuantizer(
        model_fp32,
        algo_config=matmul_nbits_quantizer.DefaultWeightOnlyQuantConfig(
            block_size=block_size, is_symmetric=True, accuracy_level=4,
            quant_format=quant_utils.QuantFormat.QOperator,
            op_types_to_quantize=("MatMul", "Gather"),
        ),
    )
    quantizer.process()
    model = quantizer.model.model
    # MatMulNBitsQuantizer in ORT 1.25 bumps the default-domain opset to 21
    # in the emitted graph, but the rest of the body is opset-14 — including
    # `ReduceMean` whose `axes` attribute moved to a runtime input at opset
    # 18. The mismatch makes ORT refuse the graph at load. We can't easily
    # migrate ReduceMean (onnx.version_converter doesn't), so we lower the
    # default opset back to 17 — the latest version that still accepts
    # `axes` as an attribute. The com.microsoft domain (where MatMulNBits
    # and GatherBlockQuantized live) is independent of the default opset.
    SAFE_OPSET = 17
    for o in model.opset_import:
        if o.domain == "" and o.version > SAFE_OPSET:
            o.version = SAFE_OPSET

    inits = {i.name: i for i in model.graph.initializer}
    GATHER_W = f"{embed_name}_Q4"           # int4  [V, H]    — embedding
    GATHER_S = f"{embed_name}_scales"       # fp16  [V, n_blocks]
    MNB_W    = f"{lm_head_name}_Q4"         # uint8 [V, n_blocks, blob_size] — lm head
    MNB_S    = f"{lm_head_name}_scales"

    g_int4 = onnx.numpy_helper.to_array(inits[GATHER_W]).astype(np.int32)  # [V, H] signed
    g_scales = onnx.numpy_helper.to_array(inits[GATHER_S])
    m_scales = onnx.numpy_helper.to_array(inits[MNB_S])
    assert np.array_equal(g_scales, m_scales), "scales differ; dedupe unsafe"
    u4 = ((g_int4 + 8) % 16).astype(np.uint8)
    packed = (u4[:, 0::2] | (u4[:, 1::2] << 4)).astype(np.uint8)
    mnb_bytes = (bytes(inits[MNB_W].raw_data)
                 if inits[MNB_W].raw_data
                 else onnx.numpy_helper.to_array(inits[MNB_W]).tobytes())
    mnb_flat = np.frombuffer(mnb_bytes, dtype=np.uint8).reshape(vocab, hidden // 2)
    agree = (packed == mnb_flat).mean()
    print(f"gather(+8) vs matmulnbits packed bytes agree on {agree * 100:.2f}% of "
          f"{vocab * hidden // 2 / 1e6:.1f}M bytes "
          f"(remainder differs by ≤ one int4 step, FP rounding in independent codepaths)")
    shared_bytes = mnb_bytes  # use the MatMulNBits bytes as the canonical shared copy

    # Build a uint4 zero_point initializer for GBQ: shape [V, n_blocks], every
    # value 8 (symmetric uint4). ONNX packs uint4 two-per-byte in flat element
    # order; for odd total counts the trailing byte's high nibble is unused.
    total_zp = vocab * n_blocks
    zp_flat = np.full(total_zp, 8, dtype=np.uint8)
    zp_packed = np.zeros((total_zp + 1) // 2, dtype=np.uint8)
    zp_packed[:total_zp // 2] = (
        zp_flat[0:total_zp - (total_zp % 2):2] | (zp_flat[1::2] << 4)
    )
    if total_zp % 2 == 1:
        zp_packed[-1] = zp_flat[-1]
    zp_init = make_uint4_tensor(
        f"{embed_name}_zero_point",
        [vocab, n_blocks], zp_packed.tobytes(),
    )

    new_gather_weight = make_uint4_tensor(GATHER_W, [vocab, hidden], shared_bytes)

    # Drop old Gather int4 + old MNB uint8; add new uint4 + zero_point.
    drop_names = {GATHER_W, MNB_W}
    new_inits = [init for init in model.graph.initializer if init.name not in drop_names]
    new_inits.append(new_gather_weight)
    new_inits.append(zp_init)

    # Recreate MNB.B as uint8 [V, n_blocks, blob_size] referencing the same bytes.
    mnb_tensor = TensorProto()
    mnb_tensor.name = MNB_W
    mnb_tensor.data_type = TensorProto.UINT8
    mnb_tensor.dims.extend([vocab, n_blocks, block_size // 2])
    mnb_tensor.raw_data = shared_bytes
    new_inits.append(mnb_tensor)

    del model.graph.initializer[:]
    model.graph.initializer.extend(new_inits)

    # Wire GatherBlockQuantized's zero_point input.
    for node in model.graph.node:
        if node.op_type == "GatherBlockQuantized":
            while len(node.input) < 4:
                node.input.append("")
            node.input[3] = zp_init.name
            break

    print("rewrote Gather's int4 weight → uint4 (values +=8 mod 16, zero_point=8); "
          "MatMulNBits weight unchanged")

    # External data writeout. Both shared-bytes initializers (GATHER_W and
    # MNB_W) reference the same (location, offset, length) — embedding stored
    # exactly once on disk.
    for p in out_dir.glob(base_name + "*"):
        p.unlink()

    EXTERNAL_THRESHOLD = 1024
    total_external = len(shared_bytes)
    for init in model.graph.initializer:
        if init.name in (GATHER_W, MNB_W):
            continue
        if init.raw_data and len(init.raw_data) >= EXTERNAL_THRESHOLD:
            total_external += len(init.raw_data)

    ideal_per_chunk = (total_external + target_n_chunks - 1) // target_n_chunks
    chunk_cap = int(ideal_per_chunk * 1.02)  # 2% slack so greedy packing fits
    assert chunk_cap <= chunk_hard_cap, (
        f"total external {total_external/1e6:.1f} MB too large for "
        f"{target_n_chunks} chunks ≤ {chunk_hard_cap/1e6:.0f} MB each"
    )
    print(f"external data total {total_external / 1e6:.2f} MB → "
          f"chunk cap {chunk_cap / 1e6:.2f} MB ({target_n_chunks} chunks target)")

    writer = ChunkWriter(out_dir, base_name, chunk_cap)
    shared_loc, shared_off, shared_len = writer.write(shared_bytes)

    for init in model.graph.initializer:
        if init.name in (GATHER_W, MNB_W):
            init.ClearField("raw_data")
            init.data_location = TensorProto.EXTERNAL
            del init.external_data[:]
            for k, v in [("location", shared_loc),
                         ("offset", str(shared_off)),
                         ("length", str(shared_len))]:
                e = init.external_data.add()
                e.key = k
                e.value = v
            continue
        if init.raw_data and len(init.raw_data) >= EXTERNAL_THRESHOLD:
            b = init.raw_data
            init.ClearField("raw_data")
            init.data_location = TensorProto.EXTERNAL
            del init.external_data[:]
            loc, off, ln = writer.write(b)
            for k, v in [("location", loc), ("offset", str(off)), ("length", str(ln))]:
                e = init.external_data.add()
                e.key = k
                e.value = v

    chunk_sizes = writer.close()
    onnx.save(model, str(dst))
    print(f"done — {summarize(dst)}")
    print(f"external data split across {len(chunk_sizes)} files:")
    for name, size in chunk_sizes:
        print(f"  {name}: {size / 1e6:.2f} MB")
    print(f"shared embedding blob ({shared_len / 1e6:.2f} MB) in {shared_loc}, "
          f"referenced by both {GATHER_W} and {MNB_W}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="fp32 model.onnx (from optimum-cli export)")
    p.add_argument("--out-dir", required=True, help="output directory")
    p.add_argument("--vocab", type=int, default=49152, help="vocab size (49152 for SmolLM2 family)")
    p.add_argument("--hidden", type=int, required=True,
                   help="hidden dim (576 for SmolLM2-135M, 960 for SmolLM2-360M)")
    p.add_argument("--block-size", type=int, default=64)
    args = p.parse_args()
    quantize(src=Path(args.src), out_dir=Path(args.out_dir),
             vocab=args.vocab, hidden=args.hidden, block_size=args.block_size)


if __name__ == "__main__":
    main()
