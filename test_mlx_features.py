#!/usr/bin/env python3
"""Tests for all features added to train_gpt_mlx.py."""
import math
import os
import pickle
import sys

os.environ["ITERATIONS"] = "0"
os.environ["MLX_EAGER_EVAL"] = "1"

import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from train_gpt_mlx import (
    GPT, Hyperparameters, CastedLinear, Block, MLP, CausalSelfAttention,
    ValueEmbedding, BigramHashEmbedding, SmearGate,
    zeropower_newtonschulz5, _PE_COEFFS,
    quantize_state_dict_int8, dequantize_state_dict_int8,
    gptq_lite_quantize_state_dict, dequantize_gptq_lite_state_dict,
    slot_loss_for_chunk,
    byte_shuffle, byte_unshuffle, compress_best, decompress_auto,
    COMPUTE_DTYPE,
)

PASS = 0
FAIL = 0

def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS: {name}")
    else:
        FAIL += 1
        print(f"  FAIL: {name} {detail}")


def make_model(**kwargs):
    defaults = dict(
        vocab_size=1024, num_layers=11, dim=512, num_heads=8, num_kv_heads=4,
        mlp_mult=3, logit_chunk_tokens=0, logit_softcap=30.0, rope_base=10000.0,
        tied_embed_init_std=0.005, qk_gain_init=1.5, rope_dims=16,
        z_loss_weight=0.0, bigram_hash_size=4096,
    )
    defaults.update(kwargs)
    return GPT(**defaults)


def test_ln_scale():
    print("\n=== LN Scale ===")
    model = make_model(ln_scale=True)
    for i, b in enumerate(model.blocks):
        expected = 1.0 / math.sqrt(i + 1)
        check(f"block {i} scale={b._ln_scale_factor:.4f}", abs(b._ln_scale_factor - expected) < 1e-6)
    model_no = make_model(ln_scale=False)
    check("ln_scale=False all 1.0", all(b._ln_scale_factor == 1.0 for b in model_no.blocks))


def test_xsa_all_layers():
    print("\n=== XSA All Layers ===")
    model = make_model()
    check("all layers have XSA", all(b.attn.use_xsa for b in model.blocks))


def test_mtp():
    print("\n=== MTP (Multi-Token Prediction) ===")
    model = make_model(mtp_num_heads=3, mtp_loss_weight=0.15)
    check("3 MTP heads created", len(model.mtp_heads) == 3)

    x = mx.random.randint(0, 1024, (2, 32))
    loss_train = model.loss(x[:, :-1], x[:, 1:], training=True)
    loss_eval = model.loss(x[:, :-1], x[:, 1:], training=False)
    mx.eval(loss_train, loss_eval)
    check("MTP train loss > eval loss", float(loss_train.item()) > float(loss_eval.item()),
          f"train={loss_train.item():.4f} eval={loss_eval.item():.4f}")

    # MTP heads stripped
    model.mtp_heads = []
    model.mtp_num_heads = 0
    mtp_params = sum(1 for k, _ in tree_flatten(model.parameters()) if "mtp" in k)
    check("MTP heads strippable", mtp_params == 0)


def test_slot():
    print("\n=== SLOT Eval ===")
    model = make_model(mtp_num_heads=0)
    x = mx.random.randint(0, 1024, (1, 64))
    y = mx.random.randint(0, 1024, (1, 64))
    normal_loss = model.loss(x, y, training=False)
    slot_loss = slot_loss_for_chunk(model, x, y, slot_steps=4, slot_lr=0.005)
    mx.eval(normal_loss, slot_loss)
    check("SLOT loss <= normal loss", float(slot_loss.item()) <= float(normal_loss.item()) + 0.01,
          f"normal={normal_loss.item():.4f} slot={slot_loss.item():.4f}")


def test_window_attention():
    print("\n=== Window Attention ===")
    model = make_model(window_size=512, window_attn_layers="2,4,6,8,10")
    windowed = [i for i, b in enumerate(model.blocks) if b.attn.window_size > 0]
    check("windowed layers correct", windowed == [2, 4, 6, 8, 10])
    full = [i for i, b in enumerate(model.blocks) if b.attn.window_size == 0]
    check("full attention layers", full == [0, 1, 3, 5, 7, 9])

    x = mx.random.randint(0, 1024, (1, 128))
    loss = model.loss(x[:, :-1], x[:, 1:], training=False)
    mx.eval(loss)
    check("window attention forward pass", loss.item() > 0 and not math.isnan(loss.item()))


def test_value_embedding_wired():
    print("\n=== ValueEmbedding Wiring ===")
    model = make_model()
    check("ve_deep_0 exists", hasattr(model, "ve_deep_0"))
    check("ve_deep_1 exists", hasattr(model, "ve_deep_1"))
    check("VE layer 0 = num_layers-2", model._ve_layer_0 == 9)
    check("VE layer 1 = num_layers-1", model._ve_layer_1 == 10)

    # Check VE params are in the model
    ve_params = sum(int(p.size) for k, p in tree_flatten(model.parameters()) if "ve_deep" in k)
    check("VE params > 0", ve_params > 0, f"got {ve_params}")

    # Check forward pass works (VE should affect output)
    x = mx.random.randint(0, 1024, (1, 32))
    loss = model.loss(x[:, :-1], x[:, 1:], training=False)
    mx.eval(loss)
    check("VE forward pass OK", not math.isnan(loss.item()))


def test_polar_express_ns():
    print("\n=== Polar Express Newton-Schulz ===")
    check("5 PE coefficients", len(_PE_COEFFS) == 5)

    g = mx.random.normal((512, 1536))
    o4 = zeropower_newtonschulz5(g, 4)
    o5 = zeropower_newtonschulz5(g, 5)
    mx.eval(o4, o5)

    # Check orthogonality: O @ O^T should be close to identity
    ident4 = np.array(o4 @ o4.T)
    off4 = float(np.mean(np.abs(ident4 - np.eye(512))))
    ident5 = np.array(o5 @ o5.T)
    off5 = float(np.mean(np.abs(ident5 - np.eye(512))))
    check(f"4-step ortho error < 0.05 (got {off4:.4f})", off4 < 0.05)
    check(f"5-step ortho error < 0.02 (got {off5:.4f})", off5 < 0.02)
    check("5 steps more precise than 4", off5 < off4)


def test_brotli_compression():
    print("\n=== Brotli + Byte Shuffle ===")
    data = os.urandom(10000)
    shuffled = byte_shuffle(data, element_size=2)
    unshuffled = byte_unshuffle(shuffled, element_size=2)
    check("byte_shuffle roundtrip", data == unshuffled)

    for es in [2, 3]:
        s = byte_shuffle(data, element_size=es)
        u = byte_unshuffle(s, element_size=es)
        check(f"byte_shuffle es={es} roundtrip", data == u)

    blob, method = compress_best(data)
    recovered = decompress_auto(blob, method)
    check(f"compress_best roundtrip (method={method})", recovered == data)


def test_quantization_roundtrip():
    print("\n=== Quantization Roundtrip ===")
    model = make_model(mtp_num_heads=0, ln_scale=True)
    flat_state = {k: v for k, v in tree_flatten(model.state)}

    # Check no _ve_layer keys leak into state
    ve_keys = [k for k in flat_state if "ve_layer" in k]
    check("no _ve_layer in state", len(ve_keys) == 0, f"found: {ve_keys}")

    # Int8 roundtrip
    quant_obj, stats = quantize_state_dict_int8(flat_state)
    quant_flat = dequantize_state_dict_int8(quant_obj)
    model.update(tree_unflatten(list(quant_flat.items())))
    mx.eval(model.state)
    x = mx.random.randint(0, 1024, (1, 32))
    loss = model.loss(x[:, :-1], x[:, 1:], training=False)
    mx.eval(loss)
    check("int8 roundtrip OK", not math.isnan(loss.item()), f"loss={loss.item():.4f}")

    # GPTQ-lite int6 roundtrip
    flat_state2 = {k: v for k, v in tree_flatten(model.state)}
    gptq_obj = gptq_lite_quantize_state_dict(flat_state2)
    gptq_flat = dequantize_gptq_lite_state_dict(gptq_obj)
    model.update(tree_unflatten(list(gptq_flat.items())))
    mx.eval(model.state)
    loss2 = model.loss(x[:, :-1], x[:, 1:], training=False)
    mx.eval(loss2)
    check("int6 GPTQ roundtrip OK", not math.isnan(loss2.item()), f"loss={loss2.item():.4f}")

    # Artifact size check
    raw = pickle.dumps(gptq_obj, protocol=pickle.HIGHEST_PROTOCOL)
    blob, method = compress_best(raw)
    code_bytes = 70000  # approximate
    total = len(blob) + code_bytes
    check(f"artifact under 16MB ({total/1e6:.2f}MB)", total < 16_000_000)


def test_skip_weights_no_sigmoid():
    print("\n=== Skip Weights (no sigmoid) ===")
    model = make_model()
    # Skip weights should be initialized to 1.0 (linear, no sigmoid)
    sw = np.array(model.skip_weights)
    check("skip_weights init ~1.0", np.allclose(sw, 1.0),
          f"mean={sw.mean():.4f}")


def test_model_line_count():
    print("\n=== Line Count ===")
    with open("train_gpt_mlx.py") as f:
        lines = len(f.readlines())
    check(f"under 1500 lines ({lines})", lines <= 1500)


if __name__ == "__main__":
    test_ln_scale()
    test_xsa_all_layers()
    test_mtp()
    test_slot()
    test_window_attention()
    test_value_embedding_wired()
    test_polar_express_ns()
    test_brotli_compression()
    test_quantization_roundtrip()
    test_skip_weights_no_sigmoid()
    test_model_line_count()

    print(f"\n{'='*50}")
    print(f"Results: {PASS} passed, {FAIL} failed out of {PASS+FAIL} tests")
    if FAIL > 0:
        sys.exit(1)
    print("ALL TESTS PASSED")
