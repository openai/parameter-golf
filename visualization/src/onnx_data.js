// Reconstruct model.onnx.data from a dequantized safetensors state-dict
// + onnx_data_manifest.json (see scripts/build_onnx_data_manifest.py).
//
// This is the JS counterpart of scripts/reconstruct_onnx_data.py — same
// algorithm, byte-identical output. node_parity_safetensors.js asserts the
// SHA-256 matches the Python reference and the committed model.onnx.data.
//
// Usage:
//   const buf = reconstructOnnxData(manifest, dequantTensors);
//   await ort.InferenceSession.create(onnxGraphBytes, {
//     externalData: [{ path: "model.onnx.data", data: buf }],
//     executionProviders: [...],
//   });

/** Build a fresh Uint8Array of `manifest.total_bytes` populated according
 *  to the manifest's layout. `tensors` is the {base → {data, shape}} map
 *  produced by dequantSafetensors().
 */
export function reconstructOnnxData(manifest, tensors) {
  const buf = new Uint8Array(manifest.total_bytes);

  for (const e of manifest.layout) {
    if (e.kind === "zeros") continue;  // buf is zero-initialized by spec
    if (e.kind === "gap") {
      const blob = base64ToBytes(e.data_b64);
      if (blob.byteLength !== e.length) {
        throw new Error(`gap length mismatch at ${e.offset}: ${blob.byteLength} vs ${e.length}`);
      }
      buf.set(blob, e.offset);
      continue;
    }
    if (e.kind !== "tensor") {
      throw new Error(`unknown layout kind: ${e.kind}`);
    }

    if (e.source === "constant") {
      const blob = base64ToBytes(e.data_b64);
      if (blob.byteLength !== e.length) {
        throw new Error(`constant length mismatch for ${e.name}`);
      }
      buf.set(blob, e.offset);
      continue;
    }
    if (e.source !== "weight") {
      throw new Error(`unknown source: ${e.source} on ${e.name}`);
    }

    const wrapped = tensors[e.safetensors_base];
    if (!wrapped) throw new Error(`safetensors_base ${e.safetensors_base} missing`);
    const arr = wrapped.data;
    let viewBytes;
    if (e.transform === "identity") {
      viewBytes = new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength);
    } else if (e.transform === "transpose") {
      // 2-D transpose. The manifest's `dims` is the post-transpose ONNX shape;
      // the safetensors source has shape (inDim, outDim).
      const [outDim, inDim] = e.dims;
      if (outDim * inDim !== arr.length) {
        throw new Error(
          `transpose for ${e.name}: dims ${e.dims} require ${outDim * inDim} elems, got ${arr.length}`
        );
      }
      const transposed = new Float32Array(arr.length);
      for (let i = 0; i < inDim; i++) {
        for (let o = 0; o < outDim; o++) {
          transposed[o * inDim + i] = arr[i * outDim + o];
        }
      }
      viewBytes = new Uint8Array(transposed.buffer);
    } else {
      throw new Error(`unknown transform: ${e.transform} on ${e.name}`);
    }

    if (viewBytes.byteLength !== e.length) {
      throw new Error(
        `length mismatch for ${e.name}: blob=${viewBytes.byteLength} expected=${e.length}`
      );
    }
    buf.set(viewBytes, e.offset);
  }

  return buf;
}

function base64ToBytes(b64) {
  // Browser path: atob() returns latin-1 string; convert to Uint8Array.
  // Node 22 also supports atob globally.
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}
