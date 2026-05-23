// Node-side loader that mirrors src/loader.js for use in the parity
// scripts. Reads `model.json` + `model.safetensors.gz` from the filesystem
// and runs the same parser/dequantizer the browser does.

import { readFile } from "node:fs/promises";
import { gunzipSync } from "node:zlib";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const { parseSafetensors, dequantSafetensors } = await import(`${ROOT}/src/safetensors.js`);

export async function loadModelFromDir(baseDir) {
  const [manifestText, gz] = await Promise.all([
    readFile(`${baseDir}/model.json`, "utf-8"),
    readFile(`${baseDir}/model.safetensors.gz`),
  ]);
  const manifest = JSON.parse(manifestText);
  const stBytes = gunzipSync(gz);
  const parsed = parseSafetensors(stBytes);
  const tensors = dequantSafetensors(parsed);
  return { manifest, tensors };
}
