// Smoke-test sentencepiece-js against Python sentencepiece on the
// canonical Transformer abstract: do the token IDs match byte-perfect?

import { readFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const sp = await import("sentencepiece-js");
console.log("sentencepiece-js exports:", Object.keys(sp).join(", "));

const SentencePieceProcessor = sp.SentencePieceProcessor || sp.default?.SentencePieceProcessor;
if (!SentencePieceProcessor) {
  console.log("full module:", JSON.stringify(sp, null, 2));
  process.exit(1);
}

const proc = new SentencePieceProcessor();
const modelPath = resolve(ROOT, "..", "data", "tokenizers", "fineweb_1024_bpe.model");
console.log("loading", modelPath);
const r = await proc.load(modelPath);
console.log("load() returned:", r);
console.log("vocab size:", proc.getPieceSize ? proc.getPieceSize() : "?");

const tokens = JSON.parse(await readFile(`${ROOT}/goldens/tokens.json`, "utf-8"));
const text = await readFile(`${ROOT}/goldens/transformer_abstract.txt`, "utf-8");

const ids = proc.encodeIds ? proc.encodeIds(text) : proc.encode(text, "ID");
console.log("first 12 ids (JS):", ids.slice(0, 12));
console.log("first 12 ids (Py):", tokens.token_ids.slice(0, 12));

let mismatch = -1;
for (let i = 0; i < Math.min(ids.length, tokens.token_ids.length); i++) {
  if (ids[i] !== tokens.token_ids[i]) { mismatch = i; break; }
}
if (ids.length === tokens.token_ids.length && mismatch === -1) {
  console.log(`\n✓ Transformer abstract: all ${ids.length} tokens match`);
} else {
  console.log(`\n✗ mismatch at ${mismatch}: JS=${ids[mismatch]}, Py=${tokens.token_ids[mismatch]}`);
  console.log("JS length:", ids.length, "Py length:", tokens.token_ids.length);
}

// --- Additional stress-tests ---
// Compare against Python by running SP via a subprocess for each sample.
import { spawnSync } from "node:child_process";

const PY_HELPER = `
import sys, json, sentencepiece as spm
sp = spm.SentencePieceProcessor(model_file=sys.argv[1])
text = sys.stdin.read()
print(json.dumps(sp.encode_as_ids(text)))
`;
function pyEncode(text) {
  const r = spawnSync(
    resolve(ROOT, "..", ".venv", "bin", "python"),
    ["-c", PY_HELPER, modelPath],
    { input: text, encoding: "utf-8" },
  );
  if (r.status !== 0) throw new Error(r.stderr || `python exit ${r.status}`);
  return JSON.parse(r.stdout);
}

const samples = [
  "Hello, world!",
  "  leading spaces and trailing   ",
  "newlines\nin\nthe\ntext",
  "tab\tcharacters\there",
  "Émoji-ish: ™ © ¶ Ω Δ",
  "1234567890.,;:!?",
  "A very long word: pneumonoultramicroscopicsilicovolcanoconiosis",
  "",  // empty
  " ",  // just space
  "A",  // single char
  "The dominant sequence transduction models",  // Transformer abstract first sentence
  // Decoded fineweb val sample if present
  await readFile(`${ROOT}/goldens/fineweb_val_sample.txt`, "utf-8")
    .then((t) => t.slice(0, 500))
    .catch(() => null),
].filter(Boolean);

let allPass = true;
for (const s of samples) {
  const a = proc.encodeIds(s);
  const b = pyEncode(s);
  const match = a.length === b.length && a.every((v, i) => v === b[i]);
  if (!match) {
    allPass = false;
    console.log(`✗ MISMATCH on sample (${s.length} chars): ${JSON.stringify(s.slice(0, 60))}`);
    console.log(`  JS: ${a.slice(0, 20).join(",")}${a.length > 20 ? "..." : ""}  (len ${a.length})`);
    console.log(`  Py: ${b.slice(0, 20).join(",")}${b.length > 20 ? "..." : ""}  (len ${b.length})`);
    // Find first divergence
    for (let i = 0; i < Math.min(a.length, b.length); i++) {
      if (a[i] !== b[i]) { console.log(`  first diff at ${i}: JS=${a[i]} Py=${b[i]}`); break; }
    }
  } else {
    console.log(`✓ ${a.length}t  "${s.slice(0, 50).replace(/\n/g, "\\n")}${s.length > 50 ? "…" : ""}"`);
  }
}
console.log(allPass ? "\n✓ All stress samples pass" : "\n✗ Some samples failed");
process.exit(allPass ? 0 : 1);
