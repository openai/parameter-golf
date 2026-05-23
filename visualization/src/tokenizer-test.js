import { loadTokenizer, encode } from "./tokenizer.js";

const OUT = document.getElementById("out");
function log(html, cls = "") {
  const div = document.createElement("div");
  div.className = `row ${cls}`;
  div.innerHTML = html;
  OUT.appendChild(div);
}

document.getElementById("run").onclick = async () => {
  OUT.innerHTML = "";
  log("loading tokenizer…");
  const t0 = performance.now();
  await loadTokenizer("tokenizer/fineweb_1024_bpe.model");
  log(`loaded in ${(performance.now() - t0).toFixed(0)}ms`, "ok");

  // Compare against goldens: tokens.json was created by Python sentencepiece.
  const tokens = await (await fetch("goldens/tokens.json")).json();
  const text = await (await fetch("goldens/transformer_abstract.txt")).text();
  const ids = encode(text);

  log(`text bytes: ${new Blob([text]).size}, expected tokens: ${tokens.token_ids.length}, got: ${ids.length}`);

  let mismatch = -1;
  for (let i = 0; i < Math.min(ids.length, tokens.token_ids.length); i++) {
    if (ids[i] !== tokens.token_ids[i]) { mismatch = i; break; }
  }
  if (ids.length === tokens.token_ids.length && mismatch === -1) {
    log(`✓ all ${ids.length} tokens match Python sentencepiece byte-perfect`, "ok");
  } else {
    log(`✗ mismatch at ${mismatch}: js=${ids[mismatch]}, py=${tokens.token_ids[mismatch]}`, "fail");
  }

  // Also exercise some live samples
  for (const s of [
    "Hello, world!",
    "Émoji ™ Δ",
    "  weird   spacing  ",
    "newlines\nhere\n",
    text.slice(0, 200),
  ]) {
    const ids = encode(s);
    log(`encode(${JSON.stringify(s.slice(0, 40))}…) → ${ids.length} tokens: [${ids.slice(0, 12).join(",")}${ids.length > 12 ? ",…" : ""}]`);
  }
};
