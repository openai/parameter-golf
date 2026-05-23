import { loadSmolLM2, smollm2BPB } from "./smollm2.js";

const STATUS = document.getElementById("status");
const OUT = document.getElementById("out");

function log(html, cls = "") {
  const div = document.createElement("div");
  div.className = `row ${cls}`;
  div.innerHTML = html;
  OUT.appendChild(div);
}

document.getElementById("load").onclick = async () => {
  STATUS.textContent = "loading…";
  const t0 = performance.now();
  await loadSmolLM2({
    onProgress: (p) => {
      // p.info from Transformers.js is an object {status, name, file, progress, loaded, total}
      const info = p.info;
      if (info && typeof info === "object" && info.progress != null) {
        STATUS.textContent = `${info.file ?? p.status}: ${info.progress.toFixed(1)}% (${info.loaded}/${info.total})`;
      } else {
        STATUS.textContent = `${p.status}: ${typeof info === "object" ? JSON.stringify(info) : info}`;
      }
    },
  });
  STATUS.textContent = `loaded in ${((performance.now() - t0) / 1000).toFixed(1)}s`;
};

async function runOn(text, label) {
  log(`<b>${label}</b>: ${text.length} chars (${new Blob([text]).size} bytes)…`);
  const t0 = performance.now();
  const result = await smollm2BPB(text);
  const dt = ((performance.now() - t0) / 1000).toFixed(2);
  log(`  ${result.n_tokens} tokens, BPB = <b>${result.bpb.toFixed(4)}</b>  (${dt}s)`,
      result.bpb > 0 && result.bpb < 8 ? "ok" : "fail");
  return result;
}

document.getElementById("run-abstract").onclick = async () => {
  const text = await (await fetch("goldens/transformer_abstract.txt")).text();
  await runOn(text, "Transformer abstract");
};
document.getElementById("run-fineweb").onclick = async () => {
  const text = await (await fetch("goldens/fineweb_val_sample.txt")).text();
  await runOn(text, "FineWeb val sample");
};
document.getElementById("run-custom").onclick = async () => {
  const text = document.getElementById("text").value;
  await runOn(text, "custom");
};
