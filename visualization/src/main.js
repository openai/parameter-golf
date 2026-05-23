// Main visualization entry. Loads model weights once, then on every "Compute"
// click runs the enabled methods on the current text and renders per-byte bars.

// IMPORTANT: keep ort_setup.js as the very first import. It pins
// ONNX_ENV.wasm.wasmPaths to the jsep wasm. transformers.js has module-level
// code that defaults wasmPaths to the asyncify CDN URL the moment its module
// is evaluated; if any other import drags in transformers.js before
// ort_setup.js runs, we lose. See PROGRESS.md "WASM EP and GBQ".
import "./ort_setup.js";

import { loadModel } from "./loader.js";
import { gzipBPB, arBPB, mdlmBPB, smollm2BPBWrap } from "./methods.js";
import { loadTokenizer, encode, pieceForId, getProcessor } from "./tokenizer.js";
import { gzipGenerate, arGenerate, mdlmGenerate, smollm2Generate } from "./generate.js";
import { smollm2Piece } from "./smollm2.js";

const STATUS = document.getElementById("status");
const SUMMARY = document.getElementById("summary");
const VIZ = document.getElementById("viz");

let arWeights = null, arArch = null;
let mdlmWeights = null, mdlmArch = null;

function status(msg, kind = "") {
  STATUS.textContent = msg;
  STATUS.className = `status ${kind}`;
}

async function ensureLoaded() {
  if (arWeights && mdlmWeights) return;
  status("loading models + tokenizer…", "busy");
  const [{ tensors: arT, manifest: arM },
         { tensors: mT, manifest: mM }] = await Promise.all([
    loadModel("models/ar"),
    loadModel("models/mdlm"),
    loadTokenizer("tokenizer/fineweb_1024_bpe.model"),
  ]);
  arWeights = arT; arArch = { ...arM.arch, label: arM.label };
  mdlmWeights = mT; mdlmArch = { ...mM.arch, label: mM.label };
  status("ready");
}

// Map a per-byte BPB value into a [0,1] opacity given the current display range.
function opacityFor(bpb, lo, hi) {
  const range = (hi - lo) || 0.001;
  return Math.max(0, Math.min(1, (bpb - lo) / range)) * 0.94 + 0.06;
}

// _lastResults is a fixed-length array, one slot per method, in canonical
// order: gzip / AR / MDLM / SmolLM2. Slots that aren't enabled this run are
// null. Each enabled slot is { label, color, bpb, n_tokens, per_byte }.
let _lastResults = [null, null, null, null];
// Per-slot note shown in the Progress column ("queued" / "running…" / "0.5s").
// Lives outside _lastResults so the methods' onProgress snapshots can't blow
// it away when they replace the whole result object on every tick.
let _slotNote = [null, null, null, null];
// Per-slot history of {k, K, bpb} samples captured from onProgress, used to
// draw the bpb sparkline in the summary table. Reset at the start of each
// run; appended to as snapshots arrive.
let _bpbHistory = [[], [], [], []];
let _lastText = "";

// In-place render state. Built once per (text, enabled-mask), then patched as
// onProgress callbacks arrive. _vizState.bars is a 2D array indexed by
// [resultIdx][charIdx]. Each bar element carries a `_flashed` boolean so we
// flash exactly once on first-arrival; runAll forces a rebuild per Compute
// click, which gives us fresh bars with `_flashed = undefined`.
let _vizState = { text: null, mask: "", charByteSpans: null, bars: null };

function buildViz(text, results) {
  // Layout: per character, render the glyph + one bar per active result.
  VIZ.innerHTML = "";
  const enc = new TextEncoder();
  const charByteSpans = [];      // [start, end) bytes covered by this char
  const bars = results.map(() => []);  // per result, a bar element per char (or null for newlines)

  let bytePos = 0;
  for (const ch of text) {
    if (ch === "\n") {
      const nl = document.createElement("div");
      nl.className = "nl";
      VIZ.appendChild(nl);
      bytePos += 1;
      // No bars for newlines; push placeholders so indices stay aligned.
      charByteSpans.push([bytePos - 1, bytePos]);
      for (let r = 0; r < results.length; r++) bars[r].push(null);
      continue;
    }
    const charNBytes = enc.encode(ch).length;
    const start = bytePos;
    const end = bytePos + charNBytes;

    const cell = document.createElement("div");
    cell.className = "c";
    const chDiv = document.createElement("div");
    chDiv.className = "ch";
    chDiv.textContent = ch === " " ? " " : ch;
    cell.appendChild(chDiv);

    for (let r = 0; r < results.length; r++) {
      const res = results[r];
      if (!res) { bars[r].push(null); continue; }
      const bar = document.createElement("div");
      bar.className = "b";
      bar.style.background = res.color;
      bar.style.opacity = "0.06";
      // Carry the slot index + char index so the popover can resolve which
      // method+token was hovered without an outer-scope lookup.
      bar.dataset.slot = String(r);
      bar.dataset.ci = String(charByteSpans.length);  // soon-to-be-pushed index
      // bar._flashed defaults to undefined → first-arrival flash fires once.
      cell.appendChild(bar);
      bars[r].push(bar);
    }
    VIZ.appendChild(cell);
    bytePos = end;
    charByteSpans.push([start, end]);
  }
  return { charByteSpans, bars };
}

// Briefly flash a bar: black outline + magenta→yellow fade, both decay to
// transparent over ~2.5 s. Highlights newly-predicted tokens. The animation
// is created once per bar's lifetime and re-played on subsequent flashes —
// far cheaper than re-creating Animation objects on every progress tick.
const FLASH_DURATION_MS = 2500;
const FLASH_KEYFRAMES = [
  { outlineColor: "rgba(0,0,0,1)",   boxShadow: "inset 0 0 0 100px rgba(255,0,255,0.85)" },
  { outlineColor: "rgba(0,0,0,0.5)", boxShadow: "inset 0 0 0 100px rgba(255,255,0,0.55)", offset: 0.45 },
  { outlineColor: "rgba(0,0,0,0)",   boxShadow: "inset 0 0 0 100px rgba(255,255,0,0)" },
];
function flashBar(bar) {
  if (!bar) return;
  let anim = bar._flashAnim;
  if (!anim) {
    anim = bar.animate(FLASH_KEYFRAMES, { duration: FLASH_DURATION_MS, easing: "ease-out", fill: "none" });
    bar._flashAnim = anim;
    return;
  }
  // Re-trigger: rewind and play. Cheaper than creating a new Animation.
  try { anim.currentTime = 0; anim.play(); } catch {}
}

// For each result, compute the set of char indices that mark a token start
// (the first char whose byteStart equals the token's byteStart). Used to
// paint a 1px white border-left on those bars so token boundaries are
// obvious. Cheap: one Map of byteStart→ci, then a per-token lookup.
function _tokenStartSets(results, charByteSpans) {
  const out = results.map(() => null);
  if (!charByteSpans) return out;
  const byteToCi = new Map();
  for (let ci = 0; ci < charByteSpans.length; ci++) {
    const sp = charByteSpans[ci];
    if (sp) byteToCi.set(sp[0], ci);
  }
  for (let r = 0; r < results.length; r++) {
    const res = results[r];
    if (!res || !res.per_token) continue;
    const set = new Set();
    for (const tok of res.per_token) {
      if (!tok) continue;
      const ci = byteToCi.get(tok.byteStart);
      if (ci != null) set.add(ci);
    }
    out[r] = set;
  }
  return out;
}

function patchBars(results) {
  const lo = parseFloat(document.getElementById("rmin").value);
  let hi = parseFloat(document.getElementById("rmax").value);
  if (hi <= lo) hi = lo + 0.1;

  const spans = _vizState.charByteSpans;
  if (!spans) return;
  const FLASH_EPS = 1e-9;
  const tokStarts = _tokenStartSets(results, spans);

  for (let r = 0; r < results.length; r++) {
    const res = results[r];
    if (!res) continue;
    const barRow = _vizState.bars[r];
    const pb = res.per_byte;
    const starts = tokStarts[r];
    for (let ci = 0; ci < spans.length; ci++) {
      const bar = barRow[ci];
      if (!bar) continue;
      const [s, e] = spans[ci];
      let sum = 0;
      const span = Math.min(e, pb.length) - s;
      for (let b = s; b < Math.min(e, pb.length); b++) sum += pb[b];
      const bpb = span > 0 ? sum / span : 0;
      bar.style.opacity = opacityFor(bpb, lo, hi).toFixed(3);
      if (starts) bar.classList.toggle("tok-start", starts.has(ci));
      // First-arrival flash: fires once per bar lifetime, the moment the bar
      // gains a real (nonzero) BPB. Slider drags repaint without flashing
      // because _flashed is already true; MDLM's K-pass refinements don't
      // re-flash for the same reason.
      if (!bar._flashed && bpb > FLASH_EPS) {
        bar._flashed = true;
        flashBar(bar);
      }
    }
  }
}

function render() {
  document.getElementById("vmin").textContent = parseFloat(document.getElementById("rmin").value).toFixed(1);
  document.getElementById("vmax").textContent = parseFloat(document.getElementById("rmax").value).toFixed(1);
  const mask = _lastResults.map((r) => (r ? "1" : "0")).join("");
  if (_vizState.text !== _lastText || _vizState.mask !== mask) {
    const built = buildViz(_lastText, _lastResults);
    _vizState = { text: _lastText, mask, ...built };
  }
  patchBars(_lastResults);
}

// Coalesce render+renderSummary calls. Many onProgress events can arrive
// between frames (gzip yields per byte, MDLM yields per K pass). Rendering
// synchronously each time stalls the main thread; deferring to the next
// animation frame batches them at a maximum of one paint per frame.
let _renderScheduled = false;
function scheduleRender() {
  if (_renderScheduled) return;
  _renderScheduled = true;
  requestAnimationFrame(() => {
    _renderScheduled = false;
    renderSummary();
    render();
  });
}

// Render a small SVG sparkline of bpb-over-progress for one method.
//   x = k/K        (compute fraction)
//   y = log(bpb)   (log-scale; bpb<=0 points are dropped — early MDLM
//                   snapshots can have bpb=0 before the first commit)
// Each point gets an invisible (but pointer-event-active) circle with a
// native <title> child so hovering reveals "k/K kind: bpb".
function _renderSparkline(points, color) {
  const W = 96, H = 18;
  const boxStyle =
    `display:inline-block; vertical-align:middle; width:${W}px; height:${H}px; ` +
    `border:1px solid var(--border); border-radius:3px; margin-right:0.4em; ` +
    `box-sizing:border-box; background:transparent;`;
  if (!points || points.length === 0) {
    return `<span style="${boxStyle}"></span>`;
  }
  // Log scale on bpb. Drop non-positive points (zero / NaN) — log isn't
  // defined there. The dropped points are typically the very first snapshot
  // before any token has been scored.
  const valid = points.filter((p) => p.bpb > 0 && isFinite(p.bpb));
  if (valid.length === 0) {
    return `<span style="${boxStyle}"></span>`;
  }
  let yMin = Infinity, yMax = -Infinity;
  for (const p of valid) {
    const ly = Math.log(p.bpb);
    if (ly < yMin) yMin = ly;
    if (ly > yMax) yMax = ly;
  }
  if (yMax - yMin < 0.05) yMax = yMin + 0.05;
  const padX = 1, padY = 1;
  const innerW = W - 2 * padX, innerH = H - 2 * padY;
  const lastK = valid[valid.length - 1].K || 1;
  const xFor = (k) => padX + (Math.min(1, k / Math.max(1, lastK))) * innerW;
  const yFor = (bpb) =>
    padY + innerH - ((Math.log(bpb) - yMin) / (yMax - yMin)) * innerH;
  let d = "";
  for (let i = 0; i < valid.length; i++) {
    const p = valid[i];
    d += (i === 0 ? "M" : "L") + xFor(p.k).toFixed(1) + "," + yFor(p.bpb).toFixed(1);
  }
  const last = valid[valid.length - 1];
  const lastX = xFor(last.k).toFixed(1), lastY = yFor(last.bpb).toFixed(1);
  // Invisible hit-circles for each point — fill="transparent" alone wouldn't
  // catch hover, so we pin pointer-events="all". r=2 gives ~5px hit area
  // across the dense end of the curve.
  const dots = valid.map((p) => {
    const x = xFor(p.k).toFixed(1);
    const y = yFor(p.bpb).toFixed(1);
    const kindStr = p.kind ? " " + p.kind : "";
    const txt = `${p.k}/${p.K}${kindStr}: ${p.bpb.toFixed(3)} bpb`;
    return `<circle cx="${x}" cy="${y}" r="2" fill="transparent" ` +
           `pointer-events="all"><title>${_escapeHtml(txt)}</title></circle>`;
  }).join("");
  return (
    `<svg width="${W}" height="${H}" style="vertical-align:middle; margin-right:0.4em;">` +
      `<rect x="0.5" y="0.5" width="${W - 1}" height="${H - 1}" fill="none" ` +
        `stroke="var(--border)" stroke-width="1" rx="3"/>` +
      `<path d="${d}" stroke="${color}" stroke-width="1.4" fill="none" ` +
        `stroke-linejoin="round" stroke-linecap="round" pointer-events="none"/>` +
      `<circle cx="${lastX}" cy="${lastY}" r="1.6" fill="${color}" pointer-events="none"/>` +
      dots +
    `</svg>`
  );
}

function renderSummary() {
  // Preserve canonical order — do NOT sort by BPB. User wants the rows to stay
  // in the gzip / AR / MDLM / SmolLM2 order they appear above.
  const visible = [];
  for (let i = 0; i < _lastResults.length; i++) {
    if (_lastResults[i]) {
      visible.push({ slot: i, r: _lastResults[i], note: _slotNote[i] });
    }
  }
  if (visible.length === 0) { SUMMARY.innerHTML = ""; return; }
  let h = '<table>';
  h += '<tr><td><b>Method</b></td><td><b>BPB</b></td><td><b>Tokens</b></td><td><b>Progress</b></td></tr>';
  for (const { slot, r, note } of visible) {
    const parts = [];
    const hist = _bpbHistory[slot];
    const hasHist = hist && hist.length > 0;
    // Show the sparkline whenever history exists — during compute and after.
    // The "k/K kind" progress text only makes sense while r.progress is set
    // (a method's running snapshots), so it sits next to the sparkline only
    // during the run.
    if (hasHist || r.progress) {
      let progText = "";
      if (r.progress) {
        const { k, K, kind } = r.progress;
        progText = `<span style="font-variant-numeric: tabular-nums;">${k}/${K} ${kind}</span>`;
      }
      const spark = hasHist ? _renderSparkline(hist, r.color) : "";
      parts.push(spark + progText);
    }
    if (note) parts.push(`<span style="color:#6b7280; font-family: monospace;">${note}</span>`);
    const progressCell = parts.length > 0 ? parts.join(' &middot; ') : "—";
    h += `<tr><td><span class="legend-swatch" style="background:${r.color}"></span>${r.label}</td>` +
         `<td style="font-variant-numeric: tabular-nums;">${r.bpb.toFixed(4)}</td>` +
         `<td>${r.n_tokens || "—"}</td>` +
         `<td>${progressCell}</td></tr>`;
  }
  h += '</table>';
  SUMMARY.innerHTML = h;
}

function setSlotNote(idx, note) {
  _slotNote[idx] = note;
  scheduleRender();
}

async function runAll() {
  const btn = document.getElementById("run");
  btn.disabled = true;
  try {
    await ensureLoaded();
    const text = document.getElementById("text").value;
    if (!text) { status("(empty input)", "err"); return; }
    _lastText = text;

    const enabled = {
      gzip: document.getElementById("enable-gzip").checked,
      ar: document.getElementById("enable-ar").checked,
      mdlm: document.getElementById("enable-mdlm").checked,
      smollm2: document.getElementById("enable-smollm2").checked,
    };
    const K = parseInt(document.getElementById("kSlider").value, 10);

    const textBytes = new TextEncoder().encode(text);
    const N = textBytes.length;

    // Pre-allocate slots in CANONICAL order. Each slot stays at this index
    // for the whole run, so even if methods finish out of order the table /
    // bar stack stays in the order: gzip, AR, MDLM, SmolLM2.
    const SLOT = { gzip: 0, ar: 1, mdlm: 2, smollm2: 3 };
    _lastResults = [
      enabled.gzip    ? { bpb: 0, per_byte: new Float64Array(N), n_tokens: 0, label: "gzip", color: "#6b7280" } : null,
      enabled.ar      ? { bpb: 0, per_byte: new Float64Array(N), n_tokens: 0, label: arArch.label ?? "AR", color: "#3b82f6" } : null,
      enabled.mdlm    ? { bpb: 0, per_byte: new Float64Array(N), n_tokens: 0, label: mdlmArch.label ?? "MDLM", color: "#f97316" } : null,
      enabled.smollm2 ? { bpb: 0, per_byte: new Float64Array(N), n_tokens: 0, label: "SmolLM2-135M", color: "#22c55e" } : null,
    ];
    // All enabled methods start "queued"; the active one flips to "running…"
    // when its turn comes; the global #status is reserved for cross-run state.
    _slotNote = _lastResults.map((r) => (r ? "queued" : null));
    _bpbHistory = _lastResults.map(() => []);
    // Force a fresh build of the viz on the first render of this run.
    _vizState.text = null;
    scheduleRender();

    function progressFor(slotIdx) {
      return (partial) => {
        // Keep the canonical color/label even if the method overrides them.
        const cur = _lastResults[slotIdx];
        const merged = { ...partial, color: cur.color, label: cur.label };
        _lastResults[slotIdx] = merged;
        // Capture bpb history for the sparkline. Skip duplicates with the
        // same progress count (some methods emit the final result and a
        // post-final wrap-up tick that share k/K).
        if (merged.progress && merged.bpb != null && isFinite(merged.bpb)) {
          const hist = _bpbHistory[slotIdx];
          const k = merged.progress.k || 0;
          const K = merged.progress.K || 1;
          const kind = merged.progress.kind || "";
          const last = hist[hist.length - 1];
          if (!last || last.k !== k || last.bpb !== merged.bpb) {
            hist.push({ k, K, bpb: merged.bpb, kind });
          }
        }
        scheduleRender();
      };
    }

    // Run sequentially. Per-method status lives in the row's Progress cell;
    // the top-line #status is only used for cross-method state below.
    async function runOne(slotIdx, runningNote, fn) {
      setSlotNote(slotIdx, runningNote);
      const t0 = performance.now();
      const onProgress = progressFor(slotIdx);
      const r = await fn(onProgress);
      onProgress(r);
      setSlotNote(slotIdx, `${((performance.now() - t0) / 1000).toFixed(1)}s`);
    }

    if (enabled.gzip) {
      await runOne(SLOT.gzip, "running…", (onProgress) =>
        gzipBPB(textBytes, { onProgress }));
    }
    if (enabled.ar) {
      await runOne(SLOT.ar, "running…", (onProgress) =>
        arBPB(arWeights, arArch, text, { onProgress }));
    }
    if (enabled.mdlm) {
      const forceFirstUnmasked = document.getElementById("mdlm-force-first").checked;
      const note = `running… (K=${K}${forceFirstUnmasked ? ", force-leftmost" : ""})`;
      await runOne(SLOT.mdlm, note, (onProgress) =>
        mdlmBPB(mdlmWeights, mdlmArch, text, { K, forceFirstUnmasked, onProgress }));
    }
    if (enabled.smollm2) {
      await runOne(SLOT.smollm2, "loading on first run…", (onProgress) =>
        smollm2BPBWrap(text, { onProgress }));
    }
    status("done");
  } catch (e) {
    console.error(e);
    status(`error: ${e.message}`, "err");
  } finally {
    btn.disabled = false;
  }
}

document.getElementById("run").addEventListener("click", runAll);
document.getElementById("kSlider").addEventListener("input", () => {
  document.getElementById("kVal").textContent = document.getElementById("kSlider").value;
});

// Auto-size the diffusion-steps slider to the current text. Capped at the
// per-chunk size (1024) since K applies per chunk; for shorter inputs the
// max is exactly the token count, which makes K=N (pure AOAR) reachable.
async function _updateKSliderMax() {
  const text = document.getElementById("text").value;
  if (!text) return;
  try {
    await loadTokenizer("tokenizer/fineweb_1024_bpe.model");
  } catch { return; }
  let n;
  try { n = encode(text).length; } catch { return; }
  const slider = document.getElementById("kSlider");
  const newMax = Math.max(1, Math.min(n, 1024));
  if (parseInt(slider.max, 10) === newMax) return;
  const curVal = parseInt(slider.value, 10);
  slider.max = String(newMax);
  if (curVal > newMax) {
    slider.value = String(newMax);
    document.getElementById("kVal").textContent = String(newMax);
  }
}
let _autoMaxTimer = null;
document.getElementById("text").addEventListener("input", () => {
  clearTimeout(_autoMaxTimer);
  _autoMaxTimer = setTimeout(_updateKSliderMax, 300);
});
// Kick off on page load (tokenizer loads in the background; once it's ready
// the slider re-sizes to fit the default Transformer abstract).
_updateKSliderMax();
document.getElementById("rmin").addEventListener("input", render);
document.getElementById("rmax").addEventListener("input", render);

// ---- Generate mode ----
const GEN_OUTPUT = document.getElementById("genOutput");
const GEN_STATUS = document.getElementById("genStatus");

function genStatus(msg, kind = "") {
  GEN_STATUS.textContent = msg;
  GEN_STATUS.className = `status ${kind}`;
}

const GEN_METHODS = [
  { key: "gzip",     label: "gzip",         color: "#6b7280" },
  { key: "ar",       label: "AR",           color: "#3b82f6" },
  { key: "mdlm",     label: "MDLM",         color: "#f97316" },
  { key: "smollm2",  label: "SmolLM2-135M", color: "#22c55e" },
];

// Build the three rows once per Generate click. We don't lay down the prompt
// text here — the first onToken with phase: "prompt" carries the canonical
// decoded prompt (e.g. SP encode/decode normalization may differ slightly
// from the raw textarea).
function buildGenRows(enabled) {
  GEN_OUTPUT.innerHTML = "";
  const rows = {};
  for (const m of GEN_METHODS) {
    if (!enabled[m.key]) continue;
    const row = document.createElement("div");
    row.className = "gen-row";
    row.innerHTML =
      `<div class="label"><span class="legend-swatch" style="background:${m.color}"></span>${m.label}` +
        `<div class="meta" id="gen-meta-${m.key}">queued</div></div>` +
      `<div class="body" id="gen-body-${m.key}"></div>`;
    GEN_OUTPUT.appendChild(row);
    rows[m.key] = {
      body: row.querySelector(".body"),
      meta: row.querySelector(".meta"),
      promptSpan: null,    // populated on the prompt-phase emit
      suffix: "",
      mode: m.key === "mdlm" ? "replace" : "append",
    };
  }
  return rows;
}

// Attach the per-step popover payload (chosen + topk + aboveThresh + tokenizer)
// to a span DOM node so the popover handler can read it later. We stash on
// `_genInfo` rather than `dataset.*` because dataset can only hold strings —
// the topk array would round-trip through JSON otherwise.
function _attachGenInfo(span, methodKey, info) {
  if (!info || info.phase === "prompt") return;
  span.classList.add("tok");
  span._genInfo = {
    method: methodKey,
    chosen: info.chosen,
    topk: info.topk,
    aboveThresh: info.aboveThresh,
    tokenizer: info.tokenizer,
    vocabSize: info.vocabSize,
  };
}

// Patch a generate-row's body with the latest token. AR / SmolLM2 append
// fresh spans; MDLM rebuilds the entire suffix because positions arrive out
// of order and we draw "▁_" placeholders for not-yet-revealed slots.
function patchGenRow(rowState, fullText, piece, info, methodKey) {
  // First emit (phase: "prompt") sets up the greyed-prompt span so the user
  // sees where generation begins, using whatever prompt-string the generator
  // canonically decoded (may differ slightly from the textarea, e.g. SP
  // whitespace normalization).
  if (info?.phase === "prompt" || !rowState.promptSpan) {
    rowState.body.innerHTML = "";
    const ps = document.createElement("span");
    ps.style.opacity = "0.55";
    ps.textContent = fullText.slice(0, info?.promptLen ?? fullText.length);
    rowState.body.appendChild(ps);
    rowState.promptSpan = ps;
    rowState.suffix = "";
    rowState.revealedData = rowState.revealedData || {};  // pos → genInfo for MDLM
    if (info?.phase === "prompt") return;
  }
  if (rowState.mode === "append") {
    if (!piece) return;
    const span = document.createElement("span");
    span.className = "new";
    span.textContent = piece;
    _attachGenInfo(span, methodKey, info);
    rowState.body.appendChild(span);
    return;
  }
  // MDLM replace mode. The generator ships info.suffixPlan: a position-ordered
  // list of {kind, text, pos, chosen?, topk?, aboveThresh?} entries. "tok"
  // entries are revealed slots; "placeholder" entries are still-masked slots
  // (rendered as a faded " _"). We render one DOM span per entry so each
  // committed slot has its own popover surface; spacing comes from the
  // generator (which uses run-level decoding so SP boundary marks survive).
  const plan = info?.suffixPlan;
  if (!plan) return;
  while (rowState.body.children.length > 1) rowState.body.removeChild(rowState.body.lastChild);
  const wrap = document.createElement("span");
  wrap.classList.add("new");
  for (const entry of plan) {
    const span = document.createElement("span");
    span.textContent = entry.text;
    if (entry.kind === "tok") {
      span.classList.add("tok");
      span._genInfo = {
        method: methodKey,
        chosen: entry.chosen,
        topk: entry.topk,
        aboveThresh: entry.aboveThresh,
        tokenizer: entry.tokenizer,
        vocabSize: entry.vocabSize,
      };
    } else {
      span.style.opacity = "0.55";
    }
    wrap.appendChild(span);
  }
  rowState.body.appendChild(wrap);
  rowState.suffix = wrap.textContent;
}

async function runGenerate() {
  const btn = document.getElementById("generate");
  btn.disabled = true;
  try {
    await ensureLoaded();
    const promptText = document.getElementById("prompt").value;
    if (!promptText) { genStatus("(empty prompt)", "err"); return; }
    const maxTokens = parseInt(document.getElementById("genTokens").value, 10);
    const temperature = parseFloat(document.getElementById("genTemp").value);
    const topP = parseFloat(document.getElementById("genTopP").value);
    const repetitionPenalty = parseFloat(document.getElementById("genRepPen").value);
    const enabled = {
      gzip: document.getElementById("gen-enable-gzip").checked,
      ar: document.getElementById("gen-enable-ar").checked,
      mdlm: document.getElementById("gen-enable-mdlm").checked,
      smollm2: document.getElementById("gen-enable-smollm2").checked,
    };
    const rows = buildGenRows(enabled);
    genStatus("ready");

    async function runOne(key, label, fn) {
      const row = rows[key];
      if (!row) return;
      row.meta.textContent = "running…";
      const t0 = performance.now();
      const onToken = (piece, fullText, info) => {
        patchGenRow(row, fullText, piece, info, key);
        if (info && info.K) {
          row.meta.textContent = `${info.k}/${info.K} tokens`;
        }
      };
      try {
        await fn({ maxTokens, temperature, topP, repetitionPenalty, onToken });
        row.meta.textContent = `${((performance.now() - t0) / 1000).toFixed(1)}s · ${maxTokens} tokens`;
      } catch (e) {
        row.meta.textContent = `error: ${e.message}`;
        console.error(`${label} generate:`, e);
      }
    }

    if (enabled.gzip) {
      await runOne("gzip", "gzip", (opts) => gzipGenerate(promptText, opts));
    }
    if (enabled.ar) {
      await runOne("ar", "AR", (opts) => arGenerate(arWeights, arArch, promptText, opts));
    }
    if (enabled.mdlm) {
      await runOne("mdlm", "MDLM", (opts) =>
        mdlmGenerate(mdlmWeights, mdlmArch, promptText, opts));
    }
    if (enabled.smollm2) {
      await runOne("smollm2", "SmolLM2", (opts) => {
        const row = rows.smollm2;
        const onLoadingProgress = (p) => {
          // p is { status: "tokenizer"|"model"|"ready", info: ... }
          if (!row || p.status === "ready") return;
          // info can be a string or a {status, name, file, ...} object from
          // transformers.js's progress events.
          const detail = typeof p.info === "string"
            ? p.info
            : (p.info?.file ?? p.info?.status ?? "");
          row.meta.textContent = `loading ${p.status}${detail ? ": " + detail : ""}`;
        };
        return smollm2Generate(promptText, { ...opts, onLoadingProgress });
      });
    }
    genStatus("done");
  } catch (e) {
    console.error(e);
    genStatus(`error: ${e.message}`, "err");
  } finally {
    btn.disabled = false;
  }
}

document.getElementById("generate").addEventListener("click", runGenerate);
document.getElementById("genTokens").addEventListener("input", () => {
  document.getElementById("genTokensVal").textContent = document.getElementById("genTokens").value;
});
document.getElementById("genTemp").addEventListener("input", () => {
  document.getElementById("genTempVal").textContent =
    parseFloat(document.getElementById("genTemp").value).toFixed(2);
});
document.getElementById("genTopP").addEventListener("input", () => {
  document.getElementById("genTopPVal").textContent =
    parseFloat(document.getElementById("genTopP").value).toFixed(2);
});
document.getElementById("genRepPen").addEventListener("input", () => {
  document.getElementById("genRepPenVal").textContent =
    parseFloat(document.getElementById("genRepPen").value).toFixed(2);
});

// ===== Popover (top-K alternatives) =====
//
// Hover (desktop) shows a transient popover; click/tap pins it so the user can
// read it without holding the cursor still. Click outside (or click the same
// surface again) dismisses. On mobile, hover events don't fire reliably for
// touch — but `click` is synthesized from tap, so the pin path is the only
// one that runs, which is what we want.
//
// Two surfaces: BPB bars (`.b` inside #viz) carry slot+ci dataset, and we
// look up token info by binary-searching `result.per_token` for the byte. Gen
// tokens (`.tok` inside #genOutput) carry the entire popover payload on a
// `_genInfo` property attached at span-creation time.

const POPOVER = document.getElementById("popover");
let _popoverPinned = false;
let _popoverTarget = null;

function _hidePopover() {
  POPOVER.style.display = "none";
  POPOVER.classList.remove("pinned");
  POPOVER.setAttribute("aria-hidden", "true");
  _popoverPinned = false;
  _popoverTarget = null;
}

// Position the popover above (or, if no room, below) the target rect, clamped
// inside the viewport. Used for both hover and pin paths.
function _positionPopover(targetRect) {
  POPOVER.style.left = "0px";
  POPOVER.style.top = "0px";
  POPOVER.style.display = "block";
  const pw = POPOVER.offsetWidth;
  const ph = POPOVER.offsetHeight;
  const margin = 8;
  let left = targetRect.left + targetRect.width / 2 - pw / 2;
  let top = targetRect.top - ph - margin;
  if (top < margin) top = targetRect.bottom + margin;
  left = Math.max(margin, Math.min(window.innerWidth - pw - margin, left));
  top = Math.max(margin, Math.min(window.innerHeight - ph - margin, top));
  POPOVER.style.left = left + "px";
  POPOVER.style.top = top + "px";
}

// Display string for a piece — preserve whitespace, collapse newlines into ↵
// so single-line popover rows stay readable. Wraps in a span with monospace.
function _formatPiece(piece) {
  if (piece == null) return "";
  return piece.replace(/\n/g, "↵").replace(/\t/g, "→");
}

// Format a probability as a percentage with enough resolution at small p
// (rank-out tokens are typically <1%). Falls back to scientific for tiny p.
function _formatProb(prob) {
  if (prob == null || !isFinite(prob)) return "?";
  const pct = prob * 100;
  if (pct >= 1) return pct.toFixed(1) + "%";
  if (pct >= 0.01) return pct.toFixed(2) + "%";
  return pct.toExponential(1) + "%";
}

function _escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

// Resolve a piece string for a token id given its source tokenizer. SP path
// (fineweb_sp) reads from the cached pieceForId table; HF path (smollm2_hf)
// goes through smollm2Piece (decode([id])). Falls back to a literal id label
// if the tokenizer isn't loaded yet (shouldn't happen post-Compute).
function _resolvePiece(id, tokenizerKey) {
  try {
    if (tokenizerKey === "smollm2_hf") return smollm2Piece(id);
    return pieceForId(id);
  } catch {
    return `<id:${id}>`;
  }
}

// Render the top-K body. token has { id, piece, bits?, rank?, topk, aboveThresh, vocabSize, tokenizer }
// chosenId picks out which row to highlight (BPB → token.id; Gen → chosen.id).
function _renderTopK(token, chosenId, color, tokenizerKey) {
  const topk = token.topk || [];
  const rows = topk.map((entry, idx) => {
    const prob = Math.exp(entry.lp);
    const pct = (prob * 100).toFixed(prob >= 0.1 ? 1 : 2);
    const widthPct = Math.max(2, Math.min(100, prob * 100));
    const isChosen = entry.id === chosenId;
    const rowCls = isChosen ? "popover-row chosen" : "popover-row";
    const pieceStr = entry.piece || _resolvePiece(entry.id, tokenizerKey);
    const piece = _escapeHtml(_formatPiece(pieceStr));
    return `<div class="${rowCls}">` +
           `<span class="rank">${idx + 1}</span>` +
           `<span class="piece">${piece}</span>` +
           `<span class="pbar"><span style="width:${widthPct}%; background:${color}"></span></span>` +
           `<span class="pct">${pct}%</span>` +
           `</div>`;
  }).join("");
  // If chosen / actual is outside top-K, append an explicit row with rank
  // and probability. Probability is recovered from `token.bits` (which is
  // -log2 p_actual in BPB mode and is now also carried on the chosen object
  // in Gen mode).
  let extra = "";
  const inTopK = chosenId != null && topk.some((e) => e.id === chosenId);
  if (chosenId != null && !inTopK && token.rank != null) {
    const piece = _escapeHtml(_formatPiece(token.chosenPiece || token.piece || `<id:${chosenId}>`));
    let probStr = "";
    if (token.bits != null && isFinite(token.bits)) {
      probStr = ` · ${_formatProb(Math.pow(2, -token.bits))}`;
    }
    extra = `<div class="popover-rank-out">rank ${token.rank}: ${piece}${probStr}</div>`;
  }
  // Footer: how many tokens above the threshold beyond the top-K rendered.
  let footer = "";
  if (token.aboveThresh != null) {
    const beyond = Math.max(0, token.aboveThresh - topk.length);
    if (beyond > 0) {
      footer = `<div class="popover-footer">+${beyond.toLocaleString()} more above 0.1% probability</div>`;
    } else if (token.aboveThresh <= topk.length) {
      footer = `<div class="popover-footer">no other options &gt; 0.1%</div>`;
    }
  }
  return rows + extra + footer;
}

// Build & show the popover for a BPB bar. Returns false if no token info is
// available for this byte (e.g. gzip — falls through to a byte-only card).
function _resolveBpbToken(slotIdx, ci) {
  const res = _lastResults[slotIdx];
  if (!res) return null;
  const spans = _vizState.charByteSpans;
  if (!spans || ci >= spans.length) return null;
  const [byteStart] = spans[ci];
  const perToken = res.per_token;
  if (!perToken || perToken.length === 0) return null;
  // Binary search: token.byteStart ≤ byteStart < token.byteEnd.
  let lo = 0, hi = perToken.length - 1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    const t = perToken[mid];
    if (!t) { lo = mid + 1; continue; }
    if (byteStart < t.byteStart) hi = mid - 1;
    else if (byteStart >= t.byteEnd) lo = mid + 1;
    else return t;
  }
  return null;
}

function _showBpbPopover(bar) {
  const slotIdx = parseInt(bar.dataset.slot, 10);
  const ci = parseInt(bar.dataset.ci, 10);
  const res = _lastResults[slotIdx];
  if (!res) return;
  const tok = _resolveBpbToken(slotIdx, ci);
  let body;
  if (tok && tok.topk && tok.topk.length > 0) {
    const totalBytes = Math.max(0, (tok.byteEnd ?? 0) - (tok.byteStart ?? 0));
    const bpb = totalBytes > 0 ? tok.bits / totalBytes : 0;
    // step = which K-diffusion step committed this token (parallel commits
    // share a step). order = the linear commit index across the chunk
    // (1-based) — answers "this token was the Nth committed."
    const stepBit = tok.commitStep != null ? ` · step ${tok.commitStep + 1}` : "";
    // Denominator = total tokens in the run (constant); using the running
    // committed-so-far count made the denominator wobble while compute was
    // streaming and confused the comparison across tokens.
    const totalTokens = (res.per_token || []).length;
    const orderBit = tok.committedAt != null
      ? ` · committed #${tok.committedAt + 1}/${totalTokens}`
      : "";
    // Rank lives in the rank-out / chosen row in the body; no need to
    // duplicate it in the header.
    const headerMeta =
      `${tok.bits.toFixed(2)} bits · ${bpb.toFixed(2)} bpb · ${totalBytes}B` +
      stepBit + orderBit;
    body =
      `<div class="popover-header">` +
        `<span class="legend-swatch" style="background:${res.color}"></span>` +
        `<span>${_escapeHtml(res.label)}</span>` +
        `<span class="meta">${headerMeta}</span>` +
        `<span class="popover-pin" data-action="close" title="dismiss">×</span>` +
      `</div>` +
      _renderTopK({ ...tok, chosenPiece: tok.piece }, tok.id, res.color, res.tokenizer);
  } else {
    // gzip or first-of-chunk fallback: show only what we have. Pulls the
    // per-byte rate at the hovered byte to give the user some signal.
    const spans = _vizState.charByteSpans;
    const [byteStart, byteEnd] = spans[ci] ?? [0, 0];
    const pb = res.per_byte;
    let bits = 0;
    if (pb && byteEnd > byteStart) {
      let s = 0;
      const end = Math.min(byteEnd, pb.length);
      for (let b = byteStart; b < end; b++) s += pb[b];
      bits = end > byteStart ? s / (end - byteStart) : 0;
    }
    body =
      `<div class="popover-header">` +
        `<span class="legend-swatch" style="background:${res.color}"></span>` +
        `<span>${_escapeHtml(res.label)}</span>` +
        `<span class="meta">${bits.toFixed(2)} bits</span>` +
        `<span class="popover-pin" data-action="close" title="dismiss">×</span>` +
      `</div>` +
      `<div class="popover-footer">no per-token distribution for this method</div>`;
  }
  POPOVER.innerHTML = body;
  POPOVER.setAttribute("aria-hidden", "false");
  _positionPopover(bar.getBoundingClientRect());
}

function _showGenPopover(span) {
  const data = span._genInfo;
  if (!data) return;
  const color = (GEN_METHODS.find((m) => m.key === data.method) || {}).color || "#3b82f6";
  const label = (GEN_METHODS.find((m) => m.key === data.method) || {}).label || data.method;
  const chosen = data.chosen || {};
  const tokenForRender = {
    topk: data.topk || [],
    aboveThresh: data.aboveThresh,
    rank: chosen.rank,
    piece: chosen.piece,
    chosenPiece: chosen.piece,
    bits: chosen.bits,
  };
  // Rank now lives in the rank-out / chosen row in the body; the Gen header
  // is empty unless we add other meta later.
  const body =
    `<div class="popover-header">` +
      `<span class="legend-swatch" style="background:${color}"></span>` +
      `<span>${_escapeHtml(label)}</span>` +
      `<span class="meta"></span>` +
      `<span class="popover-pin" data-action="close" title="dismiss">×</span>` +
    `</div>` +
    _renderTopK(tokenForRender, chosen.id, color, data.tokenizer);
  POPOVER.innerHTML = body;
  POPOVER.setAttribute("aria-hidden", "false");
  _positionPopover(span.getBoundingClientRect());
}

// ----- Event wiring -----

function _onPointerOver(e) {
  if (_popoverPinned) return;
  const bar = e.target.closest(".b");
  if (bar && VIZ.contains(bar)) { _popoverTarget = bar; _showBpbPopover(bar); return; }
  const tok = e.target.closest(".tok");
  if (tok && GEN_OUTPUT.contains(tok)) { _popoverTarget = tok; _showGenPopover(tok); return; }
}

function _onPointerOut(e) {
  if (_popoverPinned) return;
  // Hide only if leaving the active target without entering the popover.
  const to = e.relatedTarget;
  if (to && (POPOVER.contains(to) || (_popoverTarget && _popoverTarget.contains(to)))) return;
  _hidePopover();
}

function _onClick(e) {
  // Click inside the popover: only the explicit close affordance dismisses.
  if (POPOVER.contains(e.target)) {
    if (e.target.closest('[data-action="close"]')) _hidePopover();
    return;
  }
  const bar = e.target.closest(".b");
  const tok = e.target.closest(".tok");
  const surface = bar || tok;
  if (!surface) {
    _hidePopover();
    return;
  }
  // Toggle if re-clicking the same pinned surface; otherwise repin.
  if (_popoverPinned && _popoverTarget === surface) {
    _hidePopover();
    return;
  }
  _popoverPinned = true;
  _popoverTarget = surface;
  POPOVER.classList.add("pinned");
  if (bar && VIZ.contains(bar)) _showBpbPopover(bar);
  else if (tok && GEN_OUTPUT.contains(tok)) _showGenPopover(tok);
}

document.addEventListener("pointerover", _onPointerOver);
document.addEventListener("pointerout", _onPointerOut);
document.addEventListener("click", _onClick);
window.addEventListener("scroll", () => { if (!_popoverPinned) _hidePopover(); }, true);
window.addEventListener("resize", _hidePopover);

// Debug hook: expose the latest results so the JS console can audit per-token
// math (e.g. MDLM trace[]). Lives behind a "_pg" namespace so it's obviously
// debug-only.
window._pg = {
  results: () => _lastResults,
  mdlm: () => _lastResults[2],
  trace: (tokenIdx) => {
    const r = _lastResults[2];
    if (!r || !r.per_token || !r.per_token[tokenIdx]) return null;
    return r.per_token[tokenIdx];
  },
  pieceForId,
  // Encode an input via the loaded SP processor and return ids+pieces. Pieces
  // are the SP-internal strings (with ▁ for word-boundary), so we can see
  // *exactly* what the encoder picked, not the decoded form.
  encode: (s) => {
    const proc = getProcessor();
    if (!proc) return null;
    return { ids: proc.encodeIds(s), pieces: proc.encodePieces(s) };
  },
  // Walk the vocab and return [{id, piece}] entries matching a regex.
  vocabSearch: (pattern, limit = 4096) => {
    const re = pattern instanceof RegExp ? pattern : new RegExp(pattern);
    const out = [];
    for (let i = 0; i < limit; i++) {
      let p;
      try { p = pieceForId(i); } catch { break; }
      if (p == null) break;
      if (re.test(p)) out.push({ id: i, piece: p });
    }
    return out;
  },
};
