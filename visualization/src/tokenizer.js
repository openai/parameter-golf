// Wrap sentencepiece-js so the rest of the code can call a simple
// `await tokenizer.encode(text) -> number[]`.
//
// sentencepiece-js bundles SP as Emscripten WASM, but its thin TS wrapper's
// `load(url)` calls Node's `fs.readFileSync(url)` to slurp the model file
// before piping the bytes into the WASM virtual filesystem (FS.writeFile).
// In the browser there's no `fs`. We work around this by:
//   1. Pre-fetching the model bytes ourselves via the platform's fetch.
//   2. Stashing them under a sentinel URL in a tiny in-memory map.
//   3. Monkey-patching the `fs` namespace that the wrapper reaches for so
//      `readFileSync(url)` returns our cached bytes when the URL matches.
// Once the wrapper hands those bytes to FS.writeFile, the rest of the load
// (StringView → absl_string_view → processor.Load) works unmodified.

import * as spjs from "sentencepiece-js";

const Processor = spjs.SentencePieceProcessor || spjs.default?.SentencePieceProcessor;

const SENTINEL_PATH = "__pg_sp_model__";
const _bytesByPath = new Map();

// Patch the internal Node-style fs that the bundle's IIFE captured.
// In ESM the captured `fs` is unreachable from outside, but the wrapper
// references it via the bundle's local `fs__namespace` after rolling up
// `var fs = require('fs')`. We can't reach that scope. So instead, we
// override `Processor.prototype.load` to do the same work ourselves.
let _patched = false;
function patchProcessor() {
  if (_patched) return;
  _patched = true;
  const proto = Processor.prototype;
  const origLoad = proto.load;

  proto.load = async function (urlOrPath) {
    // If we have cached bytes for this path, do the inner WASM work directly
    // and skip the fs.readFileSync call.
    const bytes = _bytesByPath.get(urlOrPath);
    if (bytes) {
      // Mirror the wrapper's body: writeFile → StringView → Load
      // We need `this.sentencepiece` (the Module) to be set first. The
      // wrapper sets it lazily inside load(). Trigger that by calling the
      // original load() with a path it can't read; the Module promise
      // resolves before the readFileSync call, so this.sentencepiece is set.
      try {
        await origLoad.call(this, "/__pg_init_only__");
      } catch (_e) {
        // expected — readFileSync threw, but this.sentencepiece is now set.
      }
      const M = this.sentencepiece;
      if (!M || !M.FS) throw new Error("WASM module not initialised");
      M.FS.writeFile("sentencepiece.model", new Uint8Array(bytes));
      const sv = new M.StringView("sentencepiece.model");
      const av = sv.getView();
      this.processor = new M.SentencePieceProcessor();
      const status = this.processor.Load(av);
      status.delete();
      av.delete();
      sv.delete();
      return;
    }
    return origLoad.call(this, urlOrPath);
  };
}

let _proc = null;
export async function loadTokenizer(modelUrl) {
  if (_proc) return _proc;
  patchProcessor();
  // Fetch bytes ourselves and stash them under a sentinel path
  const res = await fetch(modelUrl);
  if (!res.ok) throw new Error(`tokenizer fetch ${modelUrl} → ${res.status}`);
  const bytes = await res.arrayBuffer();
  _bytesByPath.set(SENTINEL_PATH, bytes);

  _proc = new Processor();
  await _proc.load(SENTINEL_PATH);
  return _proc;
}

// Encode text to token ids. Returns just the ids (mirrors sp.encode_as_ids).
export function encode(text) {
  if (!_proc) throw new Error("tokenizer not loaded — call loadTokenizer first");
  return _proc.encodeIds(text);
}

// Encode + return ids, pieces, and per-token UTF-8 byte lengths in one call.
// We need pieces (with ▁ word markers preserved) to compute byte attribution
// the same way bpb_compare.sp_token_byte_lengths does — and the underlying
// sentencepiece-js wrapper doesn't expose `IdToPiece`, only `EncodeAsPieces`
// for the whole string. So we encode the text twice (ids + pieces) and zip.
const TE = new TextEncoder();
const BYTE_FALLBACK_RE = /^<0x[0-9A-Fa-f]{2}>$/;
const CONTROL_PIECES = new Set(["<pad>", "<s>", "</s>", "<unk>", "<sep>", "<cls>", "<mask>"]);

function pieceByteLength(piece) {
  if (BYTE_FALLBACK_RE.test(piece)) return 1;
  if (CONTROL_PIECES.has(piece)) return 0;
  return TE.encode(piece.replace(/▁/g, " ")).length;
}

export function encodeFull(text) {
  if (!_proc) throw new Error("tokenizer not loaded — call loadTokenizer first");
  const ids = _proc.encodeIds(text);
  const pieces = _proc.encodePieces(text);
  if (ids.length !== pieces.length) {
    throw new Error(`encodeIds/encodePieces length mismatch: ${ids.length} vs ${pieces.length}`);
  }
  const byteLens = pieces.map(pieceByteLength);
  return { ids, pieces, byteLens };
}

export function tokenByteLengths(ids, pieces = null) {
  if (!_proc) throw new Error("tokenizer not loaded");
  if (!pieces) {
    throw new Error("pieces must be provided (use encodeFull instead of encode)");
  }
  return pieces.map(pieceByteLength);
}

export function getProcessor() {
  return _proc;
}

// Return every piece's decoded text in vocab order (length = vocab size).
// The sentencepiece-js wrapper doesn't bind GetPieceSize/IdToPiece, so we
// reconstruct the vocab by decoding each id individually. For our 1024-vocab
// FineWeb tokenizer this is ~1024 cheap WASM calls, done once at startup.
export function getAllPieces(maxVocab = 4096) {
  if (!_proc) throw new Error("tokenizer not loaded");
  const out = [];
  for (let i = 0; i < maxVocab; i++) {
    let piece;
    try { piece = _proc.decodeIds([i]); }
    catch (_e) { break; } // ran past vocab end
    if (piece === null || piece === undefined) break;
    out.push(piece);
  }
  return out;
}

// Decode a list of token ids back to text via the SP processor.
export function decode(ids) {
  if (!_proc) throw new Error("tokenizer not loaded — call loadTokenizer first");
  return _proc.decodeIds(ids);
}

// Cache piece-strings per id (for popover display). Lazily populated from the
// already-fetched getAllPieces() vocab the first time it's asked, then queried
// directly. Pieces still carry the SP ▁ marker — render-time decides how to
// display it.
let _pieceById = null;
function _ensurePieceMap() {
  if (_pieceById) return _pieceById;
  if (!_proc) throw new Error("tokenizer not loaded");
  _pieceById = getAllPieces();
  return _pieceById;
}
export function pieceForId(id) {
  const m = _ensurePieceMap();
  return id >= 0 && id < m.length ? m[id] : `<id:${id}>`;
}
