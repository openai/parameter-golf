# Lexis-E -- Linguistics-Driven Text Compression

**Repository:** https://github.com/shasankp000/Lexis (branch: `lexis-e`)

Lexis-E is a research compressor that exploits the linguistic structure of
English text instead of treating it as a raw byte stream. Every token passes
through a 12-stage pipeline before a single bit is written.

---

## How it works

### Stage 1 -- Text normalisation
BOM stripping, whitespace collapsing, and idempotent Unicode cleanup via
`normalize_text`.

### Stage 2 -- Morphological analysis
`MorphologicalAnalyser` (spaCy + lemminflect) reduces each surface token to
a lowercase root and a morph code (0-12) encoding the inflection needed to
recover the original form: plural, past tense, present participle, past
participle, third-singular, comparative, superlative, adverbial, negation,
agent nominalisation, common nominalisation, and irregular.

### Stage 3 -- Syntactic analysis
`analyse_sentence` extracts Universal POS tags via spaCy's dependency parser.

### Stage 4 -- Coreference / discourse analysis
`DiscourseAnalyser` runs the `biu-nlp/f-coref` model to find repeated
named-entity coreference chains. Each unique chain is assigned a compact
symbol (`SS-E0`, `SS-E1`, ...) and the first mention is kept verbatim while
all later mentions are replaced with the symbol. This is losslessly reversed
at decode time using the stored symbol table.

### Stage 5 -- Character-class encoding
Each root character is mapped to a (phonetic-class, position) coordinate
pair from `PHONETIC_CLASSES`. Classes group characters by articulatory
similarity (labial stops, fricatives, nasals, etc.), keeping delta values
small. Position deltas within a class are stored as a separate stream.
Case is encoded independently as a 2-bit flag (lower / title / upper /
mixed) plus a per-character uppercase bitmap -- enough to recover `eBook`,
`iPhone`, and `NATO` exactly.

### Stage 6 -- Context-mixing probability model
`ContextMixingModel` maintains three context maps: character n-gram history,
current morph code, and current POS tag. Their probability estimates are
blended to form a single distribution over the character-class alphabet.

### Stage 7 -- Arithmetic coding
`ArithmeticEncoder` / `ArithmeticDecoder` compress the class stream and the
position-delta stream independently using the context-mixing model and a
unigram model respectively.

### Stage 8-9 -- Reconstruction
`_reconstruct_chars` turns decoded classes + deltas back into the sentinel-
delimited root stream (`^root$`). `_split_roots` extracts individual roots,
`apply_morph` re-inflects them, and `apply_case_flag` restores the original
surface casing.

### Stage 10 -- LEXI binary format
`encode_metadata` / `decode_metadata` pack every field (bitstreams, POS
tags, morph codes, root lengths, model weights, context maps, case flags,
case bitmaps, symbol table) into a compact binary container with a 4-byte
magic header. If `zstandard` is installed and post-compression saves bytes
the payload is wrapped with an additional `LXZ1` zstd header.

---

## Test suite

Run `python pipeline_trace.py` from the project root. All 12 stages pass
with zero FAILs.

python pipeline_trace.py 2>&1 | tee pipeline_trace.log

See `pipeline_trace.log` in this folder for the full captured output.

---

## Dependencies

| Package | Role |
|---------|------|
| spaCy + en_core_web_lg | Tokenisation, POS tagging, dependency parsing |
| lemminflect | Morphological inflection |
| f-coref (biu-nlp) | Coreference resolution |
| zstandard (optional) | Post-compression of LEXI payloads |

Install:

```bash
pip install spacy lemminflect zstandard
python -m spacy download en_core_web_lg
```

---

## Compression results

Tested on the first 500 lines of Moby Dick (`moby500.txt`, 3101 bytes).
Detailed stats are printed by `compress_to_file` and stored in the stats
dict returned from that call.

