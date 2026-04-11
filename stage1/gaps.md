# Stage 1 Gaps

After the public-record pass, the ontology is healthier but not stable yet.
The main change is that evaluation is now known to be a real surface.

## Five Set-Level Gaps

### G01: We originally under-modeled evaluation as a first-class surface

- uncovered area: evaluation-time context and scoring policy
- why it matters: public records show large gains from sliding-window eval and eval-length changes
- current portfolio fix: `M11` is now explicit and active
- remaining question: how much of that gain is robust, portable, and worth optimizing further on our path
- promotion trigger: if the running A100 eval experiments continue to replicate the public direction

### G02: Structural families are now under-supported by public evidence

- uncovered area: reuse and control simplification
- why it matters: these were part of our original generic Enigma doctrine
- current portfolio change: `M03` and `M05` are now deferred
- what evidence is missing: whether the public frontier is merely ignoring them or actually dominating them
- promotion trigger: if public-frontier moves saturate quickly

### G03: Export now splits into more subfamilies than before

- uncovered area: fp16 embedding passthrough, late-K passthrough, int6/int8 trade, keep-float threshold, scale scheme
- why it matters: `M09` is now one of the strongest surfaces, but it is too broad internally
- current portfolio miss: `M09` still bundles multiple winning subfamilies
- what evidence is missing: which export submove is actually carrying the gain
- promotion trigger: split `M09` into explicit children after the next evidence round

### G04: Long-context geometry is supported, but its portable optimum is not known

- uncovered area: exact seq length and batch geometry that transfer from A100 to `8xH100`
- why it matters: public wins use `2048` and `4096`, but hardware sensitivity is likely real
- current portfolio fix: `M02` is active
- what evidence is missing: which geometry actually survives promotion
- promotion trigger: once A100 results rank the family cleanly enough

### G05: Tokenizer remains the largest untouched high-upside frontier

- uncovered area: tokenizer family and vocab co-design
- why it matters: public records moved many other frontiers, but tokenizer is still mostly untouched
- current portfolio miss: `M10` remains deferred
- what evidence is missing: whether the current public frontier is just a local optimum under SP-1024
- promotion trigger: after export/eval gains stop compounding

## Coverage Audit

- Which move family was overrepresented in the old Stage 1?
  - config retunes inside `Hyperparameters`

- Which move family was missing entirely before the record pass?
  - evaluation-time context and scoring policy

- What is now the main discipline rule?
  - count neighborhoods, not knobs

- Which metrics still need explicit skepticism?
  - round-tripped `val_bpb`
  - step time under the real wallclock
  - artifact bytes after any structural change
  - whether an eval-only gain is still in-bounds and reproducible
