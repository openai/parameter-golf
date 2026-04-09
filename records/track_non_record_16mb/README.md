# Non-Record Experiments

This tree is intentionally broader and messier than a polished submission folder.

The point of `track_non_record_16mb` is to preserve the actual research arc:

- experiments that looked promising and later stalled
- hardware-specific runner folders that made pod time less wasteful
- structured result files that explain why we pivoted

In other words, this directory keeps the failures on purpose. The cleanup goal is not to erase them. It is to make them legible.

## Main Arcs

### March 18-24

Earlier compact-model record attempts live here as dated folders. These are closer to conventional submission branches.

### March 31 - April 1: Spectral Flood Walk

These folders track the "coprocessor grows the model at eval time" line:

- [2026-03-31_SpectralFloodWalk_v0](/Users/kennethmalloy/Local%20Documents/Developer/parameter-golf/records/track_non_record_16mb/2026-03-31_SpectralFloodWalk_v0)
- [2026-03-31_SpectralFloodWalk_v1a](/Users/kennethmalloy/Local%20Documents/Developer/parameter-golf/records/track_non_record_16mb/2026-03-31_SpectralFloodWalk_v1a)
- [2026-03-31_SpectralFloodWalk_v1b](/Users/kennethmalloy/Local%20Documents/Developer/parameter-golf/records/track_non_record_16mb/2026-03-31_SpectralFloodWalk_v1b)
- [2026-03-31_SpectralFloodWalk_v2a](/Users/kennethmalloy/Local%20Documents/Developer/parameter-golf/records/track_non_record_16mb/2026-03-31_SpectralFloodWalk_v2a)
- [2026-03-31_SpectralFloodWalk_v2a1_host1233](/Users/kennethmalloy/Local%20Documents/Developer/parameter-golf/records/track_non_record_16mb/2026-03-31_SpectralFloodWalk_v2a1_host1233)
- [2026-04-01_SpectralFloodWalk_v2b](/Users/kennethmalloy/Local%20Documents/Developer/parameter-golf/records/track_non_record_16mb/2026-04-01_SpectralFloodWalk_v2b)

The final folder in that line, [2026-04-01_SpectralFloodWalk_v2b](/Users/kennethmalloy/Local%20Documents/Developer/parameter-golf/records/track_non_record_16mb/2026-04-01_SpectralFloodWalk_v2b), also has a short [RESULTS.md](/Users/kennethmalloy/Local%20Documents/Developer/parameter-golf/records/track_non_record_16mb/2026-04-01_SpectralFloodWalk_v2b/RESULTS.md) so the dead ends are easier to read later.

### April 2: Evolutionary Benchmark

[2026-04-02_EvolutionaryBenchmark](/Users/kennethmalloy/Local%20Documents/Developer/parameter-golf/records/track_non_record_16mb/2026-04-02_EvolutionaryBenchmark) is the current large benchmarking harness for:

- population throughput on H100
- committee schedules
- adaptive widen/narrow decisions
- tokenizer comparisons
- recipe-gene evolution
- compressed-committee submission tests

That folder also keeps a concise [RESULTS.md](/Users/kennethmalloy/Local%20Documents/Developer/parameter-golf/records/track_non_record_16mb/2026-04-02_EvolutionaryBenchmark/RESULTS.md) next to the raw JSON outputs.

## What Gets Checked In

The rule of thumb in this tree is:

- keep runner scripts, READMEs, plans, and structured result JSON
- ignore ephemeral pod launcher logs when they do not add real signal

That keeps the experimental record honest without letting every transient shell transcript become repo history.
