# Research notes: Compression-aware QAT campaign

This directory contains supporting research notes from the same Parameter Golf experiment campaign.

The official scoreable submission is:

[`records/track_non_record_16mb/2026-04-22_SP8192_CompQAT_PR1493/`](../../records/track_non_record_16mb/2026-04-22_SP8192_CompQAT_PR1493/)

These notes document the broader research process: what was tested, what did not work, what was learned, and why the final submission pivoted to CompQAT on the PR #1493 SP8192 stack.

Included notes:

1. **LoRA / TTT inference-mode bugfix note**  
   Documents an inference-mode interaction discovered during LoRA/TTT experimentation.

2. **3DCF compression-vs-packing note**  
   Explains why semantic/document compression ideas did not directly transfer to Parameter Golf binary artifact packing.

3. **Full campaign article**  
   Longer write-up of the experiment path, pivot, and final submission.

Only the `records/.../2026-04-22_SP8192_CompQAT_PR1493/` folder is intended as the scoreable Parameter Golf submission. The files in this research directory are included for context and are not separate leaderboard submissions.
