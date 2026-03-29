# Session 05: Throughput, Pre-TTT, and TTT Audit

Preferred mode: Planning

Use `/research-engineer` if it exists locally and helps with audit structure.

Goal:
- Decide how to move beyond the Session 03 anchor by auditing:
  - the throughput gap to the faster local public stack
  - the portable pre-TTT stack differences
  - TTT correctness, legality, engineering cost, and expected upside

Read these first:
- @docs/campaign/artifacts/04_targeted_delta_sweep.md
- @records/track_10min_16mb/2026-03-17_LoRA_TTT/README.md
- @records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md
- @records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py
- @records/track_non_record_16mb/2026-03-28_pre_ttt_anchor/train_gpt.py

Constraints:
- This starts as an audit / planning session.
- Be strict about challenge legality and evaluation leakage.
- Separate score gain from engineering overhead.
- Distinguish portable first-wave changes from harder architectural rewrites.

Required workflow:
1. Inspect the local `1.1194` public stack and explain the major measured gaps versus the anchor:
   - pre-TTT base (`1.1218` vs `1.1290`)
   - throughput (`83.4 ms` vs `91.37 ms`)
   - TTT gain (`-0.0025`)
2. Audit the throughput gap:
   - identify likely contribution from FA3
   - identify harder coupled pieces such as parameter banking / Parallel Muon
   - decide whether FA3 is a first-wave portable change on Pegasus / NGC
3. Audit the pre-TTT stack gap:
   - list easy portable pieces from the public stack
   - rank them by effort and likely upside
4. Audit the public TTT implementation:
   - trace exactly how score-first avoids leakage
   - document the evaluation-time cost budget
   - decide what parts are portable to the anchor stack
5. Produce a recommendation:
   - what to implement first
   - what to defer
   - what should stay out of scope for now

Deliverables:
- `docs/campaign/artifacts/05_ttt_correctness_audit.md`
- one concise ranked plan for:
  - throughput work
  - pre-TTT stack work
  - TTT work

Definition of done:
- The audit explicitly states whether the public TTT path appears challenge-compliant.
- The audit states the estimated engineering cost to integrate it into your stack.
- The audit gives a ranked recommendation for the next session.

Commit message:
- `docs(campaign): add ttt correctness audit`
