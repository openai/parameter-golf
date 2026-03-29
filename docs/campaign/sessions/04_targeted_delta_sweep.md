# Session 04: Targeted Delta Sweep

Preferred mode: Execution

Status:
- Complete
- Outcome: Delta 1 failed, Delta 2 neutral, no standalone graduating delta

Use `/research-engineer` if it exists locally for prioritization, but keep the actual work narrow and measured.

Goal:
- Test a small number of low-complexity deltas on top of the anchor and identify whether any are worth carrying forward.

Read these first:
- @docs/campaign/artifacts/03_pre_ttt_anchor_summary.md
- @records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md
- @records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md
- @records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/README.md
- @records/track_non_record_16mb/<today>_pre_ttt_anchor/README.md
- @records/track_non_record_16mb/<today>_pre_ttt_anchor/train_gpt.py

Constraints:
- Test at most three deltas.
- Favor the cheapest high-signal levers first.
- Do not add TTT in this session.

Required workflow:
1. Choose at most three deltas from this pool unless the anchor evidence strongly suggests otherwise:
   - LeakyReLU squared in the MLP
   - GPTQ-lite percentile clip search
   - warmdown or EMA threshold tuning
   - one bigram or smear change
2. Implement the deltas in isolated branches or isolated experiment folders so their effects stay attributable.
3. Run enough measurement to compare against the anchor credibly.
4. Write one short comparison note that ranks the deltas by expected value.

Deliverables:
- one or more self-contained experiment folders under `records/track_non_record_16mb/`
- `docs/campaign/artifacts/04_targeted_delta_sweep.md`

Definition of done:
- Each tested delta has a measured outcome or a clearly documented blocked reason.
- The comparison note says which single delta should graduate next, if any.

Measured outcome:
- Delta 1 (GPTQ-lite percentile clip search): failed, worse BPB, over cap
- Delta 2 (LeakyReLU^2): neutral/tie, slightly better quantization and artifact size, slower throughput
- Session 04 closes without a graduating standalone delta; the next phase is Session 05 throughput + pre-TTT + TTT audit

Commit message:
- `feat(campaign): add targeted delta sweep results`
