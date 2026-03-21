# TPI Integration Memo

- Theory source repo: `golf-soris-tpi-lab`
- Implementation repo: `parameter-golf`
- First packet to integrate: `TPI-001`
- Initial target: a non-record candidate first, not a record attempt
- Constraint for the first pass: do not make heavy changes to the model, optimizer, or tokenizer until the baseline and evaluation path are grounded

## Operating flow

1. Thesis / spec / inspection packet is authored in `golf-soris-tpi-lab`.
2. Codex receives that packet and makes the smallest justified implementation branch in `parameter-golf`.
3. Run evidence and size accounting are collected under local `runs/`.
4. Findings feed back into inspection notes and failure / decision ledgers in the lab repo.
