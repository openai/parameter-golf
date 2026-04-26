# References — index of code resources for SSM work

## Vendored (load-bearing — read directly in editor)

| File | What it is | When to use |
|---|---|---|
| `mamba_minimal_model.py` | johnma2006/mamba-minimal `model.py`, ~340 LOC pure PyTorch. Numerically equivalent to official Mamba forward/backward. Sequential `selective_scan` (slow but correct on MPS). License: MIT. | Building any Mamba-1 block in this worktree. Adapt rather than reimplement from arxiv. |
| `selective_scan_ref.py` | Official `selective_scan_ref` from state-spaces/mamba, ~95 LOC. License: Apache-2.0. | **Correctness oracle.** Any custom selective-scan you write must pass numerical agreement with this on a small fixed input before you trust it in an experiment. |

Both files have local modifications: the upstream code uses `einops` (not installed in this worktree) — calls were replaced with native `torch.transpose / .unflatten / .repeat_interleave / .unsqueeze / torch.einsum`. The header of each vendored file documents every replacement. Functional equivalence was verified at vendor time: `mamba_minimal_model.MambaBlock.selective_scan` and `selective_scan_ref` agree to `max_abs_diff = 0.0` on a fixed seed-0 input (b=2, l=12, d_in=32, n=8).

### Oracle usage protocol

Before integrating any custom selective-scan into an experiment's `train_gpt.py`:

1. Build a small fixed input: e.g., `B=2, L=64, D=16, N=8`.
2. Run both your scan and `selective_scan_ref` on the same input + same params.
3. `torch.allclose(your_out, ref_out, atol=1e-5, rtol=1e-4)` should be True.
4. If it fails, your scan is wrong. **Debug in `scratch/` before training.** The recurrence amplifies bugs over the sequence length; a step-1 anomaly that would be a curiosity in a transformer is often a smoking gun in an SSM.

The two vendored files use different axis conventions for the scan inputs. `mamba_minimal_model.MambaBlock.selective_scan` takes `u(b, l, d_in)` (sequence-major); `selective_scan_ref` takes `u(B, D, L)` (channel-major, einops `b d l`). When wiring up the oracle on a custom scan, transpose accordingly. See the smoke-test at vendor time (`Phase B verification` in this session's commit) for a worked example.

## Curl-on-demand (don't preemptively download)

Fetch when needed; vendoring everything would clutter the worktree. `references/_external/` is gitignored, so curl-on-demand vendoring of upstream repos doesn't accidentally commit other people's code. If you fetch a substantial repo elsewhere, follow the same pattern (add to `.gitignore` before cloning).

### Mamba implementations

- mamba-2 minimal (tommyip): `git clone --depth=1 https://github.com/tommyip/mamba2-minimal references/_external/mamba2-minimal`
- mamba.py / MambaPy (alxndrTL — includes parallel associative `pscan`, MPS-compatible): `git clone --depth=1 https://github.com/alxndrTL/mamba.py references/_external/mambapy`. Suggested vendoring criterion (not a rule): vendor only when slow-but-correct mamba-minimal works AND step time is the experimental bottleneck (e.g. >30 min per experiment).
- The Annotated Mamba (Sasha Rush, Triton): https://srush.github.io/annotated-mamba/hard.html
- `state-spaces/mamba/mamba_ssm/modules/ssd_minimal.py` — Listing 1 of the Mamba-2 paper, ~30-line pure-PyTorch chunk-decomposition (matmul-only, MPS-compatible). Trade-off vs Mamba-1: scalar-per-time A_t (less expressive per state) but allows much larger N (state dim 64-256 vs Mamba-1's 16).

### S4 / S4D

- The Annotated S4 (best pedagogy, JAX): https://srush.github.io/annotated-s4/
- The Annotated S4D: https://srush.github.io/annotated-s4/s4d
- Official state-spaces/s4 (Apache-2.0): `git clone --depth=1 https://github.com/state-spaces/s4 references/_external/s4`

### Blogs (read in browser via curl + html-to-text if useful)

- Maarten Grootendorst — Visual Guide to Mamba: https://www.maartengrootendorst.com/blog/mamba/ (best intuition)
- Tri Dao — Mamba-2 Parts I-IV: https://tridao.me/blog/2024/mamba2-part1-model/
- Goomba Lab — Mamba-2 Parts I-IV: https://goombalab.github.io/blog/2024/mamba2-part1-model/
- Goomba Lab — Tradeoffs of SSMs and Transformers (2025): https://goombalab.github.io/blog/2025/tradeoffs/
- Hazy Research — Zoology / MQAR: https://hazyresearch.stanford.edu/blog/2023-12-11-zoology0-intro

### Papers

Use `PAPERS.md` as the curated index. Fetch with `curl https://arxiv.org/pdf/<id>` then process locally.

## A note on einops

Upstream Mamba code uses [einops](https://github.com/arogozhnikov/einops) extensively. einops is **not installed** in this worktree (not in `requirements.txt`, the harness's `python` doesn't have it). Both vendored files have einops calls replaced with native `torch` operations; see each file's header for the per-call mapping. If you adapt one of the curl-on-demand external sources into an experiment, expect to do the same replacement — do NOT add einops to `requirements.txt` (the canonical `train_gpt.py` is locked).
