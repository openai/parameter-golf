# Non-Record: The N-gram BPB Scores Are Not Real

**Author:** abaybektursun | **Date:** 2026-03-26 | **Track:** Non-record study

N-gram caching in Parameter Golf claims sub-0.5 BPB. The scores come from an invalid probability distribution that sums to ~410, not 1. This study presents the proof, experimental evidence, and proposed fixes.

Full analysis: [abay.tech/posts/eval-time-model-growth](https://abay.tech/posts/eval-time-model-growth)

PR discussion: [#886](https://github.com/openai/parameter-golf/pull/886)

## Credits

- [@Eppie](https://github.com/openai/parameter-golf/issues/677#issuecomment-4139902162) for identifying the probability validity issue
- Mirco (Discord) for the `P(cache_bin)` formulation
- N-gram cache concept: [PR #727](https://github.com/openai/parameter-golf/pull/727), [PR #779](https://github.com/openai/parameter-golf/pull/779), [PR #788](https://github.com/openai/parameter-golf/pull/788)
- Base model: [PR #728](https://github.com/openai/parameter-golf/pull/728)
- Code: `experiments/eval_time_mixing/`
