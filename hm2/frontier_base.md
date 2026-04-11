# hm2 Frontier Base

`hm2` now includes a second patch base:

- [base_frontier_train_gpt.py](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/hm2/base_frontier_train_gpt.py)

Source:

- upstream PR `#1416`
- branch: `erichroepke/parameter-golf:submission/sp8192-prequant-ttt-sdclip-v2`

Why this base:

- it is a real frontier `train_gpt.py`, not just a record-side helper script
- it already carries the stronger `SP8192 + pre-quant TTT + SDClip` stack
- it is a better future patch target than the older local root baseline

Current status:

- the `hm2` runner can select it with `HM2_BASE_VARIANT=frontier_pr1416`
- the active bootstrap-handoff patch family is still aligned to `current_local`
- so the frontier base is present and selectable, but not yet the default active handoff substrate

That is intentional for now:

- `hm2` needed a real frontier base file available
- but the stage also needed to remain runnable immediately on the validated local patch surface
