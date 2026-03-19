# rep_port_training_recipe_20260318_210940_4091_20260318_212445

1. Hypothesis

Replication of candidate port_training_recipe_20260318_210940_4091 from smoke_mlx_local.

2. Expected upside

3. Expected risk

4. Exact knobs changed

5. Promotion bar

6. Scheduler Metadata

family: training_recipe
operator: replicate
template_id: recipe_fast_decay
parent_candidate_id: port_training_recipe_20260318_210940_4091
env_overrides: {'MATRIX_LR': '0.05', 'SCALAR_LR': '0.04', 'WARMDOWN_ITERS': '256', 'WARMUP_STEPS': '64'}
config_hash: 5aa0acaf8374ff8b3e85241889b3e4b4ae827a6cb2319658ebc670aeb5f0abe0
mutation_hash: 35cebf0da85e6e58a7fa823808359a5f7b9d1670a7af6a54707586163b3b5075
