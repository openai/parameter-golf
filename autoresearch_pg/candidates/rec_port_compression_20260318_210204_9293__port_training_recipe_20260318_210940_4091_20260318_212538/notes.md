# rec_port_compression_20260318_210204_9293__port_training_recipe_20260318_210940_4091_20260318_212538

1. Hypothesis

Recombine primary=port_compression_20260318_210204_9293 (compression) secondary=port_training_recipe_20260318_210940_4091 conflict_policy=primary

2. Expected upside

3. Expected risk

4. Exact knobs changed

5. Promotion bar

6. Scheduler Metadata

family: compression
operator: recombine
template_id: None
parent_candidate_id: port_compression_20260318_210204_9293
env_overrides: {'LOGIT_SOFTCAP': '20.0', 'MATRIX_LR': '0.05', 'SCALAR_LR': '0.04', 'WARMDOWN_ITERS': '256', 'WARMUP_STEPS': '64'}
config_hash: 0eb14e2cf7fed8a5bf26721651d22e34ccc5fca588564a2a2c5f81686bf8847e
mutation_hash: 862fbd522ce0fd03d5ea98808e9a08539211ba501919b6fb1b79900f10668523
