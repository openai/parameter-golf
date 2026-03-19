# rec_validate_compression_recipe_warmdown

1. Hypothesis

Recombine primary=port_compression_20260318_210204_9293 (compression) secondary=port_training_recipe_20260318_204700_9293 conflict_policy=primary

2. Expected upside

3. Expected risk

4. Exact knobs changed

5. Promotion bar

6. Scheduler Metadata

family: compression
operator: recombine
template_id: None
parent_candidate_id: port_compression_20260318_210204_9293
env_overrides: {'LOGIT_SOFTCAP': '20.0', 'WARMDOWN_ITERS': '256'}
config_hash: 455370c77acd914850a28d5a55cf93c2ad19f5817d6c64db2960dcd8f5d5ffd3
mutation_hash: e70823719290156d05884bad9a4ed9c67cc86a62defffa582d824b4d817cf106
