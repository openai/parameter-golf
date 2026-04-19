# Auxiliary Supervision Probe v1

- goal: test whether moving from coarse zone classification to a slightly more gate-shaped decision target improves the current usefulness panel
- target_families: `zone_classification_v0` / `decision_shape_v1`
- observed_zone_names: `['error_zone', 'pre_error_zone']`
- observed_decision_names: `['mark_only', 'redirect']`
- note: this tiny seed set currently exposes `error_zone` and `pre_error_zone`, and decision labels currently land on `mark_only` / `redirect` without an observed `no_action` case

## Variants

### train_baseline_minimal
- target_family: `none`
- final_train_lm_loss: `2.196310`
- final_train_aux_loss: `0.000000`
- baseline avg_after_mismatch: `6.67`
- cheap_gate_v2 avg_after_mismatch: `1.83`
- cheap_gate_v2 effect_dist: `{'stable_redirect': 5, 'none': 1}`
- cheap_gate_v2 audit_dist: `{'right_place': 4, 'held': 2}`
- cheap_gate_v2 negative_flags: `-`

### train_aux_probe_v0
- target_family: `zone_classification_v0`
- final_train_lm_loss: `2.205491`
- final_train_aux_loss: `1.025694`
- baseline avg_after_mismatch: `6.00`
- cheap_gate_v2 avg_after_mismatch: `1.83`
- cheap_gate_v2 effect_dist: `{'stable_redirect': 5, 'none': 1}`
- cheap_gate_v2 audit_dist: `{'right_place': 4, 'held': 2}`
- cheap_gate_v2 negative_flags: `-`

### train_aux_probe_v1
- target_family: `decision_shape_v1`
- final_train_lm_loss: `2.223755`
- final_train_aux_loss: `1.307001`
- baseline avg_after_mismatch: `6.33`
- cheap_gate_v2 avg_after_mismatch: `1.83`
- cheap_gate_v2 effect_dist: `{'stable_redirect': 5, 'none': 1}`
- cheap_gate_v2 audit_dist: `{'right_place': 4, 'held': 2}`
- cheap_gate_v2 negative_flags: `-`
