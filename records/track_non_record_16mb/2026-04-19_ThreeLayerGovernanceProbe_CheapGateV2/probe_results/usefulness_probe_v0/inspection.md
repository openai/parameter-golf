# Very Small Usefulness Probe v0

- question: do governance-sensitive signals provide any measurable inference-time usefulness when used as a cheap gate, and can we reduce harm without losing the currently useful persist_2 signal?
- arms: baseline / cheap_gate_v0 / cheap_gate_v1a / cheap_gate_v1b / cheap_gate_v2_candidate
- gate triggers considered: persist_2, high_uncertainty, low_top_gap
- trigger selection priority: persist_2 > low_top_gap > high_uncertainty

## Aggregate Table

| arm | avg_before_mismatch | avg_after_mismatch | effect_dist | audit_dist | negative_flags |
| --- | ---: | ---: | --- | --- | --- |
| baseline | 6.83 | 6.83 | `{'none': 6}` | `{'missed': 5, 'held': 1}` | `-` |
| cheap_gate_v0 | 6.83 | 3.33 | `{'stable_redirect': 4, 'still_drifting': 1, 'none': 1}` | `{'right_place': 3, 'held': 2, 'too_much': 1}` | `{'error_zone_degraded': 1, 'pre_error_overgoverned': 1}` |
| cheap_gate_v1a | 6.83 | 1.67 | `{'stable_redirect': 5, 'none': 1}` | `{'right_place': 4, 'held': 2}` | `{'pre_error_overgoverned': 1}` |
| cheap_gate_v1b | 6.83 | 3.33 | `{'stable_redirect': 4, 'still_drifting': 1, 'none': 1}` | `{'right_place': 3, 'held': 2, 'too_much': 1}` | `{'error_zone_degraded': 1}` |
| cheap_gate_v2_candidate | 6.83 | 1.67 | `{'stable_redirect': 5, 'none': 1}` | `{'right_place': 4, 'held': 2}` | `-` |

## Cheap-Gate Read

### cheap_gate_v0
- improved_samples: `4/6`
- unchanged_samples: `1/6`
- worsened_samples: `1/6`

### cheap_gate_v1a
- improved_samples: `5/6`
- unchanged_samples: `1/6`
- worsened_samples: `0/6`

### cheap_gate_v1b
- improved_samples: `4/6`
- unchanged_samples: `1/6`
- worsened_samples: `1/6`

### cheap_gate_v2_candidate
- improved_samples: `5/6`
- unchanged_samples: `1/6`
- worsened_samples: `0/6`

## world_001

### baseline
- active_triggers: `['persist_2']`
- selected_trigger: `None`
- move: `no`
- before_mismatch -> after_mismatch: `6 -> 6`
- intervention_effect: `none`
- audit: `missed`
- post_monday_continuation: `The's. The's. The's to the following`

### cheap_gate_v0
- active_triggers: `['persist_2']`
- selected_trigger: `persist_2`
- move: `go_back`
- before_mismatch -> after_mismatch: `6 -> 0`
- intervention_effect: `stable_redirect`
- audit: `right_place`
- post_monday_continuation: `The next move changes.`

### cheap_gate_v1a
- active_triggers: `['persist_2']`
- selected_trigger: `persist_2`
- move: `go_back`
- before_mismatch -> after_mismatch: `6 -> 0`
- intervention_effect: `stable_redirect`
- audit: `right_place`
- post_monday_continuation: `The next move changes.`

### cheap_gate_v1b
- active_triggers: `['persist_2']`
- selected_trigger: `persist_2`
- move: `go_back`
- before_mismatch -> after_mismatch: `6 -> 0`
- intervention_effect: `stable_redirect`
- audit: `right_place`
- post_monday_continuation: `The next move changes.`

### cheap_gate_v2_candidate
- active_triggers: `['persist_2']`
- selected_trigger: `persist_2`
- move: `go_back`
- before_mismatch -> after_mismatch: `6 -> 0`
- intervention_effect: `stable_redirect`
- audit: `right_place`
- post_monday_continuation: `The next move changes.`

## world_002

### baseline
- active_triggers: `['persist_2']`
- selected_trigger: `None`
- move: `no`
- before_mismatch -> after_mismatch: `6 -> 6`
- intervention_effect: `none`
- audit: `missed`
- post_monday_continuation: `Then's to the following to the following to the`

### cheap_gate_v0
- active_triggers: `['persist_2']`
- selected_trigger: `persist_2`
- move: `go_back`
- before_mismatch -> after_mismatch: `6 -> 0`
- intervention_effect: `stable_redirect`
- audit: `right_place`
- post_monday_continuation: `The order is reversed.`

### cheap_gate_v1a
- active_triggers: `['persist_2']`
- selected_trigger: `persist_2`
- move: `go_back`
- before_mismatch -> after_mismatch: `6 -> 0`
- intervention_effect: `stable_redirect`
- audit: `right_place`
- post_monday_continuation: `The order is reversed.`

### cheap_gate_v1b
- active_triggers: `['persist_2']`
- selected_trigger: `persist_2`
- move: `go_back`
- before_mismatch -> after_mismatch: `6 -> 0`
- intervention_effect: `stable_redirect`
- audit: `right_place`
- post_monday_continuation: `The order is reversed.`

### cheap_gate_v2_candidate
- active_triggers: `['persist_2']`
- selected_trigger: `persist_2`
- move: `go_back`
- before_mismatch -> after_mismatch: `6 -> 0`
- intervention_effect: `stable_redirect`
- audit: `right_place`
- post_monday_continuation: `The order is reversed.`

## world_003

### baseline
- active_triggers: `['persist_2']`
- selected_trigger: `None`
- move: `no`
- before_mismatch -> after_mismatch: `6 -> 6`
- intervention_effect: `none`
- audit: `missed`
- post_monday_continuation: `Then's to the following to the following to the`

### cheap_gate_v0
- active_triggers: `['persist_2']`
- selected_trigger: `persist_2`
- move: `where`
- before_mismatch -> after_mismatch: `6 -> 0`
- intervention_effect: `stable_redirect`
- audit: `held`
- post_monday_continuation: `The lid fits the jar.`

### cheap_gate_v1a
- active_triggers: `['persist_2']`
- selected_trigger: `persist_2`
- move: `where`
- before_mismatch -> after_mismatch: `6 -> 0`
- intervention_effect: `stable_redirect`
- audit: `held`
- post_monday_continuation: `The lid fits the jar.`

### cheap_gate_v1b
- active_triggers: `['persist_2']`
- selected_trigger: `persist_2`
- move: `where`
- before_mismatch -> after_mismatch: `6 -> 0`
- intervention_effect: `stable_redirect`
- audit: `held`
- post_monday_continuation: `The lid fits the jar.`

### cheap_gate_v2_candidate
- active_triggers: `['persist_2']`
- selected_trigger: `persist_2`
- move: `where`
- before_mismatch -> after_mismatch: `6 -> 0`
- intervention_effect: `stable_redirect`
- audit: `held`
- post_monday_continuation: `The lid fits the jar.`

## world_004

### baseline
- active_triggers: `['persist_2']`
- selected_trigger: `None`
- move: `no`
- before_mismatch -> after_mismatch: `8 -> 8`
- intervention_effect: `none`
- audit: `missed`
- post_monday_continuation: `s. Then's to the following to the following`

### cheap_gate_v0
- active_triggers: `['persist_2']`
- selected_trigger: `persist_2`
- move: `too_fast`
- before_mismatch -> after_mismatch: `8 -> 10`
- intervention_effect: `still_drifting`
- audit: `too_much`
- post_monday_continuation: `It does not close yet.`
- negative_flags: `['error_zone_degraded']`

### cheap_gate_v1a
- active_triggers: `['persist_2']`
- selected_trigger: `persist_2`
- move: `leave_open`
- before_mismatch -> after_mismatch: `8 -> 0`
- intervention_effect: `stable_redirect`
- audit: `right_place`
- post_monday_continuation: `The next part stays open.`

### cheap_gate_v1b
- active_triggers: `['persist_2']`
- selected_trigger: `persist_2`
- move: `too_fast`
- before_mismatch -> after_mismatch: `8 -> 10`
- intervention_effect: `still_drifting`
- audit: `too_much`
- post_monday_continuation: `It does not close yet.`
- negative_flags: `['error_zone_degraded']`

### cheap_gate_v2_candidate
- active_triggers: `['persist_2']`
- selected_trigger: `persist_2`
- move: `leave_open`
- before_mismatch -> after_mismatch: `8 -> 0`
- intervention_effect: `stable_redirect`
- audit: `right_place`
- post_monday_continuation: `The next part stays open.`

## world_005

### baseline
- active_triggers: `['persist_2']`
- selected_trigger: `None`
- move: `no`
- before_mismatch -> after_mismatch: `5 -> 5`
- intervention_effect: `none`
- audit: `missed`
- post_monday_continuation: `Then's to the following to the following to the`

### cheap_gate_v0
- active_triggers: `['persist_2']`
- selected_trigger: `persist_2`
- move: `go_back`
- before_mismatch -> after_mismatch: `5 -> 0`
- intervention_effect: `stable_redirect`
- audit: `right_place`
- post_monday_continuation: `The relation reopened.`

### cheap_gate_v1a
- active_triggers: `['persist_2']`
- selected_trigger: `persist_2`
- move: `go_back`
- before_mismatch -> after_mismatch: `5 -> 0`
- intervention_effect: `stable_redirect`
- audit: `right_place`
- post_monday_continuation: `The relation reopened.`

### cheap_gate_v1b
- active_triggers: `['persist_2']`
- selected_trigger: `persist_2`
- move: `go_back`
- before_mismatch -> after_mismatch: `5 -> 0`
- intervention_effect: `stable_redirect`
- audit: `right_place`
- post_monday_continuation: `The relation reopened.`

### cheap_gate_v2_candidate
- active_triggers: `['persist_2']`
- selected_trigger: `persist_2`
- move: `go_back`
- before_mismatch -> after_mismatch: `5 -> 0`
- intervention_effect: `stable_redirect`
- audit: `right_place`
- post_monday_continuation: `The relation reopened.`

## world_006

### baseline
- active_triggers: `['high_uncertainty', 'low_top_gap']`
- selected_trigger: `None`
- move: `no`
- before_mismatch -> after_mismatch: `10 -> 10`
- intervention_effect: `none`
- audit: `held`
- post_monday_continuation: `Then's to the following to the following to the`

### cheap_gate_v0
- active_triggers: `['high_uncertainty', 'low_top_gap']`
- selected_trigger: `low_top_gap`
- move: `clarify_local_competition`
- before_mismatch -> after_mismatch: `10 -> 10`
- intervention_effect: `none`
- audit: `held`
- post_monday_continuation: `Then's to the following to the following to the`
- negative_flags: `['pre_error_overgoverned']`

### cheap_gate_v1a
- active_triggers: `['high_uncertainty', 'low_top_gap']`
- selected_trigger: `low_top_gap`
- move: `clarify_local_competition`
- before_mismatch -> after_mismatch: `10 -> 10`
- intervention_effect: `none`
- audit: `held`
- post_monday_continuation: `Then's to the following to the following to the`
- negative_flags: `['pre_error_overgoverned']`

### cheap_gate_v1b
- active_triggers: `['high_uncertainty', 'low_top_gap']`
- selected_trigger: `low_top_gap`
- move: `no`
- before_mismatch -> after_mismatch: `10 -> 10`
- intervention_effect: `none`
- audit: `held`
- post_monday_continuation: `Then's to the following to the following to the`

### cheap_gate_v2_candidate
- active_triggers: `['high_uncertainty', 'low_top_gap']`
- selected_trigger: `low_top_gap`
- move: `no`
- before_mismatch -> after_mismatch: `10 -> 10`
- intervention_effect: `none`
- audit: `held`
- post_monday_continuation: `Then's to the following to the following to the`
