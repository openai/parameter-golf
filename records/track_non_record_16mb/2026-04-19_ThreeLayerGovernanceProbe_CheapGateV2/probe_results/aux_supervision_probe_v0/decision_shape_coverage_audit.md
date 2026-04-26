# Decision-Shape Coverage Audit

- scope: `/Users/seryn/Documents/Parameter_Golf/repo/three_layer_aux_supervision_probe_v0/train_examples.jsonl`
- target under audit: `decision_shape_v1`
- question: is the current training-side decision-shape target fully supported by the available tiny seed set, or is class support the main bottleneck before step count becomes the primary suspect?

## Class Support

| class | count | sample_ids |
| --- | ---: | --- |
| `redirect` | 5 | `world_001`, `world_002`, `world_003`, `world_004`, `world_005` |
| `mark_only` | 1 | `world_006` |
| `no_action` | 0 | `-` |

## Zone Support

| zone | count | sample_ids |
| --- | ---: | --- |
| `error_zone` | 5 | `world_001`, `world_002`, `world_003`, `world_004`, `world_005` |
| `pre_error_zone` | 1 | `world_006` |

## Alignment With Current Useful Gate

Current `cheap_gate_v2_candidate` rows from `/Users/seryn/Documents/Parameter_Golf/repo/three_layer_usefulness_probe_v0/rows.jsonl` show:

| sample_id | selected_trigger | move | effect | audit |
| --- | --- | --- | --- | --- |
| `world_001` | `persist_2` | `go_back` | `stable_redirect` | `right_place` |
| `world_002` | `persist_2` | `go_back` | `stable_redirect` | `right_place` |
| `world_003` | `persist_2` | `where` | `stable_redirect` | `held` |
| `world_004` | `persist_2` | `leave_open` | `stable_redirect` | `right_place` |
| `world_005` | `persist_2` | `go_back` | `stable_redirect` | `right_place` |
| `world_006` | `low_top_gap` | `no` | `none` | `held` |

## Read

1. `decision_shape_v1` is not currently a real 3-way target.
   It is a skewed 2-way target with no observed `no_action` support.

2. The only non-redirect training example is `world_006`.
   That makes `mark_only` both rare and tied to a single boundary sample, which weakens its usefulness as a stable decision boundary.

3. The currently useful inference-time split is richer than the training-side target support.
   The gate already uses:
   - persistent error -> redirect
   - pre-error light anomaly -> no action / mark-only regime
   But the aux training set does not yet provide enough examples for the model to learn that full decision geometry.

## Conservative Conclusion

Primary bottleneck right now looks more like `class support / supervision geometry` than `step count`.

Step count may still matter, but on the current tiny seed set:
- `zone_classification_v0` is coarse but fully supported enough to teach a little general stability
- `decision_shape_v1` is closer to the useful gate boundary, but not fully expanded in data

So the next clean move is:
- first expand decision-shape class support in the tiniest possible way
- only then reconsider whether more steps deserve to be promoted from secondary suspect to primary suspect
