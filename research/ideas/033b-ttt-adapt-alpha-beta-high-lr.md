# Idea 033b — TTT alpha/beta adaptation with aggressive LR

## Thesis

`033` allowed TTT to adapt frozen `alpha/beta` on top of the same `026 seed_42`
checkpoint used by `028B`, but the effect was effectively negligible:

- `028B`: `1.0664948109`
- `033`: `1.0664878103`
- delta: about `-7e-06` bpb

That pattern is consistent with an underpowered adaptation:

- `recur_alpha` moved a little
- `recur_beta` did not move at measurable precision
- outcome barely changed

So the next cheap question is not architectural. It is simply:

- did `033` fail because `TTT_ALPHA_BETA_LR_SCALE=0.25` was too small?

## Mechanism

Keep everything from `033` the same:

- same checkpoint
- same hotstart path
- same TTT setup
- same LoRA warm-start behavior
- same code commit

Only change:

- `TTT_ALPHA_BETA_LR_SCALE=10.0`

Since the base `TTT_LORA_LR` in this codepath is `1e-4`, this gives an effective
alpha/beta LR of:

```text
1e-4 * 10.0 = 1e-3
```

## Why this is worth one run

`033` at `2.5e-5` was so conservative that it was close to a no-op.

An aggressive rerun answers the question quickly:

- if `beta` still does not move and result still does not improve, the line is
  probably exhausted
- if `alpha/beta` move materially and TTT improves, then `033` was just
  under-tuned
- if it destabilizes, we also learn that immediately

## Expected outcomes

### Positive

- `recur_beta_max_drift` becomes clearly nonzero
- post-TTT beats `033` by more than noise

### Null

- drift increases but result stays flat
- or `beta` still stays effectively frozen

### Negative

- TTT becomes noisy or regresses
- post-TTT degrades beyond `028A` territory

## Outcome

`033b` answered the question cleanly.

Observed parameter movement:

- `recur_alpha_max_drift = 0.240723`
- `recur_beta_max_drift = 0.062500`

So unlike `033`, both parameter sets moved materially under TTT:

- `alpha` moved a lot
- `beta` also moved meaningfully

But the final result got worse:

- `028B`: `1.0664948109`
- `033`: `1.0664878103`
- `033b`: `1.06666734`

So the aggressive LR did what it was supposed to do mechanically, but it hurt the
actual TTT outcome.

## Conclusion

This line now tells a coherent story:

- tiny alpha/beta adaptation is basically negligible
- aggressive alpha/beta adaptation is harmful

That means the flat `033` result was not simply because alpha/beta were impossible
to move. They are movable, but pushing them harder degrades quality.

## Recommendation

Do not promote TTT alpha/beta adaptation as a mainline lever.

If revisited at all, the only reasonable follow-ups are:

- `alpha`-only TTT adaptation with `beta` frozen
- or one medium-LR interpolation between `033` and `033b`

But this line should be treated as low priority now, not expanded into a broad
research branch.
