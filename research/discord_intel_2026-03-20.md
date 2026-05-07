# Discord Intelligence — 2026-03-20 4:11pm EDT

## Key Discussion: TTT Exploit Potential (sam_acqua / Larry / ZeroTheory)

### Sam_acqua (TTT author) flagging potential exploit:
> "Wait so to clarify: if the validation set is 1M tokens, then you can train on the first 999,999 tokens to predict the last 1? That seems a bit broken, that if you change the size of your validation set, then your loss ~in expectation~ changes. I would suggest instead that gradient based test-time procedures be limited to ~within the document~. That way, the expectation of the val loss is independent of # sequences in the eval set."

**Translation:** Cross-document TTT training could artificially decrease val loss. Current implementation restricts TTT to **within-document only** — which is fair and prevents this exploit.

### Larry's insight on paid prefix loophole:
> "The trained model is only saturated under the 16mb limit. If u can add parameters during the eval period, then you'd want to do extended pre-training on Val data while adding more parameters. It's not rly TTT, just working around 16mb bottleneck."

**Translation:** Paid prefix (PR #262, #168) is a **loophole exploit** — storing validation tokens as compressed blob in artifact bypasses the 16MB model limit. Not a real technique, might get disqualified.

### ZeroTheory (competitor):
> "Haha damn I've now fallen victim to telling the creator of the code to read his code. I see what you are saying from your clarification. I suppose as long as the model is able to actually learn and generalize we expect Val to decrease but otherwise it might not. So it seems fair game to me"

Seems confused about exploit vs. legitimate TTT.

## Other Intel

### Competition Fatigue (sam_acqua):
> "Also props to the maintainers, looking at the PRs, every new idea is repeated like 5 times in other PRs, I'm sure it's hard to parse"

**Confirmation:** Most PRs are copycats. Original ideas matter.

### Training Data Rules (ZeroTheory):
> "ah okay specifically it said you cannot edit the training data or change order. Okay that's different. 👍"

**Rule:** Cannot filter, edit, or reorder training data.

### Library Clarification (aegis):
> "was there any clarification about whether additional libraries (e.g. triton) are allowed?"

**Open question** — no response captured. Triton is allowed in repo but OOMs on 4090.

### Timeline Confirmation (Major):
> "hi! :hello: the challenge runs until April 30th"

**Official:** 41 days remaining.

## Strategic Takeaways

1. **Paid prefix is risky** — might get disqualified as loophole exploit
2. **Cross-document TTT is banned** — within-document only
3. **Full-model SGD TTT is legit** — PR #264 approach (2 epochs on val, lr=0.002)
4. **Copycat fatigue is real** — maintainers drowning in duplicate PRs
5. **Original ideas win** — don't just fork #198, add novel contributions

## Next Steps
- **Avoid paid prefix** — don't waste compute on a technique that might get banned
- **Focus on mixed Int5/Int6** — proven in PR #264
- **Research full-model SGD TTT** — beats LoRA TTT by ~0.005 bpb
- **Test SOTA base (PR #198)** on 4090 — see if it even runs with our constraints
