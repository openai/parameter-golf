#1868 = clean/reproducible neural baseline.
#1877 = much better PPM attempt than #1873, but still very likely not record-valid unless it can answer the normalization objection.
The key issue is exactly the comment on #1877: if you evaluate the mixed NN+PPM probability for every possible token id at a token position, do those probabilities sum to 1? The commenter’s answer is “hint: no,” and that is the central red flag.

What #1868 gives you
Your #1868 is valuable because it is a pure reproduction of #1851, not a new trick. It reports:

seed 42:   1.06128183
seed 314:  1.06086831
seed 1234: 1.06220261
mean:      1.06145 ± 0.00068
with all artifacts under 16,000,000 bytes, training under 600s, and eval under 600s.

That should remain your clean record-track baseline. It has normal full-vocab token scoring, CaseOps/SP8192 lineage, score-first TTT, LQER, SmearGate BOS fix, and no PPM ambiguity. It is “boring in the best possible way.”

What #1877 is trying to do
PR #1877 claims:

SP8192 + order-6 byte-level PPM-D mixture
3-seed mean = 0.96255352 BPB
artifact max = 15,999,992 bytes
eval time ≈ 463–474s
It says the eval path computes normal sliding-window NN NLLs, converts the scored token stream into byte contributions, and mixes an NN byte probability with an order-6 byte PPM-D probability:

p_mix = λ p_nn + (1 - λ) p_ppm
It also claims prefix-only gating, score-before-update, full validation, and a valid byte distribution.

That is a much more serious attempt than the earlier PPM PRs because it at least says the right words: full validation, score-first, byte-level, no rescoring, no subset. But the normalization issue is still the shark in the bathtub.

PPM vs PPM-D
PPM means Prediction by Partial Matching: an adaptive context model that predicts the next symbol from a suffix of the already-seen symbol history. The order, such as order-6, means it first tries a context of length 6 and backs off to shorter contexts if needed. PPM models usually work over bytes/characters and use escape probabilities for unseen symbols.

PPM-D is not a magic legality variant. It is one family of escape/novel-symbol probability rules. In common descriptions, PPM-D estimates the probability of a new symbol using the number of unique symbols seen in a context relative to total observations.

So the compliance distinction is not:

PPM = illegal
PPM-D = legal
It is:

legal-ish:
  PPM or PPM-D defines a normalized distribution over the scored alphabet
  using only already-scored prefix symbols
  then updates after scoring

invalid:
  PPM or PPM-D computes only the realized-symbol probability
  or mixes byte probabilities with token probabilities on incompatible sample spaces
  or updates before scoring
  or uses future validation bytes
PPM-D only tells you how escape mass is assigned inside the byte PPM model. It does not by itself fix the token/byte mixture problem.

The core mathematical problem
There are two different sample spaces floating around:

1. Token-level NN:
   p_nn(token_id | token_prefix)
   sums to 1 over vocab, e.g. 8192 SP tokens.

2. Byte-level PPM:
   p_ppm(byte | byte_prefix)
   sums to 1 over 256 bytes.
You cannot just take the probability of the realized token’s byte string under PPM and mix it with the NN token probability unless the PPM side has been normalized over the same token alphabet.



The invalid-looking version is:

q_ppm(target_token) = Π_i p_ppm(byte_i(target_token) | prefix + earlier bytes of target_token)

p_mix(target_token)
  = λ p_nn(target_token)
  + (1 - λ) q_ppm(target_token)
That scores the target token, but it does not define:

p_mix(v) for every token v
with Σ_v p_mix(v) = 1
And that is exactly what the #1877 comment is poking at.

The field guide’s “meaningful val_bpb” conditions are directly relevant here: before scoring, the submission must define a full normalized distribution over the official token alphabet, score before update, and run a single left-to-right pass. It also explicitly warns that mixing over latent/internal structures rather than the actual alphabet breaks the metric.

Why byte-string probabilities do not automatically normalize over tokens
Suppose the token vocabulary contains:

token A = "a"
token B = "ab"
A byte PPM model might assign:

q("a")  = P(a) = 0.8
q("ab") = P(a) P(b | a) = 0.8 × 0.8 = 0.64
Then:

q("a") + q("ab") = 1.44
before considering every other token.

That does not mean PPM is wrong. It means q(byte string) is a sequence probability, not automatically a probability distribution over token IDs. To mix with a token LM, you must either normalize over token IDs or convert the token LM to a proper byte distribution.

Two valid paths to legal PPM
There are really only two clean designs.

Path A — Token-level normalized PPM mixture
This is the safest if the contest expects an official token alphabet.

At each token position t, before seeing the target token x_t:

p_nn_t(v) = neural softmax over all token ids v
Then compute a PPM byte-string score for every token id:

q_t(v) = P_PPM(bytes(v) | byte_history_before_t)
Then normalize:

Z_t = Σ_v q_t(v)

p_ppm_t(v) = q_t(v) / Z_t
Then mix:

p_mix_t(v)
  = λ_t p_nn_t(v)
  + (1 - λ_t) p_ppm_t(v)
Then score:

loss_t = -log2 p_mix_t(x_t)
Then, and only then, update PPM:

byte_history ← byte_history + bytes(x_t)
This is legal-looking because:

Σ_v p_mix_t(v) = 1
and the distribution exists before the target token is used.

Critical details
λ_t must be prefix-only. It can depend on things like:

context length available
PPM entropy
max_v p_ppm_t(v)
number of observations in the longest context
It must not depend on:

p_ppm_t(target)
whether PPM guessed the target
post-hoc improvement over NN
any current-token property unavailable before scoring
BPB calculation
For token-level scoring:

BPB =
  Σ_t -log2 p_mix_t(x_t)
  /
  Σ_t original_byte_count(x_t)
The denominator must be actual original UTF-8 bytes, not transformed bytes, not hardcoded bytes/token. For SentencePiece, that means handling ▁, byte fallback tokens, BOS/EOS/control tokens, and CaseOps sidecars correctly. The field guide explicitly calls out byte fallback, leading-space ▁, boundary tokens, and the danger of hardcoded bytes/token ratios.

Path B — Pure byte-level predictor
This is mathematically cleanest for PPM, but may be less obviously aligned with the token-alphabet interpretation.

At each original byte position i, define:

p_ppm_i(b) over b ∈ {0..255}
If you mix with a neural model, the neural side must also define a proper 256-way byte distribution:

p_nn_i(b) over b ∈ {0..255}
Then:

p_mix_i(b)
  = λ_i p_nn_i(b)
  + (1 - λ_i) p_ppm_i(b)
and:

Σ_b p_mix_i(b) = 1
Then score the current byte, update after scoring:

loss_i = -log2 p_mix_i(byte_i)
PPM.update(byte_i)
BPB is:

BPB =
  Σ_i -log2 p_mix_i(byte_i)
  /
  number_of_original_utf8_bytes
This is extremely clean if both distributions are truly over 256 bytes.



The hard part is the neural model. A token LM does not automatically give a byte distribution. You need a token-trie/transducer marginalization:

p_nn(first byte = b)
  = Σ_{v: bytes(v) starts with b} p_nn(v)
Inside a multi-byte or multi-character token, you need to condition on the observed byte prefix of the latent token:

p_nn(next byte = c | observed token-byte prefix u)
  =
  Σ_{v: bytes(v) starts with u+c} p_nn(v)
  /
  Σ_{v: bytes(v) starts with u} p_nn(v)
This is doable, but it must be exact. “Spread token logprob over its bytes” is not a normalized byte model.

Where #1877 likely sits
#1877 claims byte-level PPM-D and says it mixes NN byte probability with PPM byte probability.

The unanswered question is:

Is the NN byte probability a real normalized 256-way byte distribution at each byte position, or is it derived from the realized token’s probability after the fact?

If it is the latter, #1877 is likely invalid for the same reason as #1873: it scores the realized path but does not define a normalized distribution over all alternatives.

The public comment on #1877 strongly suggests reviewers see the same issue: the token-wise probabilities induced by the byte-wise PPM and token-wise NN mixture do not sum to 1.

What would make #1877 defensible
For #1877-style PPM-D to become defensible, I would require these exact audits.

1. Normalization audit
Token-level version:

for t in sampled_positions:
    p_nn = softmax(logits[t])                    # [V]
    q = torch.empty(V)

    for v in range(V):
        q[v] = ppm_sequence_prob(token_bytes[v], state_before_t)

    p_ppm = q / q.sum()
    p_mix = lam * p_nn + (1 - lam) * p_ppm

    assert abs(p_mix.sum().item() - 1.0) < 1e-6
Byte-level version:

for i in sampled_byte_positions:
    p_nn_byte = neural_byte_dist(state_i)        # [256]
    p_ppm_byte = ppm_byte_dist(state_i)          # [256]
    p_mix = lam * p_nn_byte + (1 - lam) * p_ppm_byte

    assert abs(p_mix.sum().item() - 1.0) < 1e-6
If they cannot run one of those audits, the reported BPB is not provably meaningful.

2. Score-before-update trace
Log:

position,state_hash_before_score,loss,state_hash_after_score,state_hash_after_update
Require:

state_hash_before_score == state_hash_after_score
state changes only after scoring
no byte/token is rescored
3. Distributed correctness audit
This killed several earlier PPM designs. PPM is sequential. If eval is sharded across 8 ranks, you must ensure the PPM state represents the exact canonical prefix.

Acceptable options:

Option 1:
  gather all scored token/byte data to rank 0 in canonical order,
  run PPM sequentially once.

Option 2:
  pass exact PPM state between contiguous segments,
  not rank-local independent PPM.

Option 3:
  run the entire PPM scoring in a deterministic single-rank postpass
  over gathered NN logprobs and byte stream.
Required audit:

world_size=1 PPM BPB == world_size=8 PPM BPB
canonical_byte_index strictly increases
no missing positions
no rank-local reordering
4. Byte denominator audit
For every scoring mode:

total_scored_bytes == total_original_utf8_bytes
Special cases:

SentencePiece ▁
byte fallback
BOS/EOS/control tokens
CaseOps operators
private-use sentinels
document boundary bytes
For #1868/CaseOps this is already conceptually handled with sidecars. For PPM, the exact same original-byte stream must feed the PPM model.

Is eval-time PPM “training on eval data”?
Yes, but it can be legal under the adaptive compression interpretation if it is score-first. The field guide explicitly distinguishes fixed predictors from adaptive compression, and says mechanisms updated from previously scored eval tokens can be permitted if they obey causality, score-before-update, and single-pass rules. It even lists causal n-gram caches that accumulate only from already-scored tokens as permitted under that adaptive track.

So the legal objection is not simply:

PPM uses eval tokens, therefore illegal.
The real question is:

At position t, was the distribution fully defined before x_t?
Did it sum to 1 over the official alphabet?
Was the state updated only after x_t was scored?
Was the stream processed exactly once in order?
Was BPB divided by original bytes?
If yes, legal-ish. If no, invalid.

Difference between valid PPM and invalid PPM in one table
Question

Valid PPM

Invalid PPM

Alphabet

Explicitly tokens or explicitly bytes

Token NN mixed with byte PPM without conversion

Distribution

Sums to 1 before target is known

Only computes target probability

Update

After scoring current byte/token

Before scoring or within realized token in a token-level event

Gate

Prefix-only

Depends on whether target was predicted well

Distributed eval

Canonical left-to-right stream

Rank-local / missing rank gather

BPB denominator

Original UTF-8 bytes

Transformed bytes, token count, or hardcoded ratio

CaseOps/casefold

Lossless or sidecar-accounted

Lossy casefold or unaccounted sentinels

Review status

Defensible

Likely rejected

The cleanest formulation I’d recommend
Given the deadline and your #1868 baseline, I would not try to turn PPM into a record PR immediately. I’d create a non-record “legal PPM audit” first.

Non-record legal PPM spec
Name:
  Legal token-normalized PPM-D mixture over SP8192

Base:
  #1868 / #1851 neural baseline

At each token t:
  1. compute p_nn over all SP8192 token ids
  2. compute q_ppm(v) for every token v using PPM-D over canonical original bytes
  3. normalize q_ppm over all token ids
  4. mix distributions
  5. score target token
  6. update PPM with target token bytes
Formula:

q_t(v) = P_PPMD(bytes(v) | H_t)

p_ppm_t(v) = q_t(v) / Σ_u q_t(u)

p_mix_t(v) = λ_t p_nn_t(v) + (1 - λ_t) p_ppm_t(v)

loss_t = -log2 p_mix_t(x_t)

BPB = Σ_t loss_t / Σ_t original_bytes(x_t)
This directly answers the #1877 reviewer objection.

Practical note: this may be too slow
Computing q_t(v) for every vocab token at every position is expensive:

8192 token candidates × token bytes × validation tokens
You can optimize with:

byte trie over vocab
batched candidate byte scoring
prefix-state caching
C/CUDA implementation
candidate pruning only if you still compute exact residual mass
But pruning is dangerous: if you compute only top-k token probabilities and ignore the remaining Z, you are back to an invalid score unless you have an exact bound/aggregation for the omitted mass.

What about pure byte PPM-D as a record?
Pure byte PPM-D is mathematically easier:

for byte i:
  p_ppm_i(·) is a 256-way normalized distribution
  score byte_i
  update with byte_i
This gives a valid compression code. The question is whether the contest review process will accept a submission that does not define probabilities over the official token alphabet. The field guide’s Condition 2 says “official fixed token alphabet Σ,” which points toward token-normalized scoring.



So I’d treat pure byte PPM-D as:

excellent non-record science
possibly defensible if maintainers explicitly accept byte alphabet scoring
not the safest record-track move without clarification
My verdict on #1877
#1877 is an important PR because it clarifies the right battlefield. It fixes some earlier obvious sins:

full validation
score-before-update claim
order-6 byte PPM-D
3 seeds
artifact/time under cap
But unless it can prove either:

A. normalized token distribution over all SP8192 token ids
or:

B. normalized byte distribution for both PPM and neural model at every byte position
then I would not consider the 0.96255 BPB record-valid. The public reviewer comment is basically the shortest possible proof obligation: show that the distribution sums to 1 over the relevant alphabet, or the number is not meaningful.

Recommended next steps for you
With your #1868 already clean:

Leave #1868 as the record-track baseline. It is reproducible, auditable, and not entangled with PPM risk.
Do not graft #1877-style PPM onto #1868 unless you can implement normalization. A naive mixture will get a gorgeous number and then get shot out of the sky.
Write a short non-record PR/spec titled something like:
Non-record: What a legally normalized PPM-D mixture would require
Include two audits:
token-sum audit:
  Σ_v p_mix(v) = 1

byte-sum audit:
  Σ_b p_mix(b) = 1
Ask maintainers/community explicitly:
Is a pure byte-alphabet adaptive predictor acceptable for record track,
or must the full normalized distribution be over token ids?
That one answer determines whether Path B is viable for record.



Bottom line: PPM-D can be legal, but only if the scored distribution is normalized over the same alphabet used by the metric and updated score-first. PR #1877 is closer than #1873, but the token/byte normalization gap is still  fatal unless fixed.

