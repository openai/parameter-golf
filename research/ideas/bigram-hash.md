# BigramHash Embeddings

**Status:** candidate
**Expected Δ:** +0.003 to +0.005
**Source:** One prior submission used a BigramHash 3072×112 table, claimed +0.005 bpb.

## Idea
Add a small learned hash table that maps **bigrams** (pairs of adjacent tokens) to a low-dim vector, added to the token embedding. Size: 3072 hash buckets × 112 dims ≈ 344K params.

Rationale: the model already sees bigram statistics implicitly through the first attention layer, but a direct embedding gives it a shortcut for frequent pairs, freeing attention capacity for longer-range structure.

## Why it might help
- Proven +0.005 in a prior submission; one of the larger known Δs left on the table.
- Parameter budget fits: 344K params at INT6 is ~260KB, within the 16MB limit even with the full 11L / 512d model.
- Simple: hash (token_t, token_{t+1}) → bucket → add to embedding(token_{t+1}).

## Code-change sketch
- Add `bigram_hash_buckets` and `bigram_hash_dim` to Hyperparameters.
- Allocate `nn.Embedding(buckets, dim)`.
- In the forward pass, compute `hash(input_ids[:, :-1], input_ids[:, 1:]) % buckets` (shift-aligned), look up, pad first position with zeros, add to token embeddings.
- Ensure it gets quantized along with other weights at submission time.

## Risks / open questions
- **Fit check** — does the 11L / 512d / MLP4× model + BigramHash actually fit under 16MB at INT6? Need to compute.
- Which hash function? Simple modular: `(t1 * P + t2) % buckets` where P is a prime. More sophisticated hashes aren't needed.
- Which dim? 112 is what the prior submission used. Smaller (e.g., 64) might fit better; larger might help more.
- Does GPTQ INT6 quantization handle the hash table well? Embeddings are often kept at higher precision (SOTA uses 8-bit embeds).

## If this works
Biggest single Δ among the candidates. But requires full retrain and careful budget management.
