// Top-K extraction from a logits row, plus a count of "plausible" tokens
// (prob ≥ probThresh) so the popover can render "+N more options".
//
// Single linear pass over V values: maintain a sorted insertion list of length
// ≤ k, and tally tokens above threshold. For V=1024 (FineWeb tokenizer) and
// V≈49k (SmolLM2) this is microseconds per token. We work in log-prob space
// to avoid float underflow on flat distributions.

// Convert a row of logits into {topk, aboveThresh, lse}.
//   row         — Float32Array | Float64Array of length V (raw logits)
//   V           — vocab size
//   k           — how many top entries to keep
//   probThresh  — count tokens with prob ≥ probThresh into aboveThresh
//
// topk is sorted descending by logit (== descending by prob). Each entry is
// { id, lp } where lp is the log-prob (logit − lse), so callers can render
// either prob (=Math.exp(lp)) or bits (=-lp/Math.log(2)) without recomputing.
export function topKFromLogits(row, V, k, probThresh) {
  let m = -Infinity;
  for (let i = 0; i < V; i++) if (row[i] > m) m = row[i];
  let s = 0;
  for (let i = 0; i < V; i++) s += Math.exp(row[i] - m);
  const lse = m + Math.log(s);

  const lpThresh = Math.log(Math.max(probThresh, Number.MIN_VALUE));
  const top = [];   // sorted desc by lp
  let aboveThresh = 0;
  for (let i = 0; i < V; i++) {
    const lp = row[i] - lse;
    if (lp >= lpThresh) aboveThresh++;
    if (top.length < k || lp > top[top.length - 1].lp) {
      let pos = top.length;
      while (pos > 0 && top[pos - 1].lp < lp) pos--;
      top.splice(pos, 0, { id: i, lp });
      if (top.length > k) top.pop();
    }
  }
  return { topk: top, aboveThresh, lse };
}

// 1-indexed rank of `targetId` in the row (1 = best). Used to label the
// actual / chosen token in the popover when it's outside the top-K.
export function rankOfId(row, V, targetId) {
  const target = row[targetId];
  let r = 1;
  for (let i = 0; i < V; i++) if (row[i] > target) r++;
  return r;
}
