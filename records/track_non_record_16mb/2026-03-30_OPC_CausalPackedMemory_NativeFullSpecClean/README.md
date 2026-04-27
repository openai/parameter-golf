# OPC causal packed-memory legal packet stress test

This packet was rebuilt from scratch in a standalone workspace on top of the `open_predictive_coder` kernel.

- track: `track_non_record_16mb`
- run_id: `opc_native_tokens100000000_62021846`
- eval bits per token: `6.062992566022187`
- unigram bits per token: `8.649563865337807`
- bigram bits per token: `6.090255597305575`
- trigram bits per token: `7.22427841818739`
- train bits per token: `6.0696419226081915`
- mixture weights: `[0.0, 0.9, 0.1]`
- artifact bytes: `2705939`
- opc upstream: `https://github.com/asuramaya/open-predictive-coder`
- opc commit: `4072074288fa279b655c11c30f8fca2e1859f925`

Important scope note:

- this is a legal packet stress test and descendant rebuild
- it is not a leaderboard claim
- the model is an opc-native causal packed-memory descendant built in this workspace
