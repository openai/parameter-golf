#!/bin/bash
set -e

git push origin autoresearch/twopass

gh pr create \
  --base main \
  --head autoresearch/twopass \
  --title "Record: Packed N-gram + Two-Pass Dirichlet CTW — val_bpb 0.0830 (3-seed mean)" \
  --body-file records/track_10min_16mb/2026-03-27_PackedNgram_TwoPass_DirichletCTW/README.md
