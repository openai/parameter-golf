# Exploration: The Research Journey

This directory contains the curated research journey for the param-golf competition, organized into chronological phases. Each phase represents a distinct research direction with its own goals, experiments, and findings. 

## Timeline

| Phase | Dates | Direction | Best BPB | Key Outcome |
|-------|-------|-----------|----------|-------------|
| 1 | Mar 17-22 | Baseline improvements | 1.1233 | Pushed upstream transformer to SOTA |
| 2 | Mar 23 | Multiskip connections | — | Established multiskip as viable modifier |
| 3 | Mar 23-24 | SSM exploration (Mamba/Griffin) | — | Dead end under 10-min constraint |
| 4 | Mar 25 | DFS, token injection, prefix state | — | Prefill ideas, DFS exploration |
| 5 | Mar 26 | Macro-sidechannel genesis | — | Birth of the core architectural idea |
| 6 | Mar 26 | Trans-hier evolution | ~1.18 | From concept to competitive BPB |
| 7 | Mar 27 | R-series alternatives | — | Breadth of architecture exploration |
| 8 | Mar 19-21 | SSD/Mamba compiled track | 1.3196 | Parallel SSD track, champion on 8xH100 |

## Reading Order

If you want to understand how we got here, read the phases in order. Each phase README explains what was tried, what worked, what didn't, and what motivated the next phase.

If you just want the results, look at `models/` for the final clean architectures.

i will try to updated and make it insightful and organized. 
