Depth-recurrent transformer: 5 shared blocks looped 3x for 15 effective layers at dim=768 (baseline uses 9 layers at dim=512). GQA 12:6. 21.4M params, ~13.9MB compressed.

val_bpb: 1.2663 (4xH100 SXM, 2651 steps, 10min wallclock)
Baseline: 1.2244
Model still improving at cutoff, loss curve had not plateaued.

Next: tokenizer optimization (sp4096), width/depth sweep, test-time training, QAT.
