# Chimera + LeakyReLU(0.5)² + 2048 Context + Extended Warmdown

**Estimated val_bpb: ~0.5588** (see note below)

## Summary

This submission builds on the `chimera_submission` SOTA baseline:

**LeakyReLU(0.5)²**: Replaced `ReLU(x)²` activation in all MLP blocks with `LeakyReLU(x, 0.5)²`. This prevents the dead-neuron problem while preserving non-negative squared outputs. Shown in prior records to get ~0.0025 BPB improvement 
**Context Length 2048*: Increased `TRAIN_SEQ_LEN` and `TTT_EVAL_SEQ_LEN` from 1024 → 2048. Doubles the contextual horizon during both training and TTT evaluation.
**BigramHash Capacity 3072**: Expanded `BigramHashEmbedding` from 2048 → 3072 buckets reducing hash collisions.
**Warmdown 3500**: Extended `WARMDOWN_ITERS` from 3000 → 3500 to allow for more gradual learning rate decay before the quantization-aware export.

All other architecture and optimization hyperparameters unchanged (K-projection LoRA TTT, Min-NLL epoch selection, 8-epoch TTT, FA3, Muon optimizer, SWA, INT6 quantization).

## ⚠️ Note on bpb Estimate!

The RunPod instance was preempted by runpod in the final seconds of the evaluation phase, immediately after the LoRA TTT completed and before the `final_ttt_lora` line was printed. The training run itself completed fully and correctly (see `train.log`), and the `Size check PASSED` and `final_int8_zlib_roundtrip` lines confirm the model serialized correctly within the 16MB limit.

The reported `val_bpb: 0.5588` is an **estimate** derived from:
- The final `avg_loss=0.9435` at batch 60/61 (the last TTT batch logged before the crash)
- The known bytes-per-token ratio: `final_int8_zlib_roundtrip val_bpb:1.1722` / `val_loss:1.9793`

This submission is offered transparently for review. The authors intend to rerun the experiment and update this entry with a verified `final_ttt_lora_exact` line as soon as credits are available. (Iam broke currently) Since this might take a while, i offer to put this one in the Notable Non-Record Runs section (i really worked long on this so i hope i can see this somewhere)

## Hardware & Timing

- Instance: RunPod 8×H100 SXM (80GB HBM3 each)
- Training time: 600,082ms (10.00 minutes, wallclock-capped)
- Steps completed: 6,586 / 20,000
- TTT evaluation: 652,788ms (~10.9 minutes, timed out at batch 61/61 with 17 long docs base-scored)
- Total artifact size: 15,302,060 bytes (95% of 16MB limit)

## Key Training Metrics


- step:6586/20000 val_bpb:1.1586 (pre-quant, pre-TTT)
- final_int8_zlib_roundtrip val_bpb:1.1722 (post INT6 quant)
- quant_gap: 0.013429 BPB
- final_ttt_lora val_bpb:~0.5588 (estimated — pod crashed before print)
