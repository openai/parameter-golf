# Parameter Golf Research Program — Non-Record Core

## Objective
Find the best sub-16MB model family for longer training horizons, where quality-per-step and eventual convergence matter more than 10-minute throughput.

## Primary Principle
For non-record runs, slower but better-per-step models can be worthwhile if they continue improving and stay under the byte cap.

## What We Know
- The 12x448 family improves strongly with longer training.
- Its current limitations are export damage and byte overage, not purely optimization horizon.

## Priority Order
1. Keep the strong deeper family under the byte cap
2. Reduce export/quantization damage
3. Improve final convergence quality
4. Then consider longer unlimited-compute runs

## Preferred Directions
- Byte-recovery changes on strong deeper families
- Export-aware improvements
- Eval-time improvements that transfer to stronger cores

## Avoid
- Re-optimizing purely for 10-minute throughput
- Running much longer before recovering byte margin or export quality

## Guidance
- One conceptual change per experiment
- Use longer runs only when the family is already close on bytes and post-export BPB
