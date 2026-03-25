# Approach 5 Run Summary

## Result

- hardware: `1x H100 SXM 80GB`
- run type: `non-record unlimited-compute rerun`
- final sliding-window exact `val_bpb`: `1.15315937`
- final sliding-window exact `val_loss`: `1.94705614`
- final sliding eval time: `951907ms`

## Training Milestones

- wallclock stop: `4800619ms`
- stop checkpoint: `step 4121/20000`
- stop checkpoint `val_bpb`: `1.1631`
- post-EMA `val_bpb`: `1.1629`
- int6 roundtrip exact `val_bpb`: `1.17697562`

## Byte Audit

- serialized model: `110820611`
- serialized model int6+zstd: `15848577`
- code size: `75257`
- total submission size int6+zstd: `15923834`
- margin to official cap: `76166`

## Operational Closeout

- pod id: `vb8q3bp4ey2gu2`
- pod deleted promptly after log retrieval
- `pod list --all` returned `[]`
