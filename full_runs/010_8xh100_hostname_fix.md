# Experiment 010: 8xH100 with hostname monkey-patch + device_id restored

## Status: RUNNING

## What's different from 009
- Added Python monkey-patch to resolve container hostname → IP
- Restored `device_id=device` in `init_process_group`
- Added `NCCL_IB_DISABLE=1` to match baseline
- Step time: **67.5ms/step** — same as without the fix (68ms)

The `device_id` removal was NOT the source of the overhead. The 67ms vs baseline 43ms gap (~58% slower) must be from something else — possibly:
- Thunder Compute H100 PCIe vs baseline's NVLink topology
- Different CUDA/driver versions
- torch.compile cache differences

## Results
*Running — will update when complete*
