# BW16_DepthSweep_2k — Hypothesis

Goal: run a direct flat-depth sweep for `NUM_FLAT_LAYERS={6,7,8,9,10,11}` and measure delta against 6F control.

Setup:
- Tap-off Nightcrawler stack
- 2k steps per arm on 4 GPUs
- All non-depth knobs fixed

Primary question:
- Does depth beyond 6F continue to improve `int6_sw_bpb`, or is 6F already near the knee?
