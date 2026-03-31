This branch starts from PR 1179, with its `train_gpt.py` copied to the repo root for local iteration.
We then ported mixed-precision GPTQ from PR 1105 on top, and will append experiment results here as we run them.

On 2026-03-31, a 12-layer run with `N_INT6_LAYERS=10` finished at `final_int6_sliding_window_exact val_loss:1.87586457 val_bpb:1.11099560` with `Total submission size int6+brotli: 14512536` bytes. A follow-up 12-layer run with `N_INT6_LAYERS=20` improved to `final_int6_sliding_window_exact val_loss:1.87439804 val_bpb:1.11012704` with `Total submission size int6+brotli: 15022864` bytes, still comfortably under the 15.9 MB limit.
Also on 2026-03-31, a DR run with only repeated layer `5` MLP untied finished at `final_int8_zlib_roundtrip_exact val_loss:1.87141502 val_bpb:1.10836032` with `Total submission size int6+brotli: 15574562` bytes.
