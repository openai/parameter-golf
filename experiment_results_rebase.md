## Quick note

- Disabling layer-0 attention looks safe and saves time. Its learned `attn_scale` was only about `0.08`, versus roughly `0.4-0.8` in later layers.
- Increasing depth still helped with layer-0 attention removed, which supports the conclusion that extra layers are still buying useful capacity.
- Logged metrics for the layer-0-attention-disabled run:
  `DIAGNOSTIC post_ema val_loss:1.9102 val_bpb:1.1313 eval_time:2140ms`
  `final_int8_zlib_roundtrip_exact val_loss:1.87718387 val_bpb:1.11177697`
- New baseline comparison:
  `DIAGNOSTIC post_ema val_loss:1.9146 val_bpb:1.1340 eval_time:2062ms`
  `final_int8_zlib_roundtrip_exact val_loss:1.88156874 val_bpb:1.11437394`
- Relative to the new baseline, disabling layer-0 attention improved post-EMA by `0.0027` bpb and final roundtrip by `0.0026` bpb.
- Final recurrence rebase result:
  `DIAGNOSTIC post_ema val_loss:1.9117 val_bpb:1.1322 eval_time:2306ms`
  `final_int8_zlib_roundtrip_exact val_loss:1.88039677 val_bpb:1.11367983`
- Relative to the fresh baseline, this is `0.0018` bpb better post-EMA and `0.0007` bpb better on the final roundtrip metric.
- Full untie of the repeated layer-4 MLP improved further:
  `DIAGNOSTIC post_ema val_loss:1.9077 val_bpb:1.1299 eval_time:2310ms`
  `final_int8_zlib_roundtrip_exact val_loss:1.87680398 val_bpb:1.11155197`
- Relative to the fresh baseline, this is `0.0041` bpb better post-EMA and `0.0028` bpb better on the final roundtrip metric.
- Relative to the previous recurrence rebase result, this is another `0.0023` bpb better post-EMA and `0.0021` bpb better on the final roundtrip metric.
