#!/bin/bash
# Parse experiment results — extract final BPB scores
echo "=== PARAMETER GOLF RESULTS ==="
echo ""
printf "%-30s %12s %12s %12s\n" "EXPERIMENT" "POST_EMA_BPB" "INT6_SW_BPB" "LEGAL_TTT"
printf "%-30s %12s %12s %12s\n" "----------" "------------" "-----------" "---------"

for log in logs/*.log; do
    name=$(basename "$log" .log)
    ema_bpb=$(grep "DIAGNOSTIC post_ema" "$log" | grep -oP 'val_bpb:\K[0-9.]+' | tail -1)
    sw_bpb=$(grep "final_int6_sliding_window_exact" "$log" | grep -oP 'val_bpb:\K[0-9.]+' | tail -1)
    ttt_bpb=$(grep "legal_ttt_exact" "$log" | grep -oP 'val_bpb:\K[0-9.]+' | tail -1)
    printf "%-30s %12s %12s %12s\n" "$name" "${ema_bpb:-N/A}" "${sw_bpb:-N/A}" "${ttt_bpb:-N/A}"
done

echo ""
echo "CURRENT SOTA: 1.1194"
echo "TARGET: < 1.1144 (need 0.005 improvement with p < 0.01)"
