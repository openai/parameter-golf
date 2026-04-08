#!/bin/bash
ssh winlaptop 'powershell -Command "New-Item -ItemType Directory -Force -Path C:\Users\Public\parameter-golf\data\datasets\fineweb10B_sp1024"'
# Push tokenizer model (single file, small)
scp /Users/akhileshgogikar/parameter-golf/records/track_10min_16mb/2026-03-25_Ternary_Feedback_TTT/fineweb_8192_bpe.model winlaptop:C:/Users/Public/parameter-golf/records/track_10min_16mb/2026-03-25_Ternary_Feedback_TTT/
scp /Users/akhileshgogikar/parameter-golf/records/track_10min_16mb/2026-03-25_Ternary_Feedback_TTT/run_hybrid_win.bat winlaptop:C:/Users/Public/parameter-golf/records/track_10min_16mb/2026-03-25_Ternary_Feedback_TTT/
# Push dataset (324MB)
scp /Users/akhileshgogikar/parameter-golf/data/datasets/fineweb10B_sp1024/* winlaptop:C:/Users/Public/parameter-golf/data/datasets/fineweb10B_sp1024/

echo "Data transfer complete."
# Run benchmark
ssh winlaptop 'cmd.exe /c "cd C:\Users\Public\parameter-golf\records\track_10min_16mb\2026-03-25_Ternary_Feedback_TTT && run_hybrid_win.bat"' > win_ssh.out 2>&1 &
echo "Benchmark launched."
