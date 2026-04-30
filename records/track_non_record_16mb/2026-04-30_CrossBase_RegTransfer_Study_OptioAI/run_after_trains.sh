#!/usr/bin/env bash
# Auto-runs after EmbStudy_aos completes: extract hidden states, run quant matrix, build figures.
set -e
ROOT=/workspace/parameter-golf
LOG=/tmp/embstudy_analysis.log

echo "[$(date -u +%H:%M:%SZ)] watcher start" | tee -a "$LOG"

# Wait for AOS to be in consumed (meaning daemon completed it)
while ! grep -q "EmbStudy_aos" "${ROOT}/parameter-golf/auto/combo_consumed.txt" 2>/dev/null; do
  sleep 30
done
echo "[$(date -u +%H:%M:%SZ)] EmbStudy_aos consumed; checking final_model.pt files exist" | tee -a "$LOG"

# Wait for the final_model.pt to be stable (not still being written)
sleep 30
for reg_dir in nosimctg baseline qahsp es hsu aos; do
  pt="${ROOT}/candidate_pack/N18_baseA_${reg_dir}/final_model.pt"
  if [ ! -f "$pt" ]; then
    echo "[$(date -u +%H:%M:%SZ)] WARN: missing $pt" | tee -a "$LOG"
  else
    sz=$(stat -c%s "$pt")
    echo "[$(date -u +%H:%M:%SZ)] OK   $reg_dir final_model.pt = $sz bytes" | tee -a "$LOG"
  fi
done

echo "[$(date -u +%H:%M:%SZ)] running run_reg_quant_matrix.py" | tee -a "$LOG"
cd "${ROOT}"
python3 submissions/C_CrossBase_RegTransfer_Study/run_reg_quant_matrix.py 2>&1 | tee -a "$LOG"

echo "[$(date -u +%H:%M:%SZ)] running build_synergy_figures.py" | tee -a "$LOG"
python3 submissions/C_CrossBase_RegTransfer_Study/build_synergy_figures.py 2>&1 | tee -a "$LOG"

echo "[$(date -u +%H:%M:%SZ)] done. results in submissions/C_CrossBase_RegTransfer_Study/" | tee -a "$LOG"
