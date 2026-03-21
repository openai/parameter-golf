#!/bin/bash
# Usage: ./launch_experiment.sh <exp_number> <exp_name> <instance_id> <port> <key_file> <hypothesis> <script> <env_overrides...>
# Example: ./launch_experiment.sh 008 mtp_normalized_softcap15 0 31531 /tmp/thunder_keys/instance_0.pem "Test normalized MTP" train_gpt.py "LOGIT_SOFTCAP=15 ADAM_EPS=1e-10 MTP_ENABLED=1"
#
# This script:
# 1. Creates experiments/<NNN>_<name>.sh with the run command
# 2. Creates experiments/<NNN>_<name>.md with the hypothesis
# 3. Copies the .py script to experiments/<NNN>_<script>
# 4. Uploads the script to the remote instance
# 5. Launches the experiment via nohup

EXP_NUM=$1
EXP_NAME=$2
INSTANCE_ID=$3
PORT=$4
KEY_FILE=$5
HYPOTHESIS=$6
SCRIPT=$7
shift 7
ENV_OVERRIDES="$@"

RUN_ID="exp${EXP_NUM}_${EXP_NAME}_2k"
WANDB_KEY="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"
IP="185.216.20.240"
DIR="$(cd "$(dirname "$0")/.." && pwd)"
EXP_DIR="${DIR}/experiments"

echo "=== Launching Experiment ${EXP_NUM}: ${EXP_NAME} ==="

# 1. Save the .sh run script
cat > "${EXP_DIR}/${EXP_NUM}_${EXP_NAME}_2k.sh" << RUNEOF
#!/bin/bash
# Experiment ${EXP_NUM}: ${EXP_NAME}
cd /home/ubuntu/parameter-golf
export WANDB_API_KEY=${WANDB_KEY}
export WANDB_PROJECT=parameter-golf
export CUDA_VISIBLE_DEVICES=0
export ITERATIONS=2000
export MAX_WALLCLOCK_SECONDS=0
export VAL_LOSS_EVERY=500
export TRAIN_LOG_EVERY=100
export RUN_ID=${RUN_ID}
$(for env in ${ENV_OVERRIDES}; do echo "export ${env}"; done)
python3 ${SCRIPT}
RUNEOF

# 2. Save the .md experiment doc
cat > "${EXP_DIR}/${EXP_NUM}_${EXP_NAME}_2k.md" << MDEOF
# Experiment ${EXP_NUM}: ${EXP_NAME}

## Status: RUNNING on Instance ${INSTANCE_ID} (wandb: parameter-golf / ${RUN_ID})

## Hypothesis
${HYPOTHESIS}

## Configuration
- **Script**: ${SCRIPT}
- **Env overrides**: ${ENV_OVERRIDES}
- **wandb run**: ${RUN_ID}

## Results
*Awaiting...*
MDEOF

# 3. Copy the .py script
cp "${DIR}/${SCRIPT}" "${EXP_DIR}/${EXP_NUM}_${SCRIPT}"

echo "Saved: ${EXP_DIR}/${EXP_NUM}_${EXP_NAME}_2k.{sh,md} and ${EXP_NUM}_${SCRIPT}"
