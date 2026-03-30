#!/usr/bin/env bash
conda activate BFCL
set -euo pipefail

#JOB1_SCRIPT="bfcl-gen.slurm"
JOB1_SCRIPT="bfcl-gen-gptoss-merged.slurm"
#JOB1_SCRIPT="bfcl-gen-mistral-merged.slurm"

# 當前面有相依問題時可用下面這行測試
job1_id=$(sbatch --parsable --dependency=afterok:48417 "$JOB1_SCRIPT")
#job1_id=$(sbatch --parsable "$JOB1_SCRIPT")
echo "Submitted job1: $job1_id"

# 等 job1 結束（不在 squeue 裡表示已結束）
while squeue -j "$job1_id" -h >/dev/null 2>&1 && squeue -j "$job1_id" -h | grep -q .; do
  echo "[wait] job1 ($job1_id) still running/pending..."
  sleep 20
done

# 檢查 job1 最終狀態（COMPLETED 才繼續）
state=$(sacct -j "${job1_id}.batch" --format=State -n | head -n 1 | awk '{print $1}')
echo "[done] job1 state: $state"

if [[ "$state" == "COMPLETED" ]]; then
  sh bfcl-eval.sh
else
  echo "job1 not completed, skip bfcl-eval.sh"
  exit 1
fi
