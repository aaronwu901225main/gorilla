#!/usr/bin/env bash
conda activate BFCL
set -euo pipefail

# JOB1_SCRIPT="bfcl-gen.slurm"
JOB1_SCRIPT="bfcl-gen-gptoss-merged-1.slurm"
JOB2_SCRIPT="bfcl-gen-gptoss-merged-2.slurm"
# JOB3_SCRIPT="bfcl-gen-gptoss-merged-3.slurm"
# JOB4_SCRIPT="bfcl-gen-gptoss-merged-4.slurm"
# JOB5_SCRIPT="bfcl-gen-gptoss-merged-5.slurm"

# JOB1_SCRIPT="bfcl-gen-gemma4-fc-1.slurm"
# JOB2_SCRIPT="bfcl-gen-gemma4-fc-2.slurm"
#JOB3_SCRIPT="bfcl-gen-gemma4-fc-3.slurm"
#JOB4_SCRIPT="bfcl-gen-gemma4-fc-4.slurm"

#JOB1_SCRIPT="bfcl-gen-mistral-merged.slurm"

JOB_SCRIPTS=(
  "$JOB1_SCRIPT"
  "$JOB2_SCRIPT"
  # "$JOB3_SCRIPT"
  # "$JOB4_SCRIPT"
  # "$JOB5_SCRIPT"
)

job_ids=()

# 同時送出所有 jobs
for i in "${!JOB_SCRIPTS[@]}"; do
  script="${JOB_SCRIPTS[$i]}"
  job_id=$(sbatch --parsable "$script")
  job_ids+=("$job_id")
  echo "Submitted job$((i + 1)): $job_id ($script)"
done

all_completed=true

# 等每個 job 結束，並檢查最終狀態
for i in "${!job_ids[@]}"; do
  job_id="${job_ids[$i]}"

  while squeue -j "$job_id" -h >/dev/null 2>&1 && squeue -j "$job_id" -h | grep -q .; do
    echo "[wait] job$((i + 1)) ($job_id) still running/pending..."
    sleep 20
  done

  state=$(sacct -j "${job_id}.batch" --format=State -n | head -n 1 | awk '{print $1}')
  echo "[done] job$((i + 1)) ($job_id) state: $state"

  if [[ "$state" != "COMPLETED" ]]; then
    all_completed=false
  fi
done

if [[ "$all_completed" == "true" ]]; then
  sh bfcl-eval.sh
else
  echo "Not all jobs completed successfully, skip bfcl-eval.sh"
  exit 1
fi
