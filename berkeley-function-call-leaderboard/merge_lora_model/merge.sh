#!/usr/bin/env bash
#SBATCH -J BFCL-merge-lora
#SBATCH -p h200q
#SBATCH --gres=gpu:gpu:1
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH --time=30:00
#SBATCH -o %x-%j.out

# 進 bash 後，載入 conda hook
source /cm/shared/apps/anaconda/2024.02/etc/profile.d/conda.sh

# 重新 activate
conda deactivate 2>/dev/null || true
conda deactivate 2>/dev/null || true
conda activate BFCL

OFFLOAD_DIR="/tmp/merge_offload_${SLURM_JOB_ID}"
mkdir -p "$OFFLOAD_DIR"

# python merge.py --BASE_MODEL gemma-4-31B-it \
#   --LORA_DIR gemma4_31B_zhtw_lr5e-7_ep5_64_128_256_turn_v6_no_cot_only_miss_turn_upload_ver \
#   --OUTPUT_DIR gemma-4-31B-it-lora-5epoch-zhtw-v6-no-cot-only-miss-turn-lr5e-7 \
#   --DTYPE bf16 \
#   --DEVICE_MAP auto \
#   --OFFLOAD_DIR "$OFFLOAD_DIR" \
#   --MAX_SHARD_SIZE 2GB

# python merge.py --BASE_MODEL gemma-4-31B-it \
#   --LORA_DIR gemma4_31B_zhtw_lr5e-7_ep5_64_128_256_turn_v6_no_cot_methodB_upload_ver \
#   --OUTPUT_DIR gemma-4-31B-it-lora-5epoch-zhtw-v6-no-cot-B-method-64-128-256-lr5e-7 \
#   --DTYPE bf16 \
#   --DEVICE_MAP auto \
#   --OFFLOAD_DIR "$OFFLOAD_DIR" \
#   --MAX_SHARD_SIZE 2GB

# python merge.py --BASE_MODEL gemma-4-31B-it \
#   --LORA_DIR gemma4_31B_zhtw_lr5e-7_ep5_64_128_256_turn_v6_no_cot_methodC_upload_ver \
#   --OUTPUT_DIR gemma-4-31B-it-lora-5epoch-zhtw-v6-no-cot-C-method-64-128-256-lr5e-7 \
#   --DTYPE bf16 \
#   --DEVICE_MAP auto \
#   --OFFLOAD_DIR "$OFFLOAD_DIR" \
#   --MAX_SHARD_SIZE 2GB

python merge.py --BASE_MODEL gemma-4-31B-it \
  --LORA_DIR gemma4_31B_zhtw_lr5e-7_ep5_64_128_256_turn_v6_no_cot_methodA_upload_ver_40 \
  --OUTPUT_DIR gemma-4-31B-it-lora-5epoch-zhtw-v6-no-cot-A-method-64-128-256-lr5e-7 \
  --DTYPE bf16 \
  --DEVICE_MAP auto \
  --OFFLOAD_DIR "$OFFLOAD_DIR" \
  --MAX_SHARD_SIZE 2GB


#python merge.py --BASE_MODEL Llama-xLAM-2-8b-fc-r \
#    --LORA_DIR xlam_8B_all_zhtw_lr1e-5_ep1_16_32 \
#    --OUTPUT_DIR Llama-xLAM-2-8b-fc-r-lora-1epoch-all-zhtw-lr1e-5-16-32

#python merge.py --BASE_MODEL Qwen2.5-7B-Instruct \
#    --LORA_DIR Qwen2_5_7B_all_zhtw_turn_lr1e-5_ep1_16_32 \
#    --OUTPUT_DIR Qwen2.5-7B-Instruct-lora-1epoch-all-zhtw-split-turn-lr1e-5

#python merge.py --BASE_MODEL Qwen2.5-7B-Instruct \
#    --LORA_DIR Qwen2_5_7B_all_zhtw_turn_lr5e-7_ep1_16_64 \
#    --OUTPUT_DIR Qwen2.5-7B-Instruct-lora-1epoch-all-zhtw-split-turn-lr5e-7

#python merge.py --BASE_MODEL Llama-3.1-8B-Instruct \
#    --LORA_DIR llama3_1_8B_all_zhtw_lr1e-5_ep1_16_32_128_turn_tokenize \
#    --OUTPUT_DIR Llama-3.1-8B-Instruct-lora-1epoch-all-zhtw-lr1e-5-16-32-128-turn-tokenize

#python merge.py --BASE_MODEL Llama-3.1-8B-Instruct \
#    --LORA_DIR llama3_1_8B_all_zhtw_turn_lr1e-5_ep1_16_32_128 \
#    --OUTPUT_DIR Llama-3.1-8B-Instruct-lora-1epoch-all-zhtw-split-turn-lr1e-5-16-32-128
