# this script is for merging a base model with LoRA checkpoints into a single model.(for other model)
mkdir -p "$OFFLOAD_DIR"

python merge.py --BASE_MODEL path/to/base/model \
  --LORA_DIR path/to/lora/directory \
  --OUTPUT_DIR path/to/output/directory \
  --DTYPE bf16 \
  --DEVICE_MAP auto \
  --OFFLOAD_DIR "$OFFLOAD_DIR" \
  --MAX_SHARD_SIZE 2GB
