#!/usr/bin/env bash
# 快速檢查 LoRA 配置是否正確

echo "🔍 檢查 LoRA 部署配置..."
echo ""

# 從 slurm 腳本中提取配置
BASE_MODEL_PATH="/home/at0842/aaronwu901225master.ai13/gpt-oss-20b"

declare -a LORA_ADAPTERS=(
  "lr1e5-ckpt50|/home/at0842/aaronwu901225master.ai13/gorilla/berkeley-function-call-leaderboard/merge_lora_model/gpt-oss-20b-lora-1epoch-all-zhtw-lr1e-5/checkpoint-50|openai/gpt-oss-20b-checkpoint-50-1epoch-all-zhtw-lr1e-5"
  "lr1e5-ckpt100|/home/at0842/aaronwu901225master.ai13/gorilla/berkeley-function-call-leaderboard/merge_lora_model/gpt-oss-20b-lora-1epoch-all-zhtw-lr1e-5/checkpoint-100|openai/gpt-oss-20b-checkpoint-100-1epoch-all-zhtw-lr1e-5"
  "lr1e5-ckpt149|/home/at0842/aaronwu901225master.ai13/gorilla/berkeley-function-call-leaderboard/merge_lora_model/gpt-oss-20b-lora-1epoch-all-zhtw-lr1e-5/checkpoint-149|openai/gpt-oss-20b-checkpoint-149-1epoch-all-zhtw-lr1e-5"
  "lr5e7-ckpt50|/home/at0842/aaronwu901225master.ai13/gorilla/berkeley-function-call-leaderboard/merge_lora_model/gpt-oss-20b-lora-1epoch-all-zhtw-lr5e-7/checkpoint-50|openai/gpt-oss-20b-checkpoint-50-1epoch-all-zhtw-lr5e-7"
  "lr5e7-ckpt100|/home/at0842/aaronwu901225master.ai13/gorilla/berkeley-function-call-leaderboard/merge_lora_model/gpt-oss-20b-lora-1epoch-all-zhtw-lr5e-7/checkpoint-100|openai/gpt-oss-20b-checkpoint-100-1epoch-all-zhtw-lr5e-7"
  "lr5e7-ckpt149|/home/at0842/aaronwu901225master.ai13/gorilla/berkeley-function-call-leaderboard/merge_lora_model/gpt-oss-20b-lora-1epoch-all-zhtw-lr5e-7/checkpoint-149|openai/gpt-oss-20b-checkpoint-149-1epoch-all-zhtw-lr5e-7"
)

# 檢查 Base Model
echo "📦 檢查 Base Model..."
if [ -d "${BASE_MODEL_PATH}" ]; then
  echo "✅ Base model 存在: ${BASE_MODEL_PATH}"
  # 檢查必要文件
  if [ -f "${BASE_MODEL_PATH}/config.json" ]; then
    echo "   ✅ config.json 存在"
  else
    echo "   ⚠️  config.json 不存在"
  fi
  if [ -f "${BASE_MODEL_PATH}/tokenizer_config.json" ] || [ -f "${BASE_MODEL_PATH}/tokenizer.json" ]; then
    echo "   ✅ tokenizer 存在"
  else
    echo "   ⚠️  tokenizer 不存在"
  fi
else
  echo "❌ Base model 不存在: ${BASE_MODEL_PATH}"
  echo "   請修改 bfcl-gen-gpt-oss.slurm 中的 BASE_MODEL_PATH"
fi

echo ""
echo "🔗 檢查 LoRA Adapters..."
VALID_COUNT=0
INVALID_COUNT=0

for i in "${!LORA_ADAPTERS[@]}"; do
  ADAPTER_ENTRY="${LORA_ADAPTERS[$i]}"
  ADAPTER_NAME="${ADAPTER_ENTRY%%|*}"
  TEMP="${ADAPTER_ENTRY#*|}"
  ADAPTER_PATH="${TEMP%%|*}"
  SERVED_NAME="${TEMP##*|}"
  
  echo ""
  echo "[$((i+1))/${#LORA_ADAPTERS[@]}] ${ADAPTER_NAME}"
  echo "    Path: ${ADAPTER_PATH}"
  echo "    Served as: ${SERVED_NAME}"
  
  if [ -d "${ADAPTER_PATH}" ]; then
    echo "    ✅ 路徑存在"
    
    # 檢查 adapter 文件
    if [ -f "${ADAPTER_PATH}/adapter_config.json" ]; then
      echo "    ✅ adapter_config.json 存在"
      
      # 讀取 LoRA rank
      if command -v python &> /dev/null; then
        RANK=$(python -c "import json; print(json.load(open('${ADAPTER_PATH}/adapter_config.json'))['r'])" 2>/dev/null)
        if [ -n "$RANK" ]; then
          echo "    📊 LoRA rank: ${RANK}"
        fi
      fi
    else
      echo "    ⚠️  adapter_config.json 不存在"
    fi
    
    if [ -f "${ADAPTER_PATH}/adapter_model.safetensors" ] || [ -f "${ADAPTER_PATH}/adapter_model.bin" ]; then
      echo "    ✅ adapter weights 存在"
    else
      echo "    ⚠️  adapter weights 不存在"
    fi
    
    VALID_COUNT=$((VALID_COUNT + 1))
  else
    echo "    ❌ 路徑不存在"
    INVALID_COUNT=$((INVALID_COUNT + 1))
  fi
done

echo ""
echo "=========================================="
echo "📊 統計結果"
echo "=========================================="
echo "總計: ${#LORA_ADAPTERS[@]} 個 adapters"
echo "✅ 有效: ${VALID_COUNT}"
echo "❌ 無效: ${INVALID_COUNT}"

if [ $INVALID_COUNT -gt 0 ]; then
  echo ""
  echo "⚠️  發現 ${INVALID_COUNT} 個無效的 adapter 配置"
  echo "   請檢查路徑是否正確"
  exit 1
fi

echo ""
echo "✅ 所有配置檢查通過！"
echo "🚀 可以執行: sbatch bfcl-gen-gpt-oss.slurm"
