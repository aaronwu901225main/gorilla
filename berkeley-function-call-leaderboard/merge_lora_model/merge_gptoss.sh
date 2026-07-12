# this merge script is for merging multiple LoRA checkpoints into a single GPT-OSS model.
BASE_MODEL="path/to/your/gpt-oss-20b"

# LoRA checkpoints 和輸出目錄
declare -A LORA_CONFIGS=(
  # 格式: ["LORA_DIR"]="OUTPUT_DIR_NAME"
  # 添加更多配置...
)

# 輸出精度（bfloat16 推薦用於 vLLM Flash Attention 3）
OUTPUT_DTYPE="bfloat16"

# 工作目錄
WORK_DIR=""
cd "${WORK_DIR}"

echo "📋 環境檢查"
echo "------------------------------------------"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import peft; print(f'PEFT: {peft.__version__}')"
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
if python -c "import torch; torch.cuda.is_available()" 2>/dev/null; then
  python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
  python -c "import torch; print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')"
fi
echo ""

echo "📦 檢查 Base Model"
echo "------------------------------------------"
if [ -d "${BASE_MODEL}" ]; then
  echo "✅ Base model 存在: ${BASE_MODEL}"
  
  # 檢查量化狀態
  if [ -f "${BASE_MODEL}/config.json" ]; then
    QUANT_METHOD=$(python -c "import json; c=json.load(open('${BASE_MODEL}/config.json')); print(c.get('quantization_config', {}).get('quant_method', 'none'))" 2>/dev/null || echo "unknown")
    echo "   量化方法: ${QUANT_METHOD}"
    
    if [ "${QUANT_METHOD}" != "none" ] && [ "${QUANT_METHOD}" != "unknown" ]; then
      echo ""
      echo "⚠️  警告: Base model 使用了量化 (${QUANT_METHOD})"
      echo "   合併可能會遇到問題。建議使用非量化的 base model。"
      echo "   如果只有量化版本，將嘗試反量化後合併。"
      echo ""
    fi
  fi
else
  echo "❌ Base model 不存在: ${BASE_MODEL}"
  exit 1
fi
echo ""

echo "🔄 開始處理 LoRA Checkpoints"
echo "=========================================="

TOTAL=${#LORA_CONFIGS[@]}
CURRENT=0
SUCCESS=0
FAILED=0

for LORA_DIR in "${!LORA_CONFIGS[@]}"; do
  OUTPUT_NAME="${LORA_CONFIGS[$LORA_DIR]}"
  CURRENT=$((CURRENT + 1))
  
  echo ""
  echo "=========================================="
  echo "[${CURRENT}/${TOTAL}] Processing: ${LORA_DIR}"
  echo "=========================================="
  
  # 檢查 LoRA 目錄
  if [ ! -d "${WORK_DIR}/${LORA_DIR}" ]; then
    echo "⚠️  LoRA 目錄不存在: ${LORA_DIR}"
    echo "⏭️  跳過..."
    FAILED=$((FAILED + 1))
    continue
  fi
  
  # 執行合併
  if python merge_gptoss.py \
    --base-model "${BASE_MODEL}" \
    --lora-dir "${WORK_DIR}/${LORA_DIR}" \
    --output-dir "${WORK_DIR}/${OUTPUT_NAME}" \
    --dtype "${OUTPUT_DTYPE}" \
    --trust-remote-code \
    --skip-existing; then
    
    echo "✅ 成功: ${LORA_DIR} -> ${OUTPUT_NAME}"
    SUCCESS=$((SUCCESS + 1))
  else
    echo "❌ 失敗: ${LORA_DIR}"
    FAILED=$((FAILED + 1))
  fi
done

echo ""
echo "=========================================="
echo "📊 處理結果總結"
echo "=========================================="
echo "總計: ${TOTAL}"
echo "✅ 成功: ${SUCCESS}"
echo "❌ 失敗: ${FAILED}"
echo ""
echo "🎉 合併任務完成！"
echo "輸出目錄: ${WORK_DIR}"
echo ""

# 列出生成的模型
echo "📁 生成的模型:"
for LORA_DIR in "${!LORA_CONFIGS[@]}"; do
  OUTPUT_NAME="${LORA_CONFIGS[$LORA_DIR]}"
  OUTPUT_PATH="${WORK_DIR}/${OUTPUT_NAME}"
  
  if [ -d "${OUTPUT_PATH}" ]; then
    # 統計 checkpoint 數量
    CKPT_COUNT=$(find "${OUTPUT_PATH}" -maxdepth 1 -type d -name "*_merged" | wc -l)
    echo "   ${OUTPUT_NAME}: ${CKPT_COUNT} checkpoints"
  fi
done

if [ ${FAILED} -gt 0 ]; then
  echo ""
  echo "⚠️  有 ${FAILED} 個配置處理失敗，請檢查日誌。"
  exit 1
fi
