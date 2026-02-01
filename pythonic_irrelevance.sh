#!/usr/bin/env bash
#################自動email#####################
set -euo pipefail

#####################################
# 使用者設定
#####################################
EMAIL="aaronwu901225main@gmail.com"
JOB_NAME="pythonic-irrelevance-job"
LOG_FILE="pythonic_irrelevance_output.log"

#####################################
# 開始資訊
#####################################
START_TIME=$(date -Is)
PID=$$
PGID=$(ps -o pgid= $$ | tr -d ' ')

echo "[job] started" >> "$LOG_FILE"
echo "[job] job_name=$JOB_NAME pid=$PID pgid=$PGID time=$START_TIME" >> "$LOG_FILE"

#####################################
# 確保「不管成功或失敗」都會寄信
#####################################
finish() {
    EXIT_CODE=$?
    END_TIME=$(date -Is)

    if [ "$EXIT_CODE" -eq 0 ]; then
        STATUS="SUCCESS"
    else
        STATUS="FAILED (exit code=$EXIT_CODE)"
    fi

    {
        echo "[job] finished"
        echo "[job] status=$STATUS"
        echo "[job] start=$START_TIME"
        echo "[job] end=$END_TIME"
        echo "[job] pid=$PID pgid=$PGID"
    } >> "$LOG_FILE"

    SUBJECT="[$JOB_NAME] $STATUS"
    BODY=$(cat <<EOF
Job name : $JOB_NAME
Status   : $STATUS
PID      : $PID
PGID     : $PGID
Start    : $START_TIME
End      : $END_TIME

Last 50 lines of log:
--------------------
$(tail -n 50 "$LOG_FILE")
EOF
)

    echo "$BODY" | mail -s "$SUBJECT" "$EMAIL"
}

trap finish EXIT
#####################################################

#######################主程式#########################
# ============================================
# 語言設定：
# LANG_CODE: 控制生成資料的語言
#   - "en" (預設): 英文版
#   - "zh_tw": 繁體中文版
# 範例: LANG_CODE=zh_tw bash pythonic_irrelevance.sh
# ============================================
export LANG_CODE="zh_tw"

# ============================================
# Irrelevance 生成設定：
# IRR_NUM_SAMPLES: 每個 curriculum row 生成幾個 irrelevance 樣本 (預設: 2)
# IRR_LIMIT_ROWS: 限制處理的 curriculum rows 數量 (可選，用於測試)
# PARALLEL_WORKERS: 平行處理工作線程數 (預設: 5)
# MAX_RETRIES: API 呼叫最大重試次數 (預設: 3)
# IRR_DEBUG: 是否啟用除錯模式 ("1" 啟用)
# CLEAN_INCREMENTAL: 是否清除增量檔案重新開始 ("1" 清除)
#
# 範例:
#   IRR_NUM_SAMPLES=3 bash pythonic_irrelevance.sh
#   IRR_LIMIT_ROWS=10 IRR_DEBUG=1 bash pythonic_irrelevance.sh
#   LANG_CODE=zh_tw IRR_NUM_SAMPLES=5 bash pythonic_irrelevance.sh
# ============================================
export IRR_NUM_SAMPLES=20
export PARALLEL_WORKERS=10
export MAX_RETRIES=3
export IRR_DEBUG=1
export CLEAN_INCREMENTAL="${CLEAN_INCREMENTAL:-0}"

echo "=========================================="
echo "  Pythonic Irrelevance Data Generator"
echo "  LANG_CODE: $LANG_CODE"
echo "  IRR_NUM_SAMPLES: $IRR_NUM_SAMPLES"
echo "  PARALLEL_WORKERS: $PARALLEL_WORKERS"
echo "  MAX_RETRIES: $MAX_RETRIES"
echo "  IRR_DEBUG: $IRR_DEBUG"
echo "  CLEAN_INCREMENTAL: $CLEAN_INCREMENTAL"
echo "=========================================="

# ============================================
# 可選：限制處理的 curriculum rows（用於測試）
# ============================================
# 單行測試用############
#export IRR_LIMIT_ROWS=5
#######################

# ============================================
# OpenAI API 設定
# ============================================
export OPENAI_API_KEYS="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,\
                        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,\
                        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,\
                        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,\
                        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,\
                        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,\
                        stop"
export API_DAILY_LIMIT_TOKENS=2500000
export API_ROTATE_MARGIN=25000
export API_ROTATE_VERBOSE=1

# ============================================
# Curriculum 設定
# ============================================
export CURRICULUM_CSV="${CURRICULUM_CSV:-pipeline/data/curriculum.csv}"

# ============================================
# 清除增量檔案（如果 CLEAN_INCREMENTAL=1）
# ============================================
clean_incremental_files() {
  local run_id="$1"
  local data_dir="pipeline/data/$run_id/irrelevance"
  if [ -d "$data_dir" ]; then
    echo "Cleaning incremental files in $data_dir..."
    find "$data_dir" -name "*.incr.jsonl" -delete
    echo "Incremental files cleaned."
  fi
}

# ============================================
# 主要生成流程
# ============================================

# 檢查是否需要清除增量檔案
if [ -f "run_id_irrelevance" ] && [ "$CLEAN_INCREMENTAL" = "1" ]; then
  RUN_ID=$(cat run_id_irrelevance)
  clean_incremental_files "$RUN_ID"
  rm -f run_id_irrelevance
  echo "Cleaned previous run, starting fresh."
fi

# 執行 irrelevance 資料生成
echo "Generating irrelevance data..."
python run_irrelevance_openai.py

# 取得 run_id
RUN_ID=$(cat run_id_irrelevance)
echo "Run ID: $RUN_ID"

# ============================================
# 轉換為最終格式
# ============================================

# 根據語言設定決定輸出檔名
if [ "$LANG_CODE" = "zh_tw" ]; then
  OUTPUT_BASE="irrelevance_zh_tw"
else
  OUTPUT_BASE="irrelevance_eng"
fi

OUTPUT_DIR="pipeline/data/$RUN_ID/irrelevance"

echo "Converting irrelevance data to final format..."
python pipeline/tools/convert_to_irrelevance.py \
  --input "${OUTPUT_DIR}/irrelevance.json" \
  --output "${OUTPUT_DIR}/${OUTPUT_BASE}.jsonl"

# 驗證輸出
if [ -f "${OUTPUT_DIR}/${OUTPUT_BASE}.jsonl" ]; then
  SAMPLE_COUNT=$(wc -l < "${OUTPUT_DIR}/${OUTPUT_BASE}.jsonl")
  echo "Generated ${SAMPLE_COUNT} irrelevance samples."
  
  # 驗證資料格式
  echo "Validating irrelevance data..."
  python pipeline/tools/validate_irrelevance.py "${OUTPUT_DIR}/${OUTPUT_BASE}.jsonl"
  
  # 計算 token 數量（可選）
  if [ -f "pipeline/tools/countoken.py" ]; then
    echo "Counting tokens..."
    python pipeline/tools/countoken.py \
      --input "${OUTPUT_DIR}/${OUTPUT_BASE}.jsonl" \
      --bin_size 100
  fi
else
  echo "Warning: Output file not found: ${OUTPUT_DIR}/${OUTPUT_BASE}.jsonl"
fi

# ============================================
# 完成
# ============================================
echo "=========================================="
echo "  Irrelevance Data Generation Complete!"
echo "  Run ID: $RUN_ID"
echo "  Output: ${OUTPUT_DIR}/${OUTPUT_BASE}.jsonl"
echo "  Language: $LANG_CODE"
echo "=========================================="
