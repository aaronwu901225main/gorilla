# Quick Start Guide - LAB5 Function Calling Leaderboard (ZHTW)

## 快速開始指南 (Traditional Chinese Quick Start)

此指南提供最常用操作的快速參考。

### 基本用法

#### 1. 使用預設優化設定運行評估 (Recommended)

```bash
python openfunctions_evaluation.py \
    --model "gpt-4o-2024-05-13-FC" \
    --test-category all \
    --language zhtw
```

#### 2. 在 H200 GPU 上運行大型本地模型

```bash
python openfunctions_evaluation.py \
    --model "meta-llama/Meta-Llama-3-70B-Instruct" \
    --test-category all \
    --language zhtw \
    --gpu-memory-utilization 0.95 \
    --num-gpus 1
```

#### 3. 快速測試（僅簡單類別）

```bash
python openfunctions_evaluation.py \
    --model "gpt-3.5-turbo-0125-FC" \
    --test-category simple \
    --language zhtw
```

#### 4. 批次評估多個模型

```bash
python openfunctions_evaluation.py \
    --model "gpt-4o-2024-05-13-FC" "claude-3-5-sonnet-20240620-FC" \
    --test-category all \
    --language zhtw
```

### 參數說明

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--model` | gpt-3.5-turbo-0125-FC | 要評估的模型名稱 |
| `--test-category` | all | 測試類別 (simple, parallel_function, etc.) |
| `--language` | zhtw | 語言 (zhtw=繁體中文, en=英文) |
| `--temperature` | 自動 | 溫度參數 (建議 0.1) |
| `--top-p` | 自動 | Top-p 參數 (建議 0.9-0.95) |
| `--max-tokens` | 自動 | 最大 token 數 (建議 1500-2000) |
| `--gpu-memory-utilization` | 0.95 | GPU 記憶體使用率 (H200: 0.95) |
| `--optimize-accuracy` | True | 使用準確度優化設定 |

### 測試類別

- `all` - 全部測試類別（推薦完整評估）
- `simple` - 簡單函數呼叫
- `parallel_function` - 並行函數呼叫
- `multiple_function` - 多個函數呼叫
- `parallel_multiple_function` - 並行多個函數呼叫
- `executable_*` - 可執行的對應類別
- `relevance` - 相關性檢測

### H200 GPU 優化設定

H200 GPU 具有 141GB HBM3e 記憶體，建議設定：

```bash
# 最大準確度設定
python openfunctions_evaluation.py \
    --model "your-model-name" \
    --test-category all \
    --language zhtw \
    --temperature 0.05 \
    --top-p 0.9 \
    --max-tokens 2000 \
    --gpu-memory-utilization 0.95 \
    --num-gpus 1 \
    --optimize-accuracy
```

### 準確度優化技巧

1. **降低溫度** - 使用 0.05-0.1 獲得更確定性的輸出
2. **使用繁體中文提示** - 設定 `--language zhtw` 使用優化的中文提示
3. **增加重試次數** - 預設已增加到 5 次重試
4. **充分利用 GPU** - H200 設定 `--gpu-memory-utilization 0.95`
5. **選擇合適的模型** - 中文模型（Qwen, GLM, Breeze）通常在中文任務上表現更好

### 常見問題排解

#### 記憶體不足錯誤
```bash
# 降低 GPU 記憶體使用率
--gpu-memory-utilization 0.85
```

#### API 速率限制
- 系統會自動重試（最多 5 次）
- 每次重試間隔 65 秒
- 檢查您的 API 配額

#### 效能緩慢
- 確認使用 GPU 加速
- 檢查 `nvidia-smi` 確認 GPU 使用情況
- 考慮使用本地模型搭配 vLLM 以獲得更快的推理速度

### 查看結果

評估結果會儲存在：
```
./result/{language}/{model_name}/
```

例如：
```
./result/zhtw/gpt-4o-2024-05-13-FC/gorilla_openfunctions_v1_test_simple_result.json
```

### 產生雷達圖

```bash
# 產生評估結果的雷達圖
python analysis/chart.py \
    --score_csv ./score/zhtw/data.csv \
    --out_chart ./score/zhtw/radar_chart.png
```

### 環境設定

#### 設定 API 金鑰
```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export MISTRAL_API_KEY="your-mistral-key"
export COHERE_API_KEY="your-cohere-key"
```

#### 安裝相依套件
```bash
pip install -r requirements.txt

# 如果使用 vLLM (推薦用於本地模型)
pip install vllm==0.4.3
```

### 推薦工作流程

1. **快速測試**（5-10 分鐘）
   ```bash
   python openfunctions_evaluation.py --test-category simple --language zhtw
   ```

2. **中型評估**（30-60 分鐘）
   ```bash
   python openfunctions_evaluation.py --test-category simple parallel_function --language zhtw
   ```

3. **完整評估**（2-4 小時）
   ```bash
   python openfunctions_evaluation.py --test-category all --language zhtw
   ```

### 更多資訊

- 詳細優化指南：`ACCURACY_OPTIMIZATION.md`
- 技術細節：`LAB5_COMPLETION.md`
- 主要 README：`README.md`

### 支援

如有問題或需要協助：
1. 查看 `ACCURACY_OPTIMIZATION.md` 中的疑難排解章節
2. 檢查 API 金鑰和相依套件是否正確安裝
3. 確認 GPU 驅動程式和 CUDA 版本相容

---

## English Quick Start

### Basic Usage

#### 1. Run with Default Optimized Settings (Recommended)

```bash
python openfunctions_evaluation.py \
    --model "gpt-4o-2024-05-13-FC" \
    --test-category all \
    --language zhtw
```

#### 2. Run on H200 GPU with Local Model

```bash
python openfunctions_evaluation.py \
    --model "meta-llama/Meta-Llama-3-70B-Instruct" \
    --test-category all \
    --language zhtw \
    --gpu-memory-utilization 0.95 \
    --num-gpus 1
```

For complete documentation, see:
- `ACCURACY_OPTIMIZATION.md` - Full optimization guide
- `LAB5_COMPLETION.md` - Technical details
- `README.md` - Main documentation
