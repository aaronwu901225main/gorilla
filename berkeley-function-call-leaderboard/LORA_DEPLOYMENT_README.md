# LoRA 部署模式說明

## 🎯 為什麼使用 LoRA 模式而非 Merge？

### **傳統 Merge 方式的問題**
```
Base Model (MXFP4 + BF16 混合量化)
      ↓ 反量化
   FP16/BF16
      ↓ + LoRA adapter (FP16/BF16)
   Merged Model
      ↓ 重新量化 (可能失敗)
   MXFP4 + BF16 ❌
```

**問題**：
- ❌ 反量化 → 合併 → 重量化過程可能導致精度損失
- ❌ MXFP4 格式不是標準格式，重量化可能不支援
- ❌ Flash Attention 3 只支援 BF16，與某些量化格式衝突
- ❌ 每個 checkpoint 都需要完整存儲（佔用大量空間）

### **LoRA 模式的優勢**
```
Base Model (MXFP4 + BF16) ──┐
                            ├─→ y = (W_q × x) + (ΔW × x)
LoRA Adapter (BF16) ────────┘
```

**優點**：
- ✅ **保留原始量化**：base model 不需要修改，保持 MXFP4 + BF16 混合格式
- ✅ **避免精度問題**：不需要反量化/重量化，沒有格式轉換風險
- ✅ **節省空間**：只需一份 base model + 多個小的 adapter（通常 < 1GB）
- ✅ **快速切換**：可以在同一個 server 上載入多個 adapter
- ✅ **動態調整**：推理時可以指定使用哪個 adapter

---

## 📋 使用方法

### **1. 配置腳本**

編輯 `bfcl-gen-gpt-oss.slurm` 的第 5 節：

```bash
# Base model 路徑（保持原始量化格式）
BASE_MODEL_PATH="/path/to/your/gpt-oss-20b"  # MXFP4 + BF16 原始模型

# LoRA checkpoints 列表
declare -a LORA_ADAPTERS=(
  "adapter_name|/path/to/lora/checkpoint|openai/served-model-name"
  # 例如：
  "lr1e5-ckpt50|/path/to/checkpoint-50|openai/gpt-oss-20b-ckpt50"
  "lr1e5-ckpt100|/path/to/checkpoint-100|openai/gpt-oss-20b-ckpt100"
)

# LoRA rank（根據你的訓練配置）
MAX_LORA_RANK=64  # 如果你用 r=32 訓練，改成 32
```

### **2. 提交作業**

```bash
sbatch bfcl-gen-gpt-oss.slurm
```

### **3. 工作流程**

1. **啟動 vLLM server**（只啟動一次）
   - 載入 base model（MXFP4 + BF16）
   - 同時載入所有 LoRA adapters
   - 使用 `--enable-lora --max-lora-rank 64`

2. **循環評測**
   - 對每個 adapter 執行 `bfcl generate`
   - 無需重啟 server，直接切換 adapter

3. **優勢**
   - 只需載入一次 base model（節省時間）
   - 無需反覆啟動/關閉 server
   - 保留原始量化精度

---

## 🔧 技術細節

### **vLLM LoRA 支援**

```bash
vllm serve /path/to/base/model \
  --enable-lora \                          # 啟用 LoRA
  --max-lora-rank 64 \                     # 最大 rank
  --lora-modules name1=/path/to/adapter1 \ # 載入 adapter
  --lora-modules name2=/path/to/adapter2 \
  --dtype bfloat16                         # 使用 BF16 避免 FA3 問題
```

### **BFCL LoRA 支援**

```bash
bfcl generate \
  --model "openai/model-with-lora" \
  --enable-lora \
  --max-lora-rank 64 \
  --lora-modules "name=/path/to/adapter"
```

### **LoRA Adapter 格式**

你的 LoRA checkpoint 目錄應該包含：
```
checkpoint-50/
├── adapter_config.json
├── adapter_model.safetensors (或 .bin)
└── (其他 PEFT 文件)
```

**重要**：不需要 `_merged` 版本！

---

## 🆚 與 Merge 方式的對比

| 特性 | LoRA 模式 | Merge 模式 |
|------|-----------|------------|
| Base Model 量化 | ✅ 保留 MXFP4+BF16 | ❌ 需反量化再重量化 |
| 磁碟空間 | ✅ 1 base + n adapters | ❌ n 個完整模型 |
| 精度損失 | ✅ 無 | ⚠️ 可能有 |
| 格式兼容性 | ✅ 無問題 | ❌ FA3 BF16 only |
| 切換速度 | ✅ 快速 | ❌ 需重啟 server |
| 推理開銷 | ⚠️ 稍高（通常 < 5%） | ✅ 無額外開銷 |

---

## 📊 範例配置

### **情境 1：測試多個 checkpoint**

```bash
declare -a LORA_ADAPTERS=(
  "ckpt50|/train/output/checkpoint-50|openai/model-ckpt50"
  "ckpt100|/train/output/checkpoint-100|openai/model-ckpt100"
  "ckpt150|/train/output/checkpoint-150|openai/model-ckpt150"
  "ckpt200|/train/output/checkpoint-200|openai/model-ckpt200"
)
```

### **情境 2：不同學習率的對比**

```bash
declare -a LORA_ADAPTERS=(
  "lr1e5|/train/lr1e-5/final|openai/model-lr1e5"
  "lr5e6|/train/lr5e-6/final|openai/model-lr5e6"
  "lr1e6|/train/lr1e-6/final|openai/model-lr1e6"
)
```

### **情境 3：不同訓練策略**

```bash
declare -a LORA_ADAPTERS=(
  "full|/train/full-data/final|openai/model-full"
  "chinese|/train/chinese-only/final|openai/model-chinese"
  "multiturn|/train/multiturn-only/final|openai/model-multiturn"
)
```

---

## 🐛 常見問題

### Q1: 如何確認 LoRA rank？

```bash
# 查看 adapter_config.json
cat /path/to/checkpoint/adapter_config.json | grep "\"r\""
# 輸出: "r": 64
```

### Q2: vLLM 啟動失敗

檢查：
- ✅ Base model 路徑正確
- ✅ Adapter 路徑正確
- ✅ `max_lora_rank` >= 實際 rank
- ✅ 使用 `--dtype bfloat16`

### Q3: BFCL 找不到模型

確保 `served_model_name` 在 vLLM 中正確註冊：
```bash
# 查看 vLLM log
tail -f vllm_server_*.log | grep "served_model"
```

### Q4: 記憶體不足

減少同時載入的 adapter 數量：
```bash
# 分批處理，每次 2-3 個 adapter
declare -a LORA_ADAPTERS=(
  # Batch 1
  "adapter1|/path1|name1"
  "adapter2|/path2|name2"
)
```

---

## 📚 參考資料

- [vLLM LoRA Documentation](https://docs.vllm.ai/en/latest/models/lora.html)
- [PEFT Library](https://github.com/huggingface/peft)
- [BFCL Evaluation Framework](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard)
