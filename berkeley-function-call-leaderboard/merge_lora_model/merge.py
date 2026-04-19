import os
import gc
import shutil
import torch
from transformers import AutoTokenizer, AutoProcessor
from peft import AutoPeftModelForCausalLM
import argparse


# 基礎模型路徑可由參數指定
DEFAULT_BASE_MODEL = "Llama-xLAM-2-8b-fc-r"

# 解析命令列參數

parser = argparse.ArgumentParser(description="Merge LoRA checkpoints into full models.")
parser.add_argument('--BASE_MODEL', type=str, default=DEFAULT_BASE_MODEL, help='基礎模型路徑')
parser.add_argument('--LORA_DIR', type=str, required=True, help='LoRA checkpoints 資料夾')
parser.add_argument('--OUTPUT_DIR', type=str, required=True, help='輸出完整模型的資料夾')
parser.add_argument('--DTYPE', type=str, default='bf16', choices=['bf16', 'fp16', 'fp32'], help='載入與儲存權重精度')
parser.add_argument('--DEVICE_MAP', type=str, default='auto', help='transformers device_map，例如 auto 或 cpu')
parser.add_argument('--OFFLOAD_DIR', type=str, default=None, help='device_map 需要時的磁碟 offload 資料夾')
parser.add_argument('--MAX_SHARD_SIZE', type=str, default='2GB', help='輸出模型分片大小')
parser.add_argument('--SAFE_SERIALIZATION', action='store_true', help='啟用 safetensors 輸出（較安全但可能更吃記憶體）')
args = parser.parse_args()


BASE_MODEL = args.BASE_MODEL
LORA_DIR = args.LORA_DIR
OUTPUT_DIR = args.OUTPUT_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)

DTYPE_MAP = {
    'bf16': torch.bfloat16,
    'fp16': torch.float16,
    'fp32': torch.float32,
}
target_dtype = DTYPE_MAP[args.DTYPE]


def _copy_config_if_exists(src_dir: str, filename: str, dst_dir: str) -> bool:
    src = os.path.join(src_dir, filename)
    if os.path.isfile(src):
        shutil.copy2(src, os.path.join(dst_dir, filename))
        return True
    return False

if args.OFFLOAD_DIR:
    os.makedirs(args.OFFLOAD_DIR, exist_ok=True)

# 掃描資料夾內所有 checkpoint
for ckpt_name in sorted(os.listdir(LORA_DIR)):
    ckpt_path = os.path.join(LORA_DIR, ckpt_name)
    if not os.path.isdir(ckpt_path):
        continue
    if ckpt_name.startswith(".") or "checkpoint" not in ckpt_name:
        continue

    print(f"正在處理 {ckpt_name}...")

    # 直接從 LoRA adapter 路徑載入，AutoPeftModelForCausalLM 會自動處理 base model
    load_kwargs = {
        'device_map': args.DEVICE_MAP,
        'dtype': target_dtype,
        'low_cpu_mem_usage': True,
    }
    if args.OFFLOAD_DIR:
        load_kwargs['offload_folder'] = args.OFFLOAD_DIR

    model = AutoPeftModelForCausalLM.from_pretrained(ckpt_path, **load_kwargs)

    # 合併並卸載 LoRA
    merged_model = model.merge_and_unload()
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 以指定精度儲存
    merged_model = merged_model.to(target_dtype)

    # tokenizer 優先使用 checkpoint 內設定，失敗時回退 BASE_MODEL
    try:
        tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    # 存成完整模型
    save_path = os.path.join(OUTPUT_DIR, ckpt_name + "_merged")
    merged_model.save_pretrained(
        save_path,
        max_shard_size=args.MAX_SHARD_SIZE,
        safe_serialization=args.SAFE_SERIALIZATION,
    )
    tokenizer.save_pretrained(save_path)  # 把 tokenizer 一起存

    # 優先嘗試儲存 processor；若模型不支援則回退為直接補 config 檔。
    processor = None
    processor_source = None
    for source in (ckpt_path, BASE_MODEL):
        try:
            processor = AutoProcessor.from_pretrained(source, trust_remote_code=True)
            processor_source = source
            break
        except Exception:
            continue

    if processor is not None:
        processor.save_pretrained(save_path)
        print(f"已儲存 processor 設定（來源: {processor_source}）")
    else:
        print("警告: 無法載入 AutoProcessor，改用檔案複製補齊 processor/preprocessor 設定。")

    # 確保 merged 目錄內有 processor/preprocessor 設定。
    for cfg_name in ("processor_config.json", "preprocessor_config.json"):
        cfg_target = os.path.join(save_path, cfg_name)
        if os.path.isfile(cfg_target):
            continue
        if _copy_config_if_exists(ckpt_path, cfg_name, save_path):
            print(f"已從 checkpoint 複製 {cfg_name}")
        elif _copy_config_if_exists(BASE_MODEL, cfg_name, save_path):
            print(f"已從 BASE_MODEL 複製 {cfg_name}")

    processor_cfg = os.path.join(save_path, "processor_config.json")
    preprocessor_cfg = os.path.join(save_path, "preprocessor_config.json")

    if (not os.path.isfile(preprocessor_cfg)) and os.path.isfile(processor_cfg):
        shutil.copy2(processor_cfg, preprocessor_cfg)
        print("已用 processor_config.json 建立 preprocessor_config.json")
    elif (not os.path.isfile(processor_cfg)) and os.path.isfile(preprocessor_cfg):
        shutil.copy2(preprocessor_cfg, processor_cfg)
        print("已用 preprocessor_config.json 建立 processor_config.json")

    if not os.path.isfile(preprocessor_cfg):
        print(f"警告: {save_path} 仍缺少 preprocessor_config.json，Gemma4 vLLM 可能無法啟動。")

    del merged_model
    del tokenizer
    if processor is not None:
        del processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"已儲存至 {save_path}")
