import os
import gc
import shutil
import torch
from peft import AutoPeftModelForCausalLM
import argparse
import sys

from disk_space_guard import (
    format_bytes,
    get_directory_size_bytes,
    require_merge_output_space,
)


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

BASE_MODEL_IS_DIR = os.path.isdir(BASE_MODEL)
if not BASE_MODEL_IS_DIR:
    print(f"錯誤: BASE_MODEL 必須是本機資料夾，才能精準檢查空間: {BASE_MODEL}")
    sys.exit(1)

BASE_MODEL_SIZE_BYTES = get_directory_size_bytes(BASE_MODEL)
print(f"Base model size: {format_bytes(BASE_MODEL_SIZE_BYTES)}")

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
    save_path = os.path.join(OUTPUT_DIR, ckpt_name + "_merged")
    try:
        available_bytes, required_bytes = require_merge_output_space(
            save_path,
            BASE_MODEL_SIZE_BYTES,
            checkpoint_name=ckpt_name,
        )
    except RuntimeError as exc:
        print(f"錯誤: {exc}")
        sys.exit(1)
    print(
        "空間檢查通過: "
        f"available={format_bytes(available_bytes)}, "
        f"required={format_bytes(required_bytes)}"
    )

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

    # 存成完整模型
    merged_model.save_pretrained(
        save_path,
        max_shard_size=args.MAX_SHARD_SIZE,
        safe_serialization=args.SAFE_SERIALIZATION,
    )

    # 非權重檔一律以 BASE_MODEL 為準，避免 checkpoint 內 tokenizer 設定漂移。
    for cfg_name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "chat_template.jinja",
        "generation_config.json",
        "processor_config.json",
        "preprocessor_config.json",
    ):
        if _copy_config_if_exists(BASE_MODEL, cfg_name, save_path):
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
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"已儲存至 {save_path}")
