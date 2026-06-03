#!/usr/bin/env python3
"""
GPT-OSS LoRA Merge Script

專為 GPT-OSS 模型設計的 LoRA 合併腳本。
處理 MXFP4 + BF16 混合量化模式。

策略說明：
1. 載入原始的非量化/BF16 base model（如果有的話）
2. 或者，載入量化模型並嘗試反量化
3. 合併 LoRA adapter
4. 保存為 BF16 格式（與 vLLM Flash Attention 3 兼容）

使用方式：
    python merge_gptoss.py \
        --base-model /path/to/gpt-oss-20b \
        --lora-dir /path/to/lora/checkpoints \
        --output-dir /path/to/merged/models \
        --dtype bfloat16
"""

import os
import sys
import json
import shutil
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import PeftModel

from disk_space_guard import (
    format_bytes,
    get_directory_size_bytes,
    require_merge_output_space,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge LoRA checkpoints into GPT-OSS models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--base-model', 
        type=str, 
        required=True, 
        help='Base model 路徑（建議使用非量化/BF16 版本）'
    )
    parser.add_argument(
        '--lora-dir', 
        type=str, 
        required=True, 
        help='LoRA checkpoints 資料夾路徑'
    )
    parser.add_argument(
        '--output-dir', 
        type=str, 
        required=True, 
        help='輸出 merged model 的資料夾'
    )
    parser.add_argument(
        '--dtype', 
        type=str, 
        default='bfloat16',
        choices=['float16', 'bfloat16', 'float32'],
        help='輸出模型的精度（預設: bfloat16，推薦用於 vLLM FA3）'
    )
    parser.add_argument(
        '--checkpoints',
        type=str,
        nargs='*',
        default=None,
        help='指定要處理的 checkpoint 名稱（例如: checkpoint-50 checkpoint-100）。不指定則處理全部。'
    )
    parser.add_argument(
        '--trust-remote-code',
        action='store_true',
        default=True,
        help='信任遠端程式碼（GPT-OSS 需要）'
    )
    parser.add_argument(
        '--device-map',
        type=str,
        default='auto',
        help='Device map 設定（預設: auto）'
    )
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='跳過已存在的輸出目錄'
    )
    parser.add_argument(
        '--copy-tokenizer-only',
        action='store_true',
        help='只複製 tokenizer（用於 debug）'
    )
    
    return parser.parse_args()


def get_torch_dtype(dtype_str: str) -> torch.dtype:
    """轉換 dtype 字串為 torch.dtype"""
    dtype_map = {
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
        'float32': torch.float32,
    }
    return dtype_map[dtype_str]


def check_model_quantization(model_path: str) -> Dict[str, Any]:
    """檢查模型的量化狀態"""
    config_path = Path(model_path) / "config.json"
    
    if not config_path.exists():
        return {"quantized": False, "method": None}
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # 檢查常見的量化配置
    quant_info = {
        "quantized": False,
        "method": None,
        "bits": None,
    }
    
    # 檢查 MXFP4 (GPT-OSS 使用的量化方法)
    if config.get("quantization_config"):
        quant_config = config["quantization_config"]
        quant_info["quantized"] = True
        quant_info["method"] = quant_config.get("quant_method", "unknown")
        quant_info["bits"] = quant_config.get("bits", quant_config.get("weight_bits", None))
    
    # 檢查 torch_dtype
    quant_info["torch_dtype"] = config.get("torch_dtype", "unknown")
    
    return quant_info


def find_checkpoints(lora_dir: str, specific_checkpoints: Optional[List[str]] = None) -> List[Path]:
    """搜尋 LoRA checkpoints"""
    lora_path = Path(lora_dir)
    
    if not lora_path.exists():
        raise FileNotFoundError(f"LoRA 目錄不存在: {lora_dir}")
    
    checkpoints = []
    for item in sorted(lora_path.iterdir()):
        if not item.is_dir():
            continue
        if item.name.startswith("."):
            continue
        
        # 檢查是否為有效的 LoRA checkpoint
        adapter_config = item / "adapter_config.json"
        if not adapter_config.exists():
            continue
        
        # 如果指定了特定的 checkpoints，只處理這些
        if specific_checkpoints:
            if item.name not in specific_checkpoints:
                continue
        
        checkpoints.append(item)
    
    return checkpoints


def get_lora_info(checkpoint_path: Path) -> Dict[str, Any]:
    """獲取 LoRA checkpoint 的資訊"""
    adapter_config_path = checkpoint_path / "adapter_config.json"
    
    with open(adapter_config_path, 'r') as f:
        config = json.load(f)
    
    return {
        "rank": config.get("r", "unknown"),
        "lora_alpha": config.get("lora_alpha", "unknown"),
        "target_modules": config.get("target_modules", []),
        "peft_type": config.get("peft_type", "unknown"),
    }


def merge_checkpoint(
    base_model_path: str,
    checkpoint_path: Path,
    output_path: Path,
    target_dtype: torch.dtype,
    trust_remote_code: bool = True,
    device_map: str = "auto",
) -> bool:
    """合併單一 checkpoint"""
    
    print(f"\n{'='*60}")
    print(f"🔄 處理: {checkpoint_path.name}")
    print(f"{'='*60}")
    
    # 1. 獲取 LoRA 資訊
    lora_info = get_lora_info(checkpoint_path)
    print(f"📊 LoRA 資訊:")
    print(f"   Rank: {lora_info['rank']}")
    print(f"   Alpha: {lora_info['lora_alpha']}")
    print(f"   Target Modules: {lora_info['target_modules'][:3]}...")  # 只顯示前3個
    
    # 2. 載入 base model
    print(f"\n📦 載入 Base Model: {base_model_path}")
    print(f"   目標精度: {target_dtype}")
    
    try:
        # 嘗試以 BF16/FP16 載入（如果模型是量化的，這可能會觸發反量化）
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=target_dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            low_cpu_mem_usage=True,
        )
        print("   ✅ Base model 載入成功")
    except Exception as e:
        print(f"   ❌ 載入失敗: {e}")
        
        # 嘗試其他載入方式
        print("   🔄 嘗試使用 float32 載入後轉換...")
        try:
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype=torch.float32,
                device_map="cpu",
                trust_remote_code=trust_remote_code,
                low_cpu_mem_usage=True,
            )
            base_model = base_model.to(target_dtype)
            print("   ✅ 使用 float32 載入並轉換成功")
        except Exception as e2:
            print(f"   ❌ 備用載入方式也失敗: {e2}")
            return False
    
    # 3. 載入 LoRA adapter
    print(f"\n🔗 載入 LoRA Adapter: {checkpoint_path}")
    try:
        model_with_lora = PeftModel.from_pretrained(
            base_model,
            str(checkpoint_path),
            torch_dtype=target_dtype,
        )
        print("   ✅ LoRA adapter 載入成功")
    except Exception as e:
        print(f"   ❌ LoRA 載入失敗: {e}")
        del base_model
        torch.cuda.empty_cache()
        return False
    
    # 4. 合併 LoRA
    print("\n🔧 合併 LoRA weights...")
    try:
        merged_model = model_with_lora.merge_and_unload()
        print("   ✅ 合併成功")
    except Exception as e:
        print(f"   ❌ 合併失敗: {e}")
        del base_model, model_with_lora
        torch.cuda.empty_cache()
        return False
    
    # 5. 確保精度正確
    print(f"\n📐 轉換精度為 {target_dtype}...")
    merged_model = merged_model.to(target_dtype)
    
    # 6. 儲存合併後的模型
    print(f"\n💾 儲存至: {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        merged_model.save_pretrained(
            output_path,
            safe_serialization=True,  # 使用 safetensors 格式
        )
        print("   ✅ 模型儲存成功")
    except Exception as e:
        print(f"   ❌ 儲存失敗: {e}")
        return False
    
    # 7. 複製 tokenizer
    print("\n📝 複製 Tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=trust_remote_code,
        )
        tokenizer.save_pretrained(output_path)
        print("   ✅ Tokenizer 儲存成功")
    except Exception as e:
        print(f"   ⚠️  Tokenizer 複製失敗: {e}")
        # 不中斷，因為可以手動複製
    
    # 8. 複製其他必要文件
    base_path = Path(base_model_path)
    files_to_copy = [
        "config.json",
        "generation_config.json",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "tokenizer.model",
    ]
    
    for filename in files_to_copy:
        src = base_path / filename
        dst = output_path / filename
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
    
    # 9. 更新 config.json 移除量化配置（如果有的話）
    config_path = output_path / "config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # 移除量化相關配置
        if "quantization_config" in config:
            del config["quantization_config"]
            print("   ℹ️  已移除 quantization_config（輸出為非量化模型）")
        
        # 更新 torch_dtype
        dtype_str = str(target_dtype).replace("torch.", "")
        config["torch_dtype"] = dtype_str
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    # 清理記憶體
    del base_model, model_with_lora, merged_model
    torch.cuda.empty_cache()
    
    print(f"\n✅ {checkpoint_path.name} 處理完成！")
    return True


def main():
    args = parse_args()
    
    print("="*60)
    print("🚀 GPT-OSS LoRA Merge Tool")
    print("="*60)
    
    # 檢查 base model
    print(f"\n📦 Base Model: {args.base_model}")
    base_model_path = Path(args.base_model)
    if not base_model_path.is_dir():
        print(f"❌ Base model 必須是本機資料夾，才能精準檢查空間: {args.base_model}")
        sys.exit(1)
    base_model_size_bytes = get_directory_size_bytes(base_model_path)
    print(f"   Base model size: {format_bytes(base_model_size_bytes)}")
    
    # 檢查量化狀態
    quant_info = check_model_quantization(args.base_model)
    print(f"   量化狀態: {'是' if quant_info['quantized'] else '否'}")
    if quant_info['quantized']:
        print(f"   量化方法: {quant_info['method']}")
        print(f"   Bits: {quant_info['bits']}")
        print(f"\n   ⚠️  警告: 模型已量化 ({quant_info['method']})")
        print(f"   將嘗試反量化並以 {args.dtype} 輸出")
        print(f"   如果失敗，請使用非量化的 base model")
    
    # 搜尋 checkpoints
    print(f"\n🔍 搜尋 LoRA Checkpoints: {args.lora_dir}")
    checkpoints = find_checkpoints(args.lora_dir, args.checkpoints)
    
    if not checkpoints:
        print("❌ 未找到任何有效的 LoRA checkpoint")
        sys.exit(1)
    
    print(f"   找到 {len(checkpoints)} 個 checkpoints:")
    for ckpt in checkpoints:
        info = get_lora_info(ckpt)
        print(f"   - {ckpt.name} (rank={info['rank']})")
    
    # 準備輸出目錄
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n📁 輸出目錄: {output_dir}")
    
    # 處理每個 checkpoint
    target_dtype = get_torch_dtype(args.dtype)
    success_count = 0
    failed = []
    
    for checkpoint in checkpoints:
        output_path = output_dir / f"{checkpoint.name}_merged"
        
        # 檢查是否已存在
        if args.skip_existing and output_path.exists():
            print(f"\n⏭️  跳過已存在: {checkpoint.name}")
            success_count += 1
            continue

        try:
            available_bytes, required_bytes = require_merge_output_space(
                output_path,
                base_model_size_bytes,
                checkpoint_name=checkpoint.name,
            )
        except RuntimeError as exc:
            print(f"❌ {exc}")
            sys.exit(1)
        print(
            "\n💽 空間檢查通過: "
            f"available={format_bytes(available_bytes)}, "
            f"required={format_bytes(required_bytes)}"
        )
        
        # 合併
        success = merge_checkpoint(
            base_model_path=args.base_model,
            checkpoint_path=checkpoint,
            output_path=output_path,
            target_dtype=target_dtype,
            trust_remote_code=args.trust_remote_code,
            device_map=args.device_map,
        )
        
        if success:
            success_count += 1
        else:
            failed.append(checkpoint.name)
    
    # 總結
    print("\n" + "="*60)
    print("📊 處理結果")
    print("="*60)
    print(f"✅ 成功: {success_count}/{len(checkpoints)}")
    if failed:
        print(f"❌ 失敗: {len(failed)}")
        for name in failed:
            print(f"   - {name}")
    
    print("\n🎉 完成！")
    print(f"合併後的模型儲存在: {output_dir}")
    print(f"使用精度: {args.dtype}")
    
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
