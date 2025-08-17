#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from typing import Any, Dict, List, Tuple
import orjson
from tqdm import tqdm
from vllm import LLM, SamplingParams
import glob
import re

def escape_braces(s: str) -> str:
    # 若要用 str.format 安全插入原文，先把原文中的 { } 轉義
    return s.replace("{", "{{").replace("}", "}}")

def load_records(path: str, input_format: str = None) -> Tuple[List[Dict[str, Any]], str]:
    if input_format is None:
        input_format = "jsonl" if path.lower().endswith(".jsonl") else "json"

    if input_format == "jsonl":
        out = []
        with open(path, "rb") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                out.append(orjson.loads(line))
        return out, "jsonl"
    elif input_format == "json":
        with open(path, "rb") as f:
            data = orjson.loads(f.read())
        if not isinstance(data, list):
            raise ValueError("JSON 需為陣列（list）。")
        return data, "json"
    else:
        raise ValueError(f"未知輸入格式：{input_format}")

def build_prompts(texts: List[str]) -> List[str]:
    tpl = (
        "你是一位專業翻譯，請將下面文字精準翻譯為繁體中文，"
        "保留專有名詞與數學符號，維持原意與語氣。\n"
        "只輸出翻譯內容，不要任何額外說明或標註，並且不要回答輸入的問題。\n"
        "將最終的輸出結果放在 \\boxed{{...}} 中。\n"
        "舉例來說，如果輸入為Calculate the area of a triangle given the base is 10 meters and height is 5 meters.\n"
        "最終請只輸出 \\boxed{{計算一個三角形的面積，已知底為 10 公尺，高為 5 公尺。}}，其他問題以此類推。\n"
        "原文：\n{content}\n"
    )
    prompts = []
    for t in texts:
        t = t if isinstance(t, str) else str(t)
        prompts.append(tpl.format(content=escape_braces(t)))
    return prompts

def save_records(path, records, fmt: str):
    if fmt == "json":
        # 輸出一個 JSON 陣列，縮排 + 自動結尾換行
        with open(path, "wb") as f:
            f.write(orjson.dumps(
                records,
                option=orjson.OPT_INDENT_2 | orjson.OPT_APPEND_NEWLINE
            ))
    elif fmt == "jsonl":
        # 一行一筆，記得用 bytes 寫，並且每行加換行
        with open(path, "wb") as f:
            for obj in records:
                f.write(orjson.dumps(obj, option=orjson.OPT_APPEND_NEWLINE))
    else:
        raise ValueError(f"Unknown output format: {fmt}")


def chunked(indices: List[int], n: int):
    for i in range(0, len(indices), n):
        yield indices[i : i + n]

def collect_question_contents(
    records: List[Dict[str, Any]],
    question_key: str = "question",
    roles: List[str] = None
) -> List[Tuple[int, Tuple[int, int], str]]:
    """
    將所有需翻譯的 (record_idx, (outer_idx, inner_idx), text) 收集起來。
    預期結構：record[question_key] -> List[List[{"role": str, "content": str}]]
    roles=None 代表不過濾角色，全部翻。
    """
    collected = []
    for ri, rec in enumerate(records):
        q = rec.get(question_key, None)
        if not isinstance(q, list):
            continue
        for oi, inner in enumerate(q):
            if not isinstance(inner, list):
                continue
            for ii, msg in enumerate(inner):
                if not isinstance(msg, dict):
                    continue
                if "content" not in msg:
                    continue
                if roles is not None:
                    r = msg.get("role")
                    if r not in roles:
                        continue
                text = msg["content"]
                if not isinstance(text, str):
                    text = str(text)
                collected.append((ri, (oi, ii), text))
    return collected

def extract_boxed_answer(text: str):
    if not text:
        return None
    # 支援跨行；取最後一個 \boxed{...}
    matches = re.findall(r'\\boxed\{([\s\S]*?)\}', text)
    return matches[-1].strip() if matches else None

def write_back_translation(
    records: List[Dict[str, Any]],
    mapping: List[Tuple[int, Tuple[int, int], str]],
    translations: List[str],
    question_key: str = "question"
):
    # 不要硬 assert；確保「對得上的」都寫回，失敗就保留原文
    n = min(len(mapping), len(translations))
    for k in range(n):
        (ri, (oi, ii), _), zh = mapping[k], translations[k]
        if isinstance(zh, str) and zh.strip():
            try:
                records[ri][question_key][oi][ii]["content"] = zh.strip()
            except Exception:
                continue  # 結構異常就略過

def process_one_file(
    llm: LLM,
    sampling: SamplingParams,
    in_path: str,
    out_path: str,
    question_key: str,
    roles: List[str],
    input_format: str = None,
    output_format: str = None,
    batch_size: int = 32,
):
    records, in_fmt = load_records(in_path, input_format)
    out_fmt = (output_format or
               ("jsonl" if out_path.lower().endswith(".jsonl")
                else ("jsonl" if in_path.lower().endswith(".jsonl") else "json")))

    if not records:
        print(f"[skip] {in_path}: 沒有可處理的資料")
        save_records(out_path, records, out_fmt)
        return

    mapping = collect_question_contents(records, question_key=question_key, roles=roles)
    if not mapping:
        print(f"[pass] {in_path}: 找不到可翻譯的 question.content")
        save_records(out_path, records, out_fmt)
        return

    all_texts = [t for _, _, t in mapping]
    translations: List[str] = []
    idxs = list(range(len(all_texts)))

    for batch in tqdm(chunked(idxs, batch_size), desc=f"Translating {os.path.basename(in_path)}"):
        texts = [all_texts[i] for i in batch]
        prompts = build_prompts(texts)
        
        # 第一次：產生草稿
        out_1 = llm.generate(prompts, sampling)
        drafts = [(o.outputs[0].text or "").strip() for o in out_1]

        # 為每一筆組第二階段的 prompt（等長 list）
        refine_prompts = [
            f"{p}{d}\n最終翻譯結果為: \\boxed{{"
            for p, d in zip(prompts, drafts)
        ]

        # 第二次：產生最終結果
        out_2 = llm.generate(refine_prompts, sampling)

        # 逐筆擷取 boxed 內容
        for o in out_2:
            raw = (o.outputs[0].text or "").strip()
            zh = extract_boxed_answer(raw) or raw
            translations.append(zh)
            write_back_translation(records, mapping, translations, question_key=question_key)
            save_records(out_path, records, out_fmt)
            print(f"[ok] {in_path} -> {out_path}  已寫回 {min(len(mapping), len(translations))} 段")

def main():
    ap = argparse.ArgumentParser(
        description="使用 vLLM 將 BFCL 類巢狀 question.content 翻譯為繁體中文（直接覆蓋原文）"
    )
    # 互斥：單檔 vs 資料夾
    mex = ap.add_mutually_exclusive_group(required=True)
    mex.add_argument("--input", help="輸入檔路徑（.json 或 .jsonl）")
    mex.add_argument("--input-folder", help="要批次處理的資料夾")

    ap.add_argument("--output", help="單檔模式輸出檔路徑（可與輸入同檔名以覆蓋）")
    ap.add_argument("--output-folder", help="資料夾模式輸出到該資料夾（保留檔名）。未提供時預設原地覆蓋")
    ap.add_argument("--inplace", action="store_true", help="資料夾模式：原地覆蓋（預設 True）")
    ap.add_argument("--glob", default="*.json,*.jsonl", help="資料夾模式的檔案樣式，多個以逗號分隔")

    ap.add_argument("--input-format", choices=["json", "jsonl"], default=None)
    ap.add_argument("--output-format", choices=["json", "jsonl"], default=None)
    ap.add_argument("--question-key", default="question")
    ap.add_argument("--roles", nargs="*", default=None)

    ap.add_argument("--model", required=True)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=32)

    args = ap.parse_args()

    # vLLM 一次初始化，所有檔共用
    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        repetition_penalty=1.0,
    )
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=args.trust_remote_code,
    )

    if args.input:
        if not args.output:
            # 單檔模式若沒給 output，就原地覆蓋
            args.output = args.input
        process_one_file(
            llm, sampling,
            in_path=args.input,
            out_path=args.output,
            question_key=args.question_key,
            roles=args.roles,
            input_format=args.input_format,
            output_format=args.output_format,
            batch_size=args.batch_size,
        )
        return

    # 資料夾模式
    patterns = [p.strip() for p in args.glob.split(",") if p.strip()]
    paths = []
    for pat in patterns:
        paths.extend(glob.glob(os.path.join(args.input_folder, pat)))
    paths = sorted(set(paths))
    if not paths:
        print("資料夾模式找不到檔案。")
        return

    # 預設資料夾模式原地覆蓋（除非指定 output-folder）
    inplace = True if (args.output_folder is None) else False
    if args.inplace:
        inplace = True

    for in_path in paths:
        if not inplace:
            os.makedirs(args.output_folder, exist_ok=True)
            out_path = os.path.join(args.output_folder, os.path.basename(in_path))
        else:
            out_path = in_path

        process_one_file(
            llm, sampling,
            in_path=in_path,
            out_path=out_path,
            question_key=args.question_key,
            roles=args.roles,
            input_format=args.input_format,
            output_format=args.output_format,
            batch_size=args.batch_size,
        )
if __name__ == "__main__":
    main()
