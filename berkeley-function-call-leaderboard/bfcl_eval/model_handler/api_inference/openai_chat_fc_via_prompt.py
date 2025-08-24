# bfcl_eval/model_handler/api_inference/openai_chat_fc_via_prompt.py
import json
import os
import time
from overrides import override
from openai import OpenAI

from bfcl_eval.model_handler.api_inference.openai_chat_handler import OpenAIChatHandler
from bfcl_eval.model_handler.utils import func_doc_language_specific_pre_processing

REASONING_LEVEL = os.getenv("GPOST_REASONING_LEVEL", "medium")  # low/medium/high

class OpenAIChatFCViaPromptHandler(OpenAIChatHandler):
    """
    透過 chat.completions + Prompt 來模擬 FC：
    - 把 function schema 放在第一個 user 訊息
    - 嚴格要求輸出 JSON（單物件或陣列皆可）
    - decode_ast / decode_execute 將 JSON 轉為評測格式
    """

    @override
    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        # 與 OSS/LLama 做法一致：先做語言別前處理，保留 function list 後續插入第一個 user 訊息
        functions: list = test_entry["function"]
        test_category: str = test_entry["id"].rsplit("_", 1)[0]
        functions = func_doc_language_specific_pre_processing(functions, test_category)
        return {"message": [], "function": functions}

    @override
    def _query_prompting(self, inference_data: dict):
        """
        直接呼叫 /v1/chat/completions
        messages[0]：system，指定 reasoning 等
        messages[1] (第一個 user)：內含 tools + 嚴格輸出要求 + 原始 user 問題
        """
        msgs = inference_data["message"]
        tools = inference_data["function"]

        # 抽出第一個 user 訊息以便插入工具清單與規則
        first_user_idx = next((i for i, m in enumerate(msgs) if m.get("role") == "user"), None)
        user_text = msgs[first_user_idx]["content"] if first_user_idx is not None else ""

        # 重組 messages
        messages = []
        messages.append({
            "role": "system",
            "content": f"You are a function-calling assistant.\nReasoning: {REASONING_LEVEL}\n"
                       "When tools are provided, you MUST decide whether to call a tool. "
                       "If a tool call is needed, reply ONLY with JSON (no prose). "
                       'Preferred formats:\n'
                       '  - Single call: {"name": "<func_name>", "parameters": {...}}\n'
                       '  - Multiple calls: [{"name": "...","parameters": {...}}, ...]\n'
                       "Do not add commentary. No code fences. No variables."
        })

        # 把工具 schema 塞到第一個 user 訊息前面，並重述輸出規範
        tool_block = "\n\n".join(json.dumps(t, indent=2, ensure_ascii=False) for t in tools)
        first_user_content = (
            "You have access to the following functions. Choose the best match and "
            "respond STRICTLY in JSON as specified. Do not explain.\n\n"
            f"{tool_block}\n\n"
            "User prompt:\n"
            f"{user_text}"
        )
        messages.append({"role": "user", "content": first_user_content})

        # 其餘訊息（若有）照常附上
        for i, m in enumerate(msgs):
            if i == first_user_idx:
                continue
            messages.append(m)

        client = OpenAI()  # 會讀 OPENAI_BASE_URL、OPENAI_API_KEY
        t0 = time.time()
        api_response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            timeout=72000,
        )
        t1 = time.time()

        # 回存一下方便除錯
        inference_data["inference_input_log"] = {
            "messages": messages,
        }
        return api_response, (t1 - t0)

    # --- 解析輸出成標準 FC 期望格式 ---
    @override
    def decode_ast(self, result, language="Python"):
        """
        將模型輸出的 JSON 轉成：
        [ {func1: {...}}, {func2: {...}} ]
        允許：單物件或陣列；也容忍被 ```json``` 包住的情況。
        """
        text = result.strip()
        if text.startswith("```json"):
            text = text[len("```json"):].strip()
        if text.endswith("```"):
            text = text[:-3].strip()

        # 允許單物件或陣列
        try:
            data = json.loads(text)
        except Exception:
            # 有些模型會輸出多個物件以分號或換行分隔，嘗試寬鬆處理
            if ";" in text:
                parts = [p for p in text.split(";") if p.strip()]
                data = [json.loads(p) for p in parts]
            else:
                # 最後手段
                data = eval(text)

        if isinstance(data, dict):
            data = [data]

        out = []
        for call in data:
            name = call.get("name")
            params = call.get("parameters", {})
            out.append({name: params})
        return out

    @override
    def decode_execute(self, result):
        """
        轉成可執行字串：
        ["func(a=1,b=2)", ...]
        """
        text = result.strip()
        if text.startswith("```json"):
            text = text[len("```json"):].strip()
        if text.endswith("```"):
            text = text[:-3].strip()
        try:
            data = json.loads(text)
        except Exception:
            if ";" in text:
                parts = [p for p in text.split(";") if p.strip()]
                data = [json.loads(p) for p in parts]
            else:
                data = eval(text)

        if isinstance(data, dict):
            data = [data]

        exec_list = []
        for call in data:
            name = call.get("name")
            params = call.get("parameters", {})
            args = ",".join(f"{k}={repr(v)}" for k, v in params.items())
            exec_list.append(f"{name}({args})")
        return exec_list
