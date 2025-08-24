# bfcl_eval/model_handler/api_inference/openai_chat_handler.py
import time
from openai import OpenAI
from bfcl_eval.model_handler.api_inference.openai_response import OpenAIResponsesHandler

class OpenAIChatHandler(OpenAIResponsesHandler):
    """
    強制使用 /v1/chat/completions。
    1) _query_prompting：回傳 SDK 原始 response 物件 + latency
    2) _parse_query_response_prompting：解析 chat.completions 的 choices/usage
    """

    def _query_prompting(self, inference_data: dict):
        messages = inference_data["message"]
        if not any(m.get("role") == "system" for m in messages):
            messages = [{"role": "system", "content": "Reasoning: medium"}] + messages

        client = OpenAI()  # 讀 OPENAI_BASE_URL、OPENAI_API_KEY
        t0 = time.time()
        api_response = client.chat.completions.create(
            model=self.model_name,           # e.g. "openai/gpt-oss-20b"
            messages=messages,
            temperature=self.temperature,
            timeout=72000,
        )
        t1 = time.time()
        return api_response, (t1 - t0)

    def _parse_query_response_prompting(self, api_response):
        # chat.completions 結構
        content = api_response.choices[0].message.content
        usage = getattr(api_response, "usage", None)
        in_tok = getattr(usage, "prompt_tokens", 0) if usage else 0
        out_tok = getattr(usage, "completion_tokens", 0) if usage else 0
        return {
            "model_responses": content,
            "input_token": in_tok,
            "output_token": out_tok,
        }
