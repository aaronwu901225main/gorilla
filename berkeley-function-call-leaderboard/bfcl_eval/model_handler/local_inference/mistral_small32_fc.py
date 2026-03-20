import json
from typing import Any

from bfcl_eval.model_handler.local_inference.mistral_fc import MistralFCHandler
from overrides import override


class MistralSmall32FCHandler(MistralFCHandler):
    @staticmethod
    def _normalize_arguments(arguments: Any) -> dict:
        if isinstance(arguments, dict):
            return arguments
        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                return {}
        return {}

    def _to_tool_call_dict(self, item: Any) -> dict | None:
        if isinstance(item, str):
            try:
                item = json.loads(item)
            except json.JSONDecodeError:
                return None

        if not isinstance(item, dict):
            return None

        # Native shape: {"name": "...", "arguments": {...}}
        if "name" in item:
            return {
                "name": item["name"],
                "arguments": self._normalize_arguments(item.get("arguments", {})),
            }

        # OpenAI/vLLM chat style: {"function": {"name": "...", "arguments": "..."}, ...}
        function_obj = item.get("function")
        if isinstance(function_obj, dict) and "name" in function_obj:
            return {
                "name": function_obj["name"],
                "arguments": self._normalize_arguments(function_obj.get("arguments", {})),
            }

        return None

    @override
    def _parse_query_response_prompting(self, api_response: Any) -> dict:
        raw_text = api_response.choices[0].text
        tool_call_ids = []

        payload_text = raw_text
        if payload_text.startswith("[TOOL_CALLS]"):
            payload_text = payload_text[len("[TOOL_CALLS]") :]
        if payload_text.endswith("[/TOOL_CALLS]"):
            payload_text = payload_text[: -len("[/TOOL_CALLS]")]

        try:
            parsed = json.loads(payload_text)
        except json.JSONDecodeError:
            return {
                "model_responses": raw_text,
                "model_responses_message_for_chat_history": raw_text,
                "tool_call_ids": tool_call_ids,
                "input_token": api_response.usage.prompt_tokens,
                "output_token": api_response.usage.completion_tokens,
            }

        # Support single dict and list payloads.
        if isinstance(parsed, dict):
            if "tool_calls" in parsed and isinstance(parsed["tool_calls"], list):
                items = parsed["tool_calls"]
            else:
                items = [parsed]
        elif isinstance(parsed, list):
            items = parsed
        elif isinstance(parsed, str):
            items = [parsed]
        else:
            items = []

        normalized = []
        for item in items:
            tool_call = self._to_tool_call_dict(item)
            if tool_call is None:
                continue
            tool_call_id = self.generate_random_string()
            tool_call["id"] = tool_call_id
            tool_call_ids.append(tool_call_id)
            normalized.append(tool_call)

        model_responses = [
            {item["name"]: item.get("arguments", {})} for item in normalized
        ]
        chat_history_content = f"[TOOL_CALLS]{json.dumps(normalized)}"

        return {
            "model_responses": model_responses,
            "model_responses_message_for_chat_history": chat_history_content,
            "tool_call_ids": tool_call_ids,
            "input_token": api_response.usage.prompt_tokens,
            "output_token": api_response.usage.completion_tokens,
        }
