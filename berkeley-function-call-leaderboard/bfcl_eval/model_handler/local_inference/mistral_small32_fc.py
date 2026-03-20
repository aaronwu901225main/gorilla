import json
import time
import re
import copy
from typing import Any

from bfcl_eval.constants.enums import ModelStyle
from bfcl_eval.constants.type_mappings import GORILLA_TO_OPENAPI
from bfcl_eval.model_handler.base_handler import BaseHandler
from bfcl_eval.model_handler.local_inference.mistral_fc import MistralFCHandler
from bfcl_eval.model_handler.utils import convert_to_tool
from overrides import override


class MistralSmall32FCHandler(MistralFCHandler):
    _CONCAT_CALL_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_\.\-]*")
    _VALID_TOOL_NAME_RE = re.compile(r"[^A-Za-z0-9_\-]")

    @override
    def inference(
        self,
        test_entry: dict,
        include_input_log: bool,
        exclude_state_log: bool,
    ):
        # Route back to BaseHandler so FC models use the FC pipeline.
        return BaseHandler.inference(
            self,
            test_entry=test_entry,
            include_input_log=include_input_log,
            exclude_state_log=exclude_state_log,
        )

    @override
    def _pre_query_processing_FC(self, inference_data: dict, test_entry: dict) -> dict:
        inference_data["message"] = []
        return inference_data

    @override
    def _compile_tools(self, inference_data: dict, test_entry: dict) -> dict:
        functions: list = test_entry["function"]
        # Keep tool schema compatible with Mistral/OpenAI chat tools.
        tools = convert_to_tool(
            functions,
            GORILLA_TO_OPENAPI,
            ModelStyle.MISTRAL,
        )

        # vLLM OpenAI-compatible chat tools reject BFCL's optional return schema field.
        # Also normalize function names to OpenAI-compatible charset.
        alias_to_original: dict[str, str] = {}
        original_to_alias: dict[str, str] = {}
        for tool in tools:
            if isinstance(tool, dict):
                function_obj = tool.get("function")
                if isinstance(function_obj, dict):
                    function_obj.pop("response", None)

                    original_name = function_obj.get("name", "")
                    alias_name = self._sanitize_tool_name(original_name)
                    function_obj["name"] = alias_name
                    alias_to_original[alias_name] = original_name
                    original_to_alias[original_name] = alias_name

        self._tool_name_alias_to_original = alias_to_original
        self._tool_name_original_to_alias = original_to_alias

        inference_data["tools"] = tools
        return inference_data

    @override
    def _query_FC(self, inference_data: dict):
        message = inference_data["message"]
        tools = inference_data["tools"]

        query_message, max_tokens = self._fit_messages_and_budget(message, tools)
        inference_data["message"] = query_message

        inference_data["inference_input_log"] = {
            "message": query_message,
            "tools": tools,
            "max_tokens": max_tokens,
        }

        start_time = time.time()
        api_response = self.client.chat.completions.create(
            model=self.model_path_or_id,
            messages=query_message,
            tools=tools,
            temperature=self.temperature,
            max_tokens=max_tokens,
            timeout=72000,
        )
        end_time = time.time()

        return api_response, end_time - start_time

    @override
    def _parse_query_response_FC(self, api_response: Any) -> dict:
        message = api_response.choices[0].message
        tool_calls = message.tool_calls or []
        content = message.content or ""

        # vLLM's mistral tool parser may fail on concatenated calls like:
        # foo{...}bar{...}. Recover calls directly from content as fallback.
        fallback_calls = self._extract_concatenated_calls(content)
        if len(fallback_calls) > len(tool_calls):
            tool_calls = fallback_calls

        model_responses = []
        tool_call_func_names = []
        tool_call_ids = []

        for tool_call in tool_calls:
            if isinstance(tool_call, dict):
                function_obj = tool_call.get("function", {})
                func_name_alias = function_obj.get("name", "")
                func_name = self._restore_tool_name(func_name_alias)
                raw_args = function_obj.get("arguments", {})
                tool_call_id = tool_call.get("id", self.generate_random_string())
            else:
                func_name_alias = tool_call.function.name
                func_name = self._restore_tool_name(func_name_alias)
                raw_args = tool_call.function.arguments
                tool_call_id = tool_call.id

            if not func_name:
                continue

            normalized_args = self._normalize_arguments(raw_args)

            model_responses.append({func_name: normalized_args})
            tool_call_func_names.append(func_name)
            tool_call_ids.append(tool_call_id)

        # Keep assistant message in OpenAI chat history format for next turn.
        assistant_message = {
            "role": "assistant",
            "content": content,
        }
        if tool_calls:
            assistant_message["tool_calls"] = [
                {
                    "id": (
                        tool_call["id"]
                        if isinstance(tool_call, dict)
                        else tool_call.id
                    ),
                    "type": "function",
                    "function": {
                        "name": (
                            tool_call["function"]["name"]
                            if isinstance(tool_call, dict)
                            else tool_call.function.name
                        ),
                        "arguments": (
                            tool_call["function"]["arguments"]
                            if isinstance(tool_call, dict)
                            else tool_call.function.arguments
                        ),
                    },
                }
                for tool_call in tool_calls
            ]

        return {
            "model_responses": model_responses if tool_calls else content,
            "model_responses_message_for_chat_history": assistant_message,
            "tool_call_func_names": tool_call_func_names,
            "tool_call_ids": tool_call_ids,
            "input_token": api_response.usage.prompt_tokens,
            "output_token": api_response.usage.completion_tokens,
        }

    def _extract_concatenated_calls(self, content: str) -> list[dict]:
        if not content:
            return []

        decoder = json.JSONDecoder()
        i = 0
        parsed_calls: list[dict] = []
        n = len(content)

        while i < n:
            while i < n and content[i].isspace():
                i += 1
            if i >= n:
                break

            name_match = self._CONCAT_CALL_NAME_RE.match(content, i)
            if not name_match:
                i += 1
                continue

            func_name = name_match.group(0)
            j = name_match.end()
            while j < n and content[j].isspace():
                j += 1
            if j >= n or content[j] != "{":
                i = name_match.end()
                continue

            try:
                args_obj, consumed = decoder.raw_decode(content[j:])
            except json.JSONDecodeError:
                i = j + 1
                continue

            if isinstance(args_obj, dict):
                func_name_alias = self._tool_name_original_to_alias.get(
                    func_name, self._sanitize_tool_name(func_name)
                )
                parsed_calls.append(
                    {
                        "id": self.generate_random_string(),
                        "type": "function",
                        "function": {
                            "name": func_name_alias,
                            "arguments": json.dumps(args_obj, ensure_ascii=True),
                        },
                    }
                )

            i = j + consumed

        return parsed_calls

    @override
    def add_first_turn_message_FC(
        self, inference_data: dict, first_turn_message: list[dict]
    ) -> dict:
        inference_data["message"].extend(first_turn_message)
        return inference_data

    @override
    def _add_next_turn_user_message_FC(
        self, inference_data: dict, user_message: list[dict]
    ) -> dict:
        inference_data["message"].extend(user_message)
        return inference_data

    @override
    def _add_assistant_message_FC(
        self, inference_data: dict, model_response_data: dict
    ) -> dict:
        assistant_message = model_response_data["model_responses_message_for_chat_history"]
        content = assistant_message.get("content", "")
        tool_calls = assistant_message.get("tool_calls", None)

        # vLLM rejects assistant messages with both empty content and no tool calls.
        if content == "" and not tool_calls:
            return inference_data

        inference_data["message"].append(assistant_message)
        return inference_data

    @override
    def _add_execution_results_FC(
        self, inference_data: dict, execution_results: list[str], model_response_data: dict
    ) -> dict:
        for execution_result, func_name, tool_call_id in zip(
            execution_results,
            model_response_data["tool_call_func_names"],
            model_response_data["tool_call_ids"],
        ):
            inference_data["message"].append(
                {
                    "role": "tool",
                    "name": func_name,
                    "content": execution_result,
                    "tool_call_id": tool_call_id,
                }
            )
        return inference_data

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
            name_alias = item["name"]
            return {
                "name": self._restore_tool_name(name_alias),
                "arguments": self._normalize_arguments(item.get("arguments", {})),
            }

        # OpenAI/vLLM chat style: {"function": {"name": "...", "arguments": "..."}, ...}
        function_obj = item.get("function")
        if isinstance(function_obj, dict) and "name" in function_obj:
            name_alias = function_obj["name"]
            return {
                "name": self._restore_tool_name(name_alias),
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

    def _sanitize_tool_name(self, name: str) -> str:
        if not isinstance(name, str) or not name:
            return "tool"

        sanitized = self._VALID_TOOL_NAME_RE.sub("_", name)
        sanitized = sanitized.strip("_")
        if not sanitized:
            sanitized = "tool"

        if not re.match(r"^[A-Za-z0-9_\-]+$", sanitized):
            sanitized = "tool"

        return sanitized[:64]

    def _restore_tool_name(self, alias_name: str) -> str:
        if not isinstance(alias_name, str):
            return ""
        alias_map = getattr(self, "_tool_name_alias_to_original", {})
        return alias_map.get(alias_name, alias_name)

    def _fit_messages_and_budget(self, message: list[dict], tools: list[dict]) -> tuple[list[dict], int]:
        max_ctx = getattr(self, "max_context_length", None)
        if not isinstance(max_ctx, int) or max_ctx <= 0:
            return message, 1024

        safe_margin = 32
        # 0.11.0 can return negative max_tokens if prompt nearly fills context.
        min_completion = 1
        max_completion_cap = 2048

        trimmed_message = copy.deepcopy(message)
        token_count = self._estimate_chat_tokens(trimmed_message, tools)

        while token_count >= (max_ctx - safe_margin) and len(trimmed_message) > 1:
            drop_idx = 0
            if trimmed_message[0].get("role") == "system":
                drop_idx = 1 if len(trimmed_message) > 1 else 0
            trimmed_message.pop(drop_idx)
            token_count = self._estimate_chat_tokens(trimmed_message, tools)

        leftover = max_ctx - token_count - safe_margin
        max_tokens = max(min_completion, min(max_completion_cap, leftover))
        return trimmed_message, max_tokens

    def _estimate_chat_tokens(self, message: list[dict], tools: list[dict]) -> int:
        try:
            payload = json.dumps(
                {
                    "messages": message,
                    "tools": tools,
                },
                ensure_ascii=True,
                separators=(",", ":"),
            )
            return len(self.tokenizer.tokenize(payload))
        except Exception:
            approx_chars = len(str(message)) + len(str(tools))
            return max(1, approx_chars // 4)
