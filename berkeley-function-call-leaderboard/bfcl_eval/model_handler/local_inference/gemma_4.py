import ast
import copy
import json
import re
import time
from typing import Any, Optional
from uuid import uuid4

from bfcl_eval.constants.enums import ModelStyle, ReturnFormat
from bfcl_eval.constants.type_mappings import GORILLA_TO_OPENAPI
from bfcl_eval.model_handler.base_handler import BaseHandler
from bfcl_eval.model_handler.local_inference.base_oss_handler import OSSHandler
from bfcl_eval.model_handler.utils import (
    combine_consecutive_user_prompts,
    convert_to_function_call,
    convert_to_tool,
    default_decode_ast_prompting,
    system_prompt_pre_processing_chat_model,
)
from overrides import override


class Gemma4Handler(OSSHandler):
    def __init__(
        self,
        model_name,
        temperature,
        registry_name,
        is_fc_model,
        dtype="bfloat16",
        **kwargs,
    ) -> None:
        super().__init__(model_name, temperature, registry_name, is_fc_model, **kwargs)

    @override
    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        functions: list = test_entry["function"]

        # Prompt model: keep BFCL's default instruction scaffolding.
        if not self.is_fc_model:
            test_entry["question"][0] = system_prompt_pre_processing_chat_model(
                test_entry["question"][0], functions, test_entry["id"]
            )

        for round_idx in range(len(test_entry["question"])):
            test_entry["question"][round_idx] = combine_consecutive_user_prompts(
                test_entry["question"][round_idx]
            )

        return {"message": [], "function": functions}

    @override
    def _format_prompt(self, messages, function):
        # Prefer tokenizer-native chat template for Gemma4.
        tools = None
        if self.is_fc_model and function:
            tools = convert_to_tool(function, GORILLA_TO_OPENAPI, ModelStyle.OSSMODEL)

        try:
            kwargs = {
                "tokenize": False,
                "add_generation_prompt": True,
            }
            if tools:
                kwargs["tools"] = tools
            return self.tokenizer.apply_chat_template(messages, **kwargs)
        except Exception:
            return self._fallback_format_prompt(messages)

    @staticmethod
    def _fallback_format_prompt(messages: list[dict]) -> str:
        formatted_prompt = "<bos>"

        if messages and messages[0]["role"] == "system":
            first_user_prefix = messages[0]["content"].strip() + "\n\n"
            messages = messages[1:]
        else:
            first_user_prefix = ""

        is_first = True
        for message in messages:
            role = message["role"]
            content = str(message.get("content", "")).strip()

            if role == "assistant":
                role = "model"
            elif role == "tool":
                role = "user"
                content = f"<tool_response>\n{content}\n</tool_response>"

            formatted_prompt += (
                f"<start_of_turn>{role}\n"
                f"{first_user_prefix if is_first else ''}{content}<end_of_turn>\n"
            )
            is_first = False

        formatted_prompt += "<start_of_turn>model\n"
        return formatted_prompt

    @override
    def decode_ast(self, result, language, has_tool_call_tag):
        return self._robust_decode(result, language, has_tool_call_tag)

    @override
    def decode_execute(self, result, has_tool_call_tag):
        decoded = self._robust_decode(result, ReturnFormat.PYTHON, has_tool_call_tag)
        return convert_to_function_call(decoded)

    def _robust_decode(
        self,
        result: str,
        language: ReturnFormat = ReturnFormat.PYTHON,
        has_tool_call_tag: bool = False,
    ) -> list[dict]:
        text = result.strip()

        # 1) Try BFCL default python-call parser first.
        try:
            decoded = default_decode_ast_prompting(text, language, has_tool_call_tag)
            if self._is_valid_decoded_output(decoded):
                return decoded
        except Exception:
            pass

        # 2) Parse <tool_call> JSON blocks.
        decoded = self._parse_tool_call_json_blocks(text)
        if decoded:
            return decoded

        # 3) Parse JSON fenced blocks that contain {name, arguments} payloads.
        decoded = self._parse_fenced_json_calls(text)
        if decoded:
            return decoded

        # 4) Parse Gemma-style call notations (call:name{...}, <call:name(...)>, etc.).
        decoded = self._parse_call_notation(text)
        if decoded:
            return decoded

        # 5) Parse loose python-style function(...) calls embedded in text.
        decoded = self._parse_python_style_calls(text)
        if decoded:
            return decoded

        return []

    @staticmethod
    def _is_valid_decoded_output(decoded: Any) -> bool:
        if type(decoded) != list:
            return False
        for item in decoded:
            if type(item) != dict or len(item) != 1:
                return False
            args = list(item.values())[0]
            if type(args) != dict:
                return False
        return True

    @staticmethod
    def _parse_tool_call_json_blocks(text: str) -> list[dict]:
        calls: list[dict] = []
        pattern = r"<tool_call>\s*(.*?)\s*</tool_call>"
        for block in re.findall(pattern, text, re.DOTALL):
            obj = Gemma4Handler._safe_json_load(block)
            if type(obj) == dict and "name" in obj and "arguments" in obj:
                name = obj["name"]
                args = obj["arguments"] if type(obj["arguments"]) == dict else {}
                calls.append({str(name): args})
        return calls

    @staticmethod
    def _parse_fenced_json_calls(text: str) -> list[dict]:
        calls: list[dict] = []
        fenced_blocks = re.findall(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        for block in fenced_blocks:
            obj = Gemma4Handler._safe_json_load(block)
            if type(obj) == dict and "name" in obj and "arguments" in obj:
                args = obj["arguments"] if type(obj["arguments"]) == dict else {}
                calls.append({str(obj["name"]): args})
            elif type(obj) == list:
                for item in obj:
                    if type(item) == dict and "name" in item and "arguments" in item:
                        args = item["arguments"] if type(item["arguments"]) == dict else {}
                        calls.append({str(item["name"]): args})
        return calls

    @staticmethod
    def _parse_call_notation(text: str) -> list[dict]:
        calls: list[dict] = []
        idx = 0

        while True:
            start = text.find("call:", idx)
            if start < 0:
                break

            pos = start + len("call:")
            while pos < len(text) and text[pos].isspace():
                pos += 1

            name_start = pos
            while pos < len(text) and re.match(r"[A-Za-z0-9_.-]", text[pos]):
                pos += 1
            func_name = text[name_start:pos]

            if not func_name:
                idx = start + 1
                continue

            while pos < len(text) and text[pos].isspace():
                pos += 1

            if pos >= len(text) or text[pos] not in "{(":
                calls.append({func_name: {}})
                idx = pos
                continue

            open_char = text[pos]
            close_char = "}" if open_char == "{" else ")"
            span_end = Gemma4Handler._find_matching_bracket(text, pos, open_char, close_char)
            if span_end is None:
                idx = pos + 1
                continue

            payload = text[pos + 1 : span_end]
            if open_char == "{":
                args = Gemma4Handler._parse_key_value_args(payload, separators=(":", "="))
            else:
                args = Gemma4Handler._parse_key_value_args(payload, separators=("=", ":"))
            calls.append({func_name: args})
            idx = span_end + 1

        return calls

    @staticmethod
    def _parse_python_style_calls(text: str) -> list[dict]:
        calls: list[dict] = []
        pattern = re.compile(r"([A-Za-z_][A-Za-z0-9_.]*)\s*\(")
        idx = 0
        while True:
            match = pattern.search(text, idx)
            if not match:
                break

            func_name = match.group(1)
            open_pos = match.end() - 1
            close_pos = Gemma4Handler._find_matching_bracket(text, open_pos, "(", ")")
            if close_pos is None:
                idx = match.end()
                continue

            payload = text[open_pos + 1 : close_pos]
            args = Gemma4Handler._parse_key_value_args(payload, separators=("=", ":"))
            calls.append({func_name: args})
            idx = close_pos + 1

        return calls

    @staticmethod
    def _find_matching_bracket(
        text: str, start_idx: int, open_char: str, close_char: str
    ) -> Optional[int]:
        depth = 0
        in_quote: Optional[str] = None
        escaped = False

        for i in range(start_idx, len(text)):
            ch = text[i]

            if in_quote:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == in_quote:
                    in_quote = None
                continue

            if ch in ("'", '"'):
                in_quote = ch
                continue

            if ch == open_char:
                depth += 1
            elif ch == close_char:
                depth -= 1
                if depth == 0:
                    return i

        return None

    @staticmethod
    def _split_top_level(payload: str) -> list[str]:
        parts: list[str] = []
        buf: list[str] = []
        depth_curly = 0
        depth_square = 0
        depth_round = 0
        in_quote: Optional[str] = None
        escaped = False

        for ch in payload:
            if in_quote:
                buf.append(ch)
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == in_quote:
                    in_quote = None
                continue

            if ch in ("'", '"'):
                in_quote = ch
                buf.append(ch)
                continue

            if ch == "{":
                depth_curly += 1
            elif ch == "}":
                depth_curly -= 1
            elif ch == "[":
                depth_square += 1
            elif ch == "]":
                depth_square -= 1
            elif ch == "(":
                depth_round += 1
            elif ch == ")":
                depth_round -= 1

            if ch == "," and depth_curly == 0 and depth_square == 0 and depth_round == 0:
                part = "".join(buf).strip()
                if part:
                    parts.append(part)
                buf = []
            else:
                buf.append(ch)

        tail = "".join(buf).strip()
        if tail:
            parts.append(tail)
        return parts

    @staticmethod
    def _parse_key_value_args(payload: str, separators: tuple[str, str]) -> dict:
        args: dict[str, Any] = {}
        for item in Gemma4Handler._split_top_level(payload):
            split_idx = -1
            split_char = None
            for sep in separators:
                candidate = item.find(sep)
                if candidate >= 0 and (split_idx < 0 or candidate < split_idx):
                    split_idx = candidate
                    split_char = sep

            if split_idx < 0 or split_char is None:
                continue

            key = item[:split_idx].strip().strip("'\"")
            raw_val = item[split_idx + len(split_char) :].strip()
            args[key] = Gemma4Handler._parse_value(raw_val)

        return args

    @staticmethod
    def _parse_value(value: str) -> Any:
        lowered = value.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        if lowered == "null" or lowered == "none":
            return None

        parsed = Gemma4Handler._safe_json_load(value)
        if parsed is not None:
            return parsed

        try:
            return ast.literal_eval(value)
        except Exception:
            return value.strip("'\"")

    @staticmethod
    def _safe_json_load(raw: str) -> Any:
        try:
            return json.loads(raw)
        except Exception:
            return None


class Gemma4FCHandler(Gemma4Handler):
    def __init__(
        self,
        model_name,
        temperature,
        registry_name,
        is_fc_model,
        dtype="bfloat16",
        **kwargs,
    ) -> None:
        super().__init__(
            model_name,
            temperature,
            registry_name,
            is_fc_model,
            dtype,
            **kwargs,
        )

    @override
    def inference(
        self,
        test_entry: dict,
        include_input_log: bool,
        exclude_state_log: bool,
    ):
        # OSSHandler always routes to prompting. FC models should use BaseHandler's FC path.
        return BaseHandler.inference(
            self,
            test_entry=test_entry,
            include_input_log=include_input_log,
            exclude_state_log=exclude_state_log,
        )

    @override
    def decode_ast(self, result, language, has_tool_call_tag):
        if isinstance(result, list):
            return result
        return super().decode_ast(result, language, has_tool_call_tag)

    @override
    def decode_execute(self, result, has_tool_call_tag):
        if isinstance(result, list):
            return convert_to_function_call(result)
        return super().decode_execute(result, has_tool_call_tag)

    @override
    def _pre_query_processing_FC(self, inference_data: dict, test_entry: dict) -> dict:
        inference_data["message"] = []
        return inference_data

    @override
    def _compile_tools(self, inference_data: dict, test_entry: dict) -> dict:
        functions: list = test_entry["function"]
        tools = convert_to_tool(functions, GORILLA_TO_OPENAPI, ModelStyle.OPENAI_COMPLETIONS)

        # Keep a reversible mapping because OpenAI-compatible tools sanitize some names.
        alias_to_original: dict[str, str] = {}
        original_to_alias: dict[str, str] = {}

        for original_func, tool in zip(functions, tools):
            if not isinstance(tool, dict):
                continue

            function_obj = tool.get("function")
            if not isinstance(function_obj, dict):
                continue

            function_obj.pop("response", None)

            original_name = str(original_func.get("name", ""))
            alias_name = str(function_obj.get("name", original_name))

            if original_name:
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
            "message_count_before_trim": len(message),
            "message_count_after_trim": len(query_message),
        }

        kwargs = {
            "model": self.model_path_or_id,
            "messages": query_message,
            "temperature": self.temperature,
            "max_tokens": max_tokens,
            "timeout": 72000,
        }
        if tools:
            kwargs["tools"] = tools

        start_time = time.time()
        api_response = self.client.chat.completions.create(**kwargs)
        end_time = time.time()

        return api_response, end_time - start_time

    @override
    def _parse_query_response_FC(self, api_response: Any) -> dict:
        message = api_response.choices[0].message
        content = message.content or ""
        parsed_calls: list[dict] = []
        tool_call_ids: list[str] = []
        tool_call_func_names: list[str] = []
        assistant_tool_calls: list[dict] = []

        raw_tool_calls = getattr(message, "tool_calls", None) or []
        for raw_tool_call in raw_tool_calls:
            if isinstance(raw_tool_call, dict):
                function_obj = raw_tool_call.get("function", {})
                alias_name = str(function_obj.get("name", ""))
                raw_arguments = function_obj.get("arguments", {})
                tool_call_id = str(raw_tool_call.get("id") or self._new_tool_call_id())
            else:
                alias_name = str(raw_tool_call.function.name)
                raw_arguments = raw_tool_call.function.arguments
                tool_call_id = str(raw_tool_call.id or self._new_tool_call_id())

            func_name = self._restore_tool_name(alias_name)
            if not func_name:
                continue

            normalized_args = self._normalize_arguments(raw_arguments)
            parsed_calls.append({func_name: normalized_args})
            tool_call_ids.append(tool_call_id)
            tool_call_func_names.append(func_name)
            assistant_tool_calls.append(
                {
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": alias_name,
                        "arguments": json.dumps(normalized_args, ensure_ascii=True),
                    },
                }
            )

        # Some responses may still return tool calls in plain text. Re-parse from content as fallback.
        if len(parsed_calls) == 0 and content:
            fallback_calls = self._robust_decode(
                result=content,
                language=ReturnFormat.PYTHON,
                has_tool_call_tag=False,
            )

            for item in fallback_calls:
                if not isinstance(item, dict) or len(item) != 1:
                    continue

                func_name = str(list(item.keys())[0])
                normalized_args = item[func_name]
                if not isinstance(normalized_args, dict):
                    normalized_args = {}

                alias_name = self._tool_name_original_to_alias.get(func_name, func_name)
                tool_call_id = self._new_tool_call_id()

                parsed_calls.append({func_name: normalized_args})
                tool_call_ids.append(tool_call_id)
                tool_call_func_names.append(func_name)
                assistant_tool_calls.append(
                    {
                        "id": tool_call_id,
                        "type": "function",
                        "function": {
                            "name": alias_name,
                            "arguments": json.dumps(normalized_args, ensure_ascii=True),
                        },
                    }
                )

        assistant_message: dict[str, Any] = {
            "role": "assistant",
            "content": content,
        }
        if assistant_tool_calls:
            assistant_message["tool_calls"] = assistant_tool_calls

        usage = getattr(api_response, "usage", None)
        input_token = getattr(usage, "prompt_tokens", 0)
        output_token = getattr(usage, "completion_tokens", 0)

        response_data = {
            "model_responses": parsed_calls if parsed_calls else content,
            "model_responses_message_for_chat_history": assistant_message,
            "tool_call_ids": tool_call_ids,
            "tool_call_func_names": tool_call_func_names,
            "input_token": input_token,
            "output_token": output_token,
        }

        reasoning_content = getattr(message, "reasoning_content", None)
        if reasoning_content:
            response_data["reasoning_content"] = reasoning_content

        return response_data

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

        # vLLM chat API rejects assistant messages with neither textual content nor tool calls.
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
            model_response_data.get("tool_call_func_names", []),
            model_response_data.get("tool_call_ids", []),
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

    @staticmethod
    def _new_tool_call_id() -> str:
        return f"call_{uuid4().hex[:24]}"

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
        min_completion = 1
        max_completion_cap = 2048

        trimmed_message = copy.deepcopy(message)
        token_count = self._estimate_chat_tokens(trimmed_message, tools)

        has_system = len(trimmed_message) > 0 and trimmed_message[0].get("role") == "system"
        min_keep = 2 if has_system else 1
        budget_limit = max_ctx - safe_margin - min_completion

        # Drop the oldest history first, but preserve the newest message.
        while token_count > budget_limit and len(trimmed_message) > min_keep:
            drop_idx = 1 if has_system else 0
            if drop_idx >= len(trimmed_message) - 1:
                break
            trimmed_message.pop(drop_idx)
            token_count = self._estimate_chat_tokens(trimmed_message, tools)

        # Edge case: if even the minimum kept messages overflow, truncate oldest kept user content.
        if token_count > budget_limit:
            candidate_idx = 1 if has_system and len(trimmed_message) > 1 else 0
            candidate = trimmed_message[candidate_idx]
            content = candidate.get("content", "")
            if isinstance(content, str) and len(content) > 32:
                low, high = 32, len(content)
                best = content[-32:]
                while low <= high:
                    mid = (low + high) // 2
                    trial_message = copy.deepcopy(trimmed_message)
                    trial_message[candidate_idx]["content"] = content[-mid:]
                    trial_tokens = self._estimate_chat_tokens(trial_message, tools)
                    if trial_tokens <= budget_limit:
                        best = content[-mid:]
                        low = mid + 1
                    else:
                        high = mid - 1

                trimmed_message[candidate_idx]["content"] = best
                token_count = self._estimate_chat_tokens(trimmed_message, tools)

        leftover = max_ctx - token_count - safe_margin
        max_tokens = max(min_completion, min(max_completion_cap, leftover))
        return trimmed_message, int(max_tokens)

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
