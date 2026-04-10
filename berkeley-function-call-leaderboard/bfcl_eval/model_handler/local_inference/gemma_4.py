import ast
import json
import re
from typing import Any, Optional

from bfcl_eval.constants.enums import ModelStyle, ReturnFormat
from bfcl_eval.constants.type_mappings import GORILLA_TO_OPENAPI
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
