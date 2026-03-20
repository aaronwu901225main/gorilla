import json
import time
from typing import Any

from bfcl_eval.constants.enums import ModelStyle
from bfcl_eval.constants.type_mappings import GORILLA_TO_OPENAPI
from bfcl_eval.model_handler.base_handler import BaseHandler
from bfcl_eval.model_handler.local_inference.mistral_fc import MistralFCHandler
from bfcl_eval.model_handler.utils import convert_to_tool
from overrides import override


class MistralSmall32FCHandler(MistralFCHandler):
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
        for tool in tools:
            if isinstance(tool, dict):
                function_obj = tool.get("function")
                if isinstance(function_obj, dict):
                    function_obj.pop("response", None)

        inference_data["tools"] = tools
        return inference_data

    @override
    def _query_FC(self, inference_data: dict):
        message = inference_data["message"]
        tools = inference_data["tools"]
        inference_data["inference_input_log"] = {
            "message": message,
            "tools": tools,
        }

        start_time = time.time()
        api_response = self.client.chat.completions.create(
            model=self.model_path_or_id,
            messages=message,
            tools=tools,
            temperature=self.temperature,
            timeout=72000,
        )
        end_time = time.time()

        return api_response, end_time - start_time

    @override
    def _parse_query_response_FC(self, api_response: Any) -> dict:
        message = api_response.choices[0].message
        tool_calls = message.tool_calls or []

        model_responses = []
        tool_call_func_names = []
        tool_call_ids = []

        for tool_call in tool_calls:
            func_name = tool_call.function.name
            raw_args = tool_call.function.arguments
            normalized_args = self._normalize_arguments(raw_args)

            model_responses.append({func_name: normalized_args})
            tool_call_func_names.append(func_name)
            tool_call_ids.append(tool_call.id)

        # Keep assistant message in OpenAI chat history format for next turn.
        assistant_message = {
            "role": "assistant",
            "content": message.content or "",
        }
        if tool_calls:
            assistant_message["tool_calls"] = [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    },
                }
                for tool_call in tool_calls
            ]

        return {
            "model_responses": model_responses if tool_calls else (message.content or ""),
            "model_responses_message_for_chat_history": assistant_message,
            "tool_call_func_names": tool_call_func_names,
            "tool_call_ids": tool_call_ids,
            "input_token": api_response.usage.prompt_tokens,
            "output_token": api_response.usage.completion_tokens,
        }

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
