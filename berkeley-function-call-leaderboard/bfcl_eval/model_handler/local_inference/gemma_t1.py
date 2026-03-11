import json
import re

from bfcl_eval.constants.enums import ModelStyle
from bfcl_eval.constants.type_mappings import GORILLA_TO_OPENAPI
from bfcl_eval.model_handler.local_inference.base_oss_handler import OSSHandler
from bfcl_eval.model_handler.utils import (
    combine_consecutive_user_prompts,
    convert_to_tool,
)
from overrides import override


class GemmaT1Handler(OSSHandler):
    """
    Handler for twinkle-ai/gemma-3-4B-T1-it model.

    This model is fine-tuned from google/gemma-3-4b-pt with Hermes-style function calling format,
    but uses Gemma-style turn tokens (<start_of_turn>/<end_of_turn>).

    Prompt format (with tools):
        <bos><start_of_turn>user
        {system_message} You are provided with function signatures within <tools> </tools> XML tags. ...
        <tools>
        {tools_json}
        </tools>
        ...
        {user_message}
        <end_of_turn>
        <start_of_turn>model

    Model output format:
        <tool_call>
        {"name": "function_name", "arguments": {...}}
        </tool_call>
    """

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
    def _format_prompt(self, messages, function):
        """
        Format prompt following the T1 chat_template:
        - Gemma-style turn tokens (<start_of_turn>/<end_of_turn>)
        - Hermes-style function calling (<tools>, <tool_call>)
        - System message merged into first user turn (Gemma convention)
        """
        # Convert BFCL function definitions to normalized format
        tools = convert_to_tool(function, GORILLA_TO_OPENAPI, ModelStyle.OSSMODEL)
        tools_json = json.dumps(tools, ensure_ascii=False)

        tool_call_format = '{"arguments": <args-dict>, "name": <function-name>}'

        # Extract system message if present
        if messages and messages[0]["role"] == "system":
            system_message = messages[0]["content"].strip()
            remaining_messages = messages[1:]
        else:
            system_message = "You are a function calling AI model."
            remaining_messages = messages

        # Extract first user message
        first_user_content = ""
        loop_messages = remaining_messages
        if remaining_messages and remaining_messages[0]["role"] == "user":
            first_user_content = remaining_messages[0]["content"].strip()
            loop_messages = remaining_messages[1:]

        # Build prompt: <bos> + first turn with system + tools + user message
        formatted_prompt = "<bos>"
        formatted_prompt += "<start_of_turn>user\n"
        formatted_prompt += (
            f"{system_message} "
            "You are provided with function signatures within <tools> </tools> XML tags. "
            "You may call one or more functions to assist with the user query. "
            "Don't make assumptions about what values to plug into functions. "
            "Here are the available tools:\n\n"
            f"<tools>\n{tools_json}\n</tools>\n\n"
            "For each function call return a json object with function name and arguments "
            "within <tool_call> </tool_call> tags with the following schema:\n"
            f"<tool_call>\n{tool_call_format}\n</tool_call>"
        )

        if first_user_content:
            formatted_prompt += f"\n\n{first_user_content}"

        formatted_prompt += "<end_of_turn>\n"

        # Add remaining conversation turns
        for message in loop_messages:
            role = message["role"]
            if role == "assistant":
                role = "model"
            content = message["content"].strip()
            formatted_prompt += f"<start_of_turn>{role}\n{content}<end_of_turn>\n"

        # Prompt model to generate
        formatted_prompt += "<start_of_turn>model\n"

        return formatted_prompt

    @override
    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        functions: list = test_entry["function"]

        # Do NOT use system_prompt_pre_processing_chat_model here.
        # The T1 handler formats tools in <tools> JSON tags within _format_prompt,
        # instead of using text-based system prompt for function definitions.

        for round_idx in range(len(test_entry["question"])):
            test_entry["question"][round_idx] = combine_consecutive_user_prompts(
                test_entry["question"][round_idx]
            )
            test_entry["question"][round_idx] = self._substitute_prompt_role(
                test_entry["question"][round_idx]
            )

        return {"message": [], "function": functions}

    @staticmethod
    def _substitute_prompt_role(prompts: list[dict]) -> list[dict]:
        """Gemma uses 'model' instead of 'assistant'."""
        for prompt in prompts:
            if prompt["role"] == "assistant":
                prompt["role"] = "model"
        return prompts

    @override
    def decode_ast(self, result, language, has_tool_call_tag):
        """Parse Hermes-style <tool_call> tags from model output."""
        func_calls = []
        pattern = r"<tool_call>\s*(.*?)\s*</tool_call>"
        matches = re.findall(pattern, result, re.DOTALL)

        for match in matches:
            try:
                tool_result = json.loads(match.strip())
                if language != "Python":
                    for key in tool_result.get("arguments", {}):
                        tool_result["arguments"][key] = str(
                            tool_result["arguments"][key]
                        )
                func_calls.append({tool_result["name"]: tool_result["arguments"]})
            except (json.JSONDecodeError, KeyError):
                continue

        return func_calls

    @override
    def decode_execute(self, result, has_tool_call_tag):
        """Parse <tool_call> tags and convert to executable function call strings."""
        pattern = r"<tool_call>\s*(.*?)\s*</tool_call>"
        matches = re.findall(pattern, result, re.DOTALL)

        execution_list = []
        for match in matches:
            try:
                tool_result = json.loads(match.strip())
                func_name = tool_result["name"]
                args = tool_result["arguments"]
                execution_list.append(
                    f"{func_name}({','.join([f'{k}={repr(v)}' for k, v in args.items()])})"
                )
            except (json.JSONDecodeError, KeyError):
                continue

        return execution_list
