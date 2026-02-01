"""
Convert Irrelevance Data to Pythonic Format

This script converts the generated irrelevance data into the same format as
the main pythonic multi-turn dialogue dataset.

Pythonic irrelevance format:
{
    "id": "run_id_000001",
    "sample_index": 1,
    "tools": [...],
    "messages": [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
    ],
    "dataset": "irrelevance",
    "total_turns": 1
}

Usage:
    python pipeline/tools/convert_to_irrelevance.py --input <input.json> --output <output.jsonl>
"""
import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from pipeline.s2_functions.parser import parse_signature


# ============================================================================
# Language utilities
# ============================================================================
def get_language() -> str:
    """獲取當前語言設定"""
    return os.getenv("LANG_CODE", "en").lower()


def get_output_filename() -> str:
    """根據語言設定獲取輸出檔名"""
    lang = get_language()
    if lang == "zh_tw":
        return "irrelevance_zh_tw.jsonl"
    return "irrelevance_eng.jsonl"


# ============================================================================
# Type conversion utilities
# ============================================================================
def python_type_to_jsonschema(t: str) -> Dict[str, Any]:
    """Convert Python type annotation to JSON Schema format."""
    t = t.strip()
    
    # Handle List[type] patterns
    list_match = re.match(r"List\[(.+)\]", t, re.IGNORECASE)
    if list_match:
        inner_type = list_match.group(1)
        return {"type": "array", "items": python_type_to_jsonschema(inner_type)}
    
    # Handle Dict[key, value] patterns
    dict_match = re.match(r"Dict\[(.+?),\s*(.+)\]", t, re.IGNORECASE)
    if dict_match:
        return {"type": "dict"}
    
    # Basic type mapping (BFCL uses "float" not "number")
    type_map = {
        "str": "string",
        "string": "string",
        "int": "integer",
        "integer": "integer",
        "float": "float",
        "number": "float",
        "double": "float",
        "bool": "boolean",
        "boolean": "boolean",
        "list": "array",
        "dict": "dict",
        "object": "dict",
        "any": "string",
    }
    
    normalized = t.lower()
    if normalized in type_map:
        result_type = type_map[normalized]
        if result_type == "array":
            return {"type": "array", "items": {"type": "string"}}
        return {"type": result_type}
    
    return {"type": "string"}


# ============================================================================
# Signature parsing and conversion
# ============================================================================
def extract_docstring(signature: str) -> Tuple[str, Dict[str, str]]:
    """
    Extract description and parameter descriptions from function signature.
    
    Returns:
        Tuple of (description, {param_name: param_description})
    """
    description_lines = []
    param_descriptions: Dict[str, str] = {}
    
    # Match docstring
    doc_match = re.search(r'"""(.*?)"""', signature, re.DOTALL) or \
                re.search(r"'''(.*?)'''", signature, re.DOTALL)
    
    if not doc_match:
        return "", {}
    
    raw_doc = doc_match.group(1)
    lines = [l.strip() for l in raw_doc.splitlines()]
    
    in_return_fields = False
    
    for line in lines:
        if not line:
            continue
        
        # Skip return_fields section
        if line.startswith(':return_fields:'):
            in_return_fields = True
            continue
        
        if in_return_fields:
            if line.startswith(':'):
                in_return_fields = False
            else:
                continue
        
        # Extract :param descriptions
        param_match = re.match(r':param\s+(\w+)\s*:\s*(.+)', line)
        if param_match:
            param_descriptions[param_match.group(1)] = param_match.group(2).strip()
            continue
        
        # Skip :return, :raises lines
        if line.startswith(':return') or line.startswith(':raises'):
            continue
        
        # Regular description line
        if not line.startswith('-'):  # Skip return field lines
            description_lines.append(line)
    
    description = ' '.join(description_lines).strip()
    return description, param_descriptions


def signature_to_pythonic_tool(signature: str) -> Optional[Dict[str, Any]]:
    """
    Convert a Python function signature to pythonic tool schema format.
    
    Pythonic tool format:
    {
        "name": "function_name",
        "description": "...",
        "parameters": {
            "type": "dict",
            "properties": {
                "param1": {
                    "type": "string",
                    "description": "..."
                }
            },
            "required": ["param1"]
        },
        "response": {
            "type": "dict",
            "properties": {...}
        }
    }
    """
    try:
        parsed = parse_signature(signature)
        if not parsed:
            return None
        
        function_name = parsed.get("function_name", "unknown")
        params = parsed.get("parameters", [])
        
        # Extract description and param descriptions from docstring
        description, param_descriptions = extract_docstring(signature)
        
        # Build properties
        properties: Dict[str, Any] = {}
        required: List[str] = []
        
        for param in params:
            param_name = param[0]
            param_type = param[1] or "string"
            default_value = param[2]
            
            prop: Dict[str, Any] = python_type_to_jsonschema(param_type)
            
            # Add description if available
            if param_name in param_descriptions:
                prop["description"] = param_descriptions[param_name]
            else:
                prop["description"] = f"The {param_name} parameter."
            
            # Add default value info if present
            if default_value is not None:
                prop["description"] += f" Default is {default_value}."
            
            properties[param_name] = prop
            
            # Add to required if no default value
            if default_value is None:
                required.append(param_name)
        
        # Build response schema from return type
        response_schema = {
            "type": "dict",
            "properties": {}
        }
        
        return {
            "name": function_name,
            "description": description or f"Function {function_name}.",
            "parameters": {
                "type": "dict",
                "properties": properties,
                "required": required,
            },
            "response": response_schema
        }
    
    except Exception as e:
        print(f"Warning: Failed to parse signature: {e}")
        return None


# ============================================================================
# Main conversion
# ============================================================================
def convert_irrelevance_data(input_path: str, output_path: str, run_id: str = None) -> int:
    """
    Convert irrelevance data from internal format to pythonic format.
    
    Args:
        input_path: Path to input JSON file
        output_path: Path to output JSONL file
        run_id: Run ID for generating sample IDs (optional)
        
    Returns:
        Number of samples converted
    """
    # Load input data
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Get run_id if not provided
    if run_id is None:
        run_id_file = "run_id_irrelevance"
        if os.path.exists(run_id_file):
            with open(run_id_file, "r") as rf:
                run_id = rf.read().strip()
        else:
            run_id = "irrelevance"
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    sample_count = 0
    
    with open(output_path, "w", encoding="utf-8") as f:
        for row in data:
            domain = row.get("domain", "")
            subdomain = row.get("subdomain", "")
            samples = row.get("samples", [])
            
            for sample in samples:
                question = sample.get("question", "")
                natural_response = sample.get("natural_response", "")
                functions = sample.get("functions", [])
                
                if not question or not functions:
                    continue
                
                # Convert functions to pythonic tool format
                tools = []
                for func in functions:
                    func_signature = func.get("function", "")
                    tool = signature_to_pythonic_tool(func_signature)
                    if tool:
                        tools.append(tool)
                
                if not tools:
                    continue
                
                # Build pythonic format sample
                # Irrelevance samples are single-turn: user asks, assistant answers in natural language
                messages = [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": natural_response}
                ]
                
                pythonic_sample = {
                    "id": f"{run_id}_{sample_count:06d}",
                    "sample_index": sample_count,
                    "tools": tools,
                    "messages": messages,
                    "dataset": "irrelevance",
                    "total_turns": 1,
                    # Extra metadata (optional)
                    "domain": domain,
                    "subdomain": subdomain
                }
                
                # Write to output
                f.write(json.dumps(pythonic_sample, ensure_ascii=False) + "\n")
                sample_count += 1
    
    print(f"Converted {sample_count} samples to pythonic format.")
    return sample_count


# ============================================================================
# CLI interface
# ============================================================================
def cli_main():
    """Command-line interface for conversion tool."""
    parser = argparse.ArgumentParser(description="Convert irrelevance data to pythonic format")
    parser.add_argument("--input", required=True, help="Input JSON file path")
    parser.add_argument("--output", required=True, help="Output JSONL file path")
    parser.add_argument("--run-id", help="Run ID for sample IDs (optional, auto-detected)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return 1
    
    count = convert_irrelevance_data(args.input, args.output, args.run_id)
    return 0


# ============================================================================
# Entry point
# ============================================================================
if __name__ == "__main__":
    exit(cli_main())

