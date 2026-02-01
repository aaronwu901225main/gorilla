"""
Validate Irrelevance Data

This script validates the generated irrelevance data to ensure:
1. Each sample has valid pythonic format
2. Each sample has tools and messages
3. Tools have proper schema format
4. Messages follow the correct structure

Usage:
    python pipeline/tools/validate_irrelevance.py <input.jsonl>
"""
import argparse
import json
import os
import sys
from typing import Any, Dict, List, Tuple


def validate_tool_schema(tool: Dict[str, Any]) -> List[str]:
    """Validate a tool schema and return list of issues."""
    issues = []
    
    if not tool.get("name"):
        issues.append("Tool missing 'name' field")
    
    if not tool.get("description"):
        issues.append("Tool missing 'description' field")
    
    params = tool.get("parameters", {})
    if not isinstance(params, dict):
        issues.append("Tool 'parameters' should be a dict")
    else:
        if params.get("type") != "dict":
            issues.append("Tool parameters 'type' should be 'dict'")
        
        properties = params.get("properties", {})
        if not isinstance(properties, dict):
            issues.append("Tool parameters 'properties' should be a dict")
        else:
            for prop_name, prop_def in properties.items():
                if not isinstance(prop_def, dict):
                    issues.append(f"Property '{prop_name}' should be a dict")
                elif "type" not in prop_def:
                    issues.append(f"Property '{prop_name}' missing 'type'")
        
        required = params.get("required", [])
        if not isinstance(required, list):
            issues.append("Tool parameters 'required' should be a list")
    
    # Check response schema (optional but should be dict if present)
    if "response" in tool:
        response = tool.get("response")
        if not isinstance(response, dict):
            issues.append("Tool 'response' should be a dict")
    
    return issues


def validate_sample(sample: Dict[str, Any], line_num: int) -> Tuple[bool, List[str]]:
    """Validate a single sample and return (is_valid, issues)."""
    issues = []
    
    # Check required fields
    if not sample.get("id"):
        issues.append("Missing 'id' field")
    
    if "sample_index" not in sample:
        issues.append("Missing 'sample_index' field")
    
    if sample.get("dataset") != "irrelevance":
        issues.append("'dataset' should be 'irrelevance'")
    
    if sample.get("total_turns") != 1:
        issues.append("'total_turns' should be 1 for irrelevance samples")
    
    # Check tools
    tools = sample.get("tools")
    if not tools:
        issues.append("Missing 'tools' field")
    elif not isinstance(tools, list):
        issues.append("'tools' should be a list")
    elif len(tools) == 0:
        issues.append("'tools' list is empty")
    else:
        for tool_idx, tool in enumerate(tools):
            tool_issues = validate_tool_schema(tool)
            for issue in tool_issues:
                issues.append(f"Tool {tool_idx}: {issue}")
    
    # Check messages
    messages = sample.get("messages")
    if not messages:
        issues.append("Missing 'messages' field")
    elif not isinstance(messages, list):
        issues.append("'messages' should be a list")
    elif len(messages) != 2:
        issues.append("'messages' should have exactly 2 messages for irrelevance (user + assistant)")
    else:
        # Check first message is user
        if messages[0].get("role") != "user":
            issues.append("First message role should be 'user'")
        if not messages[0].get("content"):
            issues.append("First message missing 'content'")
        
        # Check second message is assistant
        if messages[1].get("role") != "assistant":
            issues.append("Second message role should be 'assistant'")
        if not messages[1].get("content"):
            issues.append("Second message missing 'content'")
        
        # Ensure no tool calls in irrelevance samples
        if messages[1].get("tool_calls"):
            issues.append("Irrelevance samples should not have tool_calls in assistant message")
    
    is_valid = len(issues) == 0
    return is_valid, issues


def validate_file(input_path: str, verbose: bool = True) -> Tuple[int, int, List[str]]:
    """
    Validate an irrelevance JSONL file.
    
    Returns:
        (valid_count, total_count, all_issues)
    """
    valid_count = 0
    total_count = 0
    all_issues = []
    
    with open(input_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            total_count += 1
            
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as e:
                all_issues.append(f"Line {line_num}: Invalid JSON - {e}")
                continue
            
            is_valid, issues = validate_sample(sample, line_num)
            
            if is_valid:
                valid_count += 1
            else:
                for issue in issues:
                    all_issues.append(f"Line {line_num} (id={sample.get('id', 'unknown')}): {issue}")
    
    return valid_count, total_count, all_issues


def main():
    parser = argparse.ArgumentParser(description="Validate irrelevance data")
    parser.add_argument("input", help="Input JSONL file path")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show all issues")
    parser.add_argument("--max-issues", type=int, default=20, help="Maximum issues to display")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return 1
    
    valid_count, total_count, all_issues = validate_file(args.input, args.verbose)
    
    print(f"\n{'='*60}")
    print(f"Validation Results: {args.input}")
    print(f"{'='*60}")
    print(f"Valid samples: {valid_count}/{total_count} ({100*valid_count/total_count:.1f}%)" if total_count > 0 else "No samples found")
    
    if all_issues:
        print(f"\nIssues found: {len(all_issues)}")
        print("-" * 40)
        
        issues_to_show = all_issues[:args.max_issues]
        for issue in issues_to_show:
            print(f"  - {issue}")
        
        if len(all_issues) > args.max_issues:
            print(f"  ... and {len(all_issues) - args.max_issues} more issues")
    else:
        print("\n✓ All samples are valid!")
    
    print(f"{'='*60}\n")
    
    return 0 if valid_count == total_count else 1


if __name__ == "__main__":
    sys.exit(main())
