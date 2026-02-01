"""
Irrelevance Data Generation Script

This module generates "irrelevance" test data where:
- The user asks a question that can be answered using natural language only
- The available functions are completely unrelated to the question
- The correct LLM behavior is to NOT call any function

This is a standalone pipeline that generates data independently from the main
multi-turn dialogue generation flow.

Environment Variables:
    LANG_CODE: Language setting ("en" or "zh_tw")
    CURRICULUM_CSV: Path to curriculum CSV file
    IRR_NUM_SAMPLES: Number of irrelevance samples per curriculum row (default: 2)
    IRR_LIMIT_ROWS: Limit number of curriculum rows to process (optional)
    PARALLEL_WORKERS: Number of parallel workers (default: 5)
    MAX_RETRIES: Maximum retries for API calls (default: 3)
    IRR_DEBUG: Enable debug output ("1" to enable)
    OPENAI_API_KEYS: Comma-separated list of OpenAI API keys
    API_DAILY_LIMIT_TOKENS: Daily token limit per key (default: 2500000)
"""
import asyncio
import csv
import json
import logging
import os
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple
from tqdm import tqdm
from openai_utils import render_template, extract_tags, extract_code_fence, chat_complete
from pipeline.s2_functions.parser import parse_signature
from incremental_utils import (
    IncrementalWriter,
    load_completed_indices,
    run_parallel_tasks,
    get_parallel_workers,
    ensure_jsonl_path,
    check_final_json_exists,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ============================================================================
# Language utilities (same as other stages)
# ============================================================================
def get_language() -> str:
    """獲取當前語言設定"""
    return os.getenv("LANG_CODE", "en").lower()


def get_prompt_path(base_path: str) -> str:
    """根據語言設定獲取對應的 prompt 路徑"""
    lang = get_language()
    if lang == "zh_tw":
        path_without_ext = base_path.rsplit('.', 1)[0]
        return f"{path_without_ext}_zh_tw.md"
    return base_path


def get_system_prompt_suffix() -> str:
    """根據語言設定獲取 system prompt 的語言後綴"""
    lang = get_language()
    if lang == "zh_tw":
        return " Please write all user questions and responses in Traditional Chinese (繁體中文). Keep function names in English."
    return ""


# ============================================================================
# Debug utilities
# ============================================================================
def _write_debug(debug_enabled: bool, debug_out_path: str, record: Dict[str, Any]):
    if not debug_enabled:
        return
    try:
        os.makedirs(os.path.dirname(debug_out_path), exist_ok=True)
        with open(debug_out_path, "a", encoding="utf-8") as df:
            df.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


# ============================================================================
# Curriculum reading (similar to S1)
# ============================================================================
def read_curriculum(csv_path: str) -> List[Dict[str, str]]:
    """Read curriculum CSV file."""
    rows: List[Dict[str, str]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "domain": r["domain"],
                "subdomain": r["subdomain"],
                "entities": r.get("entities", ""),
            })
    return rows


# ============================================================================
# Sample parsing
# ============================================================================
def parse_irrelevance_samples(content: str) -> List[Dict[str, Any]]:
    """
    Parse irrelevance samples from LLM response.
    
    Returns list of samples, each containing:
    - question: The user's question
    - natural_response: Expected natural language response
    - functions: List of irrelevant function definitions
    """
    samples = []
    
    sample_blocks = extract_tags(content, "sample")
    
    for sample_block in sample_blocks:
        try:
            # Extract question
            questions = extract_tags(sample_block, "question")
            if not questions:
                continue
            question = questions[0].strip()
            
            # Extract natural response
            responses = extract_tags(sample_block, "natural_response")
            natural_response = responses[0].strip() if responses else ""
            
            # Extract functions
            functions = []
            func_blocks = extract_tags(sample_block, "function")
            
            for fb in func_blocks:
                sig_blocks = extract_tags(fb, "signature")
                if not sig_blocks:
                    continue
                
                # extract_code_fence returns a list, take first element
                code_fences = extract_code_fence(sig_blocks[0])
                if code_fences:
                    sig_code = code_fences[0]
                else:
                    sig_code = sig_blocks[0].strip()
                
                # Parse the signature
                try:
                    parsed = parse_signature(sig_code)
                    if not parsed:
                        continue
                except Exception:
                    continue
                
                # Extract expected return value
                expected_blocks = extract_tags(fb, "expected")
                expected = None
                if expected_blocks:
                    try:
                        expected = json.loads(expected_blocks[0].strip())
                    except json.JSONDecodeError:
                        expected = expected_blocks[0].strip()
                
                functions.append({
                    "function": sig_code,
                    "expected": expected,
                    "parsed": parsed,
                })
            
            if question and functions:
                samples.append({
                    "question": question,
                    "natural_response": natural_response,
                    "functions": functions,
                })
        
        except Exception as e:
            logging.warning(f"Failed to parse sample: {e}")
            continue
    
    return samples


# ============================================================================
# Processing function
# ============================================================================
def process_single_row(
    idx: int,
    task_data: Tuple[Dict[str, str], str, str, int, bool, str]
) -> Optional[Dict[str, Any]]:
    """
    Process a single curriculum row to generate irrelevance samples.
    
    Args:
        idx: Row index
        task_data: (row, template_path, system, max_retries, debug_enabled, debug_path)
        
    Returns:
        Dict containing samples for this row, or None if failed
    """
    row, template_path, system, max_retries, debug_enabled, debug_path = task_data
    num_samples = int(os.getenv("IRR_NUM_SAMPLES", "2"))
    
    if debug_enabled:
        _write_debug(debug_enabled, debug_path, {
            "row_index": idx,
            "domain": row["domain"],
            "subdomain": row["subdomain"],
            "phase": "start",
        })
    
    prompt = render_template(template_path, {
        "domain": row["domain"],
        "subdomain": row["subdomain"],
        "num_samples": str(num_samples),
    })
    
    all_samples = []
    
    for attempt in range(max_retries + 1):
        try:
            content = chat_complete(prompt=prompt, system=system)
        except Exception as e:
            logging.error(f"Row {idx} attempt {attempt}: API error: {e}")
            continue
        
        if debug_enabled:
            _write_debug(debug_enabled, debug_path, {
                "row_index": idx,
                "attempt": attempt,
                "phase": "response_received",
                "content_length": len(content),
            })
        
        samples = parse_irrelevance_samples(content)
        
        if debug_enabled:
            _write_debug(debug_enabled, debug_path, {
                "row_index": idx,
                "attempt": attempt,
                "phase": "parsed",
                "sample_count": len(samples),
            })
        
        if samples:
            all_samples.extend(samples)
            break
    
    if not all_samples:
        logging.warning(f"Row {idx}: No samples generated after {max_retries + 1} attempts")
        return None
    
    return {
        "domain": row["domain"],
        "subdomain": row["subdomain"],
        "samples": all_samples,
    }


# ============================================================================
# Main generation function
# ============================================================================
async def generate_irrelevance_data(run_id: str):
    """
    Main function to generate irrelevance data.
    
    Creates:
    - pipeline/data/{run_id}/irrelevance/irrelevance.incr.jsonl (incremental)
    - pipeline/data/{run_id}/irrelevance/irrelevance.json (final)
    """
    # Create output directory
    output_dir = f"pipeline/data/{run_id}/irrelevance"
    os.makedirs(output_dir, exist_ok=True)
    
    json_path = f"{output_dir}/irrelevance.json"
    jsonl_path = ensure_jsonl_path(json_path)
    
    # Check if already complete
    if check_final_json_exists(json_path):
        logging.info(f"irrelevance.json already exists, skipping generation")
        return
    
    # Load curriculum
    csv_path = os.getenv("CURRICULUM_CSV", "pipeline/data/curriculum.csv")
    logging.info(f"Using curriculum file: {csv_path}")
    rows = read_curriculum(csv_path)
    
    # Apply row limit if specified
    limit_rows = os.getenv("IRR_LIMIT_ROWS")
    if limit_rows:
        try:
            rows = rows[:int(limit_rows)]
            logging.info(f"Limited to {limit_rows} rows")
        except ValueError:
            pass
    
    # Setup
    template_path = get_prompt_path("pipeline/s6_irrelevance/prompt.md")
    lang = get_language()
    logging.info(f"Using language: {lang}, prompt path: {template_path}")
    
    system = (
        "You are a careful data generator for LLM evaluation. "
        "Generate irrelevance test samples where the user's question can be answered "
        "using natural language only, and the provided functions are completely unrelated. "
        "Follow the format strictly."
        + get_system_prompt_suffix()
    )
    
    max_retries = int(os.getenv("MAX_RETRIES", "3"))
    debug_enabled = os.getenv("IRR_DEBUG", "0") == "1"
    debug_path = f"{output_dir}/debug.jsonl"
    
    # Load completed indices
    completed = load_completed_indices(jsonl_path)
    
    # Prepare items to process
    items_to_process = [
        (idx, (row, template_path, system, max_retries, debug_enabled, debug_path))
        for idx, row in enumerate(rows)
        if idx not in completed
    ]
    
    if not items_to_process:
        logging.info("All rows already processed, finalizing...")
    else:
        logging.info(f"Processing {len(items_to_process)} rows (skipping {len(completed)} completed)")
        
        with IncrementalWriter(jsonl_path, mode="a") as writer:
            workers = get_parallel_workers()
            
            results = run_parallel_tasks(
                task_fn=process_single_row,
                items=items_to_process,
                max_workers=workers,
                desc="Generating irrelevance samples",
                writer=writer,
            )
            
            success_count = sum(1 for _, r in results if r is not None)
            logging.info(f"Generated {success_count}/{len(items_to_process)} rows successfully")
    
    # Finalize to JSON
    writer = IncrementalWriter(jsonl_path, mode="a")
    writer.finalize_to_json(json_path)
    logging.info(f"Finalized irrelevance data to {json_path}")


# ============================================================================
# Entry point
# ============================================================================
def main():
    # Get or create run_id
    run_id_file = "run_id_irrelevance"
    
    if os.path.exists(run_id_file):
        with open(run_id_file, "r") as f:
            run_id = f.read().strip()
        logging.info(f"Using existing irrelevance run_id: {run_id}")
    else:
        run_id = f"irrelevance_{uuid.uuid4().hex[:8]}"
        with open(run_id_file, "w") as f:
            f.write(run_id)
        logging.info(f"Created new irrelevance run_id: {run_id}")
    
    # Run generation
    asyncio.run(generate_irrelevance_data(run_id))
    
    logging.info("Irrelevance data generation complete!")


if __name__ == "__main__":
    main()
