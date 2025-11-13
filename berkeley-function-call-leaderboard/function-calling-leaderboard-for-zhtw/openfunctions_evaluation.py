import argparse, json, os, time
import sys, io
from tqdm import tqdm
# from model_handler.handler_map import handler_map
from model_handler.model_style import ModelStyle
from model_handler.constant import USE_COHERE_OPTIMIZATION
from eval_checker.eval_checker_constant import TEST_COLLECTION_MAPPING
from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING
from accuracy_config import (
    get_optimal_temperature,
    get_optimal_top_p,
    get_optimal_max_tokens,
    GPU_OPTIMIZATION,
    RETRY_CONFIG
)

# Set UTF-8 encoding for standard input, output, and error
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def get_args():
    parser = argparse.ArgumentParser()
    # Refer to model_choice for supported models.
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo-0125-FC", nargs="+")
    # Refer to test_categories for supported categories.
    parser.add_argument("--test-category", type=str, default="all", nargs="+")
    parser.add_argument("--language", type=str, default="zhtw", help="Specify the language for the test cases and results")
    # Parameters for the model that you want to test.
    # Note: Defaults will be overridden by accuracy_config.py optimizations if --optimize-accuracy is set
    parser.add_argument("--temperature", type=float, default=None, help="Temperature (default: optimized per model)")
    parser.add_argument("--top-p", type=float, default=None, help="Top-p (default: optimized per model)")
    parser.add_argument("--max-tokens", type=int, default=None, help="Max tokens (default: optimized per model)")
    parser.add_argument("--num-gpus", default=1, type=int)
    parser.add_argument("--timeout", default=None, type=int, help="Timeout in seconds (default: optimized)")
    parser.add_argument("--gpu-memory-utilization", default=None, type=float, help="GPU memory utilization (default: optimized for H200)")
    parser.add_argument("--optimize-accuracy", action="store_true", default=True, help="Use optimized accuracy settings (default: True)")
    args = parser.parse_args()
    return args


TEST_FILE_MAPPING = {
    "executable_simple": "gorilla_openfunctions_v1_test_executable_simple.json",
    "executable_parallel_function": "gorilla_openfunctions_v1_test_executable_parallel_function.json",
    "executable_multiple_function": "gorilla_openfunctions_v1_test_executable_multiple_function.json",
    "executable_parallel_multiple_function": "gorilla_openfunctions_v1_test_executable_parallel_multiple_function.json",
    "simple": "gorilla_openfunctions_v1_test_simple.json",
    "relevance": "gorilla_openfunctions_v1_test_relevance.json",
    "parallel_function": "gorilla_openfunctions_v1_test_parallel_function.json",
    "multiple_function": "gorilla_openfunctions_v1_test_multiple_function.json",
    "parallel_multiple_function": "gorilla_openfunctions_v1_test_parallel_multiple_function.json",
    "rest": "gorilla_openfunctions_v1_test_rest.json",
}


def build_handler(model_name, temperature, top_p=None, max_tokens=None):
    """
    Build a handler for the specified model with optimized parameters.
    
    Args:
        model_name (str): Name of the model
        temperature (float): Temperature parameter
        top_p (float, optional): Top-p parameter
        max_tokens (int, optional): Max tokens parameter
        
    Returns:
        Handler instance for the model
    """
    config = MODEL_CONFIG_MAPPING[model_name]
    handler = config.model_handler(model_name, temperature, top_p=top_p or 1.0, max_tokens=max_tokens or 1200)
    # Propagate config flags to the handler instance
    handler.is_fc_model = config.is_fc_model
    return handler


def parse_test_category_argument(test_category_args):
    test_name_total = set()
    test_filename_total = set()
    
    for test_category in test_category_args:
        if test_category in TEST_COLLECTION_MAPPING:
            for test_name in TEST_COLLECTION_MAPPING[test_category]:
                test_name_total.add(test_name)
                test_filename_total.add(TEST_FILE_MAPPING[test_name])
        else:
            test_name_total.add(test_category)
            test_filename_total.add(TEST_FILE_MAPPING[test_category])

    return list(test_name_total), list(test_filename_total)


def collect_test_cases(test_filename_total, model_name):
    test_cases_total = []
    for file_to_open in test_filename_total:
        test_cases = []
        with open(f"./data/{args.language}/" + file_to_open) as f:
            for line in f:
                test_cases.append(json.loads(line))

        num_existing_result = 0  # if the result file already exists, skip the test cases that have been tested.
        if os.path.exists(
            f"./result/{args.language}/"
            + model_name.replace("/", "_")
            + "/"
            + file_to_open.replace(".json", "_result.json")
        ):
            with open(
                f"./result/{args.language}/"
                + model_name.replace("/", "_")
                + "/"
                + file_to_open.replace(".json", "_result.json")
            ) as f:
                for line in f:
                    num_existing_result += 1

        test_cases_total.extend(test_cases[num_existing_result:])
    return test_cases_total


def generate_results(args, model_name, test_cases_total):
    # Use optimized retry settings if accuracy optimization is enabled
    if args.optimize_accuracy:
        RETRY_LIMIT = RETRY_CONFIG["max_retries"]
        RETRY_DELAY = RETRY_CONFIG["retry_delay"]
    else:
        RETRY_LIMIT = 3
        RETRY_DELAY = 65  # Delay in seconds
    
    handler = build_handler(model_name, args.temperature, args.top_p, args.max_tokens)

    if handler.model_style == ModelStyle.OSSMODEL:
        result, metadata = handler.inference(
            test_question=test_cases_total,
            num_gpus=args.num_gpus,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
        for test_case, res in zip(test_cases_total, result):
            result_to_write = {"id": test_case["id"], "result": res}
            handler.write(result_to_write,args.language)

    else:
        for test_case in tqdm(test_cases_total):

            user_question, functions, test_category = (
                test_case["question"],
                test_case["function"],
                test_case["id"].rsplit("_", 1)[0],
            )
            if type(functions) is dict or type(functions) is str:
                functions = [functions]

            retry_count = 0
            result = None
            metadata = None

            while retry_count < RETRY_LIMIT:
                try:
                    result, metadata = handler.inference(
                        user_question, functions, test_category
                    )
                    break  # Success, exit the loop
                except Exception as e:
                    # Improved error handling for different API providers
                    # Check for rate limiting errors (429, 503, 500) or rate limit in error message
                    is_rate_limit = (
                        "rate limit" in str(e).lower() or
                        "too many requests" in str(e).lower() or
                        (hasattr(e, "status_code") and e.status_code in [429, 503, 500])
                    )
                    
                    if is_rate_limit:
                        retry_count += 1
                        if retry_count < RETRY_LIMIT:
                            print(f"Rate limit reached for {test_case['id']}. Sleeping for {RETRY_DELAY} seconds. Retry {retry_count}/{RETRY_LIMIT}")
                            time.sleep(RETRY_DELAY)
                        else:
                            print(f"Maximum retries ({RETRY_LIMIT}) reached for {test_case['id']}. Skipping this test case.")
                            print(f"Error: {str(e)}")
                            result = "ERROR: Rate limit exceeded after retries"
                            metadata = {"input_tokens": 0, "output_tokens": 0, "latency": 0}
                            break
                    else:
                        # For non-rate-limit errors, log and skip
                        print(f"Error processing {test_case['id']}: {str(e)}")
                        result = f"ERROR: {str(e)}"
                        metadata = {"input_tokens": 0, "output_tokens": 0, "latency": 0}
                        break
            
            # Ensure result and metadata are set even if all retries fail
            if result is not None and metadata is not None:
                result_to_write = {
                    "id": test_case["id"],
                    "result": result,
                    "input_token_count": metadata["input_tokens"],
                    "output_token_count": metadata["output_tokens"],
                    "latency": metadata["latency"],
                }
                handler.write(result_to_write,args.language)


if __name__ == "__main__":
    args = get_args()

    # Apply optimized accuracy settings if enabled
    if args.optimize_accuracy:
        print("Using optimized accuracy settings...")
        for model_name in args.model if isinstance(args.model, list) else [args.model]:
            if args.temperature is None:
                args.temperature = get_optimal_temperature(model_name)
            if args.top_p is None:
                args.top_p = get_optimal_top_p(model_name)
            if args.max_tokens is None:
                args.max_tokens = get_optimal_max_tokens(model_name)
            if args.timeout is None:
                args.timeout = RETRY_CONFIG["timeout"]
            if args.gpu_memory_utilization is None:
                args.gpu_memory_utilization = GPU_OPTIMIZATION["gpu_memory_utilization"]
            
            print(f"  Model: {model_name}")
            print(f"  Temperature: {args.temperature}")
            print(f"  Top-p: {args.top_p}")
            print(f"  Max tokens: {args.max_tokens}")
            print(f"  GPU memory utilization: {args.gpu_memory_utilization}")
            print(f"  Language: {args.language}")
            break  # Just show settings once for the first model
    else:
        # Set defaults if not specified
        if args.temperature is None:
            args.temperature = 0.7
        if args.top_p is None:
            args.top_p = 1.0
        if args.max_tokens is None:
            args.max_tokens = 1200
        if args.timeout is None:
            args.timeout = 60
        if args.gpu_memory_utilization is None:
            args.gpu_memory_utilization = 0.9

    if type(args.model) is not list:
        args.model = [args.model]
    if type(args.test_category) is not list:
        args.test_category = [args.test_category]
        
    test_name_total, test_filename_total = parse_test_category_argument(args.test_category)
    
    print(f"Generating results for {args.model} on test category: {test_name_total}.")

    for model_name in args.model:
        if USE_COHERE_OPTIMIZATION and "command-r-plus" in model_name:
            model_name = model_name + "-optimized"
        
        os.makedirs(f"./data/{args.language}", exist_ok=True)
        os.makedirs(f"./result/{args.language}/{model_name.replace('/', '_')}", exist_ok=True)
        test_cases_total = collect_test_cases(test_filename_total, model_name)
        
        if len(test_cases_total) == 0:
            print(f"All selected test cases have been previously generated for {model_name}. No new test cases to generate.")
        else:
            generate_results(args, model_name, test_cases_total)