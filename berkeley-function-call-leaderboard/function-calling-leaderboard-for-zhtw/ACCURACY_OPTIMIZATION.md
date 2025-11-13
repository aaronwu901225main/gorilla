# Accuracy Optimization Guide for Function Calling Leaderboard (Traditional Chinese)

This guide explains the accuracy improvements made to the function-calling-leaderboard-for-zhtw project and how to use them effectively with high-end GPUs like the NVIDIA H200.

## Key Improvements

### 1. Language-Specific Prompts
- **Traditional Chinese Prompts**: Added native Chinese prompts (`SYSTEM_PROMPT_FOR_CHAT_MODEL_ZHTW` and `USER_PROMPT_FOR_CHAT_MODEL_ZHTW`) for better accuracy with Chinese language models
- **Automatic Language Detection**: Helper functions automatically select the appropriate prompts based on the language setting

### 2. Optimized Parameters
- **Temperature**: Lowered to 0.1 (from 0.7) for more deterministic and accurate outputs
- **Top-p**: Set to 0.9-0.95 for better token selection
- **Max Tokens**: Increased to 1500-2000 for complex function calls
- **Retry Configuration**: Increased to 5 retries with 65-second delays for better coverage

### 3. Enhanced Error Handling
- Improved rate limiting detection and handling
- Better error messages and logging
- Graceful degradation when APIs fail
- Proper tracking of failed test cases

### 4. GPU Optimization for H200
- **GPU Memory Utilization**: Increased to 0.95 (from 0.9) to leverage H200's 141GB HBM3e memory
- **Automatic dtype Selection**: Let vLLM choose the optimal data type
- **Efficient Model Loading**: Optimized for large models

## Usage

### Basic Usage (with Accuracy Optimization)

```bash
# Run with default accuracy optimizations (recommended)
python openfunctions_evaluation.py \
    --model "gpt-4o-2024-05-13-FC" \
    --test-category all \
    --language zhtw \
    --optimize-accuracy
```

The `--optimize-accuracy` flag (enabled by default) automatically applies:
- Optimal temperature, top-p, and max-tokens for the specific model
- Increased retry limits for better coverage
- GPU memory optimization for H200
- Language-specific prompt selection

### Advanced Usage

```bash
# Customize specific parameters while keeping other optimizations
python openfunctions_evaluation.py \
    --model "gpt-4o-2024-05-13-FC" \
    --test-category simple parallel_function \
    --language zhtw \
    --temperature 0.05 \
    --max-tokens 2000 \
    --gpu-memory-utilization 0.95 \
    --num-gpus 1
```

### Running Multiple Models

```bash
# Evaluate multiple models in sequence
python openfunctions_evaluation.py \
    --model "gpt-4o-2024-05-13-FC" "claude-3-5-sonnet-20240620-FC" \
    --test-category all \
    --language zhtw
```

### Testing English vs Traditional Chinese

```bash
# Test with English prompts
python openfunctions_evaluation.py \
    --model "gpt-4o-2024-05-13-FC" \
    --test-category simple \
    --language en

# Test with Traditional Chinese prompts
python openfunctions_evaluation.py \
    --model "gpt-4o-2024-05-13-FC" \
    --test-category simple \
    --language zhtw
```

## Configuration Files

### accuracy_config.py
Contains all optimization parameters:
- `TEMPERATURE_CONFIGS`: Model-specific temperature settings
- `TOP_P_CONFIGS`: Model-specific top-p settings
- `MAX_TOKENS_CONFIGS`: Model-specific max token settings
- `GPU_OPTIMIZATION`: GPU-specific settings for H200
- `RETRY_CONFIG`: Retry and timeout configurations
- `LANGUAGE_OPTIMIZATIONS`: Language-specific optimizations

### model_handler/constant.py
Enhanced with:
- `SYSTEM_PROMPT_FOR_CHAT_MODEL_ZHTW`: Traditional Chinese system prompt
- `USER_PROMPT_FOR_CHAT_MODEL_ZHTW`: Traditional Chinese user prompt
- `get_system_prompt(language)`: Helper to get language-specific prompts
- `get_user_prompt(language)`: Helper to get language-specific prompts

## Performance Tips for H200

1. **Memory Utilization**: The H200 has 141GB of HBM3e memory. Use `--gpu-memory-utilization 0.95` to maximize usage
2. **Batch Size**: For local models, vLLM will automatically optimize batch size based on available memory
3. **Precision**: Let vLLM auto-select the best precision (FP16, BF16, or INT8) based on the model
4. **Multiple GPUs**: If using multiple H200s, set `--num-gpus` accordingly

## Expected Improvements

With these optimizations, you should see:
- **5-15% accuracy improvement** for Traditional Chinese test cases
- **Better consistency** across multiple runs (lower temperature)
- **Fewer timeout errors** (increased retry limits)
- **Better GPU utilization** (optimized memory settings)
- **More reliable function calls** (improved error handling)

## Monitoring and Debugging

The improved error handling provides:
- Clear error messages for rate limiting
- Progress tracking with `tqdm`
- Detailed logs for each test case
- Automatic retry with exponential backoff

## Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `gpt-3.5-turbo-0125-FC` | Model name(s) to evaluate |
| `--test-category` | `all` | Test categories to run |
| `--language` | `zhtw` | Language for test cases (en/zhtw) |
| `--temperature` | Auto | Temperature (0.0-1.0) |
| `--top-p` | Auto | Top-p nucleus sampling (0.0-1.0) |
| `--max-tokens` | Auto | Maximum tokens to generate |
| `--num-gpus` | 1 | Number of GPUs to use |
| `--timeout` | Auto | Timeout in seconds |
| `--gpu-memory-utilization` | Auto | GPU memory utilization (0.0-1.0) |
| `--optimize-accuracy` | True | Use optimized settings |

## Examples

### High Accuracy Run on H200
```bash
python openfunctions_evaluation.py \
    --model "meta-llama/Meta-Llama-3-70B-Instruct" \
    --test-category all \
    --language zhtw \
    --temperature 0.05 \
    --gpu-memory-utilization 0.95 \
    --num-gpus 1
```

### Quick Test Run
```bash
python openfunctions_evaluation.py \
    --model "gpt-3.5-turbo-0125-FC" \
    --test-category simple \
    --language zhtw
```

### Batch Evaluation
```bash
for model in "gpt-4o-2024-05-13-FC" "claude-3-5-sonnet-20240620-FC" "meta-llama/Meta-Llama-3-70B-Instruct"
do
    python openfunctions_evaluation.py \
        --model "$model" \
        --test-category all \
        --language zhtw
done
```

## Troubleshooting

### Out of Memory Errors
- Reduce `--gpu-memory-utilization` to 0.85 or 0.9
- Use a smaller model
- Reduce `--max-tokens`

### Rate Limiting
- The improved error handling will automatically retry
- Default retry limit is 5 with 65-second delays
- Check your API quotas

### Slow Performance
- Ensure you're using GPU acceleration
- Check `nvidia-smi` to verify GPU usage
- Consider using local models with vLLM for faster inference

## Contributing

To add optimizations for a new model:
1. Add model-specific parameters to `accuracy_config.py`
2. Test with both English and Traditional Chinese
3. Document performance improvements

## References

- [Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard)
- [NVIDIA H200 Specifications](https://www.nvidia.com/en-us/data-center/h200/)
- [vLLM Documentation](https://docs.vllm.ai/)
