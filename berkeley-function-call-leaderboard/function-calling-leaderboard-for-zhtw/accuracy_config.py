"""
Accuracy Configuration for Function Calling Leaderboard (Traditional Chinese)

This configuration file contains optimized parameters for improving accuracy
in the Traditional Chinese function calling benchmark.

Optimized for high-end GPUs like H200.
"""

# Temperature settings for different model types
# Lower temperature generally improves accuracy by reducing randomness
TEMPERATURE_CONFIGS = {
    "default": 0.1,  # Very low temperature for maximum accuracy
    "gpt": 0.1,
    "claude": 0.1,
    "llama": 0.1,
    "mistral": 0.1,
    "gemini": 0.1,
    "qwen": 0.1,
}

# Top-p (nucleus sampling) settings
# Lower values focus on more likely tokens
TOP_P_CONFIGS = {
    "default": 0.95,
    "gpt": 0.95,
    "claude": 0.95,
    "llama": 0.9,
    "mistral": 0.95,
    "gemini": 0.95,
    "qwen": 0.95,
}

# Max tokens for responses
MAX_TOKENS_CONFIGS = {
    "default": 1500,  # Increased for complex function calls
    "gpt": 1500,
    "claude": 2000,
    "llama": 1500,
    "mistral": 1500,
    "gemini": 2000,
    "qwen": 1500,
}

# GPU optimization settings for H200
GPU_OPTIMIZATION = {
    "num_gpus": 1,
    "gpu_memory_utilization": 0.95,  # High utilization for H200's 141GB memory
    "tensor_parallel_size": 1,
    "max_model_len": None,  # Will use model's default
    "dtype": "auto",  # Let vLLM choose the best dtype
}

# Language-specific optimizations
LANGUAGE_OPTIMIZATIONS = {
    "zhtw": {
        "use_chinese_prompts": True,
        "temperature": 0.1,  # Even lower for better precision
        "top_p": 0.9,
        "prefer_chinese_models": ["qwen", "glm", "breeze"],  # Models trained on Chinese
    },
    "en": {
        "use_chinese_prompts": False,
        "temperature": 0.1,
        "top_p": 0.95,
    }
}

# Retry and timeout configurations
RETRY_CONFIG = {
    "max_retries": 5,  # Increased retries for better coverage
    "retry_delay": 65,  # Delay between retries (seconds)
    "timeout": 120,  # Increased timeout for complex queries
}

def get_optimal_temperature(model_name):
    """
    Get optimal temperature setting for a given model.
    
    Args:
        model_name (str): Name of the model
        
    Returns:
        float: Optimal temperature value
    """
    for key in TEMPERATURE_CONFIGS:
        if key in model_name.lower():
            return TEMPERATURE_CONFIGS[key]
    return TEMPERATURE_CONFIGS["default"]

def get_optimal_top_p(model_name):
    """
    Get optimal top-p setting for a given model.
    
    Args:
        model_name (str): Name of the model
        
    Returns:
        float: Optimal top-p value
    """
    for key in TOP_P_CONFIGS:
        if key in model_name.lower():
            return TOP_P_CONFIGS[key]
    return TOP_P_CONFIGS["default"]

def get_optimal_max_tokens(model_name):
    """
    Get optimal max tokens setting for a given model.
    
    Args:
        model_name (str): Name of the model
        
    Returns:
        int: Optimal max tokens value
    """
    for key in MAX_TOKENS_CONFIGS:
        if key in model_name.lower():
            return MAX_TOKENS_CONFIGS[key]
    return MAX_TOKENS_CONFIGS["default"]

def get_language_config(language):
    """
    Get language-specific configuration.
    
    Args:
        language (str): Language code ('en', 'zhtw')
        
    Returns:
        dict: Language-specific configuration
    """
    return LANGUAGE_OPTIMIZATIONS.get(language, LANGUAGE_OPTIMIZATIONS["en"])
