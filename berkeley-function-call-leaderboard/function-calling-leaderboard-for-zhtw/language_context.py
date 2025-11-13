"""
Language Context Manager for ZHTW Function Calling Leaderboard

This module provides context management for language-specific configurations
to ensure the correct prompts and settings are used based on the evaluation language.
"""

import threading
from contextlib import contextmanager

# Thread-local storage for language context
_language_context = threading.local()

def set_language_context(language="en"):
    """
    Set the current language context for the thread.
    
    Args:
        language (str): Language code ('en' for English, 'zhtw' for Traditional Chinese)
    """
    _language_context.language = language

def get_language_context():
    """
    Get the current language context for the thread.
    
    Returns:
        str: Current language code, defaults to 'en' if not set
    """
    return getattr(_language_context, 'language', 'en')

@contextmanager
def language_context(language):
    """
    Context manager for temporarily setting the language context.
    
    Args:
        language (str): Language code to use in this context
        
    Example:
        with language_context('zhtw'):
            # Code here will use Traditional Chinese prompts
            handler.inference(prompt, functions, test_category)
    """
    old_language = get_language_context()
    set_language_context(language)
    try:
        yield
    finally:
        set_language_context(old_language)

def get_prompt_with_language(prompt_type='system', language=None):
    """
    Get the appropriate prompt based on language context.
    
    Args:
        prompt_type (str): Type of prompt ('system' or 'user')
        language (str, optional): Language override. If None, uses current context
        
    Returns:
        str: The appropriate prompt template
    """
    from model_handler.constant import (
        SYSTEM_PROMPT_FOR_CHAT_MODEL,
        USER_PROMPT_FOR_CHAT_MODEL,
        get_system_prompt,
        get_user_prompt
    )
    
    if language is None:
        language = get_language_context()
    
    if prompt_type == 'system':
        return get_system_prompt(language)
    elif prompt_type == 'user':
        return get_user_prompt(language)
    else:
        raise ValueError(f"Unknown prompt type: {prompt_type}")

# Initialize default language context
set_language_context('en')
