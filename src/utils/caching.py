"""Caching utilities for performance optimization."""

import functools
import hashlib
import json
import os
from typing import Any, Callable

from ..config.settings import CACHE_DIR, ENABLE_CACHING


def generate_cache_key(*args) -> str:
    """Generate a unique cache key from function arguments.
    
    Args:
        *args: Function arguments
        
    Returns:
        Unique cache key string
    """
    # Convert args to string representation
    key_string = str(args)
    
    # Generate hash
    return hashlib.md5(key_string.encode()).hexdigest()


def cache_result(func: Callable) -> Callable:
    """Decorator to cache function results to disk.
    
    Args:
        func: Function to cache
        
    Returns:
        Wrapped function with caching
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not ENABLE_CACHING:
            return func(*args, **kwargs)
        
        # Generate cache key
        cache_key = f"{func.__name__}_{generate_cache_key(*args, **kwargs)}"
        cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")
        
        # Check if cached result exists
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                print(f"Cache hit for {func.__name__}")
                return cached_data
            except (json.JSONDecodeError, IOError):
                # Cache file corrupted, remove it
                os.remove(cache_file)
        
        # Compute result
        result = func(*args, **kwargs)
        
        # Cache result
        try:
            os.makedirs(CACHE_DIR, exist_ok=True)
            with open(cache_file, 'w') as f:
                json.dump(result, f, indent=2)
        except (IOError, TypeError) as e:
            print(f"Cache write error for {func.__name__}: {e}")
        
        return result
    
    return wrapper


class Timer:
    """Context manager for timing operations."""
    
    def __init__(self, description: str):
        self.description = description
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        import time
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        print(f"{self.description} completed in {duration:.3f}s")