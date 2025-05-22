"""Parallel processing utilities."""

import concurrent.futures
import multiprocessing
import os
import sys
import psutil
from typing import List, Dict, Any, Tuple

from ..config.settings import DEFAULT_MAX_WORKERS, HIGH_PRIORITY_PROCESS


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


def init_worker():
    """Initialize worker process with optimal settings."""
    if HIGH_PRIORITY_PROCESS:
        try:
            if sys.platform == 'win32':
                psutil.Process().nice(psutil.HIGH_PRIORITY_CLASS)
            else:
                os.nice(-5)  # Lower nice value = higher priority
        except (OSError, AttributeError):
            pass  # Ignore if cannot set priority


def parallel_process_pieces(piece_data: List[Dict[str, Any]], output_dirs: Tuple[str, ...], 
                          max_workers: int = None) -> List[Dict[str, Any]]:
    """Process puzzle pieces in parallel.
    
    Args:
        piece_data: List of piece data dictionaries
        output_dirs: Tuple of output directory paths
        max_workers: Maximum number of worker processes
        
    Returns:
        List of processed piece results
    """
    if max_workers is None:
        max_workers = DEFAULT_MAX_WORKERS or multiprocessing.cpu_count()
    
    # Limit to reasonable number of workers
    max_workers = min(max_workers, multiprocessing.cpu_count(), len(piece_data))
    
    print(f"Processing {len(piece_data)} pieces using {max_workers} cores...")
    
    # Import here to avoid circular imports
    from ..core.piece_detection import process_piece
    
    # Prepare arguments for parallel processing
    args_list = [(piece, output_dirs) for piece in piece_data]
    
    results = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max_workers, 
        initializer=init_worker
    ) as executor:
        # Submit all tasks
        future_to_piece = {
            executor.submit(process_piece, piece, output_dirs): i 
            for i, (piece, _) in enumerate(args_list)
        }
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_piece):
            piece_idx = future_to_piece[future]
            try:
                result = future.result()
                results.append((piece_idx, result))
                
                # Progress reporting
                if len(results) % 5 == 0:
                    print(f"Processed {len(results)}/{len(piece_data)} pieces...")
                    
            except Exception as e:
                print(f"Error processing piece {piece_idx}: {e}")
                results.append((piece_idx, None))
    
    # Sort results by original piece index
    results.sort(key=lambda x: x[0])
    return [result[1] for result in results if result[1] is not None]


def set_process_priority():
    """Set high priority for the current process."""
    if not HIGH_PRIORITY_PROCESS:
        return
    
    try:
        if sys.platform == 'win32':
            psutil.Process().nice(psutil.HIGH_PRIORITY_CLASS)
        else:
            os.nice(-10)
        print("Process priority set to high")
    except Exception as e:
        print(f"Could not set process priority: {e}")


def get_optimal_worker_count() -> int:
    """Get optimal number of workers for parallel processing.
    
    Returns:
        Optimal number of worker processes
    """
    cpu_count = multiprocessing.cpu_count()
    
    # Leave one core free for system
    optimal_count = max(1, cpu_count - 1)
    
    # Limit based on available memory (rough estimate)
    try:
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        # Assume each worker needs ~500MB
        memory_limited_workers = max(1, int(available_memory_gb / 0.5))
        optimal_count = min(optimal_count, memory_limited_workers)
    except:
        pass  # Use CPU-based count if memory check fails
    
    return optimal_count