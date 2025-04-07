"""
Main entry point for the puzzle piece detection system
"""

import os
import sys
import argparse
import time
import cv2
import numpy as np
from typing import List, Dict, Any

from src.config.settings import Config
from src.core.processor import PuzzleProcessor
from src.utils.image_utils import read_image


# Define color codes for terminal output
class TermColors:
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    PURPLE = "\033[95m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def print_info(message: str) -> None:
    """Print an info message"""
    print(f"{TermColors.BLUE}ℹ️  {message}{TermColors.RESET}")


def print_success(message: str) -> None:
    """Print a success message"""
    print(f"{TermColors.GREEN}✅ {message}{TermColors.RESET}")


def print_warning(message: str) -> None:
    """Print a warning message"""
    print(f"{TermColors.YELLOW}⚠️  {message}{TermColors.RESET}")


def print_error(message: str) -> None:
    """Print an error message"""
    print(f"{TermColors.RED}❌ {message}{TermColors.RESET}")


def print_header(message: str) -> None:
    """Print a header message"""
    print(f"{TermColors.PURPLE}{TermColors.BOLD}🔷 {message}{TermColors.RESET}")

def clean_debug_dir(debug_dir: str) -> None:
    """Clean the debug directory by removing .jpg files"""
    if not os.path.exists(debug_dir):
        return
        
    for filename in os.listdir(debug_dir):
        if filename.endswith(".jpg") or filename.endswith(".json"):
            file_path = os.path.join(debug_dir, filename)
            try:
                os.remove(file_path)
            except Exception as e:
                print_warning(f"Error removing file {file_path}: {str(e)}")
                
def main() -> None:
    """Main function"""
    # Set up command line arguments
    parser = argparse.ArgumentParser(description="Puzzle piece detector")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to the puzzle image"
    )
    parser.add_argument(
        "--pieces",
        type=int,
        default=None,
        help="Expected number of pieces in the puzzle"
    )
    parser.add_argument(
        "--debug-dir",
        type=str,
        default="debug",
        help="Directory to save debug outputs"
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Extract individual pieces to separate files"
    )
    parser.add_argument(
        "--extract-dir",
        type=str,
        default="extracted_pieces",
        help="Directory to save extracted pieces"
    )
    parser.add_argument(
        "--use-multiprocessing",
        action="store_true",
        help="Use multiprocessing for faster detection"
    )
    parser.add_argument(
        "--view",
        action="store_true",
        help="View results (opens image windows)"
    )
    parser.add_argument(
        "--auto-threshold",
        action="store_true",
        help="Automatically select the best thresholding method (Otsu or adaptive)"
    )
    parser.add_argument(
        "--no-mean-filter",
        action="store_true",
        help="Disable filtering based on mean contour area"
    )
    parser.add_argument(
        "--mean-threshold",
        type=float,
        default=2.0,
        help="Standard deviation threshold for mean area filtering (default: 2.0)"
    )
    parser.add_argument(
        "--use-sobel",
        action="store_true",
        help="Use the Sobel pipeline (Gray → Blurred → Sobel → Contrasted → Dilated → Eroded)"
    )

    args = parser.parse_args()

    # Initialize configuration
    config = Config()
    config.DEBUG_DIR = args.debug_dir
    config.USE_MULTIPROCESSING = args.use_multiprocessing
    config.USE_AUTO_THRESHOLD = args.auto_threshold
    config.USE_MEAN_FILTERING = not args.no_mean_filter
    config.MEAN_DEVIATION_THRESHOLD = args.mean_threshold
    config.USE_SOBEL_PIPELINE = args.use_sobel  # Add the new Sobel pipeline flag

    # Ensure debug directory exists
    if not os.path.exists(config.DEBUG_DIR):
        os.makedirs(config.DEBUG_DIR, exist_ok=True)
    
    # Clean debug directory
    clean_debug_dir(config.DEBUG_DIR)

    # Check if an image path was provided
    if args.image is None:
        print_error("No image path provided. Use --image to specify an image file.")
        print_info("Example: python main.py --image path/to/puzzle.jpg")
        return

    # Check if the image exists
    if not os.path.exists(args.image):
        print_error(f"Image file not found: {args.image}")
        return

    print_header("Puzzle Piece Detector")
    print_info(f"Processing image: {args.image}")
    print_info(f"Debug output directory: {config.DEBUG_DIR}")
    
    if args.pieces:
        print_info(f"Expected pieces: {args.pieces}")
        
    if args.auto_threshold:
        print_info("Using auto threshold selection")
    
    if config.USE_MEAN_FILTERING:
        print_info(f"Using mean area filtering (threshold: {config.MEAN_DEVIATION_THRESHOLD} std dev)")
    else:
        print_info("Mean area filtering disabled")
        
    if config.USE_SOBEL_PIPELINE:
        print_info("Using Sobel edge detection pipeline")

    # Start timing
    start_time = time.time()

    # Create processor and process image
    processor = PuzzleProcessor(config)
    
    try:
        results = processor.process_image(args.image, args.pieces)
        
        # Extract pieces if requested
        if args.extract:
            processor.extract_pieces(results['pieces'], args.extract_dir)
        
        # Save results
        processor.save_results(results)
        
        # Show results if requested
        if args.view:
            cv2.imshow("Summary", results['visualizations']['summary'])
            cv2.imshow("Metrics", results['visualizations']['metrics'])
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        # Calculate and display statistics
        num_pieces = len(results['pieces'])
        elapsed_time = time.time() - start_time
        
        print_header("\nPuzzle Analysis Results:")
        print_success(f"Detected {num_pieces} valid puzzle pieces")
        
        if args.pieces:
            detection_rate = (num_pieces / args.pieces) * 100
            print_info(f"Detection rate: {detection_rate:.1f}%")
        
        print_info(f"Processing time: {elapsed_time:.2f} seconds")
        print_success("Analysis complete!")
        
    except Exception as e:
        print_error(f"Error processing image: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()