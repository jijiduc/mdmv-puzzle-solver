"""
Enhanced puzzle piece detection and analysis system with adaptive parameter optimization

This script processes images of puzzle pieces on a dark background and
detects individual pieces using enhanced computer vision techniques.
All terminal output is also saved to a log file.
"""
import cProfile
import pstats
from io import StringIO
import cv2
import numpy as np
import os
import argparse
import time
import sys
import logging
import shutil
from typing import Optional
from datetime import datetime

# Import puzzle detection modules
from src.config.settings import Config
from src.core.processor import PuzzleProcessor
from src.utils.image_utils import read_image


def setup_logging(log_dir="logs"):
    """
    Set up logging to both console and file with proper encoding
    
    Args:
        log_dir: Directory to save log files
    
    Returns:
        Path to the log file
    """
    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Create a timestamped log file name
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = os.path.join(log_dir, f"puzzle_detection_{timestamp}.log")
    
    # Configure file handler with detailed logging
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s'))
    
    # Configure console handler with minimal output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.WARNING)  # Only show warnings and errors in console
    console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Create a separate progress logger for minimal console output
    progress_logger = logging.getLogger('progress')
    progress_logger.setLevel(logging.INFO)
    progress_logger.addHandler(console_handler)
    
    progress_logger.info(f"Processing started. Detailed logs at: {log_file}")
    
    return log_file


def clear_directory(directory_path):
    """
    Clear all files in a directory without removing the directory itself
    
    Args:
        directory_path: Path to the directory to clear
    """
    if os.path.exists(directory_path):
        for file_name in os.listdir(directory_path):
            file_path = os.path.join(directory_path, file_name)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                logging.error(f"Error removing {file_path}: {e}")
    else:
        # Create the directory if it doesn't exist
        os.makedirs(directory_path, exist_ok=True)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Enhanced Puzzle Piece Detector with adaptive parameter optimization"
    )
    
    # Required parameters
    parser.add_argument("--image", required=True, help="Path to the puzzle image")
    
    # Optional parameters
    parser.add_argument("--pieces", type=int, help="Expected number of pieces in the puzzle")
    parser.add_argument("--debug-dir", default="debug", help="Directory to save debug outputs")
    parser.add_argument("--log-dir", default="logs", help="Directory to save log files")
    parser.add_argument("--extract", action="store_true", help="Extract individual pieces to separate files")
    parser.add_argument("--extract-dir", default="extracted_pieces", help="Directory to save extracted pieces")
    parser.add_argument("--use-multiprocessing", action="store_true", help="Use multiprocessing for faster detection")
    parser.add_argument("--view", action="store_true", help="View results in image windows")
    parser.add_argument("--auto-threshold", action="store_true", help="Applies both Otsu and adaptive thresholding")
    parser.add_argument("--no-mean-filter", action="store_true", help="Disable filtering by mean area")
    parser.add_argument("--mean-threshold", type=float, default=1.5, help="Standard deviation threshold for mean filtering")
    parser.add_argument("--adaptive-preprocessing", action="store_true", help="Use adaptive preprocessing")
    parser.add_argument("--optimize-parameters", action="store_true", help="Optimize parameters for the image")
    parser.add_argument("--multi-pass", action="store_true", help="Use multi-pass detection")
    parser.add_argument("--analyze-image", action="store_true", help="Analyze image characteristics")
    
    # parameters for final verification
    parser.add_argument("--area-verification", action="store_true", help="Apply final area verification")
    parser.add_argument("--area-threshold", type=float, default=2.0, help="Standard deviation threshold for area verification")
    parser.add_argument("--comprehensive-verification", action="store_true", help="Apply comprehensive final verification")
    # Add performance profiling option
    parser.add_argument("--profile", action="store_true", help="Enable performance profiling")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with image saving")
    return parser.parse_args()



def create_config(args):
    """
    Create configuration based on command line arguments
    
    Args:
        args: Command line arguments
    
    Returns:
        Config object
    """
    config = Config()
    
    # Update config from command-line arguments
    config.DEBUG_DIR = args.debug_dir
    config.USE_AUTO_THRESHOLD = args.auto_threshold
    config.USE_MEAN_FILTERING = not args.no_mean_filter
    config.MEAN_DEVIATION_THRESHOLD = args.mean_threshold
    config.USE_MULTIPROCESSING = args.use_multiprocessing
    
    # New parameters for enhanced detection
    config.USE_ADAPTIVE_PREPROCESSING = args.adaptive_preprocessing
    config.USE_PARAMETER_OPTIMIZATION = args.optimize_parameters
    config.USE_MULTI_PASS_DETECTION = args.multi_pass
    
    return config


def log_message(message, console=False):
    """
    Log a message to both console and file or just file
    
    Args:
        message: Message to log
        console: Whether to show in console
    """
    if console:
        logging.getLogger('progress').info(message)
    else:
        logging.info(message)


def display_results(results, expected_pieces: Optional[int] = None):
    """
    Display processing results in a formatted way
    
    Args:
        results: Dictionary with processing results
        expected_pieces: Expected number of pieces
    """
    pieces = results['pieces']
    metrics = results['metrics']
    
    # Log detailed information to file only
    logging.info("\n=== Puzzle Analysis Results ===")
    logging.info(f"Detected {len(pieces)} valid puzzle pieces")
    
    if expected_pieces:
        detection_rate = len(pieces) / expected_pieces * 100
        logging.info(f"Detection rate: {detection_rate:.1f}%")
    
    logging.info(f"Processing time: {results['processing_time']:.2f} seconds")
    
    # Print summary to console
    progress_logger = logging.getLogger('progress')
    progress_logger.info(f"Detected {len(pieces)} puzzle pieces")
    if expected_pieces:
        detection_rate = len(pieces) / expected_pieces * 100
        progress_logger.info(f"Detection rate: {detection_rate:.1f}%")
    progress_logger.info(f"Processing time: {results['processing_time']:.2f} seconds")


def view_images(results):
    """
    Display images in OpenCV windows
    
    Args:
        results: Dictionary with processing results
    """
    # Display summary visualization
    cv2.imshow("Puzzle Analysis Summary", results['visualizations']['summary'])
    
    # Display metrics visualization
    cv2.imshow("Metrics", results['visualizations']['metrics'])
    
    # Wait for user to close windows
    log_message("Press any key to close image windows and continue...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    """Main entry point for the puzzle piece detector"""
    # Start timing
    start_time = time.time()
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Set up logging to both console and file
    log_file = setup_logging(args.log_dir)
    
    # Add performance profiling option
    if args.profile:
        profiler = cProfile.Profile()
        profiler.enable()
        
    # Clear debug and extracted_pieces directories
    log_message("Cleaning output directories...")
    clear_directory(args.debug_dir)
    if args.extract:
        clear_directory(args.extract_dir)
    
    # Print initialization message
    log_message("== Enhanced Puzzle Piece Detector ==")
    log_message(f"Processing image: {args.image}")
    log_message(f"Debug output directory: {args.debug_dir}")
    log_message(f"Log file: {log_file}")
    
    if args.pieces:
        log_message(f"Expected pieces: {args.pieces}")
    
    if args.no_mean_filter:
        log_message(f"Mean area filtering: disabled")
    else:
        log_message(f"Using mean area filtering (threshold: {args.mean_threshold} std dev)")
    
    if args.adaptive_preprocessing:
        log_message(f"Using adaptive preprocessing")
    
    if args.optimize_parameters:
        log_message(f"Using parameter optimization")
    
    if args.multi_pass:
        log_message(f"Using multi-pass detection")
    
    # Log verification settings
    if args.area_verification:
        log_message(f"Using final area verification (threshold: {args.area_threshold})")
    
    if args.comprehensive_verification:
        log_message(f"Using comprehensive final verification")
    
    # Create configuration
    config = create_config(args)
    # Skip debug image saving when not in debug mode
    if not args.debug:
        config.DEBUG = False
        # Create a no-op function to replace save_debug_image
        def dummy_save(*args, **kwargs):
            pass
        processor.detector.save_debug_image = dummy_save
    # Create processor
    processor = PuzzleProcessor(config)
    
    # Analyze image characteristics if requested
    if args.analyze_image:
        log_message("Analyzing image characteristics...")
        analysis = processor.analyze_image_characteristics(args.image)
        
        log_message(f"Image contrast: {analysis['contrast']:.2f}")
        log_message(f"Edge density: {analysis['edge_density']:.3f}")
        log_message(f"Dark background: {analysis['is_dark_background']}")
        log_message(f"Bimodal histogram: {analysis['is_bimodal']}")
        
        # Update config based on analysis
        config.optimize_for_image_characteristics(analysis)
        log_message("Configuration optimized based on image characteristics")
    
    # Optimize parameters if requested
    if args.optimize_parameters:
        expected_pieces = args.pieces if args.pieces else 24
        
        log_message("Optimizing parameters...")
        optimization = processor.optimize_parameters_for_image(args.image, expected_pieces)
        
        log_message("Optimization complete:")
        log_message(f"Best parameters found:")
        for param, value in optimization['best_params'].items():
            log_message(f"   {param}: {value}")
        log_message(f"Best detection rate: {optimization['detection_rate'] * 100:.1f}%")
        
        # Update config with best parameters
        config.update(**optimization['best_params'])
        log_message("Configuration updated with optimal parameters")
    
    # Process the image
    # Add area verification parameters if enabled
    process_kwargs = {
        'use_multi_pass': args.multi_pass
    }
    
    if args.area_verification:
        process_kwargs['area_verification_threshold'] = args.area_threshold
        process_kwargs['use_area_verification'] = True
    
    if args.comprehensive_verification:
        process_kwargs['use_comprehensive_verification'] = True
    
    results = processor.process_image(args.image, args.pieces, **process_kwargs)
    
    # Extract individual pieces if requested
    if args.extract:
        processor.extract_pieces(results['pieces'], args.extract_dir)
    
    # Display results
    display_results(results, args.pieces)
    
    # Save results
    processor.save_results(results)
    
    # View images if requested
    if args.view:
        view_images(results)
    # End profiling if enabled
    if args.profile:
        profiler.disable()
        s = StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        ps.print_stats(20)  # Print top 20 time consumers
        logging.info("Performance Profile:\n" + s.getvalue())
    
    # Report total processing time
    total_time = time.time() - start_time
    log_message(f"Total processing time: {total_time:.2f} seconds", console=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        sys.exit(1)