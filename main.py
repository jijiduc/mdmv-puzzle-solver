#!/usr/bin/env python3
"""Main entry point for the puzzle solver."""

import argparse
import sys
import os
import time
import cv2
import numpy as np
from typing import Dict, Any

# Add src to path to allow imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.config.settings import INPUT_PATH, THRESHOLD_VALUE, MIN_CONTOUR_AREA
from src.core.image_processing import detect_puzzle_pieces, setup_output_directories, preprocess_image
from src.utils.parallel import parallel_process_pieces, set_process_priority, Timer
from src.utils.io_operations import save_masks, save_pieces, create_summary_report
from src.utils.visualization import (
    create_input_visualization, 
    create_geometry_visualization,
    create_edge_classification_visualization,
    create_summary_dashboard
)
from src.utils.corner_analysis import analyze_corner_detection_method


def main():
    """Main function for puzzle solving."""
    parser = argparse.ArgumentParser(description='Puzzle Solver - Model Driven Machine Vision')
    parser.add_argument('--input', '-i', default=INPUT_PATH, 
                       help='Path to input puzzle image')
    parser.add_argument('--threshold', '-t', type=int, default=THRESHOLD_VALUE,
                       help='Binary threshold value for segmentation')
    parser.add_argument('--min-area', '-a', type=int, default=MIN_CONTOUR_AREA,
                       help='Minimum contour area for valid pieces')
    parser.add_argument('--workers', '-w', type=int, default=None,
                       help='Number of parallel workers')
    parser.add_argument('--no-cache', action='store_true',
                       help='Disable caching')
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found")
        return 1
    
    # Set process priority
    set_process_priority()
    
    # Track total processing time
    start_time = time.time()
    
    # Setup output directories
    dirs = setup_output_directories()
    output_dirs = (dirs['edges'], dirs['edge_types'], dirs['contours'])
    
    # Step 1: Input analysis
    with Timer("Input analysis"):
        img, binary_mask, processed_mask = preprocess_image(args.input, args.threshold)
        create_input_visualization(args.input, img, dirs['input'])
    
    # Step 2: Puzzle piece detection  
    with Timer("Puzzle detection"):
        puzzle_data = detect_puzzle_pieces(args.input, args.threshold, args.min_area)
        
        # Create final mask for piece extraction
        
        contours, _ = cv2.findContours(processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = [c for c in contours if cv2.contourArea(c) > args.min_area]
        
        filled_mask = np.zeros_like(processed_mask)
        cv2.drawContours(filled_mask, valid_contours, -1, 255, -1)
    
    # Extract piece data
    piece_count = puzzle_data['count']
    pieces = puzzle_data['pieces']
    
    print(f"Detected {piece_count} puzzle pieces")
    
    # Step 3: Save preprocessing and detection results
    with Timer("Saving preprocessing and detection results"):
        save_masks(img, binary_mask, processed_mask, filled_mask, valid_contours, 
                  dirs, args.threshold, args.min_area)
        save_pieces(pieces, img, filled_mask, dirs)
    
    # Step 4: Process pieces in parallel
    with Timer("Parallel piece processing"):
        piece_results = parallel_process_pieces(pieces, output_dirs, args.workers)
    
    print(f"Successfully processed {len(piece_results)} pieces")
    
    # Step 5: Create geometry visualizations
    with Timer("Creating geometry visualizations"):
        for i, result in enumerate(piece_results):
            if result:
                piece_img = np.array(pieces[i]['img'], dtype=np.uint8)
                create_geometry_visualization(result, piece_img, i, dirs['geometry'])
                
                # Add detailed corner detection analysis
                corner_analysis_dir = os.path.join(dirs['geometry'], 'corner_analysis')
                os.makedirs(corner_analysis_dir, exist_ok=True)
                analyze_corner_detection_method(pieces[i], piece_img, i, corner_analysis_dir)
    
    # Step 6: Basic edge visualization
    with Timer("Creating edge visualization"):
        # Create basic edge classification visualization
        create_edge_classification_visualization(piece_results, dirs['classification'])
    
    # Step 7: Create final summary
    total_time = time.time() - start_time
    with Timer("Creating summary dashboard"):
        create_summary_dashboard(piece_count, total_time, piece_results, dirs['base'])
        create_summary_report(piece_results, os.path.join(dirs['base'], 'detailed_report.txt'))
        
    
    # TODO: Add edge matching and puzzle assembly
    # This would involve:
    # 1. Edge matching using DTW and shape compatibility
    # 2. Puzzle assembly algorithm
    # 3. Final visualization and output
    
    print("Puzzle analysis complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())