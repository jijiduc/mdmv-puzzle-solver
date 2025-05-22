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
    create_edge_classification_visualization,
    create_summary_dashboard,
    create_shape_analysis_visualization,
    create_shape_summary_visualization
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
    
    # Update pieces with processing results (corners, edges, etc.)
    for i, result in enumerate(piece_results):
        if result and i < len(pieces):
            piece = pieces[i]
            # Update corners if they were found
            if 'corners' in result:
                piece.corners = result['corners']
            
            # Re-process edges to get updated classification data
            # Since the parallel processing doesn't return the updated piece object,
            # we need to reconstruct the edge data from the results
            if ('edge_types' in result and 'edge_deviations' in result and 
                'edge_sub_types' in result and 'edge_confidences' in result and
                'edge_points' in result):
                
                edge_types = result['edge_types'] 
                edge_deviations = result['edge_deviations']
                edge_sub_types = result['edge_sub_types']
                edge_confidences = result['edge_confidences']
                edge_points_list = result['edge_points']
                
                # If piece has no edges (parallel processing issue), create them from result data
                if len(piece.edges) == 0 and len(edge_types) > 0:
                    from src.core.piece import EdgeSegment
                    for edge_idx in range(len(edge_types)):
                        edge_segment = EdgeSegment(
                            edge_type=edge_types[edge_idx],
                            deviation=edge_deviations[edge_idx],
                            sub_type=edge_sub_types[edge_idx],
                            confidence=edge_confidences[edge_idx],
                            points=edge_points_list[edge_idx] if edge_idx < len(edge_points_list) else [],
                            length=len(edge_points_list[edge_idx]) if edge_idx < len(edge_points_list) else 0
                        )
                        piece.add_edge(edge_segment)
                else:
                    # Update existing edges with the classification results
                    for edge_idx, edge in enumerate(piece.edges):
                        if edge_idx < len(edge_types):
                            edge.edge_type = edge_types[edge_idx]
                            edge.deviation = edge_deviations[edge_idx]
                            edge.sub_type = edge_sub_types[edge_idx]
                            edge.confidence = edge_confidences[edge_idx]
                            if edge_idx < len(edge_points_list):
                                edge.points = edge_points_list[edge_idx]
                                edge.length = len(edge_points_list[edge_idx])
    
    # Step 5: Create geometry visualizations
    with Timer("Creating geometry visualizations"):
        for i, result in enumerate(piece_results):
            if result and i < len(pieces):
                piece = pieces[i]
                
                # Add detailed corner detection analysis
                corner_analysis_dir = os.path.join(dirs['geometry'], 'corner_analysis')
                os.makedirs(corner_analysis_dir, exist_ok=True)
                analyze_corner_detection_method(piece.to_dict(), piece.image, i, corner_analysis_dir)
    
    # Step 6: Shape analysis visualization
    with Timer("Creating shape analysis visualizations"):
        # Create shape analysis directory
        shape_dir = os.path.join(dirs['features'], 'shape')
        os.makedirs(shape_dir, exist_ok=True)
        
        # Create individual piece shape analysis
        for i, result in enumerate(piece_results):
            if result and i < len(pieces):
                piece = pieces[i]
                create_shape_analysis_visualization(piece, i, shape_dir)
        
        # Create shape summary visualization
        create_shape_summary_visualization(pieces, shape_dir)
        
        # Create basic edge classification visualization (legacy)
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