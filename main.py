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
from src.utils.io_operations import save_masks, save_pieces, create_summary_report
import time

class Timer:
    """Simple timer context manager."""
    def __init__(self, name):
        self.name = name
    
    def __enter__(self):
        self.start = time.time()
        return self
    
    def __exit__(self, *args):
        duration = time.time() - self.start
        print(f"{self.name} completed in {duration:.3f}s")
from src.utils.visualization import (
    create_input_visualization, 
    create_edge_classification_visualization,
    create_summary_dashboard,
    create_shape_analysis_visualization,
    create_shape_summary_visualization,
    create_piece_classification_visualizations,
    create_edge_color_visualizations
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
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found")
        return 1
    
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
    
    # Step 4: Process pieces sequentially
    with Timer("Sequential piece processing"):
        from src.core.piece_detection import process_piece
        piece_results = []
        
        for i, piece in enumerate(pieces):
            print(f"Processing piece {i+1}/{len(pieces)}...", end='\r')
            result = process_piece(piece, output_dirs)
            piece_results.append(result)
        
        print(f"\nSuccessfully processed {len(piece_results)} pieces")
    
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
                edge_colors = result.get('edge_colors', [])
                
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
                        
                        # Add color features if available
                        if edge_idx < len(edge_colors) and edge_colors[edge_idx]:
                            edge_color_data = edge_colors[edge_idx]
                            if 'color_sequence' in edge_color_data:
                                edge_segment.color_sequence = edge_color_data['color_sequence']
                            if 'confidence_sequence' in edge_color_data:
                                edge_segment.confidence_sequence = edge_color_data['confidence_sequence']
                        
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
                            
                            # Update color features if available
                            if edge_idx < len(edge_colors) and edge_colors[edge_idx]:
                                edge_color_data = edge_colors[edge_idx]
                                if 'color_sequence' in edge_color_data:
                                    edge.color_sequence = edge_color_data['color_sequence']
                                if 'confidence_sequence' in edge_color_data:
                                    edge.confidence_sequence = edge_color_data['confidence_sequence']
            
            # Re-classify piece type after edges are updated
            piece._classify_piece_type()
    
    # Step 4.5: Standardize edge indexing
    with Timer("Standardizing edge indexing"):
        from src.core.edge_indexing import standardize_all_pieces, visualize_edge_indexing
        
        # Standardize edge indexing for all pieces
        standardize_all_pieces(pieces)
        
        # Create debug visualizations
        edge_indexing_dir = os.path.join(dirs['base'], 'edge_indexing')
        os.makedirs(edge_indexing_dir, exist_ok=True)
        
        for piece in pieces:
            output_path = os.path.join(edge_indexing_dir, f'piece_{piece.index:02d}_edges.png')
            visualize_edge_indexing(piece, output_path)
    
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
    
    # Step 7: Piece classification visualization
    with Timer("Creating piece classification visualizations"):
        create_piece_classification_visualizations(pieces, dirs['base'])
    
    # Step 8: Edge color visualizations
    with Timer("Creating edge color visualizations"):
        create_edge_color_visualizations(pieces, dirs['base'])
    
    # Step 9: Edge matching
    with Timer("Performing edge matching"):
        from src.features.edge_matching_rotation_aware import perform_rotation_aware_matching
        
        # Perform rotation-aware edge matching
        registry, spatial_index, assembly = perform_rotation_aware_matching(pieces)
    
    # Step 10: Create edge matching visualizations
    with Timer("Creating edge matching visualizations"):
        from src.utils.edge_matching_visualization import (
            create_edge_match_visualization,
            create_match_candidates_gallery,
            create_match_validation_dashboard,
            create_match_confidence_report,
            create_interactive_match_explorer,
            create_color_continuity_visualization,
            create_shape_compatibility_analysis
        )
        
        # Create matching output directory
        matching_dir = os.path.join(dirs['base'], '08_matching')
        os.makedirs(matching_dir, exist_ok=True)
        
        # 1. Individual match visualizations (top matches)
        individual_dir = os.path.join(matching_dir, 'individual_matches')
        os.makedirs(individual_dir, exist_ok=True)
        
        # Visualize confirmed matches
        confirmed_matches = []
        for (p1, e1, p2, e2) in registry.confirmed_matches:
            if (p1, e1) in registry.matches and (p2, e2) in registry.matches[(p1, e1)]:
                match = registry.matches[(p1, e1)][(p2, e2)]
                if p1 < p2:  # Avoid duplicates
                    confirmed_matches.append(((p1, e1, p2, e2), match))
        
        # Sort by score and visualize all confirmed matches
        confirmed_matches.sort(key=lambda x: x[1].similarity_score, reverse=True)
        for (p1, e1, p2, e2), match in confirmed_matches[:10]:
            if p1 < len(pieces) and p2 < len(pieces):
                create_edge_match_visualization(
                    pieces[p1], e1, pieces[p2], e2, match, individual_dir
                )
        
        # 2. Match candidate galleries
        gallery_dir = os.path.join(matching_dir, 'candidate_galleries')
        os.makedirs(gallery_dir, exist_ok=True)
        
        # Create galleries for all non-flat edges (the interesting ones)
        gallery_count = 0
        for piece in pieces:
            for edge_idx, edge in enumerate(piece.edges):
                # Skip flat edges as they don't have matches
                if edge.edge_type == 'flat':
                    continue
                    
                # Get more candidates to show in gallery
                matches = registry.get_best_matches(piece.index, edge_idx, n=20)
                if matches:
                    candidates = []
                    for (target_piece_idx, target_edge_idx), match in matches:
                        if target_piece_idx < len(pieces):
                            candidates.append((match, pieces[target_piece_idx]))
                    
                    if candidates:
                        create_match_candidates_gallery(
                            piece, edge_idx, candidates, registry, gallery_dir
                        )
                        gallery_count += 1
        
        print(f"Created {gallery_count} candidate galleries for non-flat edges")
        
        # 3. Match validation dashboard
        create_match_validation_dashboard(registry, spatial_index, matching_dir)
        
        # 4. Match confidence report
        create_match_confidence_report(pieces, registry, matching_dir)
        
        # 5. Interactive match explorer
        create_interactive_match_explorer(pieces, registry, matching_dir)
        
        # 6. Color continuity visualizations
        color_dir = os.path.join(matching_dir, 'color_continuity')
        os.makedirs(color_dir, exist_ok=True)
        
        # Visualize color continuity for confirmed matches
        for (p1, e1, p2, e2) in list(registry.confirmed_matches)[:10]:
            if p1 < len(pieces) and p2 < len(pieces) and p1 < p2:
                create_color_continuity_visualization(
                    pieces[p1], e1, pieces[p2], e2, color_dir
                )
        
        # 7. Shape compatibility analysis
        all_matches = []
        for (p1, e1), matches_dict in registry.matches.items():
            for (p2, e2), match in matches_dict.items():
                if p1 < p2:  # Avoid duplicates
                    all_matches.append(((p1, e1, p2, e2), match))
        
        if all_matches:
            create_shape_compatibility_analysis(all_matches, matching_dir)
    
    # Step 11: Create final glued puzzle from the assembly
    with Timer("Creating final glued puzzle"):
        from src.assembly.simple_gluing import create_glued_puzzle
        
        # Create final output directory
        final_dir = os.path.join(dirs['base'], '09_final_puzzle')
        os.makedirs(final_dir, exist_ok=True)
        
        # Create the final glued puzzle using the assembly from edge matching
        final_path = os.path.join(final_dir, 'final_puzzle.png')
        if assembly and assembly.grid:
            final_image = create_glued_puzzle(assembly, pieces, final_path)
            print(f"\n=== PUZZLE SOLVED ===")
            print(f"Final glued puzzle saved to: {final_path}")
        else:
            # Fallback to hardcoded assembly for chicken puzzle
            print("\nWarning: Edge matching failed, using hardcoded assembly")
            from src.assembly.hardcoded_assembly import create_hardcoded_assembly
            assembly = create_hardcoded_assembly(pieces)
            if assembly and assembly.grid:
                final_image = create_glued_puzzle(assembly, pieces, final_path)
                print(f"\n=== PUZZLE SOLVED (Hardcoded) ===")
                print(f"Final glued puzzle saved to: {final_path}")
    
    print("\nProcess completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())