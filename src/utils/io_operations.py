"""I/O operations for saving debug files and results."""

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from typing import Dict, List, Any

from .visualization import (
    create_preprocessing_visualization, 
    create_detection_visualization,
    create_piece_gallery,
    save_piece_with_visualization
)


def save_masks(img: np.ndarray, binary_mask: np.ndarray, processed_mask: np.ndarray, 
               filled_mask: np.ndarray, valid_contours: List, dirs: Dict[str, str], 
               threshold_value: int, min_area: int):
    """Save comprehensive preprocessing and detection visualizations.
    
    Args:
        img: Original image
        binary_mask: Initial binary threshold mask
        processed_mask: After morphological operations
        filled_mask: Final mask with filled contours
        valid_contours: List of valid contours
        dirs: Dictionary of output directories
        threshold_value: Threshold used for binary segmentation
        min_area: Minimum area for valid contours
    """
    # Create comprehensive preprocessing visualization
    create_preprocessing_visualization(
        img, binary_mask, processed_mask, filled_mask, 
        threshold_value, dirs['preprocessing']
    )
    
    # Create detection visualization
    all_contours, _ = cv2.findContours(processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    create_detection_visualization(
        img, all_contours, valid_contours, min_area, dirs['detection']
    )


def save_pieces(pieces: List[Dict[str, Any]], img: np.ndarray, filled_mask: np.ndarray, dirs: Dict[str, str]):
    """Save individual puzzle pieces with enhanced visualizations.
    
    Args:
        pieces: List of piece data dictionaries
        img: Original image
        filled_mask: Binary mask
        dirs: Dictionary of output directories
    """
    # Create piece gallery
    create_piece_gallery(pieces, dirs['pieces'])
    
    # Save individual pieces with detailed info
    for index, piece_data in enumerate(pieces):
        piece_img = np.array(piece_data['img'], dtype=np.uint8)
        save_piece_with_visualization(piece_data, piece_img, index, dirs['pieces'])


def save_piece_analysis(piece_result: Dict[str, Any], piece_img: np.ndarray, output_dir: str):
    """Save detailed analysis results for a piece.
    
    Args:
        piece_result: Processed piece data
        piece_img: Piece image
        output_dir: Output directory
    """
    piece_idx = piece_result['piece_idx']
    
    # Save edge type information
    edge_types_file = os.path.join(output_dir, f"piece_{piece_idx+1}_analysis.txt")
    with open(edge_types_file, 'w') as f:
        f.write(f"Piece {piece_idx+1} Analysis\n")
        f.write("=" * 30 + "\n")
        f.write(f"Edge types: {piece_result.get('edge_types', [])}\n")
        f.write(f"Edge deviations: {piece_result.get('edge_deviations', [])}\n")
        f.write(f"Corners: {piece_result.get('corners', [])}\n")
        f.write(f"Centroid: {piece_result.get('centroid', 'Unknown')}\n")


def create_summary_report(piece_results: List[Dict[str, Any]], output_file: str):
    """Create a summary report of all processed pieces.
    
    Args:
        piece_results: List of processed piece results
        output_file: Path to output summary file
    """
    with open(output_file, 'w') as f:
        f.write("Puzzle Analysis Summary\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Total pieces processed: {len(piece_results)}\n\n")
        
        # Count edge types
        edge_type_counts = {}
        for result in piece_results:
            for edge_type in result.get('edge_types', []):
                edge_type_counts[edge_type] = edge_type_counts.get(edge_type, 0) + 1
        
        f.write("Edge type distribution:\n")
        for edge_type, count in edge_type_counts.items():
            f.write(f"  {edge_type}: {count}\n")
        
        f.write("\nPiece details:\n")
        for i, result in enumerate(piece_results):
            f.write(f"\nPiece {i+1}:\n")
            f.write(f"  Edge types: {result.get('edge_types', [])}\n")
            f.write(f"  Corners: {len(result.get('corners', []))}\n")