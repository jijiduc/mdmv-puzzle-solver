"""
Puzzle processing pipeline and analysis
"""

from src.utils.image_utils import read_image, save_image
from src.utils.visualization import (
    create_processing_visualization, display_metrics, draw_contours
)
from src.core.piece import PuzzlePiece
from src.core.detector import PuzzleDetector
from src.config.settings import Config
import cv2
import numpy as np
import os
import sys
import json
import time
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter

# Add parent directory to path to allow imports from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class PuzzleProcessor:
    """
    Main processor for puzzle detection and analysis
    """

    def __init__(self, config: Config = None):
        """
        Initialize the processor

        Args:
            config: Configuration parameters
        """
        self.config = config or Config()
        self.detector = PuzzleDetector(config)

        # Ensure debug directory exists
        if self.config.DEBUG:
            os.makedirs(self.config.DEBUG_DIR, exist_ok=True)

    def process_image(self, 
                        image_path: str, 
                        expected_pieces: Optional[int] = None) -> Dict[str, Any]:
        """
        Process an image to detect and analyze puzzle pieces
        No image resizing is performed.
        
        Args:
            image_path: Path to the image file
            expected_pieces: Expected number of pieces (for metrics)
        
        Returns:
            Dictionary with processing results
        """
        print(f"🔍 Processing image: {image_path}")
        
        # Read the image - without any resizing
        image = read_image(image_path)
        
        # Log original image dimensions
        print(f"📏 Original image dimensions: {image.shape[1]}x{image.shape[0]}")
        
        # Detect puzzle pieces
        pieces, debug_images = self.detector.detect(image)
        
        # Calculate metrics
        metrics = self.calculate_metrics(pieces, image, expected_pieces)
        
        # Save metrics report
        metrics_path = os.path.join(self.config.DEBUG_DIR, "metrics_report.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"📊 Metrics saved to {metrics_path}")
        
        # Create piece visualizations - now showing only the detected piece
        piece_visualizations = []
        for i, piece in enumerate(pieces):
            # Create two types of visualizations:
            # 1. Full image with piece contour for the summary
            full_vis = piece.draw(image)
            
            # 2. Just the extracted piece image for individual piece visualization
            if piece.extracted_image is not None:
                piece_only_vis = piece.extracted_image.copy()
                
                # Save just the piece image
                if self.config.DEBUG:
                    save_path = os.path.join(self.config.DEBUG_DIR, f"piece_{i}.jpg")
                    save_image(piece_only_vis, save_path)
            else:
                # Fallback if extraction failed
                piece_only_vis = full_vis
                print(f"⚠️ Warning: Could not extract clean image for piece {i}")
                
            piece_visualizations.append(full_vis)  # Keep using full_vis for summary
        
        # Create summary visualization
        pieces_info = []
        for i, piece in enumerate(pieces):
            pieces_info.append({
                'id': i,
                'visualization': piece_visualizations[i],
                'is_valid': piece.is_valid,
                'border_types': piece.border_types
            })
        
        summary_vis = create_processing_visualization(
            image,
            debug_images['preprocessed'],
            debug_images['binary'],
            debug_images['piece_visualization'],
            pieces_info,
            os.path.join(self.config.DEBUG_DIR, "processing_summary.jpg")
        )
        
        # Create metrics visualization
        metrics_vis = display_metrics(metrics)
        save_image(metrics_vis, os.path.join(self.config.DEBUG_DIR, "metrics_visualization.jpg"))
        
        # Return results
        return {
            'image_path': image_path,
            'pieces': pieces,
            'metrics': metrics,
            'visualizations': {
                'summary': summary_vis,
                'metrics': metrics_vis,
                'pieces': piece_visualizations
            }
        }

    def calculate_metrics(self,
                          pieces: List[PuzzlePiece],
                          image: np.ndarray,
                          expected_count: Optional[int] = None) -> Dict[str, Any]:
        """
        Calculate metrics for detected puzzle pieces

        Args:
            pieces: List of detected pieces
            image: Original image
            expected_count: Expected number of pieces

        Returns:
            Dictionary of metrics
        """
        print("📐 Calculating metrics...")

        metrics = {}

        # Count-based metrics
        metrics['detected_count'] = len(pieces)
        metrics['expected_count'] = expected_count

        if expected_count:
            metrics['detection_rate'] = len(pieces) / expected_count

        # Area-based metrics
        image_area = image.shape[0] * image.shape[1]

        if pieces:
            areas = [piece.features['area'] for piece in pieces]
            metrics['mean_area'] = np.mean(areas)
            metrics['std_area'] = np.std(areas)
            metrics['min_area'] = np.min(areas)
            metrics['max_area'] = np.max(areas)
            metrics['total_piece_area'] = sum(areas)
            metrics['area_coverage'] = sum(areas) / image_area
            metrics['area_distribution'] = areas
        else:
            metrics['mean_area'] = 0
            metrics['std_area'] = 0
            metrics['min_area'] = 0
            metrics['max_area'] = 0
            metrics['total_piece_area'] = 0
            metrics['area_coverage'] = 0
            metrics['area_distribution'] = []

        # Shape metrics
        if pieces:
            solidities = [piece.features['solidity'] for piece in pieces]
            metrics['mean_solidity'] = np.mean(solidities)
            metrics['std_solidity'] = np.std(solidities)
            metrics['solidity_distribution'] = solidities

            # Equivalent diameters
            eq_diameters = [piece.features['equivalent_diameter']
                            for piece in pieces]
            metrics['mean_equivalent_diameter'] = np.mean(eq_diameters)
            metrics['std_equivalent_diameter'] = np.std(eq_diameters)
        else:
            metrics['mean_solidity'] = 0
            metrics['std_solidity'] = 0
            metrics['solidity_distribution'] = []
            metrics['mean_equivalent_diameter'] = 0
            metrics['std_equivalent_diameter'] = 0

        # Border type distribution
        if pieces:
            all_border_types = []
            for piece in pieces:
                if piece.border_types:
                    all_border_types.extend(piece.border_types)

            border_counter = Counter(all_border_types)
            metrics['border_types'] = dict(border_counter)
        else:
            metrics['border_types'] = {}

        # Edge alignment metrics (how well contours align with actual edges)
        edge_map = cv2.Canny(image, 50, 150)
        if pieces:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            for piece in pieces:
                cv2.drawContours(mask, [piece.contour], -1, 255, 2)

            # Calculate overlap between detected contours and edge map
            overlap = cv2.bitwise_and(edge_map, mask)
            metrics['edge_alignment'] = np.sum(
                overlap > 0) / (np.sum(mask > 0) + 1e-6)
        else:
            metrics['edge_alignment'] = 0

        return metrics

    def extract_pieces(self, 
                        pieces: List[PuzzlePiece], 
                        output_dir: str = "extracted_pieces") -> List[str]:
        """
        Extract individual puzzle pieces to separate image files
        
        Args:
            pieces: List of puzzle pieces
            output_dir: Directory to save extracted pieces
        
        Returns:
            List of paths to saved piece images
        """
        print(f"✂️ Extracting {len(pieces)} pieces to {output_dir}...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        saved_paths = []
        for i, piece in enumerate(pieces):
            # Use the new get_extracted_image method with clean background
            if hasattr(piece, 'get_extracted_image'):
                piece_image = piece.get_extracted_image(clean_background=True)
            else:
                # Fallback to the old method if the new one isn't available
                if piece.extracted_image is not None:
                    piece_image = piece.extracted_image
                else:
                    print(f"⚠️ Warning: No extracted image for piece {i}, skipping.")
                    continue
                    
            path = os.path.join(output_dir, f"piece_{i:03d}.jpg")
            save_image(piece_image, path)
            saved_paths.append(path)
        
        print(f"✅ Saved {len(saved_paths)} pieces to {output_dir}")
        return saved_paths

    def save_results(self,
                     results: Dict[str, Any],
                     output_dir: str = "results") -> str:
        """
        Save processing results to a directory

        Args:
            results: Processing results
            output_dir: Directory to save results

        Returns:
            Path to saved results
        """
        os.makedirs(output_dir, exist_ok=True)

        # Create a timestamp-based directory
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        result_dir = os.path.join(output_dir, f"puzzle_{timestamp}")
        os.makedirs(result_dir, exist_ok=True)

        # Save visualizations
        vis_dir = os.path.join(result_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)

        save_image(results['visualizations']['summary'],
                   os.path.join(vis_dir, "summary.jpg"))

        save_image(results['visualizations']['metrics'],
                   os.path.join(vis_dir, "metrics.jpg"))

        # Save piece visualizations
        pieces_dir = os.path.join(vis_dir, "pieces")
        os.makedirs(pieces_dir, exist_ok=True)

        for i, vis in enumerate(results['visualizations']['pieces']):
            save_image(vis, os.path.join(pieces_dir, f"piece_{i:03d}.jpg"))

        # Save metrics as JSON
        with open(os.path.join(result_dir, "metrics.json"), 'w') as f:
            json.dump(results['metrics'], f, indent=4)

        # Save pieces data
        pieces_data = []
        for piece in results['pieces']:
            pieces_data.append(piece.to_dict())

        with open(os.path.join(result_dir, "pieces.json"), 'w') as f:
            json.dump(pieces_data, f, indent=4)

        print(f"💾 Results saved to {result_dir}")
        return result_dir
