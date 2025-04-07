"""
Enhanced puzzle processing pipeline with adaptive parameter optimization
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
import logging

# Add parent directory to path to allow imports from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class PuzzleProcessor:
    """
    Enhanced processor for puzzle detection and analysis with adaptive capabilities
    """

    def __init__(self, config: Config = None):
        """
        Initialize the processor

        Args:
            config: Configuration parameters
        """
        self.config = config or Config()
        self.detector = PuzzleDetector(config)
        self.logger = self._setup_logger()

        # Ensure debug directory exists
        if self.config.DEBUG:
            os.makedirs(self.config.DEBUG_DIR, exist_ok=True)
    
    def _setup_logger(self) -> logging.Logger:
        """Set up a logger for the processor"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG if self.config.DEBUG else logging.INFO)
        
        # Add console handler if no handlers exist
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger

    def process_image(self, 
                     image_path: str, 
                     expected_pieces: Optional[int] = None,
                     use_multi_pass: bool = True) -> Dict[str, Any]:
        """
        Process an image to detect and analyze puzzle pieces
        with enhanced adaptive optimization
        
        Args:
            image_path: Path to the image file
            expected_pieces: Expected number of pieces (for metrics and optimization)
            use_multi_pass: Whether to use multi-pass detection for improved results
        
        Returns:
            Dictionary with processing results
        """
        self.logger.info(f"Processing image: {image_path}")
        
        # Read the image - without any resizing
        image = read_image(image_path)
        
        # Log original image dimensions
        self.logger.info(f"Original image dimensions: {image.shape[1]}x{image.shape[0]}")
        
        # Get start time for performance measurement
        start_time = time.time()
        
        # Detect puzzle pieces using enhanced methods
        if use_multi_pass:
            # Use multi-pass detection for better results
            pieces, debug_images = self.detector.multi_pass_detection(image, expected_pieces)
        else:
            # Use single-pass detection (still with adaptive parameters)
            pieces, debug_images = self.detector.detect(image, expected_pieces)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        self.logger.info(f"Detection completed in {processing_time:.2f} seconds")
        
        # Calculate metrics
        metrics = self.calculate_metrics(pieces, image, expected_pieces)
        metrics['processing_time'] = processing_time
        
        # Save metrics report
        metrics_path = os.path.join(self.config.DEBUG_DIR, "metrics_report.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        self.logger.info(f"Metrics saved to {metrics_path}")
        
        # Create piece visualizations - showing the detected pieces
        piece_visualizations = []
        for i, piece in enumerate(pieces):
            # Create two types of visualizations:
            # 1. Full image with piece contour for the summary
            full_vis = piece.draw(image)
            
            # 2. Just the extracted piece image for individual piece visualization
            piece_only_vis = piece.get_extracted_image(clean_background=True)
            
            # Save just the piece image for debugging
            if self.config.DEBUG:
                save_path = os.path.join(self.config.DEBUG_DIR, f"piece_{i}.jpg")
                save_image(piece_only_vis, save_path)
                
            piece_visualizations.append(full_vis)  # Keep using full_vis for summary
        
        # Create summary visualization
        pieces_info = []
        for i, piece in enumerate(pieces):
            pieces_info.append({
                'id': i,
                'visualization': piece_visualizations[i],
                'is_valid': piece.is_valid,
                'border_types': piece.border_types,
                'validation_score': piece.validation_score if hasattr(piece, 'validation_score') else 0.0
            })
        
        # Create and save summary visualization
        summary_vis = create_processing_visualization(
            image,
            debug_images.get('preprocessed', np.zeros_like(image)),
            debug_images.get('binary', np.zeros_like(image)),
            debug_images.get('piece_visualization', np.zeros_like(image)),
            pieces_info,
            os.path.join(self.config.DEBUG_DIR, "processing_summary.jpg")
        )
        
        # Create metrics visualization
        metrics_vis = display_metrics(metrics)
        save_image(metrics_vis, os.path.join(self.config.DEBUG_DIR, "metrics_visualization.jpg"))
        
        # If we have combined visualization from multi-pass, include it
        if 'combined_visualization' in debug_images:
            save_image(debug_images['combined_visualization'], 
                     os.path.join(self.config.DEBUG_DIR, "combined_detection.jpg"))
        
        # Return results
        return {
            'image_path': image_path,
            'pieces': pieces,
            'metrics': metrics,
            'visualizations': {
                'summary': summary_vis,
                'metrics': metrics_vis,
                'pieces': piece_visualizations
            },
            'processing_time': processing_time
        }

    def calculate_metrics(self,
                          pieces: List[PuzzlePiece],
                          image: np.ndarray,
                          expected_count: Optional[int] = None) -> Dict[str, Any]:
        """
        Calculate comprehensive metrics for detected puzzle pieces
        
        Args:
            pieces: List of detected pieces
            image: Original image
            expected_count: Expected number of pieces
        
        Returns:
            Dictionary of metrics
        """
        self.logger.info("Calculating metrics...")

        metrics = {}

        # Count-based metrics
        metrics['detected_count'] = len(pieces)
        metrics['expected_count'] = expected_count
        metrics['valid_pieces'] = sum(1 for p in pieces if p.is_valid)

        if expected_count:
            metrics['detection_rate'] = len(pieces) / expected_count
            metrics['valid_detection_rate'] = metrics['valid_pieces'] / expected_count

        # Area-based metrics
        image_area = image.shape[0] * image.shape[1]

        if pieces:
            areas = [piece.features['area'] for piece in pieces]
            metrics['mean_area'] = np.mean(areas)
            metrics['median_area'] = np.median(areas)
            metrics['std_area'] = np.std(areas)
            metrics['min_area'] = np.min(areas)
            metrics['max_area'] = np.max(areas)
            metrics['total_piece_area'] = sum(areas)
            metrics['area_coverage'] = sum(areas) / image_area
            
            # Coefficient of variation (normalized measure of dispersion)
            metrics['area_cv'] = metrics['std_area'] / metrics['mean_area'] if metrics['mean_area'] > 0 else 0
        else:
            metrics['mean_area'] = 0
            metrics['median_area'] = 0
            metrics['std_area'] = 0
            metrics['min_area'] = 0
            metrics['max_area'] = 0
            metrics['total_piece_area'] = 0
            metrics['area_coverage'] = 0
            metrics['area_cv'] = 0

        # Shape metrics
        if pieces:
            # Get shape characteristics
            solidities = [piece.features['solidity'] for piece in pieces]
            metrics['mean_solidity'] = np.mean(solidities)
            metrics['std_solidity'] = np.std(solidities)
            
            # Compactness metrics (perimeter^2 / area)
            compactness = [piece.features.get('compactness', 0) for piece in pieces]
            metrics['mean_compactness'] = np.mean(compactness)
            metrics['std_compactness'] = np.std(compactness)

            # Equivalent diameters
            eq_diameters = [piece.features['equivalent_diameter'] for piece in pieces]
            metrics['mean_equivalent_diameter'] = np.mean(eq_diameters)
            metrics['std_equivalent_diameter'] = np.std(eq_diameters)
            
            # Validation scores if available
            validation_scores = [
                piece.validation_score for piece in pieces 
                if hasattr(piece, 'validation_score')
            ]
            if validation_scores:
                metrics['mean_validation_score'] = np.mean(validation_scores)
                metrics['min_validation_score'] = np.min(validation_scores)
                metrics['max_validation_score'] = np.max(validation_scores)
        else:
            metrics['mean_solidity'] = 0
            metrics['std_solidity'] = 0
            metrics['mean_compactness'] = 0
            metrics['std_compactness'] = 0
            metrics['mean_equivalent_diameter'] = 0
            metrics['std_equivalent_diameter'] = 0
            metrics['mean_validation_score'] = 0
            metrics['min_validation_score'] = 0
            metrics['max_validation_score'] = 0

        # Border type distribution
        if pieces:
            all_border_types = []
            for piece in pieces:
                if piece.border_types:
                    all_border_types.extend(piece.border_types)

            border_counter = Counter(all_border_types)
            metrics['border_types'] = dict(border_counter)
            
            # Calculate tab-to-pocket ratio (should be close to 1 for valid puzzles)
            tab_count = border_counter.get('tab', 0)
            pocket_count = border_counter.get('pocket', 0)
            if pocket_count > 0:
                metrics['tab_pocket_ratio'] = tab_count / pocket_count
            else:
                metrics['tab_pocket_ratio'] = 0
        else:
            metrics['border_types'] = {}
            metrics['tab_pocket_ratio'] = 0

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
        
        # Calculate metrics for validation scores if available
        valid_pieces = [p for p in pieces if p.is_valid]
        if valid_pieces:
            metrics['valid_piece_count'] = len(valid_pieces)
            metrics['valid_piece_ratio'] = len(valid_pieces) / len(pieces)
        else:
            metrics['valid_piece_count'] = 0
            metrics['valid_piece_ratio'] = 0

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
        self.logger.info(f"Extracting {len(pieces)} pieces to {output_dir}...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        saved_paths = []
        for i, piece in enumerate(pieces):
            # Use the enhanced extraction with clean background
            piece_image = piece.get_extracted_image(clean_background=True)
                    
            path = os.path.join(output_dir, f"piece_{i:03d}.jpg")
            save_image(piece_image, path)
            saved_paths.append(path)
        
        self.logger.info(f"Saved {len(saved_paths)} pieces to {output_dir}")
        return saved_paths
    
    def analyze_piece_matches(self, pieces: List[PuzzlePiece]) -> Dict[str, Any]:
        """
        Analyze potential matches between pieces
        
        Args:
            pieces: List of puzzle pieces
        
        Returns:
            Dictionary with match analysis results
        """
        self.logger.info(f"Analyzing potential matches between {len(pieces)} pieces...")
        
        # Only analyze valid pieces
        valid_pieces = [p for p in pieces if p.is_valid]
        
        if len(valid_pieces) < 2:
            return {'matches': []}
        
        # Calculate match scores between all pairs of pieces
        matches = []
        
        for i, piece1 in enumerate(valid_pieces):
            for j, piece2 in enumerate(valid_pieces):
                if i >= j:  # Skip self-matches and duplicates
                    continue
                
                # Calculate match score
                match_scores = piece1.calculate_match_score(piece2)
                
                # Only keep significant matches
                if match_scores['overall'] > 0.5:
                    matches.append({
                        'piece1_id': piece1.id,
                        'piece2_id': piece2.id,
                        'match_score': match_scores['overall'],
                        'match_details': match_scores
                    })
        
        # Sort matches by score (descending)
        matches.sort(key=lambda x: x['match_score'], reverse=True)
        
        # Limit to top matches for clarity
        top_matches = matches[:min(20, len(matches))]
        
        self.logger.info(f"Found {len(top_matches)} significant matches")
        
        return {
            'matches': top_matches,
            'match_count': len(top_matches),
            'analyzed_pieces': len(valid_pieces)
        }

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

        self.logger.info(f"Results saved to {result_dir}")
        return result_dir
    
    def optimize_parameters_for_image(self, 
                                     image_path: str, 
                                     expected_pieces: int,
                                     parameter_grid: Optional[Dict[str, List[Any]]] = None) -> Dict[str, Any]:
        """
        Optimize detection parameters for a specific image
        
        Args:
            image_path: Path to the image file
            expected_pieces: Expected number of pieces
            parameter_grid: Grid of parameters to try
        
        Returns:
            Dictionary with optimal parameters and results
        """
        self.logger.info(f"Optimizing parameters for {image_path}...")
        
        # Read the image
        image = read_image(image_path)
        
        # Define default parameter grid if not provided
        if parameter_grid is None:
            parameter_grid = {
                'CORNER_APPROX_EPSILON': [0.01, 0.02, 0.03, 0.05],
                'MIN_CONTOUR_AREA': [500, 1000, 2000, 3000],
                'MEAN_DEVIATION_THRESHOLD': [1.0, 1.5, 2.0, 2.5]
            }
        
        # Track best parameters and results
        best_score = -1
        best_params = {}
        best_pieces = []
        
        # Define a small subset of combinations to try
        # (full grid search would be too slow)
        total_combinations = 1
        for param_values in parameter_grid.values():
            total_combinations *= len(param_values)
        
        # Limit combinations to a reasonable number
        max_combinations = 12
        sample_ratio = min(1.0, max_combinations / total_combinations)
        
        self.logger.info(f"Testing {max_combinations} parameter combinations...")
        
        # Keep track of original parameter values
        original_params = {}
        for param_name in parameter_grid.keys():
            original_params[param_name] = getattr(self.config, param_name)
        
        # Try parameter combinations (sampling to limit combinations)
        import random
        combinations_to_try = []
        
        # Generate parameter combinations
        for _ in range(min(max_combinations, total_combinations)):
            combination = {}
            for param_name, param_values in parameter_grid.items():
                combination[param_name] = random.choice(param_values)
            combinations_to_try.append(combination)
        
        # Ensure we try both adaptive and standard preprocessing
        if 'USE_ADAPTIVE_PREPROCESSING' not in parameter_grid:
            for i, combo in enumerate(combinations_to_try):
                if i % 2 == 0:
                    combo['USE_ADAPTIVE_PREPROCESSING'] = True
                else:
                    combo['USE_ADAPTIVE_PREPROCESSING'] = False
        
        # Try each combination
        for i, param_set in enumerate(combinations_to_try):
            # Update config with current parameters
            for param_name, param_value in param_set.items():
                setattr(self.config, param_name, param_value)
            
            self.logger.info(f"Testing combination {i+1}/{len(combinations_to_try)}: {param_set}")
            
            # Run detection
            pieces, _ = self.detector.detect(image, expected_pieces)
            
            # Calculate score (detection rate and quality)
            detection_rate = len(pieces) / expected_pieces if expected_pieces > 0 else 0
            valid_pieces = sum(1 for p in pieces if p.is_valid)
            valid_rate = valid_pieces / expected_pieces if expected_pieces > 0 else 0
            
            # Average validation score
            validation_scores = [
                piece.validation_score for piece in pieces 
                if hasattr(piece, 'validation_score') and piece.is_valid
            ]
            avg_validation = np.mean(validation_scores) if validation_scores else 0
            
            # Combined score
            score = 0.5 * detection_rate + 0.3 * valid_rate + 0.2 * avg_validation
            
            self.logger.info(f"  Results: {len(pieces)}/{expected_pieces} pieces, " +
                           f"score: {score:.3f}")
            
            # Track best parameters
            if score > best_score:
                best_score = score
                best_params = param_set.copy()
                best_pieces = pieces
        
        # Restore original parameters
        for param_name, param_value in original_params.items():
            setattr(self.config, param_name, param_value)
        
        # Log best parameters
        self.logger.info(f"Best parameters: {best_params}")
        self.logger.info(f"Best score: {best_score:.3f}")
        
        # Return optimization results
        return {
            'best_params': best_params,
            'best_score': best_score,
            'best_piece_count': len(best_pieces),
            'expected_pieces': expected_pieces,
            'detection_rate': len(best_pieces) / expected_pieces if expected_pieces > 0 else 0
        }
    
    def analyze_image_characteristics(self, image_path: str) -> Dict[str, Any]:
        """
        Analyze image characteristics to determine optimal processing parameters
        
        Args:
            image_path: Path to the image file
        
        Returns:
            Dictionary with image analysis results
        """
        self.logger.info(f"Analyzing image characteristics: {image_path}")
        
        # Read the image
        image = read_image(image_path)
        
        # Convert to grayscale for analysis
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Calculate basic statistics
        mean = np.mean(gray)
        std = np.std(gray)
        median = np.median(gray)
        
        # Calculate histogram
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()  # Normalize
        
        # Calculate peaks (modes) in histogram
        peak_indices = []
        for i in range(1, 255):
            if hist[i] > hist[i-1] and hist[i] > hist[i+1] and hist[i] > 0.01:
                peak_indices.append(i)
        
        # Get peak values
        peaks = [(i, hist[i]) for i in peak_indices]
        peaks.sort(key=lambda x: x[1], reverse=True)  # Sort by height
        
        # Check for bimodal histogram (typical for puzzle on contrasting background)
        is_bimodal = len(peaks) >= 2
        
        # Calculate contrast
        p5 = np.percentile(gray, 5)
        p95 = np.percentile(gray, 95)
        contrast = (p95 - p5) / 255.0
        
        # Detect edges
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.count_nonzero(edges) / (gray.shape[0] * gray.shape[1])
        
        # Estimate background color
        # Assuming darker regions are background for puzzle pieces
        dark_ratio = np.sum(gray < 50) / gray.size
        is_dark_background = dark_ratio > 0.3
        
        # Determine recommended parameters based on analysis
        recommended_params = {}
        
        # Adaptive preprocessing if image has low contrast or uneven lighting
        recommended_params['USE_ADAPTIVE_PREPROCESSING'] = contrast < 0.5
        
        # Use Sobel pipeline for images with subtle edge features
        recommended_params['USE_SOBEL_PIPELINE'] = edge_density < 0.05
        
        # Adjust contour area threshold based on image size
        img_area = image.shape[0] * image.shape[1]
        estimated_piece_area = img_area / 30  # Rough estimate for 24-piece puzzle
        recommended_params['MIN_CONTOUR_AREA'] = max(500, int(estimated_piece_area * 0.1))
        
        # Adjust mean filtering threshold based on contrast
        if contrast < 0.4:
            # More permissive for low-contrast images
            recommended_params['MEAN_DEVIATION_THRESHOLD'] = 2.0
        else:
            # More strict for high-contrast images
            recommended_params['MEAN_DEVIATION_THRESHOLD'] = 1.5
        
        analysis_results = {
            'mean': float(mean),
            'std': float(std),
            'median': float(median),
            'contrast': float(contrast),
            'edge_density': float(edge_density),
            'is_bimodal': is_bimodal,
            'is_dark_background': is_dark_background,
            'peaks': peaks[:3] if peaks else [],  # Top 3 peaks
            'recommended_params': recommended_params
        }
        
        self.logger.info(f"Analysis complete: contrast={contrast:.2f}, " +
                       f"edge_density={edge_density:.3f}, " +
                       f"dark_background={is_dark_background}")
        
        return analysis_results