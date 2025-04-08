"""
Enhanced puzzle piece detection algorithms with adaptive parameter optimization
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import os
import sys
import time
import logging
from multiprocessing import Pool, cpu_count
import math
from itertools import product

# Add parent directory to path to allow imports from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.image_utils import (
    preprocess_image, adaptive_threshold, apply_morphology, detect_edges,
    multi_channel_preprocess, analyze_image, adaptive_preprocess,
    find_optimal_threshold_parameters, compare_threshold_methods,
    fuse_binary_images, multi_scale_edge_detection
)
from src.utils.contour_utils import (
    find_contours, filter_contours, calculate_contour_features, 
    enhanced_find_corners, extract_borders, classify_border,
    cluster_contours, validate_shape_as_puzzle_piece
)
from src.config.settings import Config
from src.core.piece import PuzzlePiece


class PuzzleDetector:
    """
    Enhanced detector for puzzle pieces in images with adaptive parameter optimization
    """
    
    def __init__(self, config: Config = None):
        """
        Initialize the detector
        
        Args:
            config: Configuration parameters
        """
        self.config = config or Config()
        self.logger = self._setup_logger()
        
        # Track detection performance for parameter optimization
        self.detection_stats = {
            'params': {},
            'results': {}
        }
    
    def _setup_logger(self) -> logging.Logger:
        """Set up a logger for the detector/processor"""
        logger = logging.getLogger(__name__)
        # Don't add handlers - use the root logger configuration
        return logger
    
    def save_debug_image(self, image: np.ndarray, filename: str) -> None:
        """
        Save an image for debugging purposes
        
        Args:
            image: Image to save
            filename: Filename (will be saved to debug directory)
        """
        if not self.config.DEBUG:
            return
            
        os.makedirs(self.config.DEBUG_DIR, exist_ok=True)
        path = os.path.join(self.config.DEBUG_DIR, filename)
        cv2.imwrite(path, image)
        self.logger.debug(f"Saved debug image to {path}")
    
    def preprocess_with_sobel(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Preprocess an image using the Sobel pipeline
        
        Args:
            image: Input color image
        
        Returns:
            Tuple of (preprocessed image, binary image, edge image)
        """
        self.logger.info("Preprocessing image with Sobel pipeline...")
        
        # No resizing - use original image dimensions
        h, w = image.shape[:2]
        self.logger.info(f"Processing original image dimensions: {w}x{h}")
        
        # 1. Conversion to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.save_debug_image(gray, "01_gray.jpg")
        
        # 2. Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, self.config.BLUR_KERNEL_SIZE, 0)
        self.save_debug_image(blurred, "02_blurred.jpg")
        
        # 3. Apply Sobel filter for edge detection
        sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=self.config.SOBEL_KSIZE)
        sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=self.config.SOBEL_KSIZE)
        sobel_combined = cv2.magnitude(sobelx, sobely)
        sobel_8u = cv2.normalize(sobel_combined, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        self.save_debug_image(sobel_8u, "03_sobel.jpg")
        
        # 4. Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=self.config.CLAHE_CLIP_LIMIT, 
                                tileGridSize=self.config.CLAHE_GRID_SIZE)
        contrasted = clahe.apply(sobel_8u)
        self.save_debug_image(contrasted, "04_contrasted.jpg")
        
        # 5. Apply dilation to thicken edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, 
                                        (self.config.MORPH_KERNEL_SIZE_SOBEL, 
                                        self.config.MORPH_KERNEL_SIZE_SOBEL))
        dilated = cv2.dilate(contrasted, kernel, iterations=self.config.DILATE_ITERATIONS)
        self.save_debug_image(dilated, "05_dilated.jpg")
        
        # 6. Apply erosion to refine edges
        eroded = cv2.erode(dilated, kernel, iterations=self.config.ERODE_ITERATIONS)
        self.save_debug_image(eroded, "06_eroded.jpg")
        
        # 7. Apply Otsu's thresholding as the final step
        _, binary = cv2.threshold(eroded, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        self.save_debug_image(binary, "07_threshold.jpg")
        
        # Edge detection with Canny (optional, for compatibility with existing code)
        edges = cv2.Canny(binary, self.config.CANNY_LOW_THRESHOLD, self.config.CANNY_HIGH_THRESHOLD)
        self.save_debug_image(edges, "08_edges.jpg")
        
        return gray, binary, edges
    
    def preprocess_adaptive(self, image: np.ndarray, expected_pieces: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Advanced adaptive preprocessing with image analysis and multi-channel processing
        
        Args:
            image: Input color image
            expected_pieces: Expected number of pieces (for optimization)
        
        Returns:
            Tuple of (preprocessed image, binary image, edge image)
        """
        self.logger.info("Using advanced adaptive preprocessing...")
        
        # Use original image dimensions
        h, w = image.shape[:2]
        self.logger.info(f"Processing original image dimensions: {w}x{h}")
        
        # Analyze image properties
        analysis = analyze_image(image)
        self.logger.info(f"Image analysis: contrast={analysis['contrast']:.2f}, background={analysis['background_value']:.2f}")
        
        # Use multi-channel preprocessing to create the best grayscale representation
        best_preprocessed, all_channels = multi_channel_preprocess(image)
        self.save_debug_image(best_preprocessed, "01_best_preprocessed.jpg")
        
        # Save diagnostic images of different channels
        for name, channel in all_channels.items():
            if len(channel.shape) == 2:  # Only grayscale channels
                self.save_debug_image(channel, f"01_channel_{name}.jpg")
        
        # Find optimal threshold parameters
        threshold_params = find_optimal_threshold_parameters(best_preprocessed)
        self.logger.info(f"Selected threshold method: {threshold_params['method']}")
        
        # Apply the optimal thresholding
        if threshold_params['method'] == 'otsu':
            _, binary = cv2.threshold(best_preprocessed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif threshold_params['method'] == 'adaptive':
            binary = cv2.adaptiveThreshold(
                best_preprocessed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, threshold_params['block_size'], threshold_params['c']
            )
        elif threshold_params['method'] == 'hybrid':
            # Hybrid approach
            _, otsu_binary = cv2.threshold(best_preprocessed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            adaptive_binary = cv2.adaptiveThreshold(
                best_preprocessed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 35, 10
            )
            binary = cv2.bitwise_or(otsu_binary, adaptive_binary)
        
        self.save_debug_image(binary, "02_binary.jpg")
        
        # Apply morphological operations to clean up binary image
        # Use adaptive kernel size based on image size
        kernel_size = max(3, min(7, int(min(h, w) / 500)))
        morph = apply_morphology(
            binary, 
            operation=cv2.MORPH_CLOSE, 
            kernel_size=kernel_size,
            iterations=2
        )
        self.save_debug_image(morph, "03_morphology.jpg")
        
        # Enhanced edge detection
        edges = multi_scale_edge_detection(best_preprocessed)
        self.save_debug_image(edges, "04_edges.jpg")
        
        return best_preprocessed, morph, edges
    
    def preprocess(self, image: np.ndarray, expected_pieces: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Preprocess an image for puzzle piece detection
        Uses either original pipeline, Sobel pipeline, or new adaptive preprocessing
        
        Args:
            image: Input color image
            expected_pieces: Optional expected number of pieces for optimization
        
        Returns:
            Tuple of (preprocessed image, binary image, edge image)
        """
        # Choose which pipeline to use based on configuration or auto-detection
        if hasattr(self.config, 'USE_ADAPTIVE_PREPROCESSING') and self.config.USE_ADAPTIVE_PREPROCESSING:
            return self.preprocess_adaptive(image, expected_pieces)
        elif self.config.USE_SOBEL_PIPELINE:
            return self.preprocess_with_sobel(image)
            
        # Original preprocessing pipeline follows
        self.logger.info("Preprocessing image with standard pipeline...")
        
        # No resizing - use original image dimensions
        h, w = image.shape[:2]
        self.logger.info(f"Processing original image dimensions: {w}x{h}")
        
        # Use simplified preprocessing approach
        preprocessed = preprocess_image(image)
        self.save_debug_image(preprocessed, "01_preprocessed.jpg")
        
        # Determine the best thresholding method if auto-threshold is enabled
        if self.config.USE_AUTO_THRESHOLD:
            self.logger.info("Using auto threshold selection...")
            
            # Use the enhanced comparison method
            binary, method, metrics = compare_threshold_methods(
                preprocessed,
                self.config.ADAPTIVE_BLOCK_SIZE,
                self.config.ADAPTIVE_C
            )
            
            self.logger.info(f"Selected {method} thresholding method")
            self.save_debug_image(binary, f"02_{method}_binary.jpg")
        else:
            # Default to Otsu if auto-threshold is not enabled
            _, binary = cv2.threshold(preprocessed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            self.save_debug_image(binary, "02_binary.jpg")
        
        # Morphological operations to clean up the binary image
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        self.save_debug_image(morph, "03_morphology.jpg")
        
        # Edge detection with Canny
        edges = cv2.Canny(morph, 30, 200)
        self.save_debug_image(edges, "04_edges.jpg")
        
        return preprocessed, morph, edges
    
    def try_parameter_combinations(self, binary_image: np.ndarray, original_image: np.ndarray, 
                                 expected_pieces: Optional[int] = None) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """
        Try multiple parameter combinations for contour detection and select the best
        
        Args:
            binary_image: Binary input image
            original_image: Original image
            expected_pieces: Expected number of pieces
        
        Returns:
            Tuple of (list of best contours, parameter statistics)
        """
        self.logger.info("Optimizing contour detection parameters...")
        
        # Define parameter grid
        param_grid = {
            'min_area': [500, 1000, 2000, 3000],
            'solidity_min': [0.5, 0.6, 0.7],
            'aspect_ratio_range': [(0.2, 5.0), (0.25, 4.0), (0.3, 3.0)]
        }
        
        # Generate all parameter combinations
        param_combinations = []
        for min_area in param_grid['min_area']:
            for solidity_min in param_grid['solidity_min']:
                for aspect_ratio_range in param_grid['aspect_ratio_range']:
                    params = {
                        'min_area': min_area,
                        'solidity_range': (solidity_min, 0.99),
                        'aspect_ratio_range': aspect_ratio_range
                    }
                    param_combinations.append(params)
        
        # Calculate max contour area based on image size
        img_area = original_image.shape[0] * original_image.shape[1]
        max_area = self.config.MAX_CONTOUR_AREA_RATIO * img_area
        
        # Find contours once
        initial_contours, _ = cv2.findContours(binary_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.logger.info(f"Found {len(initial_contours)} initial contours")
        
        # If we have too few initial contours, try alternative methods
        if len(initial_contours) < (expected_pieces or 10):
            # Try different contour finding modes
            alt_contours, _ = cv2.findContours(binary_image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            initial_contours.extend([c for c in alt_contours if cv2.contourArea(c) > 100])
            self.logger.info(f"Added alternative contours, now have {len(initial_contours)}")
        
        results = []
        
        # Try each parameter combination
        for params in param_combinations:
            # Add max_area to params
            params['max_area'] = max_area
            params['min_perimeter'] = self.config.MIN_CONTOUR_PERIMETER
            
            # Filter contours with current parameters
            filtered = filter_contours(
                initial_contours,
                **params
            )
            
            # Apply statistical filtering if enabled
            if self.config.USE_MEAN_FILTERING and len(filtered) > 1:
                areas = np.array([cv2.contourArea(cnt) for cnt in filtered])
                mean_area = np.mean(areas)
                std_area = np.std(areas)
                
                min_acceptable = mean_area - self.config.MEAN_DEVIATION_THRESHOLD * std_area
                max_acceptable = mean_area + self.config.MEAN_DEVIATION_THRESHOLD * std_area
                
                mean_filtered = [cnt for cnt in filtered if min_acceptable <= cv2.contourArea(cnt) <= max_acceptable]
                final_contours = mean_filtered
            else:
                final_contours = filtered
            
            # Calculate score based on contour quality and expected count
            score = self._evaluate_contour_set(final_contours, expected_pieces)
            
            results.append({
                'params': params,
                'contours': final_contours,
                'count': len(final_contours),
                'mean_area': np.mean([cv2.contourArea(c) for c in final_contours]) if final_contours else 0,
                'score': score
            })
        
        # Sort by score (descending)
        results.sort(key=lambda x: x['score'], reverse=True)
        
        # Log results
        self.logger.info(f"Parameter optimization results:")
        for i, result in enumerate(results[:3]):  # Log top 3
            self.logger.info(f"  #{i+1}: score={result['score']:.2f}, count={result['count']}, " +
                           f"min_area={result['params']['min_area']}, " +
                           f"solidity={result['params']['solidity_range'][0]}")
        
        # Record detection statistics
        detection_stats = {
            'total_combinations': len(param_combinations),
            'best_params': results[0]['params'] if results else None,
            'best_score': results[0]['score'] if results else 0,
            'best_count': results[0]['count'] if results else 0,
        }
        
        # Return the best contours and stats
        return results[0]['contours'] if results else [], detection_stats
    
    def _evaluate_contour_set(self, contours: List[np.ndarray], expected_pieces: Optional[int] = None) -> float:
        """
        Evaluate the quality of a set of contours
        
        Args:
            contours: List of contours
            expected_pieces: Expected number of pieces
        
        Returns:
            Quality score (higher is better)
        """
        if not contours:
            return 0.0
        
        # Calculate metrics
        areas = [cv2.contourArea(c) for c in contours]
        perimeters = [cv2.arcLength(c, True) for c in contours]
        
        mean_area = np.mean(areas)
        std_area = np.std(areas)
        cv_area = std_area / mean_area if mean_area > 0 else float('inf')
        
        # Calculate shape complexity (higher for puzzle pieces)
        complexity = [p**2 / (4 * np.pi * a) if a > 0 else 0 for p, a in zip(perimeters, areas)]
        mean_complexity = np.mean(complexity)
        
        # Calculate score components
        
        # 1. Count score - how close we are to expected count
        if expected_pieces:
            count_ratio = len(contours) / expected_pieces
            # Penalize both too few and too many contours
            count_score = 1.0 - min(abs(1.0 - count_ratio), 1.0)
        else:
            # Without expected count, moderate numbers are better
            # (too few probably misses pieces, too many probably has noise)
            count_score = min(len(contours) / 30.0, 1.0)
        
        # 2. Area consistency - puzzle pieces should be similar in size
        consistency_score = 1.0 - min(cv_area, 1.0)
        
        # 3. Shape complexity - puzzle pieces should have tabs and indentations
        # Typical range for puzzle pieces is 3-7
        complexity_score = 0.0
        if 2.5 <= mean_complexity <= 8.0:
            # Peak score at around 5.0 complexity
            complexity_score = 1.0 - abs(mean_complexity - 5.0) / 5.0
        
        # 4. Validation score - check what percentage look like puzzle pieces
        valid_pieces = sum(1 for c in contours if validate_shape_as_puzzle_piece(c))
        validation_score = valid_pieces / len(contours) if contours else 0.0
        
        # Weight the components
        if expected_pieces:
            # When we have expected count, prioritize getting close to that number
            final_score = (
                0.4 * count_score +
                0.3 * consistency_score +
                0.2 * complexity_score +
                0.1 * validation_score
            )
        else:
            # Without expected count, prioritize consistency and shape
            final_score = (
                0.2 * count_score +
                0.4 * consistency_score +
                0.3 * complexity_score +
                0.1 * validation_score
            )
        
        return final_score
    
    def detect_contours(self, binary_image: np.ndarray, original_image: np.ndarray,
                        expected_pieces: Optional[int] = None) -> List[np.ndarray]:
        """
        Enhanced contour detection with parameter optimization
        
        Args:
            binary_image: Binary input image
            original_image: Original image (for size-based filtering)
            expected_pieces: Expected number of pieces
        
        Returns:
            List of detected contours
        """
        self.logger.info("Detecting contours with enhanced approach...")
        
        # Try multiple parameter combinations if time permits
        if hasattr(self.config, 'USE_PARAMETER_OPTIMIZATION') and self.config.USE_PARAMETER_OPTIMIZATION:
            contours, stats = self.try_parameter_combinations(binary_image, original_image, expected_pieces)
            self.detection_stats['params'] = stats
            
            if len(contours) > 0:
                self.logger.info(f"Parameter optimization found {len(contours)} contours")
                # Create contour visualization
                contour_vis = original_image.copy()
                for i, contour in enumerate(contours):
                    # Generate random color for each contour
                    color = (
                        np.random.randint(0, 255),
                        np.random.randint(0, 255),
                        np.random.randint(0, 255)
                    )
                    cv2.drawContours(contour_vis, [contour], -1, color, 2)
                    
                    # Add contour index
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        cv2.putText(contour_vis, str(i), (cx, cy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                self.save_debug_image(contour_vis, "05_contours_optimized.jpg")
                return contours
        
        # If optimization is disabled or failed, use standard approach
        # Calculate max contour area based on image size
        img_area = original_image.shape[0] * original_image.shape[1]
        max_area = self.config.MAX_CONTOUR_AREA_RATIO * img_area
        
        # Find contours using comprehensive methods
        contours = find_contours(binary_image)
        self.logger.info(f"Found {len(contours)} initial contours")
        
        # Filter contours with more lenient parameters
        filtered_contours = filter_contours(
            contours,
            min_area=self.config.MIN_CONTOUR_AREA,
            max_area=max_area,
            min_perimeter=self.config.MIN_CONTOUR_PERIMETER,
            solidity_range=self.config.SOLIDITY_RANGE,
            aspect_ratio_range=self.config.ASPECT_RATIO_RANGE,
            use_statistical_filtering=True,
            expected_piece_count=expected_pieces
        )
        self.logger.info(f"After filtering: {len(filtered_contours)} contours")
        
        # Apply mean-based filtering if enabled and we have enough contours
        if self.config.USE_MEAN_FILTERING and len(filtered_contours) > 1:
            # Calculate areas
            areas = [cv2.contourArea(cnt) for cnt in filtered_contours]
            
            # Use more robust statistics: median and median absolute deviation
            from scipy import stats
            median_area = np.median(areas)
            mad = stats.median_abs_deviation(areas)
            
            # Define acceptable range using median absolute deviation
            # (more robust to outliers than mean/stddev)
            deviation_threshold = self.config.MEAN_DEVIATION_THRESHOLD
            min_acceptable = median_area - deviation_threshold * mad
            max_acceptable = median_area + deviation_threshold * mad
            
            # Filter contours based on area deviation from median
            mean_filtered_contours = []
            rejected_contours = []
            
            for i, cnt in enumerate(filtered_contours):
                area = cv2.contourArea(cnt)
                if min_acceptable <= area <= max_acceptable:
                    mean_filtered_contours.append(cnt)
                else:
                    rejected_contours.append(cnt)
            
            # Log statistics
            self.logger.info(f"Median contour area: {median_area:.2f}, MAD: {mad:.2f}")
            self.logger.info(f"Acceptable area range: {min_acceptable:.2f} to {max_acceptable:.2f}")
            self.logger.info(f"Rejected {len(rejected_contours)} contours with outlier areas")
            self.logger.info(f"After median-based filtering: {len(mean_filtered_contours)} contours")
            
            filtered_contours = mean_filtered_contours
            
            # Recovery step: check if any rejected contours could actually be valid pieces
            if expected_pieces and len(filtered_contours) < expected_pieces * 0.9:
                # Check if any rejected contours are actually valid puzzle pieces
                valid_rejects = []
                for cnt in rejected_contours:
                    # Apply more sophisticated validation
                    if validate_shape_as_puzzle_piece(cnt):
                        valid_rejects.append(cnt)
                
                if valid_rejects:
                    self.logger.info(f"Recovered {len(valid_rejects)} valid pieces from rejected contours")
                    filtered_contours.extend(valid_rejects)
        
        # Create contour visualization
        contour_vis = original_image.copy()
        for i, contour in enumerate(filtered_contours):
            # Generate random color for each contour
            color = (
                np.random.randint(0, 255),
                np.random.randint(0, 255),
                np.random.randint(0, 255)
            )
            cv2.drawContours(contour_vis, [contour], -1, color, 2)
            
            # Add contour index
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.putText(contour_vis, str(i), (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        self.save_debug_image(contour_vis, "05_contours.jpg")
        
        return filtered_contours

    # Create a modified version of recover_missed_pieces function in detector.py
    def recover_missed_pieces(self, binary_image: np.ndarray, 
                            detected_contours: List[np.ndarray],
                            original_image: np.ndarray,
                            expected_pieces: Optional[int] = None) -> List[np.ndarray]:
        """
        Attempt to recover pieces that were missed in initial detection
        with improved deduplication
        
        Args:
            binary_image: Binary input image
            detected_contours: Already detected contours
            original_image: Original image
            expected_pieces: Expected number of pieces
        
        Returns:
            List of additional detected contours
        """
        # Only attempt recovery if we have expected_pieces and are missing some
        if not expected_pieces or len(detected_contours) >= expected_pieces:
            return []
        
        self.logger.info(f"Attempting to recover missed pieces. " +
                    f"Found {len(detected_contours)}/{expected_pieces} expected pieces.")
        
        # Create a mask to exclude already detected pieces with padding
        mask = np.ones_like(binary_image)
        for contour in detected_contours:
            # Create a slightly expanded contour to avoid detecting the same piece
            x, y, w, h = cv2.boundingRect(contour)
            # Add padding around the bounding box
            padding = 15  # Increased padding
            x_min = max(0, x - padding)
            y_min = max(0, y - padding)
            x_max = min(binary_image.shape[1], x + w + padding)
            y_max = min(binary_image.shape[0], y + h + padding)
            
            # Block out this region in the mask
            mask[y_min:y_max, x_min:x_max] = 0
        
        # Apply mask to the binary image
        masked_binary = cv2.bitwise_and(binary_image, binary_image, mask=mask)
        self.save_debug_image(masked_binary, "06_masked_binary.jpg")
        
        # Initialize the recovered_contours list
        recovered_contours = []
        
        # Try recovery with more lenient parameters
        recovery_params = [
            # More lenient parameters for small or irregular pieces
            {'min_area': self.config.MIN_CONTOUR_AREA * 0.7, 
            'solidity_range': (0.5, 0.99),
            'aspect_ratio_range': (0.2, 5.0)},
            
            # Try with morphological opening to separate touching pieces
            {'min_area': self.config.MIN_CONTOUR_AREA * 0.8,
            'solidity_range': (0.6, 0.99),
            'aspect_ratio_range': (0.25, 4.0)},
        ]
        
        for params in recovery_params:
            # Find contours in the masked binary image
            mask_contours, _ = cv2.findContours(masked_binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Calculate max area
            img_area = original_image.shape[0] * original_image.shape[1]
            max_area = self.config.MAX_CONTOUR_AREA_RATIO * img_area
            
            # Filter with current parameters
            filtered = filter_contours(
                mask_contours,
                min_area=params['min_area'],
                max_area=max_area,
                min_perimeter=self.config.MIN_CONTOUR_PERIMETER,
                solidity_range=params['solidity_range'],
                aspect_ratio_range=params['aspect_ratio_range']
            )
            
            # Validate each candidate recovery
            valid_recoveries = []
            for contour in filtered:
                # Check if it's likely a valid puzzle piece
                if validate_shape_as_puzzle_piece(contour):
                    valid_recoveries.append(contour)
            
            self.logger.info(f"Recovery attempt: found {len(valid_recoveries)} potential pieces")
            recovered_contours.extend(valid_recoveries)
            
            # If we've recovered enough pieces, stop
            if len(detected_contours) + len(recovered_contours) >= expected_pieces:
                break
        
        # Add stricter deduplication check before returning
        final_recovered = []
        for new_contour in recovered_contours:
            is_duplicate = False
            for existing_contour in detected_contours:
                # Check for substantial overlap using IoU
                if self._contours_match(new_contour, existing_contour, threshold=0.3):
                    is_duplicate = True
                    break
            
            # Also check against already recovered contours
            for existing_recovered in final_recovered:
                if self._contours_match(new_contour, existing_recovered, threshold=0.3):
                    is_duplicate = True
                    break
                    
            if not is_duplicate:
                final_recovered.append(new_contour)
        
        # Create visualization of recovered contours
        if final_recovered:
            recovery_vis = original_image.copy()
            
            # Draw original contours in green
            cv2.drawContours(recovery_vis, detected_contours, -1, (0, 255, 0), 2)
            
            # Draw recovered contours in red
            cv2.drawContours(recovery_vis, final_recovered, -1, (0, 0, 255), 2)
            
            self.save_debug_image(recovery_vis, "07_recovered_contours.jpg")
        
        self.logger.info(f"Recovered {len(final_recovered)} additional pieces after deduplication")
        return final_recovered
    
    def _process_contour(self, args: Tuple[np.ndarray, np.ndarray, int]) -> Optional[PuzzlePiece]:
        """
        Process a single contour to create a puzzle piece
        
        Args:
            args: Tuple of (contour, image, index)
        
        Returns:
            PuzzlePiece object or None if invalid
        """
        contour, image, idx = args
        try:
            piece = PuzzlePiece(image, contour, self.config)
            piece.id = idx
            return piece
        except Exception as e:
            self.logger.error(f"Error processing contour {idx}: {str(e)}")
            return None
    
    def process_contours(self, contours: List[np.ndarray], image: np.ndarray) -> List[PuzzlePiece]:
        """
        Process contours to create puzzle piece objects
        
        Args:
            contours: List of contours
            image: Original image
        
        Returns:
            List of valid puzzle pieces
        """
        self.logger.info("Processing contours to create pieces...")
        
        pieces = []
        
        if self.config.USE_MULTIPROCESSING and len(contours) > 1:
            # Prepare arguments for multiprocessing
            args = [(contour, image, i) for i, contour in enumerate(contours)]
            
            # Use process pool for parallel processing
            with Pool(processes=min(self.config.NUM_PROCESSES, cpu_count())) as pool:
                results = pool.map(self._process_contour, args)
                pieces = [p for p in results if p is not None and p.is_valid]
        else:
            # Sequential processing
            for i, contour in enumerate(contours):
                piece = self._process_contour((contour, image, i))
                if piece is not None and piece.is_valid:
                    pieces.append(piece)
        
        self.logger.info(f"Found {len(pieces)} valid pieces")
        return pieces
    
    def detect(self, image: np.ndarray, expected_pieces: Optional[int] = None) -> Tuple[List[PuzzlePiece], Dict[str, np.ndarray]]:
        """
        Enhanced puzzle piece detection with parameter optimization and recovery
        
        Args:
            image: Input color image
            expected_pieces: Expected number of pieces
        
        Returns:
            Tuple of (list of puzzle pieces, dict of debug images)
        """
        start_time = time.time()
        self.logger.info("Starting enhanced puzzle piece detection")
        
        # Preprocess the image
        preprocessed, binary, edges = self.preprocess(image, expected_pieces)
        
        # Detect contours with parameter optimization
        contours = self.detect_contours(binary, image, expected_pieces)
        
        # Attempt to recover missed pieces if appropriate
        if expected_pieces and len(contours) < expected_pieces:
            recovered_contours = self.recover_missed_pieces(binary, contours, image, expected_pieces)
            if recovered_contours:
                self.logger.info(f"Recovered {len(recovered_contours)} additional pieces")
                contours.extend(recovered_contours)
        
        # Process contours to create puzzle pieces
        pieces = self.process_contours(contours, image)
        
        # Create a visualization of all valid pieces
        piece_vis = image.copy()
        for piece in pieces:
            piece_vis = piece.draw(piece_vis)
        
        self.save_debug_image(piece_vis, "08_detected_pieces.jpg")
        
        # If we still have fewer pieces than expected, consider adaptive thresholding
        if expected_pieces and len(pieces) < expected_pieces * 0.8 and not hasattr(self.config, 'USE_ADAPTIVE_PREPROCESSING'):
            self.logger.info(f"Low detection rate ({len(pieces)}/{expected_pieces}). Trying adaptive preprocessing.")
            
            # Try with adaptive preprocessing
            adaptive_preprocessed, adaptive_binary, adaptive_edges = self.preprocess_adaptive(image, expected_pieces)
            
            # Detect contours with the new binary image
            adaptive_contours = self.detect_contours(adaptive_binary, image, expected_pieces)
            
            # Process these contours
            adaptive_pieces = self.process_contours(adaptive_contours, image)
            
            # If adaptive approach found more pieces, use those results
            if len(adaptive_pieces) > len(pieces):
                self.logger.info(f"Adaptive preprocessing found {len(adaptive_pieces)} pieces (vs. {len(pieces)} with standard approach)")
                
                # Update visualizations
                adaptive_piece_vis = image.copy()
                for piece in adaptive_pieces:
                    adaptive_piece_vis = piece.draw(adaptive_piece_vis)
                
                self.save_debug_image(adaptive_piece_vis, "09_adaptive_detected_pieces.jpg")
                
                # Update our results
                pieces = adaptive_pieces
                preprocessed = adaptive_preprocessed
                binary = adaptive_binary
                edges = adaptive_edges
                piece_vis = adaptive_piece_vis
        
        elapsed_time = time.time() - start_time
        self.logger.info(f"Detection completed in {elapsed_time:.2f} seconds")
        
        # Record detection results
        self.detection_stats['results'] = {
            'found_pieces': len(pieces),
            'expected_pieces': expected_pieces,
            'detection_rate': len(pieces) / expected_pieces if expected_pieces else None,
            'elapsed_time': elapsed_time
        }
        
        # Return the pieces and debug images
        debug_images = {
            'preprocessed': preprocessed,
            'binary': binary,
            'edges': edges,
            'piece_visualization': piece_vis
        }
        
        return pieces, debug_images

    def multi_pass_detection(self, image: np.ndarray, expected_pieces: Optional[int] = None) -> Tuple[List[PuzzlePiece], Dict[str, np.ndarray]]:
        """
        Multi-pass detection with different parameters to maximize piece detection
        
        Args:
            image: Input color image
            expected_pieces: Expected number of pieces
        
        Returns:
            Tuple of (list of puzzle pieces, dict of debug images)
        """
        # Initialize storage for results from different passes
        all_pieces = []
        all_contours = []
        best_debug_images = {}
        
        # Pass 1: Standard detection
        pieces1, debug_images1 = self.detect(image, expected_pieces)
        all_pieces.extend(pieces1)
        
        # Extract contours from pieces for uniqueness checking
        contours1 = [piece.contour for piece in pieces1]
        all_contours.extend(contours1)
        
        self.logger.info(f"Pass 1: Detected {len(pieces1)} pieces")
        
        # If we already found all expected pieces, skip additional passes
        if expected_pieces and len(all_pieces) >= expected_pieces:
            return all_pieces, debug_images1
        
        # Pass 2: Try with different preprocessing parameters
        # Temporarily modify config for second pass
        original_settings = {}
        
        # If we used standard pipeline first, try adaptive now
        if not hasattr(self.config, 'USE_ADAPTIVE_PREPROCESSING'):
            original_settings['USE_ADAPTIVE_PREPROCESSING'] = getattr(self.config, 'USE_ADAPTIVE_PREPROCESSING', False)
            self.config.USE_ADAPTIVE_PREPROCESSING = True
        # If we used adaptive first, try standard with Sobel
        else:
            original_settings['USE_SOBEL_PIPELINE'] = self.config.USE_SOBEL_PIPELINE
            self.config.USE_SOBEL_PIPELINE = not self.config.USE_SOBEL_PIPELINE
        
        # Modify mean threshold for second pass
        original_settings['MEAN_DEVIATION_THRESHOLD'] = self.config.MEAN_DEVIATION_THRESHOLD
        self.config.MEAN_DEVIATION_THRESHOLD = self.config.MEAN_DEVIATION_THRESHOLD * 1.5  # More permissive
        
        # Run second pass detection
        pieces2, debug_images2 = self.detect(image, expected_pieces)
        
        # Restore original settings
        for setting, value in original_settings.items():
            setattr(self.config, setting, value)
        
        # Filter out duplicate pieces
        unique_pieces2 = []
        for piece in pieces2:
            is_duplicate = False
            for contour in all_contours:
                if self._contours_match(piece.contour, contour):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_pieces2.append(piece)
                all_contours.append(piece.contour)
        
        self.logger.info(f"Pass 2: Detected {len(pieces2)} pieces, {len(unique_pieces2)} unique")
        all_pieces.extend(unique_pieces2)
        
        # Combine debug images (use first pass as base, add second pass if it found unique pieces)
        best_debug_images = debug_images1
        if unique_pieces2:
            # Create a combined visualization
            combined_vis = image.copy()
            
            # Draw first pass pieces in green
            for piece in pieces1:
                cv2.drawContours(combined_vis, [piece.contour], -1, (0, 255, 0), 2)
            
            # Draw unique second pass pieces in blue
            for piece in unique_pieces2:
                cv2.drawContours(combined_vis, [piece.contour], -1, (255, 0, 0), 2)
            
            best_debug_images['combined_visualization'] = combined_vis
            self.save_debug_image(combined_vis, "10_combined_detection.jpg")
        
        return all_pieces, best_debug_images
    
    def _contours_match(self, contour1: np.ndarray, contour2: np.ndarray, 
                    threshold: float = 0.3) -> bool:
        """
        Enhanced check if two contours match (represent same object)
        with more strict criteria to prevent duplicates
        
        Args:
            contour1: First contour
            contour2: Second contour
            threshold: Matching threshold (0.0 to 1.0)
        
        Returns:
            True if contours likely represent the same piece
        """
        # Get bounding boxes
        x1, y1, w1, h1 = cv2.boundingRect(contour1)
        x2, y2, w2, h2 = cv2.boundingRect(contour2)
        
        # Calculate IoU of bounding boxes
        # Intersection rectangle
        x_inter = max(x1, x2)
        y_inter = max(y1, y2)
        w_inter = min(x1 + w1, x2 + w2) - x_inter
        h_inter = min(y1 + h1, y2 + h2) - y_inter
        
        # If there's no overlap, they don't match
        if w_inter <= 0 or h_inter <= 0:
            return False
        
        # Calculate areas
        area_inter = w_inter * h_inter
        area1 = w1 * h1
        area2 = w2 * h2
        area_union = area1 + area2 - area_inter
        
        # Calculate IoU
        iou = area_inter / area_union
        
        # Check centroid distance
        m1 = cv2.moments(contour1)
        m2 = cv2.moments(contour2)
        
        if m1["m00"] > 0 and m2["m00"] > 0:
            cx1 = m1["m10"] / m1["m00"]
            cy1 = m1["m01"] / m1["m00"]
            cx2 = m2["m10"] / m2["m00"]
            cy2 = m2["m01"] / m2["m00"]
            
            # Calculate distance between centroids
            distance = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
            
            # If centroids are very close, likely same piece
            if distance < 50:  # Adjust this threshold as needed
                return True
        
        # Return true if IoU exceeds threshold
        return iou > threshold
    def validate_puzzle_piece(contour, image):
        """
        Comprehensive validation of puzzle piece characteristics
        
        Args:
            contour: Contour to validate
            image: Original image
        
        Returns:
            True if contour represents a valid puzzle piece
        """
        # Basic shape validation
        if not validate_shape_as_puzzle_piece(contour):
            return False
            
        # Extract the piece image
        x, y, w, h = cv2.boundingRect(contour)
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], 0, 255, -1)
        
        # Create ROI
        roi = image[y:y+h, x:x+w]
        mask_roi = mask[y:y+h, x:x+w]
        
        # Check color variation within the piece (puzzle pieces should have meaningful content)
        if len(image.shape) == 3:  # Color image
            # Convert to grayscale
            if len(roi.shape) == 3:
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            else:
                gray_roi = roi
                
            # Calculate standard deviation of pixel values within the piece
            piece_pixels = gray_roi[mask_roi > 0]
            if len(piece_pixels) > 0:
                std_dev = np.std(piece_pixels)
                # Shadow artifacts typically have very low color variation
                if std_dev < 15.0:  # Adjust this threshold as needed
                    return False
        
        # Check for abnormal darkness (shadows are typically very dark)
        if len(piece_pixels) > 0:
            mean_value = np.mean(piece_pixels)
            if mean_value < 30:  # Very dark pieces are likely shadows
                return False
        
        return True