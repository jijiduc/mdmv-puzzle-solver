"""
Puzzle piece detection algorithms and utilities
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import os
import sys
import time
import logging
from multiprocessing import Pool, cpu_count

# Add parent directory to path to allow imports from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.image_utils import (
    preprocess_image, adaptive_threshold, apply_morphology, detect_edges
)
from src.utils.contour_utils import find_contours, filter_contours
from src.config.settings import Config
from src.core.piece import PuzzlePiece


class PuzzleDetector:
    """
    Detector for puzzle pieces in images
    """
    
    def __init__(self, config: Config = None):
        """
        Initialize the detector
        
        Args:
            config: Configuration parameters
        """
        self.config = config or Config()
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Set up a logger for the detector"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG if self.config.DEBUG else logging.INFO)
        
        # Add console handler if no handlers exist
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
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
        
    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Preprocess an image for puzzle piece detection
        Uses either original pipeline or Sobel pipeline based on config
        
        Args:
            image: Input color image
        
        Returns:
            Tuple of (preprocessed image, binary image, edge image)
        """
        # Choose which pipeline to use based on configuration
        if self.config.USE_SOBEL_PIPELINE:
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
            
            # Apply Otsu thresholding
            _, otsu_binary = cv2.threshold(preprocessed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            self.save_debug_image(otsu_binary, "02a_otsu_binary.jpg")
            
            # Apply adaptive thresholding
            adaptive_binary = cv2.adaptiveThreshold(
                preprocessed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, self.config.ADAPTIVE_BLOCK_SIZE, self.config.ADAPTIVE_C
            )
            self.save_debug_image(adaptive_binary, "02b_adaptive_binary.jpg")
            
            # Evaluate which method is better by counting significant contours
            otsu_contours = cv2.findContours(otsu_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            adaptive_contours = cv2.findContours(adaptive_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            
            # Filter to significant contours
            min_area = self.config.MIN_CONTOUR_AREA / 2  # Lower threshold for comparison
            otsu_filtered = [c for c in otsu_contours if cv2.contourArea(c) > min_area]
            adaptive_filtered = [c for c in adaptive_contours if cv2.contourArea(c) > min_area]
            
            self.logger.info(f"Otsu thresholding found {len(otsu_filtered)} significant contours")
            self.logger.info(f"Adaptive thresholding found {len(adaptive_filtered)} significant contours")
            
            # Use method that found more contours or has better contour quality
            if len(otsu_filtered) >= len(adaptive_filtered):
                self.logger.info("Selected Otsu thresholding (more contours)")
                binary = otsu_binary
            else:
                self.logger.info("Selected adaptive thresholding (more contours)")
                binary = adaptive_binary
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
    
    def detect_contours(self, binary_image: np.ndarray, original_image: np.ndarray) -> List[np.ndarray]:
        """
        Detect contours in a binary image using improved approach
        
        Args:
            binary_image: Binary input image
            original_image: Original image (for size-based filtering)
        
        Returns:
            List of detected contours
        """
        self.logger.info("Detecting contours...")
        
        # Find contours using the improved method
        # Using the morphed image from preprocessing
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.logger.info(f"Found {len(contours)} initial contours")
        
        # Calculate max contour area based on image size
        img_area = original_image.shape[0] * original_image.shape[1]
        max_area = self.config.MAX_CONTOUR_AREA_RATIO * img_area
        
        # Filter contours with more lenient parameters
        filtered_contours = filter_contours(
            contours,
            min_area=self.config.MIN_CONTOUR_AREA,
            max_area=max_area,
            min_perimeter=self.config.MIN_CONTOUR_PERIMETER,
            solidity_range=self.config.SOLIDITY_RANGE,
            aspect_ratio_range=self.config.ASPECT_RATIO_RANGE
        )
        self.logger.info(f"After filtering: {len(filtered_contours)} contours")
        
        # Additional size-based filtering for when we have too many contours
        if len(filtered_contours) > 10:  # If we have too many potential pieces
            # Calculate areas
            areas = np.array([cv2.contourArea(cnt) for cnt in filtered_contours])
            # Use median instead of mean (more robust to outliers)
            median_area = np.median(areas)
            
            # Keep only contours that are at least 30% of the median area
            size_filtered = []
            for cnt in filtered_contours:
                if cv2.contourArea(cnt) >= 0.3 * median_area:
                    size_filtered.append(cnt)
            
            self.logger.info(f"After size-based filtering: {len(size_filtered)} contours")
            filtered_contours = size_filtered
        
        # Calculate mean area and standard deviation if mean filtering is enabled
        if self.config.USE_MEAN_FILTERING and len(filtered_contours) > 1:
            areas = np.array([cv2.contourArea(cnt) for cnt in filtered_contours])
            mean_area = np.mean(areas)
            std_area = np.std(areas)
            
            # Define acceptable range (default: within 2 standard deviations)
            deviation_threshold = self.config.MEAN_DEVIATION_THRESHOLD
            min_acceptable = mean_area - deviation_threshold * std_area
            max_acceptable = mean_area + deviation_threshold * std_area
            
            # Filter contours based on area deviation from mean
            mean_filtered_contours = []
            rejected_contours = []
            
            for cnt in filtered_contours:
                area = cv2.contourArea(cnt)
                if min_acceptable <= area <= max_acceptable:
                    mean_filtered_contours.append(cnt)
                else:
                    rejected_contours.append(cnt)
            
            # Log statistics
            self.logger.info(f"Mean contour area: {mean_area:.2f}, Std dev: {std_area:.2f}")
            self.logger.info(f"Acceptable area range: {min_acceptable:.2f} to {max_acceptable:.2f}")
            self.logger.info(f"Rejected {len(rejected_contours)} contours with outlier areas")
            self.logger.info(f"After mean-based filtering: {len(mean_filtered_contours)} contours")
            
            # Use the mean-filtered contours
            filtered_contours = mean_filtered_contours
        
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
    
    def detect(self, image: np.ndarray) -> Tuple[List[PuzzlePiece], Dict[str, np.ndarray]]:
        """
        Detect puzzle pieces in an image
        
        Args:
            image: Input color image
        
        Returns:
            Tuple of (list of puzzle pieces, dict of debug images)
        """
        start_time = time.time()
        self.logger.info("Starting puzzle piece detection")
        
        # Preprocess the image
        preprocessed, binary, edges = self.preprocess(image)
        
        # Detect contours
        contours = self.detect_contours(binary, image)
        
        # Process contours to create puzzle pieces
        pieces = self.process_contours(contours, image)
        
        # Create a visualization of all valid pieces
        piece_vis = image.copy()
        for piece in pieces:
            piece_vis = piece.draw(piece_vis)
        
        self.save_debug_image(piece_vis, "06_detected_pieces.jpg")
        
        elapsed_time = time.time() - start_time
        self.logger.info(f"Detection completed in {elapsed_time:.2f} seconds")
        
        # Return the pieces and debug images
        debug_images = {
            'preprocessed': preprocessed,
            'binary': binary,
            'edges': edges,
            'piece_visualization': piece_vis
        }
        
        return pieces, debug_images