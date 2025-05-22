"""Core image processing functions for puzzle detection."""

import cv2
import numpy as np
import os
from typing import Dict, List, Tuple, Any

from ..utils.caching import cache_result
from ..utils.parallel import Timer


@cache_result
def detect_puzzle_pieces(img_path: str, threshold_value: int, min_area: int) -> Dict[str, Any]:
    """Detect and extract puzzle pieces from an image.
    
    Args:
        img_path: Path to the input image
        threshold_value: Binary threshold value for segmentation
        min_area: Minimum contour area to consider as a valid piece
        
    Returns:
        Dictionary containing piece count and piece data
    """
    with Timer("Image loading and processing"):
        # Read image
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Could not read image from {img_path}")
        
        # Convert to grayscale and apply threshold
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary_mask = cv2.threshold(gray_img, threshold_value, 255, cv2.THRESH_BINARY)
        binary_mask = np.uint8(binary_mask)
        
        # Morphological operations
        closing_kernel = np.ones((9, 9), np.uint8)
        dilation_kernel = np.ones((3, 3), np.uint8)
        
        closed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, closing_kernel)
        processed_mask = cv2.dilate(closed_mask, dilation_kernel, iterations=1)
        
        # Find and filter contours
        contours, _ = cv2.findContours(processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
        
        # Create final mask
        filled_mask = np.zeros_like(processed_mask)
        cv2.drawContours(filled_mask, valid_contours, -1, 255, -1)
        
        # Extract puzzle pieces
        pieces = []
        padding = 5  # Small padding around each piece
        
        for i, contour in enumerate(valid_contours):
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Add padding (while staying within bounds)
            x1, y1 = max(0, x-padding), max(0, y-padding)
            x2, y2 = min(img.shape[1], x+w+padding), min(img.shape[0], y+h+padding)
            
            # Extract piece
            piece_img = img[y1:y2, x1:x2].copy()
            piece_mask = filled_mask[y1:y2, x1:x2].copy()
            
            # Apply mask to piece (isolate from background)
            masked_piece = cv2.bitwise_and(piece_img, piece_img, mask=piece_mask)
            
            # Add to list
            pieces.append({
                'index': i,
                'img': masked_piece.tolist(),  # Convert to list for JSON
                'mask': piece_mask.tolist(),
                'bbox': (x1, y1, x2, y2)
            })
    
    return {
        'count': len(pieces),
        'pieces': pieces
    }


def setup_output_directories() -> Dict[str, str]:
    """Create and return output directory paths.
    
    Returns:
        Dictionary mapping directory names to paths
    """
    from ..config.settings import DEBUG_DIRS
    
    dirs = {}
    for name, path in DEBUG_DIRS.items():
        os.makedirs(path, exist_ok=True)
        dirs[name] = path
    
    return dirs


def preprocess_image(img_path: str, threshold_value: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Preprocess image for puzzle piece detection.
    
    Args:
        img_path: Path to input image
        threshold_value: Binary threshold value
        
    Returns:
        Tuple of (original_image, binary_mask, processed_mask)
    """
    # Read image
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not read image from {img_path}")
    
    # Convert to grayscale and threshold
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary_mask = cv2.threshold(gray_img, threshold_value, 255, cv2.THRESH_BINARY)
    
    # Morphological operations
    closing_kernel = np.ones((9, 9), np.uint8)
    dilation_kernel = np.ones((3, 3), np.uint8)
    
    closed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, closing_kernel)
    processed_mask = cv2.dilate(closed_mask, dilation_kernel, iterations=1)
    
    return img, binary_mask, processed_mask