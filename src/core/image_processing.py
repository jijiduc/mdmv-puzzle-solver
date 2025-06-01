"""Core image processing functions for puzzle detection."""

import cv2
import numpy as np
import os
from typing import Dict, List, Tuple, Any

from ..utils.parallel import Timer
from .piece import Piece


def fill_holes_flood_fill(binary_mask: np.ndarray) -> np.ndarray:
    """Fill holes in binary mask using flood fill algorithm.
    
    Args:
        binary_mask: Binary mask with potential holes
        
    Returns:
        Binary mask with holes filled
    """
    # Create a copy for flood filling
    filled = binary_mask.copy()
    
    # Create a mask that is 2 pixels larger in each dimension for flood fill
    h, w = binary_mask.shape
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    
    # Flood fill from the corner (background region)
    # This will fill all background areas connected to the border
    cv2.floodFill(filled, flood_mask, (0, 0), 255)
    
    # Invert to get filled areas
    filled_inverted = cv2.bitwise_not(filled)
    
    # Combine with original to fill holes while preserving original foreground
    result = cv2.bitwise_or(binary_mask, filled_inverted)
    
    return result


def improve_segmentation_with_color(img: np.ndarray, threshold_value: int) -> np.ndarray:
    """Improve segmentation using color information to better capture puzzle pieces.
    
    Args:
        img: Original BGR image
        threshold_value: Base threshold value
        
    Returns:
        Improved binary mask
    """
    # Method 1: Multi-channel thresholding
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Standard grayscale threshold
    _, gray_mask = cv2.threshold(gray_img, threshold_value, 255, cv2.THRESH_BINARY)
    
    # Method 2: Color channel analysis
    b, g, r = cv2.split(img)
    
    # Balanced color channel thresholds for proper piece extraction
    # Red channel - moderate enhancement for red chicken areas
    _, red_mask = cv2.threshold(r, max(75, threshold_value - 15), 255, cv2.THRESH_BINARY)
    
    # Blue and green channels - slightly lower thresholds 
    _, blue_mask = cv2.threshold(b, max(110, threshold_value), 255, cv2.THRESH_BINARY)
    _, green_mask = cv2.threshold(g, max(110, threshold_value), 255, cv2.THRESH_BINARY)
    
    # Method 3: Background detection and subtraction
    # Assume corners are background (white/light areas)
    h, w = img.shape[:2]
    corner_samples = [
        img[0:20, 0:20],           # Top-left
        img[0:20, w-20:w],         # Top-right  
        img[h-20:h, 0:20],         # Bottom-left
        img[h-20:h, w-20:w]        # Bottom-right
    ]
    
    # Calculate background color (mean of corners)
    background_pixels = np.concatenate([corner.reshape(-1, 3) for corner in corner_samples])
    bg_color = np.mean(background_pixels, axis=0)
    
    # Distance from background color
    color_diff = np.linalg.norm(img - bg_color, axis=2)
    
    # Balanced color distance threshold
    adaptive_threshold = max(40, threshold_value * 0.55)
    _, color_distance_mask = cv2.threshold(color_diff.astype(np.uint8), 
                                         adaptive_threshold, 255, cv2.THRESH_BINARY)
    
    # Method 4: Selective combination to avoid over-segmentation
    # Focus mainly on red channel enhancement for chicken areas
    red_enhancement = cv2.bitwise_and(red_mask, color_distance_mask)
    
    # Combine selectively - primarily grayscale with targeted red enhancement
    final_mask = cv2.bitwise_or(gray_mask, red_enhancement)
    
    return final_mask


def fill_holes_contour_based(binary_mask: np.ndarray) -> np.ndarray:
    """Fill holes using contour hierarchy analysis.
    
    Args:
        binary_mask: Binary mask with potential holes
        
    Returns:
        Binary mask with holes filled
    """
    # Find all contours with hierarchy
    contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    
    if hierarchy is None:
        return binary_mask
    
    # Create filled mask
    filled_mask = binary_mask.copy()
    
    # Fill internal contours (holes)
    for i, contour in enumerate(contours):
        # Check if this contour is a hole (has a parent)
        if hierarchy[0][i][3] != -1:  # Has parent, so it's a hole
            cv2.drawContours(filled_mask, [contour], -1, 255, -1)
    
    return filled_mask


def gentle_hole_filling(binary_mask: np.ndarray) -> np.ndarray:
    """Gentle hole filling that preserves piece shape while filling internal holes.
    
    Args:
        binary_mask: Binary mask with potential holes
        
    Returns:
        Binary mask with gentle hole filling
    """
    # Start with the input mask
    result = binary_mask.copy()
    
    # Method 1: Only contour-based hole filling (most conservative)
    result = fill_holes_contour_based(result)
    
    # Method 2: Very small morphological closing for tiny holes only
    small_kernel = np.ones((4, 4), np.uint8)
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, small_kernel)
    
    return result


def aggressive_hole_filling(binary_mask: np.ndarray) -> np.ndarray:
    """Aggressively fill holes using multiple techniques.
    
    Args:
        binary_mask: Binary mask with potential holes
        
    Returns:
        Binary mask with aggressive hole filling
    """
    # Start with the input mask
    result = binary_mask.copy()
    
    # Method 1: Contour-based hole filling
    result = fill_holes_contour_based(result)
    
    # Method 2: Multiple flood fill attempts
    result = fill_holes_flood_fill(result)
    
    # Method 3: Morphological closing with moderate kernel
    large_kernel = np.ones((18, 18), np.uint8)
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, large_kernel)
    
    # Method 4: Fill remaining holes using convex hull approach
    contours, _ = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        # Get convex hull
        hull = cv2.convexHull(contour)
        
        # Create mask from convex hull
        hull_mask = np.zeros_like(result)
        cv2.fillPoly(hull_mask, [hull], 255)
        
        # Combine with result (more conservative - only fill obvious internal areas)
        # Check if hull area is not too much larger than original contour area
        contour_area = cv2.contourArea(contour)
        hull_area = cv2.contourArea(hull)
        
        if hull_area > 0 and contour_area / hull_area > 0.8:  # If original contour is at least 80% of hull (more conservative)
            result = cv2.bitwise_or(result, hull_mask)
    
    return result


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
        
        # Improved segmentation using color information
        binary_mask = improve_segmentation_with_color(img, threshold_value)
        binary_mask = np.uint8(binary_mask)
        
        # Conservative morphological operations to preserve puzzle piece shapes
        # 1. Minimal closing to connect only very close regions
        small_closing_kernel = np.ones((3, 3), np.uint8)
        closed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, small_closing_kernel)
        
        # 2. Gentle hole filling for internal regions only
        filled_mask = gentle_hole_filling(closed_mask)
        
        # 3. Very small additional closing for tiny gaps only
        tiny_closing_kernel = np.ones((5, 5), np.uint8)
        filled_mask = cv2.morphologyEx(filled_mask, cv2.MORPH_CLOSE, tiny_closing_kernel)
        
        # 4. Skip dilation to maintain precise edge boundaries
        # This ensures color sampling stays within piece boundaries
        processed_mask = filled_mask
        
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
            
            # Create Piece object
            piece = Piece(
                index=i,
                image=masked_piece,
                mask=piece_mask,
                bbox=(x1, y1, x2 - x1, y2 - y1)  # Convert to x, y, width, height format
            )
            pieces.append(piece)
    
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
    
    # Improved segmentation using color information
    binary_mask = improve_segmentation_with_color(img, threshold_value)
    
    # Conservative morphological operations to preserve puzzle piece shapes
    # 1. Minimal closing to connect only very close regions
    small_closing_kernel = np.ones((3, 3), np.uint8)
    closed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, small_closing_kernel)
    
    # 2. Gentle hole filling for internal regions only
    filled_mask = gentle_hole_filling(closed_mask)
    
    # 3. Very small additional closing for tiny gaps only
    tiny_closing_kernel = np.ones((5, 5), np.uint8)
    filled_mask = cv2.morphologyEx(filled_mask, cv2.MORPH_CLOSE, tiny_closing_kernel)
    
    # 4. Minimal dilation to clean up edges without distortion
    small_dilation_kernel = np.ones((2, 2), np.uint8)
    processed_mask = cv2.dilate(filled_mask, small_dilation_kernel, iterations=1)
    
    return img, binary_mask, processed_mask