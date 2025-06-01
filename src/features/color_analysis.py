"""Color analysis and extraction functions for puzzle edges."""

import cv2
import numpy as np
from typing import List, Tuple, Union, Optional

from ..config.settings import DEFAULT_COLOR_RADIUS


def detect_background_color(image: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float]:
    """Detect the background color of the image.
    
    Args:
        image: Source image (BGR)
        mask: Optional mask where 255 = piece, 0 = background
        
    Returns:
        Tuple of (background_color_bgr, confidence)
    """
    if mask is not None:
        # Use inverse mask to sample only background
        background_mask = cv2.bitwise_not(mask)
        background_pixels = image[background_mask > 0]
        
        if len(background_pixels) > 0:
            # Use median for robustness against outliers
            bg_color = np.median(background_pixels, axis=0).astype(np.uint8)
            # Calculate confidence based on color uniformity
            std_dev = np.std(background_pixels, axis=0)
            confidence = 1.0 / (1.0 + np.mean(std_dev) / 10.0)
            return bg_color, confidence
    
    # Fallback: sample image borders
    h, w = image.shape[:2]
    border_size = 5
    
    # Sample pixels from all borders
    border_pixels = []
    # Top and bottom borders
    border_pixels.extend(image[0:border_size, :].reshape(-1, 3))
    border_pixels.extend(image[-border_size:, :].reshape(-1, 3))
    # Left and right borders (excluding corners already sampled)
    border_pixels.extend(image[border_size:-border_size, 0:border_size].reshape(-1, 3))
    border_pixels.extend(image[border_size:-border_size, -border_size:].reshape(-1, 3))
    
    if border_pixels:
        border_pixels = np.array(border_pixels)
        # Use mode for most common color (likely background)
        unique_colors, counts = np.unique(border_pixels, axis=0, return_counts=True)
        bg_color = unique_colors[np.argmax(counts)].astype(np.uint8)
        confidence = 0.8
    else:
        # Ultimate fallback
        bg_color = np.array([255, 255, 255], dtype=np.uint8)  # Assume white
        confidence = 0.5
    
    return bg_color, confidence


def compute_edge_normal(edge_points: List[Tuple[int, int]], point_idx: int, 
                       window_size: int = 5, piece_mask: Optional[np.ndarray] = None) -> Tuple[float, float]:
    """Compute the inward normal vector at a point on the edge.
    
    Args:
        edge_points: Ordered list of edge points
        point_idx: Index of the point to compute normal for
        window_size: Number of points to use for tangent estimation
        piece_mask: Optional mask to determine inward direction
        
    Returns:
        Tuple of (normal_x, normal_y) pointing inward
    """
    n_points = len(edge_points)
    if n_points < 3:
        return (0, 0)
    
    # Get current point
    x, y = edge_points[point_idx]
    
    # Get points for tangent calculation
    start_idx = max(0, point_idx - window_size // 2)
    end_idx = min(n_points - 1, point_idx + window_size // 2)
    
    if end_idx - start_idx < 2:
        return (0, 0)
    
    # Calculate tangent using finite differences
    if point_idx == 0:
        next_pt = np.array(edge_points[1])
        curr_pt = np.array(edge_points[0])
        tangent = next_pt - curr_pt
    elif point_idx == n_points - 1:
        curr_pt = np.array(edge_points[-1])
        prev_pt = np.array(edge_points[-2])
        tangent = curr_pt - prev_pt
    else:
        # Central difference
        prev_idx = max(0, point_idx - 1)
        next_idx = min(n_points - 1, point_idx + 1)
        tangent = np.array(edge_points[next_idx]) - np.array(edge_points[prev_idx])
    
    # Normalize tangent
    tangent_norm = np.linalg.norm(tangent)
    if tangent_norm > 0:
        tangent = tangent / tangent_norm
        
        # Get two possible normals by rotating tangent 90 degrees
        normal1 = np.array([-tangent[1], tangent[0]])  # Left rotation
        normal2 = np.array([tangent[1], -tangent[0]])  # Right rotation
        
        # If mask provided, test which normal points inward
        if piece_mask is not None:
            h, w = piece_mask.shape
            # Test both directions
            test_dist = 3
            
            # Test normal1
            test_x1 = int(x + normal1[0] * test_dist)
            test_y1 = int(y + normal1[1] * test_dist)
            inside1 = (0 <= test_x1 < w and 0 <= test_y1 < h and 
                      piece_mask[test_y1, test_x1] > 0)
            
            # Test normal2
            test_x2 = int(x + normal2[0] * test_dist)
            test_y2 = int(y + normal2[1] * test_dist)
            inside2 = (0 <= test_x2 < w and 0 <= test_y2 < h and 
                      piece_mask[test_y2, test_x2] > 0)
            
            # Choose the normal that points inside
            if inside1 and not inside2:
                return tuple(normal1)
            elif inside2 and not inside1:
                return tuple(normal2)
            else:
                # If both or neither are inside, use default (left rotation)
                return tuple(normal1)
        else:
            # Default to left rotation if no mask
            return tuple(normal1)
    
    return (0, 0)


def extract_inward_color(image: np.ndarray, mask: np.ndarray, x: int, y: int, 
                        edge_points: List[Tuple[int, int]], point_idx: int,
                        inward_offset: int = 5, radius: int = 1) -> Tuple[np.ndarray, float]:
    """Extract color by sampling inward from the edge.
    
    Args:
        image: Source image (BGR)
        mask: Binary mask (255 = piece, 0 = background)
        x, y: Edge point coordinates
        edge_points: Full list of edge points
        point_idx: Index of current point in edge_points
        inward_offset: Pixels to move inward from edge (increased to 5)
        radius: Sampling radius at the inward point
        
    Returns:
        Tuple of (color_bgr, confidence)
    """
    h, w = image.shape[:2]
    
    # Compute inward normal with mask awareness
    normal = compute_edge_normal(edge_points, point_idx, piece_mask=mask)
    
    # Sample at multiple inward distances, starting from 2 pixels in
    best_color = None
    best_confidence = 0.0
    
    for offset in range(2, inward_offset + 1):
        # Calculate inward point
        sample_x = int(x + normal[0] * offset)
        sample_y = int(y + normal[1] * offset)
        
        # Check bounds
        if not (0 <= sample_x < w and 0 <= sample_y < h):
            continue
            
        # Check if point is inside mask
        if mask[sample_y, sample_x] == 0:
            continue
            
        # Extract color at this point
        color = extract_robust_color(image, sample_x, sample_y, radius)
        conf = color_confidence(image, sample_x, sample_y, radius)
        
        # Additional confidence based on mask coverage
        region_mask = mask[max(0, sample_y-radius):min(h, sample_y+radius+1),
                          max(0, sample_x-radius):min(w, sample_x+radius+1)]
        if region_mask.size > 0:
            mask_coverage = np.mean(region_mask) / 255.0
            conf *= mask_coverage
        
        if conf > best_confidence:
            best_color = color
            best_confidence = conf
    
    # Fallback to original point if no valid inward point found
    if best_color is None:
        best_color = extract_robust_color(image, x, y, radius)
        best_confidence = 0.5  # Lower confidence for edge point
        
    return best_color, best_confidence


def extract_robust_color(image: np.ndarray, x: int, y: int, radius: int = DEFAULT_COLOR_RADIUS) -> np.ndarray:
    """Extract average color from a small region to reduce noise.
    
    Args:
        image: Source image (BGR)
        x, y: Center coordinates
        radius: Radius of sampling region
        
    Returns:
        Average color of the region (BGR)
    """
    x, y = int(x), int(y)
    # Ensure coordinates are within image bounds
    if not (0 <= y < image.shape[0] and 0 <= x < image.shape[1]):
        return np.array([0, 0, 0], dtype=np.uint8)
        
    # Extract region
    region = image[max(0, y-radius):min(image.shape[0], y+radius+1), 
                  max(0, x-radius):min(image.shape[1], x+radius+1)]
    
    if region.size > 0:
        return np.mean(region, axis=(0, 1)).astype(np.uint8)
    return image[y, x]  # Fallback to single pixel


def color_confidence(image: np.ndarray, x: int, y: int, radius: int = DEFAULT_COLOR_RADIUS) -> float:
    """Calculate confidence based on color variance in local region.
    
    Args:
        image: Source image
        x, y: Center coordinates
        radius: Radius of sampling region
        
    Returns:
        Confidence score between 0 and 1
    """
    x, y = int(x), int(y)
    # Ensure coordinates are within image bounds
    if not (0 <= y < image.shape[0] and 0 <= x < image.shape[1]):
        return 0.5
        
    # Extract region
    region = image[max(0, y-radius):min(image.shape[0], y+radius+1), 
                  max(0, x-radius):min(image.shape[1], x+radius+1)]
    
    if region.size > 0:
        # Calculate variance in each channel
        std_dev = np.std(region, axis=(0, 1))
        # Lower variance = higher confidence
        return 1.0 / (1.0 + np.mean(std_dev))
    return 0.5  # Default confidence


def normalize_edge_colors(colors: List[np.ndarray]) -> np.ndarray:
    """Apply gentle color normalization to reduce lighting variations.
    
    Args:
        colors: List of color values (any color space)
        
    Returns:
        Normalized color array
    """
    if len(colors) == 0:
        return np.array([])
        
    colors_array = np.array(colors)
    
    # Skip normalization if too few colors
    if len(colors_array) < 3:
        return colors_array
    
    # Don't normalize LAB colors - they're already perceptually uniform
    # Just return as-is to preserve actual color differences
    return colors_array
    
    # Alternative: gentle normalization could be added here if needed
    # For example, histogram equalization or gamma correction


def color_distance(color1: np.ndarray, color2: np.ndarray) -> float:
    """Calculate perceptual distance between two colors in LAB space.
    
    Args:
        color1: First color in LAB space
        color2: Second color in LAB space
        
    Returns:
        Perceptual distance
    """
    # Simple Euclidean distance in LAB space is a good approximation of perceptual distance
    return np.sqrt(np.sum((color1 - color2)**2))


def extract_edge_color_sequence(piece_img: np.ndarray, edge_points: List[Tuple[int, int]], 
                               corner1: Tuple[int, int], corner2: Tuple[int, int],
                               piece_mask: Optional[np.ndarray] = None,
                               background_color: Optional[np.ndarray] = None) -> Tuple[List[np.ndarray], List[float]]:
    """Extract color sequence and confidence values along an edge using inward sampling.
    
    Args:
        piece_img: Source image containing the puzzle piece
        edge_points: List of (x, y) coordinates along the edge
        corner1: First corner coordinates
        corner2: Second corner coordinates
        piece_mask: Optional binary mask (255 = piece, 0 = background)
        background_color: Optional detected background color (BGR)
        
    Returns:
        Tuple of (color_sequence, confidence_sequence)
    """
    if len(edge_points) == 0:
        return [], []
    
    # Sort points along edge path
    from ..core.geometry import sort_edge_points
    sorted_points = sort_edge_points(edge_points, corner1, corner2)
    
    # Create mask if not provided
    if piece_mask is None:
        # Simple threshold-based mask as fallback
        gray = cv2.cvtColor(piece_img, cv2.COLOR_BGR2GRAY)
        _, piece_mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    
    # Detect background if not provided
    if background_color is None:
        background_color, _ = detect_background_color(piece_img, piece_mask)
    
    # Extract color and confidence for each point using inward sampling
    bgr_sequence = []
    confidence_sequence = []
    
    for i, (x, y) in enumerate(sorted_points):
        # Use inward sampling
        color, conf = extract_inward_color(piece_img, piece_mask, x, y, sorted_points, i)
        
        # Check if color is too similar to background (likely a bad sample)
        bg_distance = np.linalg.norm(color.astype(float) - background_color.astype(float))
        if bg_distance < 30:  # Threshold for background similarity
            conf *= 0.3  # Reduce confidence significantly
        
        bgr_sequence.append(color)
        confidence_sequence.append(conf)
    
    # Convert BGR to LAB color space for better perceptual matching
    lab_sequence = []
    for bgr_color in bgr_sequence:
        # Convert single pixel BGR to LAB
        bgr_pixel = bgr_color.reshape(1, 1, 3)
        lab_pixel = cv2.cvtColor(bgr_pixel, cv2.COLOR_BGR2LAB)
        lab_sequence.append(lab_pixel[0, 0])
    
    return lab_sequence, confidence_sequence


def bgr_to_lab_sequence(bgr_sequence: List[np.ndarray]) -> List[np.ndarray]:
    """Convert BGR color sequence to LAB color space.
    
    Args:
        bgr_sequence: List of BGR color values
        
    Returns:
        List of LAB color values
    """
    lab_sequence = []
    for bgr_color in bgr_sequence:
        # Ensure we have a 3-channel color
        if len(bgr_color.shape) == 1 and bgr_color.shape[0] == 3:
            bgr_pixel = bgr_color.reshape(1, 1, 3)
            lab_pixel = cv2.cvtColor(bgr_pixel, cv2.COLOR_BGR2LAB)
            lab_sequence.append(lab_pixel[0, 0])
        else:
            lab_sequence.append(bgr_color)  # Already in correct format
    
    return lab_sequence


def calculate_color_similarity(color_seq1: List[np.ndarray], color_seq2: List[np.ndarray]) -> float:
    """Calculate similarity between two color sequences.
    
    Args:
        color_seq1: First color sequence
        color_seq2: Second color sequence
        
    Returns:
        Similarity score between 0 and 1
    """
    if not color_seq1 or not color_seq2:
        return 0.0
    
    # Resample sequences to same length
    from .edge_extraction import resample_sequence
    target_length = min(len(color_seq1), len(color_seq2), 20)
    
    resampled_seq1 = resample_sequence(color_seq1, target_length)
    resampled_seq2 = resample_sequence(color_seq2, target_length)
    
    # Calculate average color distance
    total_distance = 0
    for c1, c2 in zip(resampled_seq1, resampled_seq2):
        total_distance += color_distance(c1, c2)
    
    avg_distance = total_distance / len(resampled_seq1)
    
    # Convert distance to similarity (0-1 scale)
    # Assuming max reasonable LAB distance is around 100
    similarity = max(0, 1 - (avg_distance / 100))
    
    return similarity