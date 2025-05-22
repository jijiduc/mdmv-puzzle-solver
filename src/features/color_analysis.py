"""Color analysis and extraction functions for puzzle edges."""

import cv2
import numpy as np
from typing import List, Tuple, Union

from ..config.settings import DEFAULT_COLOR_RADIUS


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
    """Apply color normalization to make matching more robust.
    
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
    
    # Simple normalization: scale to use full range
    normalized = np.zeros_like(colors_array, dtype=np.float32)
    
    # Normalize each channel independently
    for channel in range(colors_array.shape[1]):
        channel_data = colors_array[:, channel].astype(np.float32)
        channel_min = np.min(channel_data)
        channel_max = np.max(channel_data)
        
        # Avoid division by zero
        if channel_max > channel_min:
            # Scale to [0, 255]
            normalized[:, channel] = ((channel_data - channel_min) * 255.0 / 
                                     (channel_max - channel_min))
        else:
            normalized[:, channel] = channel_data
    
    return normalized


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
                               corner1: Tuple[int, int], corner2: Tuple[int, int]) -> Tuple[List[np.ndarray], List[float]]:
    """Extract color sequence and confidence values along an edge.
    
    Args:
        piece_img: Source image containing the puzzle piece
        edge_points: List of (x, y) coordinates along the edge
        corner1: First corner coordinates
        corner2: Second corner coordinates
        
    Returns:
        Tuple of (color_sequence, confidence_sequence)
    """
    if len(edge_points) == 0:
        return [], []
    
    # Sort points along edge path
    from ..core.geometry import sort_edge_points
    sorted_points = sort_edge_points(edge_points, corner1, corner2)
    
    # Extract color and confidence for each point
    bgr_sequence = []
    confidence_sequence = []
    
    for x, y in sorted_points:
        robust_color = extract_robust_color(piece_img, x, y)
        bgr_sequence.append(robust_color)
        confidence_sequence.append(color_confidence(piece_img, x, y))
    
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