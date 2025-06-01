"""Edge feature extraction and processing functions."""

import cv2
import numpy as np
import os
from typing import Dict, List, Tuple, Any, Optional

from .color_analysis import extract_edge_color_sequence, normalize_edge_colors
from .texture_analysis import extract_edge_texture_descriptor
from ..config.settings import DEFAULT_TARGET_EDGE_POINTS


def resample_sequence(sequence: List[Any], target_length: int) -> np.ndarray:
    """Resample a sequence to a target length using linear interpolation.
    
    Args:
        sequence: Source sequence
        target_length: Desired length
        
    Returns:
        Resampled sequence
    """
    if len(sequence) == 0:
        return np.array([])
    
    if len(sequence) == target_length:
        return np.array(sequence)
    
    sequence = np.array(sequence)
    
    # Handle 1D sequences
    if len(sequence.shape) == 1:
        orig_indices = np.arange(len(sequence))
        target_indices = np.linspace(0, len(sequence) - 1, target_length)
        return np.interp(target_indices, orig_indices, sequence)
    
    # Handle multi-dimensional sequences
    orig_indices = np.arange(len(sequence))
    target_indices = np.linspace(0, len(sequence) - 1, target_length)
    
    # Interpolate each channel separately
    result = np.zeros((target_length, sequence.shape[1]), dtype=sequence.dtype)
    
    for channel in range(sequence.shape[1]):
        result[:, channel] = np.interp(
            target_indices, orig_indices, sequence[:, channel])
    
    return result


def normalize_edge_points(points: List[Tuple[int, int]], target_points: int = DEFAULT_TARGET_EDGE_POINTS, 
                         flip_for_matching: bool = True) -> np.ndarray:
    """Normalize edge points for consistent matching.
    
    Args:
        points: List of edge point coordinates
        target_points: Target number of points after normalization
        flip_for_matching: Whether to create flipped version for matching
        
    Returns:
        Normalized edge points array
    """
    if len(points) == 0:
        return np.array([])
    
    points_array = np.array(points)
    
    # Resample to target number of points
    if len(points_array) != target_points:
        # Interpolate along the edge
        indices = np.linspace(0, len(points_array) - 1, target_points)
        resampled_points = np.zeros((target_points, 2))
        
        resampled_points[:, 0] = np.interp(indices, np.arange(len(points_array)), points_array[:, 0])
        resampled_points[:, 1] = np.interp(indices, np.arange(len(points_array)), points_array[:, 1])
        
        points_array = resampled_points
    
    # Center the points around origin
    centroid = np.mean(points_array, axis=0)
    centered_points = points_array - centroid
    
    # Normalize scale
    max_distance = np.max(np.linalg.norm(centered_points, axis=1))
    if max_distance > 0:
        normalized_points = centered_points / max_distance
    else:
        normalized_points = centered_points
    
    return normalized_points


def extract_dtw_edge_features(piece_img: np.ndarray, edge_points: List[Tuple[int, int]], 
                             corner1: Tuple[int, int], corner2: Tuple[int, int], 
                             edge_index: int, piece_mask: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Extract comprehensive edge features for DTW matching.
    
    Args:
        piece_img: Source image containing the puzzle piece
        edge_points: List of edge coordinates
        corner1: First corner coordinates
        corner2: Second corner coordinates
        edge_index: Index of the edge
        piece_mask: Optional binary mask (255 = piece, 0 = background)
        
    Returns:
        Dictionary containing edge features
    """
    if len(edge_points) == 0:
        return {}
    
    # Extract color sequence and confidence with improved sampling
    lab_sequence, confidence_sequence = extract_edge_color_sequence(
        piece_img, edge_points, corner1, corner2, piece_mask)
    
    # Normalize color sequence
    if len(lab_sequence) > 0:
        lab_sequence = normalize_edge_colors(lab_sequence)
    
    # Extract geometric features
    normalized_points = normalize_edge_points(edge_points)
    
    # Calculate edge statistics
    edge_length = calculate_edge_length(edge_points)
    curvature = calculate_edge_curvature(edge_points)
    
    # Extract texture features
    texture_descriptor = extract_edge_texture_descriptor(piece_img, edge_points, piece_mask)
    
    return {
        'color_sequence': lab_sequence.tolist() if len(lab_sequence) > 0 else [],
        'confidence_sequence': confidence_sequence,
        'geometric_points': normalized_points.tolist() if len(normalized_points) > 0 else [],
        'edge_length': edge_length,
        'curvature': curvature,
        'corner1': corner1,
        'corner2': corner2,
        'edge_index': edge_index,
        'num_points': len(edge_points),
        'texture_descriptor': texture_descriptor
    }


def extract_edge_color_features(piece_img: np.ndarray, edge_points: List[Tuple[int, int]], 
                               corner1: Tuple[int, int], corner2: Tuple[int, int], 
                               edge_index: int) -> Dict[str, Any]:
    """Extract color-specific features from an edge.
    
    Args:
        piece_img: Source image
        edge_points: Edge coordinates
        corner1: First corner
        corner2: Second corner
        edge_index: Edge index
        
    Returns:
        Dictionary containing color features
    """
    if len(edge_points) == 0:
        return {}
    
    # Extract color sequence
    lab_sequence, confidence_sequence = extract_edge_color_sequence(
        piece_img, edge_points, corner1, corner2)
    
    if len(lab_sequence) == 0:
        return {}
    
    # Calculate color statistics
    mean_color = np.mean(lab_sequence, axis=0)
    std_color = np.std(lab_sequence, axis=0)
    
    # Calculate color gradient (change along edge)
    color_gradient = calculate_color_gradient(lab_sequence)
    
    return {
        'lab_sequence': lab_sequence.tolist(),
        'confidence_sequence': confidence_sequence,
        'mean_color': mean_color.tolist(),
        'std_color': std_color.tolist(),
        'color_gradient': color_gradient,
        'edge_index': edge_index
    }


def calculate_edge_length(edge_points: List[Tuple[int, int]]) -> float:
    """Calculate the total length of an edge.
    
    Args:
        edge_points: List of edge coordinates
        
    Returns:
        Total edge length in pixels
    """
    if len(edge_points) < 2:
        return 0.0
    
    total_length = 0.0
    for i in range(len(edge_points) - 1):
        p1 = edge_points[i]
        p2 = edge_points[i + 1]
        length = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        total_length += length
    
    return total_length


def calculate_edge_curvature(edge_points: List[Tuple[int, int]]) -> float:
    """Calculate the average curvature of an edge.
    
    Args:
        edge_points: List of edge coordinates
        
    Returns:
        Average curvature value
    """
    if len(edge_points) < 3:
        return 0.0
    
    curvatures = []
    for i in range(1, len(edge_points) - 1):
        p1 = np.array(edge_points[i-1])
        p2 = np.array(edge_points[i])
        p3 = np.array(edge_points[i+1])
        
        # Calculate vectors
        v1 = p2 - p1
        v2 = p3 - p2
        
        # Calculate angle change
        if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1, 1)
            angle = np.arccos(cos_angle)
            curvatures.append(angle)
    
    return np.mean(curvatures) if curvatures else 0.0


def calculate_color_gradient(color_sequence: List[np.ndarray]) -> float:
    """Calculate color gradient along an edge.
    
    Args:
        color_sequence: Sequence of LAB color values
        
    Returns:
        Average color gradient magnitude
    """
    if len(color_sequence) < 2:
        return 0.0
    
    gradients = []
    for i in range(len(color_sequence) - 1):
        color1 = np.array(color_sequence[i])
        color2 = np.array(color_sequence[i + 1])
        gradient = np.linalg.norm(color2 - color1)
        gradients.append(gradient)
    
    return np.mean(gradients) if gradients else 0.0


def extract_edge_points_from_image(piece_idx: int, edge_idx: int, debug_dir: str = 'debug') -> Optional[List[Tuple[int, int]]]:
    """Extract edge points from saved debug image.
    
    Args:
        piece_idx: Index of the piece
        edge_idx: Index of the edge
        debug_dir: Debug directory path
        
    Returns:
        List of edge points or None if not found
    """
    edge_file = os.path.join(debug_dir, 'edges', f'piece_{piece_idx+1}_edge_{edge_idx+1}.png')
    
    if not os.path.exists(edge_file):
        return None
    
    # Load edge image
    edge_img = cv2.imread(edge_file, cv2.IMREAD_GRAYSCALE)
    if edge_img is None:
        return None
    
    # Find edge points
    edge_points = np.where(edge_img > 0)
    y_coords, x_coords = edge_points
    
    # Return as list of (x, y) tuples
    return list(zip(x_coords, y_coords))