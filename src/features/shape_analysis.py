#!/usr/bin/env python3
"""Shape analysis functions for edge classification and matching."""

import numpy as np
from typing import Tuple, List, Optional, Dict
from scipy import signal
from scipy.spatial.distance import cdist
import cv2


def calculate_curvature_profile(edge_points: np.ndarray, window_size: int = 5, 
                               smooth: bool = True) -> np.ndarray:
    """
    Calculate curvature at each point along the edge with optional smoothing.
    
    Args:
        edge_points: Array of shape (N, 2) containing edge points
        window_size: Window size for smoothing derivatives
        smooth: Whether to apply additional smoothing
        
    Returns:
        Array of curvature values at each point
    """
    if len(edge_points) < 3:
        return np.zeros(len(edge_points))
    
    # Smooth the points first using Savitzky-Golay filter
    window = min(window_size, len(edge_points))
    if window % 2 == 0:
        window -= 1  # Ensure odd window size
    window = max(3, window)  # Minimum window of 3
    
    x = signal.savgol_filter(edge_points[:, 0], window, min(3, window-1), mode='nearest')
    y = signal.savgol_filter(edge_points[:, 1], window, min(3, window-1), mode='nearest')
    
    # Calculate first and second derivatives
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    
    # Curvature formula: |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
    numerator = np.abs(dx * ddy - dy * ddx)
    denominator = np.power(dx**2 + dy**2, 1.5)
    
    # Avoid division by zero
    curvature = np.zeros_like(numerator)
    valid_mask = denominator > 1e-6
    curvature[valid_mask] = numerator[valid_mask] / denominator[valid_mask]
    
    # Apply additional smoothing if requested
    if smooth and len(curvature) > 10:
        # Use median filter to remove spikes
        from scipy.ndimage import median_filter
        curvature = median_filter(curvature, size=min(5, len(curvature)//4))
        
        # Follow up with gaussian smoothing
        from scipy.ndimage import gaussian_filter1d
        curvature = gaussian_filter1d(curvature, sigma=1.5, mode='nearest')
    
    return curvature


def calculate_turning_angles(edge_points: np.ndarray) -> np.ndarray:
    """
    Calculate cumulative turning angles along the edge.
    
    Args:
        edge_points: Array of shape (N, 2) containing edge points
        
    Returns:
        Array of cumulative turning angles
    """
    if len(edge_points) < 3:
        return np.zeros(len(edge_points))
    
    # Calculate vectors between consecutive points
    vectors = np.diff(edge_points, axis=0)
    
    # Calculate angles between consecutive vectors
    angles = []
    for i in range(len(vectors) - 1):
        v1 = vectors[i]
        v2 = vectors[i + 1]
        
        # Calculate angle using cross product and dot product
        cross = np.cross(v1, v2)
        dot = np.dot(v1, v2)
        angle = np.arctan2(cross, dot)
        angles.append(angle)
    
    # Pad to match original length
    angles = [0] + angles + [0]
    
    # Calculate cumulative sum
    cumulative_angles = np.cumsum(angles)
    
    return cumulative_angles


def calculate_shape_symmetry(edge_points: np.ndarray, num_samples: int = 50) -> float:
    """
    Calculate symmetry score of the edge shape.
    
    Args:
        edge_points: Array of shape (N, 2) containing edge points
        num_samples: Number of points to sample for symmetry calculation
        
    Returns:
        Symmetry score between 0 (asymmetric) and 1 (symmetric)
    """
    if len(edge_points) < 10:
        return 0.0
    
    # Resample edge to fixed number of points
    indices = np.linspace(0, len(edge_points) - 1, num_samples).astype(int)
    sampled_points = edge_points[indices]
    
    # Center the points
    center = np.mean(sampled_points, axis=0)
    centered_points = sampled_points - center
    
    # Find principal axis using PCA
    cov_matrix = np.cov(centered_points.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    principal_axis = eigenvectors[:, np.argmax(eigenvalues)]
    
    # Project points onto principal axis
    projections = np.dot(centered_points, principal_axis)
    
    # Split points into two halves based on projection
    median_proj = np.median(projections)
    left_mask = projections < median_proj
    right_mask = projections >= median_proj
    
    left_points = centered_points[left_mask]
    right_points = centered_points[right_mask]
    
    # Reflect right points across the perpendicular to principal axis
    perpendicular = np.array([-principal_axis[1], principal_axis[0]])
    
    # For each point on the right, find its reflection and compare to left
    symmetry_scores = []
    
    for point in right_points:
        # Reflect point across the line perpendicular to principal axis
        proj_on_perp = np.dot(point, perpendicular)
        reflected = point - 2 * proj_on_perp * perpendicular
        
        # Find closest point on left side
        if len(left_points) > 0:
            distances = cdist([reflected], left_points)[0]
            min_distance = np.min(distances)
            
            # Normalize distance by edge scale
            edge_scale = np.std(centered_points)
            normalized_distance = min_distance / (edge_scale + 1e-6)
            
            # Convert to score (closer = higher score)
            score = np.exp(-normalized_distance)
            symmetry_scores.append(score)
    
    # Average symmetry score
    if symmetry_scores:
        return np.mean(symmetry_scores)
    else:
        return 0.0


def calculate_shape_compactness(edge_points: np.ndarray) -> float:
    """
    Calculate compactness of the edge shape (perimeter^2 / area).
    Lower values indicate more compact shapes.
    
    Args:
        edge_points: Array of shape (N, 2) containing edge points
        
    Returns:
        Compactness score
    """
    if len(edge_points) < 3:
        return 0.0
    
    # Calculate perimeter
    perimeter = 0.0
    for i in range(len(edge_points)):
        j = (i + 1) % len(edge_points)
        perimeter += np.linalg.norm(edge_points[j] - edge_points[i])
    
    # Calculate area using shoelace formula
    area = 0.0
    for i in range(len(edge_points)):
        j = (i + 1) % len(edge_points)
        area += edge_points[i, 0] * edge_points[j, 1]
        area -= edge_points[j, 0] * edge_points[i, 1]
    area = abs(area) / 2.0
    
    # Avoid division by zero
    if area < 1e-6:
        return float('inf')
    
    # Compactness = perimeter^2 / area
    # Normalize by 4π (compactness of a circle = 4π)
    compactness = (perimeter ** 2) / (area * 4 * np.pi)
    
    return compactness


def analyze_edge_shape(edge_points: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Perform comprehensive shape analysis on an edge.
    
    Args:
        edge_points: Array of shape (N, 2) containing edge points
        
    Returns:
        Dictionary containing various shape descriptors
    """
    return {
        'curvature_profile': calculate_curvature_profile(edge_points),
        'turning_angles': calculate_turning_angles(edge_points),
        'symmetry_score': calculate_shape_symmetry(edge_points),
        'compactness': calculate_shape_compactness(edge_points),
        'length': calculate_edge_length(edge_points),
        'mean_curvature': np.mean(np.abs(calculate_curvature_profile(edge_points))),
        'max_curvature': np.max(np.abs(calculate_curvature_profile(edge_points)))
    }


def calculate_edge_length(edge_points: np.ndarray) -> float:
    """Calculate the total length of an edge."""
    if len(edge_points) < 2:
        return 0.0
    
    length = 0.0
    for i in range(len(edge_points) - 1):
        length += np.linalg.norm(edge_points[i + 1] - edge_points[i])
    
    return length


def classify_edge_shape(edge_points: np.ndarray, 
                       reference_line: Tuple[np.ndarray, np.ndarray],
                       threshold_ratio: float = 0.005) -> Tuple[str, Optional[str], float]:
    """
    Classify edge shape with primary type and sub-type.
    
    Args:
        edge_points: Array of shape (N, 2) containing edge points
        reference_line: Tuple of (start_point, end_point) for the reference line
        threshold_ratio: Threshold for flat classification as ratio of edge length
        
    Returns:
        Tuple of (primary_type, sub_type, confidence)
        primary_type: "flat", "convex", or "concave"
        sub_type: "symmetric", "asymmetric", or None
        confidence: Classification confidence score (0-1)
    """
    if len(edge_points) < 5:
        return "flat", None, 1.0
    
    # Calculate perpendicular distances from points to reference line
    start, end = reference_line
    line_vec = end - start
    line_length = np.linalg.norm(line_vec)
    line_unit = line_vec / line_length
    
    # Calculate signed distances
    distances = []
    for point in edge_points:
        # Vector from start to point
        to_point = point - start
        
        # Project onto line
        projection_length = np.dot(to_point, line_unit)
        projection = start + projection_length * line_unit
        
        # Calculate perpendicular vector
        perp_vec = point - projection
        
        # Determine sign using cross product
        cross = np.cross(line_vec, to_point)
        sign = np.sign(cross)
        
        # Signed distance
        distance = sign * np.linalg.norm(perp_vec)
        distances.append(distance)
    
    distances = np.array(distances)
    
    # Calculate distance-based statistics
    max_distance = np.max(np.abs(distances))
    mean_distance = np.mean(distances)
    edge_length = calculate_edge_length(edge_points)
    
    # Calculate curvature-based statistics (filtering corner artifacts)
    try:
        curvature = calculate_curvature_profile(edge_points)
        # Filter out extreme corner artifacts (top/bottom 10%)
        filtered_curvature = curvature[int(len(curvature)*0.1):int(len(curvature)*0.9)]
        if len(filtered_curvature) > 0:
            mean_curvature = np.mean(np.abs(filtered_curvature))
            max_curvature = np.max(np.abs(filtered_curvature))
            curvature_std = np.std(filtered_curvature)
        else:
            mean_curvature = max_curvature = curvature_std = 0
    except:
        mean_curvature = max_curvature = curvature_std = 0
    
    # Combined thresholds
    distance_threshold = threshold_ratio * edge_length
    curvature_threshold = 0.1  # Absolute curvature threshold
    
    # Multi-metric classification
    distance_score = max_distance / distance_threshold if distance_threshold > 0 else 0
    curvature_score = mean_curvature / curvature_threshold if curvature_threshold > 0 else 0
    
    # Combined score (weighted average)
    combined_score = 0.6 * distance_score + 0.4 * curvature_score
    
    # Debug output for problematic edges
    debug = False
    # Enable debug for edges with borderline scores
    if combined_score > 0.3 and combined_score < 0.5:
        debug = False  # Disabled for now
    if debug:
        print(f"Edge classification debug (score={combined_score:.3f}):")
        print(f"  Distance score: {distance_score:.3f}, Curvature score: {curvature_score:.3f}")
        print(f"  Max distance: {max_distance:.1f}, Edge length: {edge_length:.1f}")
        print(f"  Mean curvature: {mean_curvature:.3f}, Mean dist: {mean_distance:.3f}")
    
    # Primary classification using combined metrics
    # Adjusted threshold to better separate flat from curved edges
    if combined_score < 0.4:  # Increased threshold for better flat detection
        primary_type = "flat"
        sub_type = None
        confidence = max(0, 1.0 - combined_score)
    else:
        # Determine convex vs concave using area analysis
        positive_area = np.sum(distances[distances > 0])
        negative_area = np.abs(np.sum(distances[distances < 0]))
        
        # Also consider the mean distance for better classification
        mean_dist = np.mean(distances)
        
        # Additional check: if the edge has very low curvature variation, it might still be flat
        # Also check for low confidence curved edges that might be flat
        if (curvature_std < 0.05 and abs(mean_dist) < 2.0) or \
           (combined_score < 0.45 and abs(mean_dist) < 3.0):
            primary_type = "flat"
            confidence = 0.6  # Lower confidence for this secondary classification
        else:
            # Improved logic considering both area and mean distance
            if positive_area > negative_area * 1.1 or (positive_area > negative_area * 0.9 and mean_dist > 0.5):
                primary_type = "convex"
            elif negative_area > positive_area * 1.1 or (negative_area > positive_area * 0.9 and mean_dist < -0.5):
                primary_type = "concave"
            else:
                # Use mean distance for ambiguous cases
                if mean_dist > 0:
                    primary_type = "convex"
                else:
                    primary_type = "concave"
        
        # Calculate symmetry for sub-type using improved method
        symmetry_score = calculate_shape_symmetry(edge_points)
        
        # Adaptive symmetry threshold based on edge complexity
        base_symmetry_threshold = 0.7
        complexity_factor = min(curvature_std / 0.5, 1.0)  # More complex edges need lower threshold
        symmetry_threshold = base_symmetry_threshold - (complexity_factor * 0.2)
        
        if symmetry_score > symmetry_threshold:
            sub_type = "symmetric"
        else:
            sub_type = "asymmetric"
        
        # Confidence based on classification strength
        total_area = positive_area + negative_area
        if total_area > 0:
            area_confidence = abs(positive_area - negative_area) / total_area
        else:
            area_confidence = 0.5
            
        # Combine area confidence with curvature consistency
        curvature_confidence = min(1.0, curvature_score / 2.0)  # Scale curvature contribution
        confidence = 0.7 * area_confidence + 0.3 * curvature_confidence
        confidence = max(0.1, min(1.0, confidence))  # Clamp to reasonable range
    
    return primary_type, sub_type, confidence