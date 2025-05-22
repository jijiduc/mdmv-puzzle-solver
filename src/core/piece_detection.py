"""Individual puzzle piece processing and analysis."""

import cv2
import numpy as np
import math
import gc
import os
from typing import Dict, List, Tuple, Any

from .geometry import extract_edge_between_corners, classify_edge
from ..features.edge_extraction import extract_dtw_edge_features


def process_piece(piece_data: Dict[str, Any], output_dirs: Tuple[str, ...]) -> Dict[str, Any]:
    """Process a single puzzle piece - optimized for performance.
    
    Args:
        piece_data: Dictionary containing piece image data and metadata
        output_dirs: Tuple of output directory paths
        
    Returns:
        Dictionary containing processed piece data
    """
    piece_index = piece_data['index']
    
    # Get output paths
    edges_dir, edge_types_dir, contours_dir = output_dirs
    
    # Convert lists to NumPy arrays
    piece_img = np.array(piece_data['img'], dtype=np.uint8)
    piece_mask = np.array(piece_data['mask'], dtype=np.uint8)
    
    # Detect contours and centroid
    edges = cv2.Canny(piece_mask, 50, 150)
    
    # Find edge points
    edge_points = np.where(edges > 0)
    y_edge, x_edge = edge_points[0], edge_points[1]
    edge_coordinates = np.column_stack((x_edge, y_edge))
    
    # Calculate centroid
    moments = cv2.moments(piece_mask)
    if moments["m00"] != 0:
        centroid_x = int(moments["m10"] / moments["m00"])
        centroid_y = int(moments["m01"] / moments["m00"])
    else:
        centroid_x = piece_mask.shape[1] // 2
        centroid_y = piece_mask.shape[0] // 2
    centroid = (centroid_x, centroid_y)
    
    # Calculate distances and angles from centroid
    distances = []
    angles = []
    coords = []
    
    for x, y in edge_coordinates:
        # Distance from centroid
        dist = math.sqrt((x - centroid_x)**2 + (y - centroid_y)**2)
        distances.append(dist)
        
        # Angle from centroid
        angle = math.atan2(y - centroid_y, x - centroid_x)
        angles.append(angle)
        coords.append((x, y))
    
    # Sort by angle for proper ordering
    if angles:
        sorted_data = sorted(zip(angles, distances, coords))
        sorted_angles, sorted_distances, sorted_coords = zip(*sorted_data)
        sorted_angles = list(sorted_angles)
        sorted_distances = list(sorted_distances)
        sorted_coords = list(sorted_coords)
    else:
        sorted_angles, sorted_distances, sorted_coords = [], [], []
    
    # Find corners using distance peaks
    corners = find_corners(sorted_distances, sorted_coords, sorted_angles)
    
    # Extract edges between consecutive corners
    edge_types = []
    edge_deviations = []
    edge_colors = []
    
    for i in range(len(corners)):
        next_i = (i + 1) % len(corners)
        
        # Extract edge points between corners
        edge_points = extract_edge_between_corners(corners, i, next_i, 
                                                  np.array(sorted_coords), centroid)
        
        if len(edge_points) > 0:
            # Classify edge type (pass piece and edge indices for debug)
            edge_type, deviation = classify_edge(edge_points, corners[i], corners[next_i], centroid, 
                                                piece_index, i)
            edge_types.append(edge_type)
            edge_deviations.append(deviation)
            
            # Extract edge features
            edge_features = extract_dtw_edge_features(piece_img, edge_points, 
                                                    corners[i], corners[next_i], i)
            edge_colors.append(edge_features)
        else:
            edge_types.append("unknown")
            edge_deviations.append(0)
            edge_colors.append({})
    
    # Memory cleanup
    gc.collect()
    
    return {
        'piece_idx': piece_index,
        'edge_types': edge_types,
        'edge_deviations': edge_deviations,
        'edge_colors': edge_colors,
        'corners': corners,
        'centroid': centroid,
        'img': piece_img.tolist(),  # Add piece image for visualization
        'mask': piece_mask.tolist()  # Add piece mask
    }


def find_corners(distances: List[float], coords: List[Tuple[int, int]], 
                angles: List[float]) -> List[Tuple[int, int]]:
    """Find corner points using geometric constraint-based analysis.
    
    This method uses a two-stage approach:
    1. Detect all potential corner candidates using peak detection
    2. Find the best 4-point combination that forms a rectangle/square
    
    Args:
        distances: List of distances from centroid
        coords: List of coordinate pairs
        angles: List of angles from centroid
        
    Returns:
        List of corner coordinates (exactly 4 points)
    """
    if len(distances) < 4:
        return coords[:4] if len(coords) >= 4 else coords
    
    # Stage 1: Candidate Detection
    corner_candidates = detect_corner_candidates(distances, coords, angles)
    
    if len(corner_candidates) < 4:
        # Not enough candidates - return best available
        return [coords[i] for i in corner_candidates] + coords[:max(0, 4-len(corner_candidates))]
    
    # Stage 2: Find best 4-point rectangular combination
    best_combination = find_best_rectangular_combination(corner_candidates, distances, angles)
    
    # Return corner coordinates
    corners = [coords[i] for i in best_combination]
    return corners


def detect_corner_candidates(distances: List[float], coords: List[Tuple[int, int]], 
                           angles: List[float]) -> List[int]:
    """Detect all potential corner candidates using peak detection.
    
    Args:
        distances: List of distances from centroid
        coords: List of coordinate pairs  
        angles: List of angles from centroid
        
    Returns:
        List of indices of potential corner candidates
    """
    from scipy.signal import savgol_filter, find_peaks
    
    # Smooth distances to reduce noise
    if len(distances) > 5:
        window_length = min(len(distances)//4*2+1, 11)
        smoothed_distances = savgol_filter(distances, window_length, 2)
    else:
        smoothed_distances = distances
    
    # Find all peaks above mean distance (potential corners)
    mean_dist = np.mean(smoothed_distances)
    peaks, peak_properties = find_peaks(smoothed_distances, 
                                      height=mean_dist,
                                      distance=len(distances)//20)  # Minimum separation
    
    if len(peaks) == 0:
        # No peaks found - use highest distance points
        sorted_indices = sorted(range(len(distances)), key=lambda i: distances[i], reverse=True)
        return sorted_indices[:min(20, len(distances))]
    
    # Filter and rank candidates by peak prominence
    candidate_scores = []
    for peak_idx in peaks:
        # Score based on distance and prominence
        distance_score = smoothed_distances[peak_idx] / max(smoothed_distances)
        prominence_score = peak_properties['peak_heights'][list(peaks).index(peak_idx)] / max(peak_properties['peak_heights'])
        combined_score = 0.7 * distance_score + 0.3 * prominence_score
        candidate_scores.append((combined_score, peak_idx))
    
    # Sort by score and return top candidates (limit to reasonable number)
    candidate_scores.sort(reverse=True)
    max_candidates = min(30, len(candidate_scores))  # Limit to prevent combinatorial explosion
    
    return [idx for _, idx in candidate_scores[:max_candidates]]


def find_best_rectangular_combination(candidates: List[int], distances: List[float], 
                                    angles: List[float]) -> List[int]:
    """Find the best 4-point combination that forms a rectangle/square.
    
    Args:
        candidates: List of candidate corner indices
        distances: List of distances from centroid
        angles: List of angles from centroid
        
    Returns:
        List of 4 indices representing the best corner combination
    """
    from itertools import combinations
    
    if len(candidates) < 4:
        return candidates
    
    best_score = -1
    best_combination = None
    
    # Evaluate all possible 4-point combinations
    for combo in combinations(candidates, 4):
        score = evaluate_rectangular_score(combo, distances, angles)
        
        if score > best_score:
            best_score = score
            best_combination = combo
    
    # Sort combination by angle to maintain consistent ordering
    if best_combination:
        sorted_combo = sorted(best_combination, key=lambda i: angles[i])
        return sorted_combo
    
    # Fallback: return first 4 candidates
    return candidates[:4]


def evaluate_rectangular_score(combination: Tuple[int, int, int, int], 
                              distances: List[float], angles: List[float]) -> float:
    """Evaluate how well a 4-point combination forms a rectangle/square.
    
    Args:
        combination: Tuple of 4 indices
        distances: List of distances from centroid
        angles: List of angles from centroid
        
    Returns:
        Score between 0 and 1 (higher = more rectangular)
    """
    combo_list = list(combination)
    
    # Sort by angle to get proper ordering around the perimeter
    sorted_combo = sorted(combo_list, key=lambda i: angles[i])
    
    # Calculate angular spacings between consecutive corners
    angular_spacings = []
    for i in range(4):
        curr_angle = angles[sorted_combo[i]]
        next_angle = angles[sorted_combo[(i + 1) % 4]]
        
        # Handle angle wraparound
        spacing = next_angle - curr_angle
        if spacing < 0:
            spacing += 2 * np.pi
        
        angular_spacings.append(spacing)
    
    # 1. Angular Uniformity Score (how close to 90° each angle is)
    target_spacing = np.pi / 2  # 90 degrees in radians
    angular_deviations = [abs(spacing - target_spacing) for spacing in angular_spacings]
    max_allowed_deviation = np.pi / 6  # 30 degrees tolerance
    
    angular_score = 0
    for deviation in angular_deviations:
        if deviation <= max_allowed_deviation:
            angular_score += 1 - (deviation / max_allowed_deviation)
    angular_score /= 4  # Normalize to 0-1
    
    # 2. Distance Uniformity Score (how similar the corner distances are)
    corner_distances = [distances[i] for i in sorted_combo]
    mean_distance = np.mean(corner_distances)
    
    if mean_distance > 0:
        distance_cv = np.std(corner_distances) / mean_distance
        distance_score = max(0, 1 - (distance_cv / 0.2))  # 20% CV tolerance
    else:
        distance_score = 0
    
    # 3. Angular Coverage Score (how well the 4 points cover the full circle)
    total_coverage = sum(angular_spacings)
    coverage_score = min(1.0, total_coverage / (2 * np.pi))
    
    # 4. Geometric Regularity Score (aspect ratio consideration)
    # Calculate approximate rectangle dimensions using corner distances
    opposite_pairs = [
        (corner_distances[0], corner_distances[2]),  # Opposite corners
        (corner_distances[1], corner_distances[3])
    ]
    
    aspect_ratios = []
    for pair in opposite_pairs:
        if min(pair) > 0:
            ratio = max(pair) / min(pair)
            aspect_ratios.append(ratio)
    
    if aspect_ratios:
        # Prefer aspect ratios closer to 1 (square) but allow rectangles
        mean_aspect_ratio = np.mean(aspect_ratios)
        regularity_score = 1 / (1 + abs(mean_aspect_ratio - 1))  # Score peaks at 1 for squares
    else:
        regularity_score = 0
    
    # Combine scores with weights
    weights = {
        'angular': 0.4,      # Most important: proper angular spacing
        'distance': 0.3,     # Second: uniform distances
        'coverage': 0.2,     # Third: good coverage
        'regularity': 0.1    # Fourth: geometric regularity
    }
    
    final_score = (weights['angular'] * angular_score + 
                   weights['distance'] * distance_score +
                   weights['coverage'] * coverage_score + 
                   weights['regularity'] * regularity_score)
    
    return final_score