"""Proper corner detection for puzzle pieces using polar distance profile analysis."""

import numpy as np
from typing import List, Tuple
from scipy.signal import find_peaks, savgol_filter
from itertools import combinations


def find_puzzle_corners(distances: List[float], coords: List[Tuple[int, int]], 
                       angles: List[float]) -> List[Tuple[int, int]]:
    """Find 4 corners of a puzzle piece using polar distance profile analysis.
    
    This method:
    1. Finds all peaks in the distance profile
    2. Identifies groups of 4 peaks that are roughly equally spaced in angle (90° apart)
    3. Scores each group based on how well they form a rectangular pattern
    4. Returns the best 4 corners
    
    Args:
        distances: List of distances from centroid
        coords: List of coordinate pairs
        angles: List of angles from centroid (in radians, -π to π)
        
    Returns:
        List of 4 corner coordinates
    """
    if len(distances) < 4:
        return coords[:4] if len(coords) >= 4 else coords
    
    # Convert to numpy arrays
    distances = np.array(distances)
    angles = np.array(angles)
    coords_array = np.array(coords)
    
    # Smooth the distance profile
    if len(distances) > 20:
        window_length = min(len(distances)//10*2+1, 51)
        if window_length < 5:
            window_length = 5
        if window_length % 2 == 0:
            window_length += 1
        smoothed_distances = savgol_filter(distances, window_length, 3)
    else:
        smoothed_distances = distances
    
    # Find peaks in the distance profile
    # Use adaptive parameters based on the profile characteristics
    mean_dist = np.mean(smoothed_distances)
    std_dist = np.std(smoothed_distances)
    
    # Try different prominence values to get reasonable number of peaks
    peaks = None
    for prominence_factor in [0.5, 0.3, 0.2, 0.1, 0.05]:
        prominence = prominence_factor * std_dist
        min_distance = len(distances) // 30  # Minimum separation between peaks
        
        found_peaks, properties = find_peaks(
            smoothed_distances,
            prominence=prominence,
            distance=min_distance
        )
        
        if 6 <= len(found_peaks) <= 20:  # Good range for finding 4 corners
            peaks = found_peaks
            break
    
    # If still no good peaks, find peaks above mean
    if peaks is None or len(peaks) < 4:
        peaks, _ = find_peaks(smoothed_distances, height=mean_dist)
        
    # If still not enough peaks, take the highest points
    if len(peaks) < 4:
        sorted_indices = np.argsort(smoothed_distances)[::-1]
        peaks = sorted_indices[:min(12, len(distances))]
    
    # Now find the best 4-peak combination that forms a rectangular pattern
    best_score = -np.inf
    best_corners = None
    
    # For efficiency, if too many peaks, pre-filter to the most prominent ones
    if len(peaks) > 15:
        peak_heights = smoothed_distances[peaks]
        top_indices = np.argsort(peak_heights)[-15:]
        peaks = peaks[top_indices]
    
    # Try all combinations of 4 peaks
    for combo in combinations(peaks, 4):
        score = evaluate_rectangular_pattern(combo, angles, smoothed_distances)
        if score > best_score:
            best_score = score
            best_corners = combo
    
    if best_corners is None:
        # Fallback: take 4 highest peaks
        highest_peaks = peaks[np.argsort(smoothed_distances[peaks])[-4:]]
        best_corners = sorted(highest_peaks, key=lambda i: angles[i])
    else:
        # Sort corners by angle for consistent ordering
        best_corners = sorted(best_corners, key=lambda i: angles[i])
    
    # Convert indices to coordinates
    corner_coords = [tuple(coords_array[i]) for i in best_corners]
    
    return corner_coords


def evaluate_rectangular_pattern(corner_indices: Tuple[int, ...], angles: np.ndarray, 
                                distances: np.ndarray) -> float:
    """Evaluate how well 4 points form a rectangular pattern in polar coordinates.
    
    A good rectangular pattern has:
    - Angular intervals close to 90° (π/2 radians)
    - Consistent pattern: opposite corners have similar distances
    - All corners are prominent peaks
    
    Args:
        corner_indices: Tuple of 4 indices
        angles: Array of angles in radians
        distances: Array of distances
        
    Returns:
        Score (higher is better)
    """
    # Sort corners by angle
    sorted_indices = sorted(corner_indices, key=lambda i: angles[i])
    corner_angles = angles[sorted_indices]
    corner_distances = distances[sorted_indices]
    
    # Calculate angular intervals between consecutive corners
    angular_intervals = []
    for i in range(4):
        angle1 = corner_angles[i]
        angle2 = corner_angles[(i + 1) % 4]
        
        # Handle wraparound from π to -π
        interval = angle2 - angle1
        if interval < 0:
            interval += 2 * np.pi
        elif interval > 2 * np.pi:
            interval -= 2 * np.pi
            
        angular_intervals.append(interval)
    
    # Score 1: How close are the intervals to 90° (π/2)?
    target_interval = np.pi / 2
    interval_deviations = [abs(interval - target_interval) for interval in angular_intervals]
    
    # Use exponential decay for deviation penalty
    angular_score = np.mean([np.exp(-2 * dev) for dev in interval_deviations])
    
    # Score 2: Rectangular pattern - opposite corners should have similar distances
    # In a rectangle inscribed in a circle, opposite corners are equidistant from center
    diagonal_similarity1 = 1.0 / (1.0 + abs(corner_distances[0] - corner_distances[2]) / np.mean(corner_distances))
    diagonal_similarity2 = 1.0 / (1.0 + abs(corner_distances[1] - corner_distances[3]) / np.mean(corner_distances))
    pattern_score = (diagonal_similarity1 + diagonal_similarity2) / 2
    
    # Score 3: Corner prominence - prefer corners that are strong peaks
    min_distance = np.min(corner_distances)
    mean_distance = np.mean(distances)
    if mean_distance > 0:
        prominence_score = min_distance / mean_distance
    else:
        prominence_score = 0
    
    # Score 4: Angular coverage - the 4 corners should roughly cover the full circle
    total_coverage = sum(angular_intervals)
    coverage_score = 1.0 / (1.0 + abs(total_coverage - 2 * np.pi))
    
    # Combined score with weights
    weights = {
        'angular': 0.4,      # Most important: 90° spacing
        'pattern': 0.3,      # Important: rectangular pattern
        'prominence': 0.2,   # Moderate: strong peaks
        'coverage': 0.1      # Least: full coverage
    }
    
    final_score = (
        weights['angular'] * angular_score +
        weights['pattern'] * pattern_score +
        weights['prominence'] * prominence_score +
        weights['coverage'] * coverage_score
    )
    
    return final_score