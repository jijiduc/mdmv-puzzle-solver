"""
Utility functions for contour processing and analysis
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union
import math
from scipy.spatial import distance

def find_contours(binary_image: np.ndarray,
                 mode: int = cv2.RETR_EXTERNAL,
                 method: int = cv2.CHAIN_APPROX_TC89_KCOS) -> List[np.ndarray]:
    """
    Find contours in a binary image
    
    Args:
        binary_image: Binary input image
        mode: Contour retrieval mode
        method: Contour approximation method
    
    Returns:
        List of contours
    """
    contours, _ = cv2.findContours(binary_image.copy(), mode, method)
    return contours


def filter_contours(contours: List[np.ndarray],
                   min_area: float = 500,  # Reduced from original value
                   max_area: Optional[float] = None,
                   min_perimeter: float = 50,  # Reduced from original value
                   solidity_range: Tuple[float, float] = (0.6, 0.99),  # More lenient range
                   aspect_ratio_range: Tuple[float, float] = (0.2, 5.0)) -> List[np.ndarray]:  # More lenient range
    """
    Filter contours based on various criteria - with more relaxed parameters
    
    Args:
        contours: List of input contours
        min_area: Minimum contour area
        max_area: Maximum contour area (if None, no upper limit)
        min_perimeter: Minimum contour perimeter
        solidity_range: (min, max) range for solidity (area/convex_hull_area)
        aspect_ratio_range: (min, max) range for aspect ratio
    
    Returns:
        Filtered list of contours
    """
    filtered = []
    
    for contour in contours:
        # Calculate contour properties
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        # Basic size filtering
        if area < min_area or (max_area is not None and area > max_area):
            continue
        
        if perimeter < min_perimeter:
            continue
        
        perimeter_area_ratio = perimeter / (2 * math.sqrt(area * math.pi))
        if perimeter_area_ratio > 3.0:  # Too irregular for typical puzzle pieces
            continue

            
        # Convexity and solidity check
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:  # Avoid division by zero
            continue
            
        solidity = area / hull_area
        if solidity < solidity_range[0] or solidity > solidity_range[1]:
            continue
            
        # Aspect ratio check
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h > 0 else 0
        if aspect_ratio < aspect_ratio_range[0] or aspect_ratio > aspect_ratio_range[1]:
            continue
            
        filtered.append(contour)
    
    # If no contours passed the filter, return the largest 10 by area as a fallback
    if not filtered and contours:
        print("Warning: No contours passed the filter. Using largest contours as fallback.")
        contours_with_area = [(cv2.contourArea(cnt), cnt) for cnt in contours if cv2.contourArea(cnt) > 200]
        contours_with_area.sort(key=lambda x: x[0], reverse=True)
        return [cnt for _, cnt in contours_with_area[:10]]
    
    return filtered


def approximate_polygon(contour: np.ndarray, 
                       epsilon_factor: float = 0.02) -> np.ndarray:
    """
    Approximate a contour with a polygon
    
    Args:
        contour: Input contour
        epsilon_factor: Approximation accuracy factor (relative to perimeter)
    
    Returns:
        Approximated polygon vertices
    """
    perimeter = cv2.arcLength(contour, True)
    epsilon = epsilon_factor * perimeter
    return cv2.approxPolyDP(contour, epsilon, True)


def calculate_contour_features(contour: np.ndarray) -> Dict[str, Any]:
    """
    Calculate various features of a contour
    
    Args:
        contour: Input contour
    
    Returns:
        Dictionary of contour features
    """
    # Basic measurements
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    
    # Bounding shapes
    x, y, w, h = cv2.boundingRect(contour)
    rect_area = w * h
    
    min_area_rect = cv2.minAreaRect(contour)
    min_area_rect_area = min_area_rect[1][0] * min_area_rect[1][1]
    
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    
    # Shape descriptors
    extent = area / rect_area if rect_area > 0 else 0
    solidity = area / hull_area if hull_area > 0 else 0
    equivalent_diameter = np.sqrt(4 * area / np.pi)
    
    # Moments and derived features
    moments = cv2.moments(contour)
    
    centroid_x = moments['m10'] / moments['m00'] if moments['m00'] != 0 else 0
    centroid_y = moments['m01'] / moments['m00'] if moments['m00'] != 0 else 0
    
    # Calculate hu moments for shape recognition
    hu_moments = cv2.HuMoments(moments)
    
    return {
        'area': area,
        'perimeter': perimeter,
        'bbox': (x, y, w, h),
        'bbox_area': rect_area,
        'min_area_rect': min_area_rect,
        'min_area_rect_area': min_area_rect_area,
        'convex_hull': hull,
        'hull_area': hull_area,
        'extent': extent,
        'solidity': solidity,
        'equivalent_diameter': equivalent_diameter,
        'centroid': (centroid_x, centroid_y),
        'moments': moments,
        'hu_moments': hu_moments
    }


def find_corners(contour: np.ndarray,
                approx_epsilon: float = 0.02,
                angle_threshold: float = 45.0,
                min_corner_dist: float = 10.0) -> np.ndarray:
    """
    Improved corner detection in puzzle pieces
    
    Args:
        contour: Input contour
        approx_epsilon: Polygon approximation accuracy factor
        angle_threshold: Maximum deviation from 90° for corner angles
        min_corner_dist: Minimum distance between detected corners
    
    Returns:
        Array of corner points
    """
    # First, approximate the contour to reduce noise
    perimeter = cv2.arcLength(contour, True)
    epsilon = approx_epsilon * perimeter
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    # For puzzle pieces, we'll use the Douglas-Peucker algorithm with careful parameters
    corners = []
    
    # If the approximated polygon has 4 points, it's likely a good approximation already
    if len(approx) == 4:
        return approx.reshape(-1, 2)
    
    # Otherwise, use a more sophisticated approach
    # First get the convex hull
    hull = cv2.convexHull(contour)
    
    # Find convexity defects
    if len(hull) > 3:
        hull_indices = cv2.convexHull(contour, returnPoints=False)
        defects = cv2.convexityDefects(contour, hull_indices)
        
        if defects is not None:
            # Extract deep convexity defects as potential corners
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                
                # Convert depth to actual distance in pixels
                depth = d / 256.0
                
                # Only consider deep defects
                if depth > min_corner_dist:
                    start_point = tuple(contour[s][0])
                    end_point = tuple(contour[e][0])
                    far_point = tuple(contour[f][0])
                    
                    # Add start and end points to corners
                    corners.append(start_point)
                    corners.append(end_point)
    
    # If we have fewer than 4 corners, fall back to the approximated polygon
    if len(corners) < 4:
        corners = [tuple(p[0]) for p in approx]
    
    # Remove duplicate or very close corners
    filtered_corners = []
    for corner in corners:
        # Check if this corner is far enough from already filtered corners
        if not any(distance.euclidean(corner, fc) < min_corner_dist for fc in filtered_corners):
            filtered_corners.append(corner)
    
    # Sort corners clockwise
    if len(filtered_corners) >= 3:
        # Calculate centroid
        cx = sum(x for x, y in filtered_corners) / len(filtered_corners)
        cy = sum(y for x, y in filtered_corners) / len(filtered_corners)
        
        # Sort corners by angle from centroid
        filtered_corners.sort(key=lambda pt: math.atan2(pt[1] - cy, pt[0] - cx))
    
    return np.array(filtered_corners)


def extract_borders(contour: np.ndarray, 
                   corners: np.ndarray) -> List[np.ndarray]:
    """
    Extract border segments between corners
    
    Args:
        contour: Input contour
        corners: Corner points
    
    Returns:
        List of border segments (each a numpy array of points)
    """
    if len(corners) < 2:
        return []
    
    # Reshape contour to make it easier to work with
    contour_points = contour.reshape(-1, 2)
    
    # Get indices of corner points in the contour
    corner_indices = []
    for corner in corners:
        # Find the closest point in the contour to this corner
        distances = np.sqrt(np.sum((contour_points - corner) ** 2, axis=1))
        closest_idx = np.argmin(distances)
        corner_indices.append(closest_idx)
    
    # Sort corner indices
    corner_indices.sort()
    
    # Extract border segments
    borders = []
    n_contour_points = len(contour_points)
    
    for i in range(len(corner_indices)):
        start_idx = corner_indices[i]
        end_idx = corner_indices[(i + 1) % len(corner_indices)]
        
        # Handle wraparound
        if end_idx < start_idx:
            segment = np.vstack((
                contour_points[start_idx:],
                contour_points[:end_idx + 1]
            ))
        else:
            segment = contour_points[start_idx:end_idx + 1]
        
        borders.append(segment)
    
    return borders


def classify_border(border: np.ndarray,
                  complexity_threshold: float = 1.2,
                  deviation_threshold: float = 10.0) -> str:
    """
    Classify a border segment as straight, tab, or pocket
    
    Args:
        border: Border segment points
        complexity_threshold: Threshold for path complexity
        deviation_threshold: Threshold for maximum deviation
    
    Returns:
        Border type: "straight", "tab", or "pocket"
    """
    if len(border) < 3:
        return "straight"
    
    # Get start and end points
    start, end = border[0], border[-1]
    
    # Calculate direct distance between endpoints
    direct_dist = np.linalg.norm(end - start)
    
    if direct_dist < 1e-5:  # Avoid division by zero
        return "straight"
    
    # Calculate path length along the border
    path_length = 0
    for i in range(len(border) - 1):
        path_length += np.linalg.norm(border[i+1] - border[i])
    
    # Calculate complexity (ratio of path length to direct distance)
    complexity = path_length / direct_dist
    
    # Calculate maximum deviation from straight line
    deviations = []
    for p in border:
        # Calculate perpendicular distance from point to line
        dev = np.abs(np.cross(end - start, start - p)) / direct_dist
        deviations.append(dev)
    
    max_deviation = max(deviations)
    
    # Classify based on complexity and deviation
    if complexity <= complexity_threshold and max_deviation <= deviation_threshold:
        return "straight"
    else:
        # Determine if it's a tab or a pocket
        # First, define the straight line from start to end
        line_vec = end - start
        
        # Find the point with maximum deviation
        max_dev_idx = np.argmax(deviations)
        max_dev_point = border[max_dev_idx]
        
        # Vector from start to max_dev_point
        start_to_dev = max_dev_point - start
        
        # Calculate cross product to determine if point is "above" or "below" the line
        cross_prod = np.cross(line_vec, start_to_dev)
        
        # If cross product is positive, the point is "above" the line (tab)
        # If cross product is negative, the point is "below" the line (pocket)
        return "tab" if cross_prod > 0 else "pocket"