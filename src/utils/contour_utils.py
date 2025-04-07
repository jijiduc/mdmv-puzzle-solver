"""
Enhanced utility functions for contour processing and analysis
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union
import math
from scipy.spatial import distance
from scipy import stats
import logging

# Configure logging
logger = logging.getLogger(__name__)


def find_contours(binary_image: np.ndarray,
                  mode: int = cv2.RETR_EXTERNAL,
                  method: int = cv2.CHAIN_APPROX_TC89_KCOS) -> List[np.ndarray]:
    """
    Find contours in a binary image with multiple methods and combining results
    
    Args:
        binary_image: Binary input image
        mode: Contour retrieval mode
        method: Contour approximation method
    
    Returns:
        List of contours
    """
    # Use multiple methods and combine results for more robust detection
    methods = [
        cv2.CHAIN_APPROX_TC89_KCOS,  # Good for curved shapes
        cv2.CHAIN_APPROX_SIMPLE      # More efficient for straight lines
    ]
    
    all_contours = []
    
    for method in methods:
        contours, _ = cv2.findContours(binary_image.copy(), mode, method)
        all_contours.extend(contours)
    
    # Remove duplicates by checking overlap
    unique_contours = []
    for contour in all_contours:
        # Check if this contour is significantly different from all unique contours
        is_unique = True
        for unique in unique_contours:
            # If the contours have similar area and location, consider them duplicates
            if _contours_overlap(contour, unique):
                is_unique = False
                break
        
        if is_unique:
            unique_contours.append(contour)
    
    return unique_contours


def _contours_overlap(contour1: np.ndarray, contour2: np.ndarray, 
                      threshold: float = 0.7) -> bool:
    """
    Check if two contours overlap significantly
    
    Args:
        contour1: First contour
        contour2: Second contour
        threshold: Overlap threshold (0.0 to 1.0)
    
    Returns:
        True if contours overlap significantly
    """
    # Create masks for both contours
    x1, y1, w1, h1 = cv2.boundingRect(contour1)
    x2, y2, w2, h2 = cv2.boundingRect(contour2)
    
    # Determine the bounding box that encompasses both contours
    x_min = min(x1, x2)
    y_min = min(y1, y2)
    x_max = max(x1 + w1, x2 + w2)
    y_max = max(y1 + h1, y2 + h2)
    
    width = x_max - x_min
    height = y_max - y_min
    
    # Create masks
    mask1 = np.zeros((height, width), dtype=np.uint8)
    mask2 = np.zeros((height, width), dtype=np.uint8)
    
    # Draw contours on masks
    cv2.drawContours(mask1, [contour1], 0, 255, -1, offset=(-x_min, -y_min))
    cv2.drawContours(mask2, [contour2], 0, 255, -1, offset=(-x_min, -y_min))
    
    # Calculate intersection and union
    intersection = cv2.bitwise_and(mask1, mask2)
    union = cv2.bitwise_or(mask1, mask2)
    
    intersection_area = np.count_nonzero(intersection)
    union_area = np.count_nonzero(union)
    
    if union_area == 0:
        return False
    
    # Calculate IoU (Intersection over Union)
    iou = intersection_area / union_area
    
    return iou > threshold


def filter_contours(contours: List[np.ndarray],
                   min_area: float = 500,
                   max_area: Optional[float] = None,
                   min_perimeter: float = 50,
                   solidity_range: Tuple[float, float] = (0.6, 0.99),
                   aspect_ratio_range: Tuple[float, float] = (0.2, 5.0),
                   use_statistical_filtering: bool = True,
                   expected_piece_count: Optional[int] = None) -> List[np.ndarray]:
    """
    Enhanced contour filtering with statistical analysis and validation
    
    Args:
        contours: List of input contours
        min_area: Minimum contour area
        max_area: Maximum contour area (if None, no upper limit)
        min_perimeter: Minimum contour perimeter
        solidity_range: (min, max) range for solidity (area/convex_hull_area)
        aspect_ratio_range: (min, max) range for aspect ratio
        use_statistical_filtering: Whether to use statistical filtering
        expected_piece_count: Expected number of puzzle pieces (optional)
    
    Returns:
        Filtered list of contours
    """
    # Perform initial filtering with basic criteria
    initial_filtered = []
    
    for contour in contours:
        # Calculate contour properties
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        # Skip tiny contours
        if area < min_area or perimeter < min_perimeter:
            continue
            
        # Skip too large contours if max_area is specified
        if max_area is not None and area > max_area:
            continue
        
        # Calculate additional shape properties
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        
        if hull_area == 0:  # Avoid division by zero
            continue
            
        solidity = area / hull_area
        
        # Check solidity range
        if solidity < solidity_range[0] or solidity > solidity_range[1]:
            continue
            
        # Check aspect ratio
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h > 0 else 0
        
        if aspect_ratio < aspect_ratio_range[0] or aspect_ratio > aspect_ratio_range[1]:
            continue
        
        # Check if contour has enough points for a puzzle piece
        if len(contour) < 20:  # A puzzle piece should have a complex boundary
            continue
            
        # Check perimeter-to-area ratio (compactness) - puzzle pieces tend to have specific ranges
        # A perfect circle has ratio = 2*sqrt(pi) ≈ 3.54
        # Puzzle pieces tend to have higher values due to their complex boundaries
        compactness = perimeter**2 / (4 * np.pi * area) if area > 0 else float('inf')
        
        # Typical puzzle pieces have compactness between 3 (nearly circular) and 
        # 10 (very irregular with lots of tabs and pockets)
        if compactness < 1.2 or compactness > 15.0:
            continue
            
        initial_filtered.append((contour, {
            'area': area,
            'perimeter': perimeter,
            'solidity': solidity,
            'aspect_ratio': aspect_ratio,
            'compactness': compactness
        }))
    
    # If no statistical filtering or too few contours, return initial filtered contours
    if not use_statistical_filtering or len(initial_filtered) < 5:
        return [c[0] for c in initial_filtered]
    
    # Extract metrics for statistical analysis
    areas = [metrics['area'] for _, metrics in initial_filtered]
    
    # Use more robust measures: median and median absolute deviation
    median_area = np.median(areas)
    mad_area = stats.median_abs_deviation(areas)
    
    # If we have an expected piece count, we can refine our area expectations
    if expected_piece_count is not None and expected_piece_count > 0:
        # Estimate the total puzzle area (approximate)
        # We assume that filtered contours contain a mix of valid and invalid pieces
        total_filtered_area = sum(areas)
        
        # Estimate expected area per piece
        # We apply a correction factor assuming around 75% of filtered contours are valid
        estimated_area_per_piece = total_filtered_area / (len(initial_filtered) * 0.75)
        
        # Adjust our expectations based on this estimate
        expected_median = estimated_area_per_piece
        
        # Use a weighted average of our statistical measures and expected area
        confidence_in_expected = min(len(initial_filtered) / (expected_piece_count * 2), 0.8)
        adjusted_median = (median_area * (1 - confidence_in_expected) + 
                          expected_median * confidence_in_expected)
        
        # Update our median for final filtering
        median_area = adjusted_median
    
    # Define acceptance range based on statistics
    # Use a dynamic threshold based on the coefficient of variation
    cv_value = mad_area / (median_area + 1e-6)  # Coefficient of variation (using MAD)
    
    # If pieces are very consistent in size, use tighter bounds
    if cv_value < 0.2:
        deviation_factor = 2.0
    else:
        # If pieces vary a lot in size, use looser bounds
        deviation_factor = 3.0 + cv_value * 5.0  # Scale up for higher variation
    
    min_acceptable_area = median_area - deviation_factor * mad_area
    max_acceptable_area = median_area + deviation_factor * mad_area
    
    # Ensure min_area constraint is still respected
    min_acceptable_area = max(min_acceptable_area, min_area)
    
    # Two-pass filtering approach
    filtered_contours = []
    
    # First pass: strict filtering based on area
    for contour, metrics in initial_filtered:
        if min_acceptable_area <= metrics['area'] <= max_acceptable_area:
            filtered_contours.append(contour)
    
    # If we're getting too few contours, attempt a recovery pass with more lenient criteria
    if expected_piece_count is not None and len(filtered_contours) < expected_piece_count * 0.7:
        # Second pass: more lenient filtering
        for contour, metrics in initial_filtered:
            # Skip contours already accepted
            if contour in filtered_contours:
                continue
                
            # More lenient area range
            lenient_min = median_area - deviation_factor * 1.5 * mad_area
            lenient_max = median_area + deviation_factor * 1.5 * mad_area
            
            if lenient_min <= metrics['area'] <= lenient_max:
                # Additional validation for outliers: check if shape looks like a puzzle piece
                if validate_shape_as_puzzle_piece(contour):
                    filtered_contours.append(contour)
    
    # Log statistics
    logger.info(f"Contour filtering statistics:")
    logger.info(f"  Initial contours: {len(contours)}")
    logger.info(f"  After basic filtering: {len(initial_filtered)}")
    logger.info(f"  After statistical filtering: {len(filtered_contours)}")
    logger.info(f"  Median area: {median_area:.2f}, MAD: {mad_area:.2f}")
    logger.info(f"  Area acceptance range: {min_acceptable_area:.2f} to {max_acceptable_area:.2f}")
    
    # If we still don't have enough contours, use the initial filtered set
    if expected_piece_count is not None and len(filtered_contours) < expected_piece_count * 0.5:
        logger.warning(f"Low detection rate after statistical filtering. Using basic filtering.")
        return [c[0] for c in initial_filtered]
    
    return filtered_contours


def validate_shape_as_puzzle_piece(contour: np.ndarray) -> bool:
    """
    Validate if a contour's shape is consistent with a puzzle piece
    
    Args:
        contour: Input contour
    
    Returns:
        True if the contour is likely a puzzle piece
    """
    # Check if the contour has enough points for a puzzle piece
    if len(contour) < 20:
        return False
        
    # Approximate the contour to get key points
    perimeter = cv2.arcLength(contour, True)
    epsilon = 0.02 * perimeter
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    # Puzzle pieces typically have between 4 and 12 corners when approximated
    if len(approx) < 4 or len(approx) > 15:
        return False
    
    # Check the shape complexity using various metrics
    area = cv2.contourArea(contour)
    if area <= 0:
        return False
        
    # Calculate shape metrics
    compactness = perimeter**2 / (4 * np.pi * area)
    
    # Puzzle pieces typically have compactness between 3 and 8
    if not (2.0 <= compactness <= 12.0):
        return False
    
    # Check for convexity defects (tabs and pockets)
    hull = cv2.convexHull(contour, returnPoints=False)
    
    try:
        defects = cv2.convexityDefects(contour, hull)
        if defects is None:
            return False
            
        # Count significant defects
        significant_defects = 0
        for i in range(defects.shape[0]):
            _, _, _, depth = defects[i, 0]
            if depth > 300:  # Arbitrary threshold based on typical puzzle piece dimensions
                significant_defects += 1
        
        # Puzzle pieces typically have some significant convexity defects
        if significant_defects < 1:
            return False
    except:
        # If convexity defects calculation fails, be conservative
        return False
    
    return True


def approximate_polygon(contour: np.ndarray, 
                       epsilon_factor: float = 0.02) -> np.ndarray:
    """
    Approximate a contour with a polygon using adaptive epsilon
    
    Args:
        contour: Input contour
        epsilon_factor: Approximation accuracy factor (relative to perimeter)
    
    Returns:
        Approximated polygon vertices
    """
    perimeter = cv2.arcLength(contour, True)
    
    # Determine epsilon_factor based on contour complexity
    area = cv2.contourArea(contour)
    if area <= 0:
        return np.array([])
        
    compactness = perimeter**2 / (4 * np.pi * area)
    
    # For more complex shapes (higher compactness), use a smaller epsilon
    # for more accurate polygon approximation
    if compactness > 5.0:
        epsilon_factor = max(0.01, epsilon_factor * 0.7)
    elif compactness < 3.0:
        epsilon_factor = min(0.04, epsilon_factor * 1.3)
    
    epsilon = epsilon_factor * perimeter
    return cv2.approxPolyDP(contour, epsilon, True)


def calculate_contour_features(contour: np.ndarray) -> Dict[str, Any]:
    """
    Calculate comprehensive features of a contour
    
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
    
    # Additional shape features
    
    # Compactness (perimeter^2 / area)
    compactness = perimeter**2 / (4 * np.pi * area) if area > 0 else 0
    
    # Contour complexity (length of contour / perimeter of equivalent circle)
    complexity = perimeter / (2 * np.pi * equivalent_diameter / 2) if equivalent_diameter > 0 else 0
    
    # Ellipticity (ratio of major to minor axis lengths)
    if min_area_rect[1][0] > 0 and min_area_rect[1][1] > 0:
        major_axis = max(min_area_rect[1][0], min_area_rect[1][1])
        minor_axis = min(min_area_rect[1][0], min_area_rect[1][1])
        ellipticity = major_axis / minor_axis
    else:
        ellipticity = 1.0
    
    # Calculate convexity defects for tab/pocket analysis
    try:
        hull_indices = cv2.convexHull(contour, returnPoints=False)
        defects = cv2.convexityDefects(contour, hull_indices)
        
        if defects is not None:
            # Count significant defects and calculate their average depth
            significant_defects = []
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                depth = d / 256.0  # Convert to actual distance
                if depth > 5.0:  # Arbitrary threshold for significance
                    significant_defects.append(depth)
            
            defect_count = len(significant_defects)
            avg_defect_depth = np.mean(significant_defects) if significant_defects else 0
        else:
            defect_count = 0
            avg_defect_depth = 0
    except:
        defect_count = 0
        avg_defect_depth = 0
    
    # Return comprehensive feature dictionary
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
        'hu_moments': hu_moments,
        'compactness': compactness,
        'complexity': complexity,
        'ellipticity': ellipticity,
        'defect_count': defect_count,
        'avg_defect_depth': avg_defect_depth
    }


def enhanced_find_corners(contour: np.ndarray,
                          approx_epsilon: float = 0.02,
                          use_adaptive_epsilon: bool = True,
                          corner_refinement: bool = True) -> np.ndarray:
    """
    Enhanced corner detection in puzzle pieces
    
    Args:
        contour: Input contour
        approx_epsilon: Polygon approximation accuracy factor
        use_adaptive_epsilon: Whether to adapt epsilon based on contour complexity
        corner_refinement: Whether to refine corner positions
    
    Returns:
        Array of corner points
    """
    if len(contour) < 6:  # Needs minimum points
        return np.array([])
    
    # Calculate adaptive epsilon if requested
    if use_adaptive_epsilon:
        # Compute contour features to determine complexity
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        compactness = perimeter**2 / (4 * np.pi * area) if area > 0 else 0
        
        # Adjust epsilon based on compactness (more complex = smaller epsilon)
        if compactness > 6.0:
            approx_epsilon = max(0.01, approx_epsilon * 0.7)  # More precise for complex shapes
        elif compactness < 4.0:
            approx_epsilon = min(0.04, approx_epsilon * 1.3)  # Less precise for simple shapes
    
    # Initial corner detection using Douglas-Peucker algorithm
    epsilon = approx_epsilon * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    # For puzzles, we commonly expect 4 or more corners
    if 4 <= len(approx) <= 12:
        # If reasonable number of corners found and no refinement needed, use as is
        if not corner_refinement:
            return approx.reshape(-1, 2)
    
    # Advanced corner detection using multiple approaches
    corners = []
    
    # Method 1: Use convexity defects to identify potential corners
    try:
        hull_indices = cv2.convexHull(contour, returnPoints=False)
        defects = cv2.convexityDefects(contour, hull_indices)
        
        if defects is not None:
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                depth = d / 256.0  # Convert to actual distance
                
                if depth > 10.0:  # Consider only significant defects
                    # Both start and end points are potential corners
                    start_point = tuple(contour[s][0])
                    end_point = tuple(contour[e][0])
                    corners.append(start_point)
                    corners.append(end_point)
    except:
        pass  # Fallback to other methods if convexity defects fails
    
    # Method 2: Use Shi-Tomasi corner detector
    if len(corners) < 4:
        # Create a mask image for the contour
        mask = np.zeros((np.max(contour[:, :, 1]) + 10, np.max(contour[:, :, 0]) + 10), dtype=np.uint8)
        cv2.drawContours(mask, [contour], 0, 255, 1)
        
        # Apply Shi-Tomasi detector
        corners_shi = cv2.goodFeaturesToTrack(mask, maxCorners=20, qualityLevel=0.01, minDistance=10)
        
        if corners_shi is not None:
            for corner in corners_shi:
                x, y = corner.ravel()
                corners.append((int(x), int(y)))
    
    # If still not enough corners, fall back to approximation with relaxed epsilon
    if len(corners) < 4:
        larger_epsilon = approx_epsilon * 1.5
        epsilon = larger_epsilon * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        corners = [tuple(p[0]) for p in approx]
    
    # Remove duplicate or very close corners
    filtered_corners = []
    for corner in corners:
        if not any(distance.euclidean(corner, fc) < 10 for fc in filtered_corners):
            filtered_corners.append(corner)
    
    # If too many corners, keep the most significant ones
    if len(filtered_corners) > 12:
        # Use a more aggressive filtering approach
        filtered_corners.sort(key=lambda p: _corner_significance(p, contour))
        filtered_corners = filtered_corners[:12]  # Keep top 12 corners
    
    # Sort corners clockwise
    if len(filtered_corners) >= 3:
        # Calculate centroid
        cx = sum(x for x, y in filtered_corners) / len(filtered_corners)
        cy = sum(y for x, y in filtered_corners) / len(filtered_corners)
        
        # Sort corners by angle from centroid
        filtered_corners.sort(key=lambda pt: math.atan2(pt[1] - cy, pt[0] - cx))
    
    return np.array(filtered_corners)


def _corner_significance(point: Tuple[int, int], contour: np.ndarray) -> float:
    """
    Calculate the significance of a corner point based on its properties
    
    Args:
        point: Corner point (x, y)
        contour: Original contour
    
    Returns:
        Significance score (higher is more significant)
    """
    # Convert point to numpy array
    point_array = np.array(point)
    
    # Find the closest point in the contour
    distances = np.sqrt(np.sum((contour[:, 0, :] - point_array) ** 2, axis=1))
    closest_idx = np.argmin(distances)
    
    # Calculate angle at this point using neighboring points
    n = len(contour)
    prev_idx = (closest_idx - 5) % n  # Go back 5 points for stability
    next_idx = (closest_idx + 5) % n  # Go forward 5 points
    
    prev_point = contour[prev_idx][0]
    next_point = contour[next_idx][0]
    current_point = contour[closest_idx][0]
    
    # Calculate vectors
    v1 = prev_point - current_point
    v2 = next_point - current_point
    
    # Calculate angle between vectors
    dot_product = np.dot(v1, v2)
    norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
    
    if norm_product < 1e-6:  # Avoid division by zero
        return 0
        
    cos_angle = dot_product / norm_product
    cos_angle = max(-1, min(1, cos_angle))  # Ensure in range [-1, 1]
    angle = np.arccos(cos_angle)
    
    # Convert to degrees
    angle_deg = angle * 180 / np.pi
    
    # Corners have angles far from 180 degrees
    angle_significance = abs(180 - angle_deg)
    
    # Also consider the convexity/concavity
    # Convex corners (angle < 180) are more likely puzzle piece corners
    convexity_bonus = max(0, 180 - angle_deg) * 0.5
    
    return angle_significance + convexity_bonus


def find_corners(contour: np.ndarray,
                 approx_epsilon: float = 0.02,
                 angle_threshold: float = 45.0,
                 min_corner_dist: float = 10.0) -> np.ndarray:
    """
    Backwards-compatible corner detection with enhanced algorithm
    
    Args:
        contour: Input contour
        approx_epsilon: Polygon approximation accuracy factor
        angle_threshold: Maximum deviation from 90° for corner angles
        min_corner_dist: Minimum distance between detected corners
    
    Returns:
        Array of corner points
    """
    # Use the enhanced algorithm with compatible interface
    return enhanced_find_corners(
        contour, 
        approx_epsilon=approx_epsilon,
        use_adaptive_epsilon=True,
        corner_refinement=True
    )


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


def enhanced_classify_border(border: np.ndarray,
                             complexity_threshold: float = 1.2,
                             deviation_threshold: float = 10.0,
                             use_adaptive_thresholds: bool = True) -> str:
    """
    Enhanced border classification with adaptive thresholds
    
    Args:
        border: Border segment points
        complexity_threshold: Threshold for path complexity
        deviation_threshold: Threshold for maximum deviation
        use_adaptive_thresholds: Whether to use adaptive thresholds
    
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
    
    # Calculate deviations from straight line
    deviations = []
    for p in border:
        # Calculate perpendicular distance from point to line
        dev = np.abs(np.cross(end - start, start - p)) / direct_dist
        deviations.append(dev)
    
    max_deviation = max(deviations)
    
    # Adapt thresholds based on border length if enabled
    if use_adaptive_thresholds:
        # Longer borders may have more natural deviation even if straight
        border_length_factor = min(1.5, direct_dist / 100.0)
        
        # Adjust thresholds based on border length
        adaptive_complexity_threshold = complexity_threshold * border_length_factor
        adaptive_deviation_threshold = deviation_threshold * border_length_factor
    else:
        adaptive_complexity_threshold = complexity_threshold
        adaptive_deviation_threshold = deviation_threshold
    
    # Classify based on adjusted thresholds
    if complexity <= adaptive_complexity_threshold and max_deviation <= adaptive_deviation_threshold:
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


def classify_border(border: np.ndarray,
                  complexity_threshold: float = 1.2,
                  deviation_threshold: float = 10.0) -> str:
    """
    Classify a border segment as straight, tab, or pocket
    Backwards-compatible wrapper around enhanced version
    
    Args:
        border: Border segment points
        complexity_threshold: Threshold for path complexity
        deviation_threshold: Threshold for maximum deviation
    
    Returns:
        Border type: "straight", "tab", or "pocket"
    """
    return enhanced_classify_border(
        border,
        complexity_threshold=complexity_threshold,
        deviation_threshold=deviation_threshold,
        use_adaptive_thresholds=True
    )


def cluster_contours(contours: List[np.ndarray],
                     features: Optional[List[Dict[str, Any]]] = None) -> Dict[str, List[int]]:
    """
    Cluster contours based on their features
    
    Args:
        contours: List of contours
        features: List of pre-calculated features (optional)
    
    Returns:
        Dictionary of cluster_id -> list of contour indices
    """
    if len(contours) <= 1:
        return {'0': list(range(len(contours)))}
    
    # Calculate features if not provided
    if features is None:
        features = [calculate_contour_features(c) for c in contours]
    
    # Extract key metrics for clustering
    area_values = np.array([f['area'] for f in features])
    compactness_values = np.array([f['compactness'] for f in features])
    solidity_values = np.array([f['solidity'] for f in features])
    
    # Normalize features
    area_norm = (area_values - np.mean(area_values)) / (np.std(area_values) + 1e-10)
    compactness_norm = (compactness_values - np.mean(compactness_values)) / (np.std(compactness_values) + 1e-10)
    solidity_norm = (solidity_values - np.mean(solidity_values)) / (np.std(solidity_values) + 1e-10)
    
    # Combine features
    feature_matrix = np.column_stack((
        area_norm,           # Area is most important
        compactness_norm,    # Shape complexity
        solidity_norm        # How "filled" the shape is
    ))
    
    # Simple clustering approach
    # 1. Determine number of clusters (default guess is 1-3)
    n_samples = len(contours)
    if n_samples < 10:
        n_clusters = 1
    else:
        n_clusters = min(3, n_samples // 10 + 1)
    
    # 2. Use KMeans clustering from scratch (to avoid sklearn dependency)
    clusters = _simple_kmeans(feature_matrix, n_clusters)
    
    # 3. Organize results
    cluster_dict = {}
    for i, cluster_id in enumerate(clusters):
        if str(cluster_id) not in cluster_dict:
            cluster_dict[str(cluster_id)] = []
        cluster_dict[str(cluster_id)].append(i)
    
    # Identify the largest cluster (likely puzzle pieces)
    largest_cluster = max(cluster_dict.items(), key=lambda x: np.mean([area_values[i] for i in x[1]]))
    largest_cluster_id = largest_cluster[0]
    
    # Log results
    logger.info(f"Contour clustering results:")
    for cluster_id, indices in cluster_dict.items():
        mean_area = np.mean([area_values[i] for i in indices])
        logger.info(f"  Cluster {cluster_id}: {len(indices)} contours, mean area: {mean_area:.2f}")
    
    logger.info(f"Largest cluster (likely puzzle pieces): {largest_cluster_id}")
    
    return cluster_dict


def _simple_kmeans(data: np.ndarray, k: int, max_iters: int = 100) -> np.ndarray:
    """
    Simple K-means clustering implementation
    
    Args:
        data: Feature matrix (n_samples, n_features)
        k: Number of clusters
        max_iters: Maximum number of iterations
    
    Returns:
        Array of cluster indices for each sample
    """
    n_samples, n_features = data.shape
    
    # Initialize centroids randomly
    idx = np.random.choice(n_samples, k, replace=False)
    centroids = data[idx, :]
    
    # Initialize cluster assignments
    clusters = np.zeros(n_samples, dtype=int)
    
    for _ in range(max_iters):
        # Assign samples to closest centroid
        old_clusters = clusters.copy()
        
        for i in range(n_samples):
            # Calculate distances to all centroids
            distances = np.sqrt(np.sum((centroids - data[i]) ** 2, axis=1))
            clusters[i] = np.argmin(distances)
        
        # Check for convergence
        if np.all(old_clusters == clusters):
            break
        
        # Update centroids
        for j in range(k):
            cluster_points = data[clusters == j]
            if len(cluster_points) > 0:
                centroids[j] = np.mean(cluster_points, axis=0)
    
    return clusters