"""Geometric calculations and transformations for puzzle pieces."""

import math
import numpy as np
from typing import List, Tuple, Union


def extract_edge_between_corners(corners: List[Tuple[int, int]], corner_idx1: int, corner_idx2: int, 
                                edge_coords: np.ndarray, centroid: Tuple[int, int]) -> List[Tuple[int, int]]:
    """Extract edge points between two corners.
    
    Args:
        corners: List of corner coordinates
        corner_idx1: Index of first corner
        corner_idx2: Index of second corner
        edge_coords: Array of edge coordinates
        centroid: Centroid coordinates of the piece
        
    Returns:
        List of edge points between the specified corners
    """
    corner1 = corners[corner_idx1]
    corner2 = corners[corner_idx2]
    centroid_x, centroid_y = centroid
    
    if len(edge_coords) == 0:
        return []
    
    # Calculate angles of corners relative to centroid
    angle1 = math.atan2(corner1[1] - centroid_y, corner1[0] - centroid_x)
    angle2 = math.atan2(corner2[1] - centroid_y, corner2[0] - centroid_x)
    
    # Ensure angle2 > angle1 for range checking
    if angle2 < angle1:
        angle2 += 2 * math.pi
    
    # Calculate angles for all points
    all_angles = np.array([math.atan2(y - centroid_y, x - centroid_x) for x, y in edge_coords])
    all_angles_normalized = all_angles.copy()
    all_angles_normalized[all_angles_normalized < angle1] += 2 * math.pi
    
    # Mask for points in angular range
    angle_mask = (all_angles_normalized >= angle1) & (all_angles_normalized <= angle2)
    filtered_points = edge_coords[angle_mask]
    
    # If no filtered points, return empty list
    if len(filtered_points) == 0:
        return []
    
    # Sort by angle
    sorted_indices = np.argsort([math.atan2(y - centroid_y, x - centroid_x) for x, y in filtered_points])
    sorted_points = filtered_points[sorted_indices]
    
    return sorted_points.tolist()


def classify_edge(edge_points: List[Tuple[int, int]], corner1: Tuple[int, int], 
                 corner2: Tuple[int, int], centroid: Tuple[int, int]) -> Tuple[str, float]:
    """Classify an edge as straight, intrusion, or extrusion.
    
    Args:
        edge_points: List of edge point coordinates
        corner1: First corner coordinates
        corner2: Second corner coordinates
        centroid: Centroid coordinates
        
    Returns:
        Tuple of (edge_type, max_deviation)
    """
    if len(edge_points) == 0 or len(edge_points) < 5:
        return "unknown", 0
    
    # Create straight line between corners
    x1, y1 = corner1
    x2, y2 = corner2
    centroid_x, centroid_y = centroid
    
    # Line vector
    line_vec = (x2-x1, y2-y1)
    line_length = math.sqrt(line_vec[0]**2 + line_vec[1]**2)
    if line_length < 1:
        return "unknown", 0
    
    # Normal vector
    normal_vec = (-line_vec[1]/line_length, line_vec[0]/line_length)
    
    # Vector from centroid to line midpoint
    mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
    centroid_to_mid = (mid_x - centroid_x, mid_y - centroid_y)
    
    # Direction of normal vector (inward or outward)
    normal_direction = centroid_to_mid[0]*normal_vec[0] + centroid_to_mid[1]*normal_vec[1]
    
    # Ensure normal vector points outward
    outward_normal = normal_vec if normal_direction > 0 else (-normal_vec[0], -normal_vec[1])
    
    # Calculate deviations for all edge points
    deviations = []
    for x, y in edge_points:
        # Vector from first corner to point
        point_vec = (x-x1, y-y1)
        
        # Project point vector onto line vector
        if line_length > 0:
            line_dot = (point_vec[0]*line_vec[0] + point_vec[1]*line_vec[1]) / line_length
            
            # Projected point coordinates
            proj_x = x1 + line_dot * line_vec[0] / line_length
            proj_y = y1 + line_dot * line_vec[1] / line_length
            
            # Deviation vector
            dev_vec = (x-proj_x, y-proj_y)
            
            # Deviation magnitude
            deviation = math.sqrt(dev_vec[0]**2 + dev_vec[1]**2)
            
            # Deviation sign (positive if in outward normal direction)
            sign = 1 if (dev_vec[0]*outward_normal[0] + dev_vec[1]*outward_normal[1]) > 0 else -1
            
            # Signed deviation
            deviations.append(sign * deviation)
    
    # Calculate adaptive threshold
    straight_threshold = max(5, line_length * 0.05)  # At least 5px or 5% of length
    
    # Classification
    if deviations:
        # Statistical calculations
        mean_deviation = sum(deviations) / len(deviations)
        abs_deviations = [abs(d) for d in deviations]
        max_abs_deviation = max(abs_deviations)
        
        # Count significant deviations
        significant_positive = sum(1 for d in deviations if d > straight_threshold)
        significant_negative = sum(1 for d in deviations if d < -straight_threshold)
        
        # Portion of edge with significant deviations
        portion_significant = (significant_positive + significant_negative) / len(deviations)
        
        # Classification logic
        if max_abs_deviation < straight_threshold or portion_significant < 0.2:
            edge_type = "straight"
            max_deviation = mean_deviation
        elif abs(mean_deviation) < straight_threshold * 0.5:
            # If mean is close to zero but max is high
            if significant_positive > significant_negative * 2:
                edge_type = "extrusion"
                max_deviation = max([d for d in deviations if d > 0], default=0)
            elif significant_negative > significant_positive * 2:
                edge_type = "intrusion"
                max_deviation = min([d for d in deviations if d < 0], default=0)
            else:
                edge_type = "straight"  # Balanced deviations
                max_deviation = mean_deviation
        elif mean_deviation > 0:
            edge_type = "extrusion"
            max_deviation = max(deviations)
        else:
            edge_type = "intrusion"
            max_deviation = min(deviations)
    else:
        edge_type = "unknown"
        max_deviation = 0
    
    return edge_type, max_deviation


def sort_edge_points(edge_points: List[Tuple[int, int]], corner1: Tuple[int, int], 
                    corner2: Tuple[int, int]) -> List[Tuple[int, int]]:
    """Sort edge points from one corner to another.
    
    Args:
        edge_points: List of (x, y) coordinates
        corner1: Start corner coordinates
        corner2: End corner coordinates
        
    Returns:
        Sorted list of edge points
    """
    if len(edge_points) < 2:
        return edge_points
    
    # Create vector from corner1 to corner2
    corner_vec = (corner2[0] - corner1[0], corner2[1] - corner1[1])
    corner_length = np.sqrt(corner_vec[0]**2 + corner_vec[1]**2)
    
    if corner_length == 0:
        return edge_points
    
    # Project each point onto line connecting corners
    projections = []
    for x, y in edge_points:
        # Vector from corner1 to point
        point_vec = (x - corner1[0], y - corner1[1])
        # Dot product divided by line length = distance along line
        proj = (point_vec[0]*corner_vec[0] + point_vec[1]*corner_vec[1]) / corner_length
        projections.append((proj, (x, y)))
    
    # Sort by projection
    sorted_points = [p[1] for p in sorted(projections)]
    return sorted_points


def calculate_edge_straightness(edge_points: List[Tuple[int, int]]) -> float:
    """Calculate how straight an edge is.
    
    Args:
        edge_points: List of edge coordinates
        
    Returns:
        Straightness score (0 = curved, 1 = straight)
    """
    if len(edge_points) < 3:
        return 1.0
    
    # Convert to numpy array for easier calculation
    points = np.array(edge_points)
    
    # Calculate distance between first and last point
    total_distance = np.linalg.norm(points[-1] - points[0])
    
    if total_distance == 0:
        return 1.0
    
    # Calculate sum of distances between consecutive points
    path_distance = 0
    for i in range(len(points) - 1):
        path_distance += np.linalg.norm(points[i+1] - points[i])
    
    # Straightness ratio
    if path_distance == 0:
        return 1.0
    
    straightness = total_distance / path_distance
    return min(1.0, straightness)


def validate_corner_angle(edge1_points: List[Tuple[int, int]], 
                         edge2_points: List[Tuple[int, int]]) -> bool:
    """Validate if two edges form a valid corner angle.
    
    Args:
        edge1_points: Points of first edge
        edge2_points: Points of second edge
        
    Returns:
        True if angle is valid for a puzzle piece corner
    """
    if len(edge1_points) < 2 or len(edge2_points) < 2:
        return False
    
    # Get direction vectors of the edges
    edge1_vec = np.array(edge1_points[-1]) - np.array(edge1_points[0])
    edge2_vec = np.array(edge2_points[-1]) - np.array(edge2_points[0])
    
    # Calculate angle between edges
    cos_angle = np.dot(edge1_vec, edge2_vec) / (np.linalg.norm(edge1_vec) * np.linalg.norm(edge2_vec))
    cos_angle = np.clip(cos_angle, -1, 1)
    angle = np.arccos(cos_angle)
    
    # Valid puzzle piece corners are typically between 45° and 135°
    return np.pi/4 <= angle <= 3*np.pi/4