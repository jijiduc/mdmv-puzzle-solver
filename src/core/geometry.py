"""Geometric calculations and transformations for puzzle pieces."""

import math
import numpy as np
from typing import List, Tuple, Union, Optional
from src.features.shape_analysis import classify_edge_shape



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


def count_consecutive_deviations(deviations: List[float], threshold: float, direction: str) -> int:
    """Count the maximum consecutive deviations in a given direction.
    
    Args:
        deviations: List of signed deviation values
        threshold: Minimum deviation magnitude to consider significant
        direction: 'positive' for extrusions, 'negative' for intrusions
        
    Returns:
        Maximum consecutive deviation count
    """
    max_consecutive = 0
    current_consecutive = 0
    
    for deviation in deviations:
        if direction == 'positive' and deviation > threshold:
            current_consecutive += 1
            max_consecutive = max(max_consecutive, current_consecutive)
        elif direction == 'negative' and deviation < -threshold:
            current_consecutive += 1
            max_consecutive = max(max_consecutive, current_consecutive)
        else:
            current_consecutive = 0
    
    return max_consecutive


def determine_outward_direction_robust(edge_points: List[Tuple[int, int]], corner1: Tuple[int, int], 
                                      corner2: Tuple[int, int], centroid: Tuple[int, int], 
                                      normal_vec: Tuple[float, float], 
                                      piece_idx: int = -1, edge_idx: int = -1) -> Tuple[float, float]:
    """Determine the correct outward direction using multiple geometric methods.
    
    Args:
        edge_points: List of edge point coordinates
        corner1: First corner coordinates
        corner2: Second corner coordinates
        centroid: Centroid coordinates
        normal_vec: Unit normal vector to the reference line
        
    Returns:
        Tuple of outward normal vector (x, y)
    """
    if len(edge_points) < 3:
        # Fallback to simple centroid method for very short edges
        x1, y1 = corner1
        x2, y2 = corner2
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        centroid_x, centroid_y = centroid
        centroid_to_mid = (mid_x - centroid_x, mid_y - centroid_y)
        normal_direction = centroid_to_mid[0]*normal_vec[0] + centroid_to_mid[1]*normal_vec[1]
        return normal_vec if normal_direction > 0 else (-normal_vec[0], -normal_vec[1])
    
    # Method 1: Direct edge analysis - use the actual edge shape to determine direction
    edge_points_np = np.array(edge_points)
    x1, y1 = corner1
    x2, y2 = corner2
    
    # Calculate the reference line
    line_vec = (x2-x1, y2-y1)
    line_length = math.sqrt(line_vec[0]**2 + line_vec[1]**2)
    
    if line_length < 1:
        return normal_vec
    
    # Find the midpoint of the edge (actual edge, not reference line)
    edge_midpoint = np.mean(edge_points_np, axis=0)
    line_midpoint = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
    
    # Vector from reference line midpoint to actual edge midpoint
    line_to_edge_vec = edge_midpoint - line_midpoint
    line_to_edge_magnitude = np.linalg.norm(line_to_edge_vec)
    
    # Debug output for the problematic edge (disabled)
    # is_debug_edge = (piece_idx == 1 and edge_idx == 2)
    
    # If the edge midpoint is significantly displaced from the reference line
    if line_to_edge_magnitude > 2.0:  # More than 2 pixels displacement
        line_to_edge_unit = line_to_edge_vec / line_to_edge_magnitude
        
        # The outward normal should align with the direction from line to edge
        dot_positive = np.dot(line_to_edge_unit, normal_vec)
        dot_negative = np.dot(line_to_edge_unit, (-normal_vec[0], -normal_vec[1]))
        
        # Choose the direction that better aligns with the actual displacement
        if abs(dot_positive) > abs(dot_negative):
            result = normal_vec if dot_positive > 0 else (-normal_vec[0], -normal_vec[1])
            return result
        else:
            result = (-normal_vec[0], -normal_vec[1]) if dot_negative > 0 else normal_vec
            return result
    
    # Method 2: Statistical deviation analysis
    # Calculate all deviations and use the predominant direction
    total_deviation_vec = np.array([0.0, 0.0])
    valid_points = 0
    
    for x, y in edge_points:
        # Vector from corner1 to point
        point_vec = (x-x1, y-y1)
        
        # Project onto line
        line_dot = (point_vec[0]*line_vec[0] + point_vec[1]*line_vec[1]) / line_length
        proj_x = x1 + line_dot * line_vec[0] / line_length
        proj_y = y1 + line_dot * line_vec[1] / line_length
        
        # Deviation vector from projection to actual point
        dev_vec = np.array([x - proj_x, y - proj_y])
        dev_magnitude = np.linalg.norm(dev_vec)
        
        # Only consider significant deviations to avoid noise
        if dev_magnitude > 1.0:
            total_deviation_vec += dev_vec
            valid_points += 1
    
    if valid_points > 0:
        avg_deviation_direction = total_deviation_vec / valid_points
        avg_deviation_magnitude = np.linalg.norm(avg_deviation_direction)
        
        if avg_deviation_magnitude > 0.5:  # Significant average deviation
            # Normalize the average deviation direction
            avg_deviation_unit = avg_deviation_direction / avg_deviation_magnitude
            
            # The outward normal should align with the average deviation direction
            dot_positive = np.dot(avg_deviation_unit, normal_vec)
            dot_negative = np.dot(avg_deviation_unit, (-normal_vec[0], -normal_vec[1]))
            
            # Choose the direction that better aligns with the average deviation
            return normal_vec if dot_positive > dot_negative else (-normal_vec[0], -normal_vec[1])
    
    # Method 3: Piece boundary analysis
    # Use the overall shape of the edge to determine inside vs outside
    edge_centroid = np.mean(edge_points_np, axis=0)
    
    # Vector from piece centroid to edge centroid
    piece_to_edge = edge_centroid - np.array([centroid[0], centroid[1]])
    piece_to_edge_magnitude = np.linalg.norm(piece_to_edge)
    
    if piece_to_edge_magnitude > 0:
        piece_to_edge_unit = piece_to_edge / piece_to_edge_magnitude
        
        # The outward direction should generally align with the direction from piece center to edge
        dot_positive = np.dot(piece_to_edge_unit, normal_vec)
        dot_negative = np.dot(piece_to_edge_unit, (-normal_vec[0], -normal_vec[1]))
        
        if abs(dot_positive) > abs(dot_negative):
            return normal_vec if dot_positive > 0 else (-normal_vec[0], -normal_vec[1])
        else:
            return (-normal_vec[0], -normal_vec[1]) if dot_negative > 0 else normal_vec
    
    # Method 4: Fallback to improved centroid method
    # Use the midpoint of the edge instead of the line midpoint
    edge_midpoint = np.mean(edge_points_np, axis=0)
    line_midpoint = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
    
    # Vector from line midpoint to edge midpoint
    mid_to_edge = edge_midpoint - line_midpoint
    mid_to_edge_magnitude = np.linalg.norm(mid_to_edge)
    
    if mid_to_edge_magnitude > 0:
        mid_to_edge_unit = mid_to_edge / mid_to_edge_magnitude
        
        # The outward direction should align with the direction from line to actual edge
        dot_positive = np.dot(mid_to_edge_unit, normal_vec)
        dot_negative = np.dot(mid_to_edge_unit, (-normal_vec[0], -normal_vec[1]))
        
        return normal_vec if dot_positive > 0 else (-normal_vec[0], -normal_vec[1])
    
    # Final fallback
    return normal_vec


def calculate_edge_curvature(edge_points: List[Tuple[int, int]]) -> float:
    """Calculate the average curvature of an edge.
    
    Args:
        edge_points: List of edge point coordinates
        
    Returns:
        Average curvature value (higher = more curved)
    """
    if len(edge_points) < 3:
        return 0.0
    
    curvatures = []
    points = np.array(edge_points)
    
    # Calculate curvature at each interior point
    for i in range(1, len(points) - 1):
        p1, p2, p3 = points[i-1], points[i], points[i+1]
        
        # Vectors
        v1 = p2 - p1
        v2 = p3 - p2
        
        # Lengths
        len1 = np.linalg.norm(v1)
        len2 = np.linalg.norm(v2)
        
        if len1 > 0 and len2 > 0:
            # Normalize vectors
            v1_norm = v1 / len1
            v2_norm = v2 / len2
            
            # Calculate angle change (curvature indicator)
            dot_product = np.dot(v1_norm, v2_norm)
            dot_product = np.clip(dot_product, -1.0, 1.0)
            angle_change = abs(np.arccos(dot_product))
            
            # Curvature = angle change / average segment length
            avg_length = (len1 + len2) / 2
            curvature = angle_change / avg_length if avg_length > 0 else 0
            curvatures.append(curvature)
    
    return np.mean(curvatures) if curvatures else 0.0


def classify_edge(edge_points: List[Tuple[int, int]], corner1: Tuple[int, int], 
                 corner2: Tuple[int, int], centroid: Tuple[int, int], 
                 piece_idx: int = -1, edge_idx: int = -1) -> Tuple[str, float, Optional[str], float]:
    """Classify an edge as flat, convex, or concave with sub-type classification.
    
    Args:
        edge_points: List of edge point coordinates
        corner1: First corner coordinates
        corner2: Second corner coordinates
        centroid: Centroid coordinates
        
    Returns:
        Tuple of (primary_type, max_deviation, sub_type, confidence)
        - primary_type: "flat", "convex", or "concave"
        - max_deviation: Maximum deviation from reference line
        - sub_type: "symmetric", "asymmetric", or None
        - confidence: Classification confidence (0-1)
    """
    if len(edge_points) == 0 or len(edge_points) < 5:
        return "unknown", 0, None, 0.0
    
    # Convert edge points to numpy array
    edge_points_np = np.array(edge_points)
    
    # Create reference line from corner1 to corner2
    reference_line = (np.array(corner1), np.array(corner2))
    
    # Use the new shape-based classification
    primary_type, sub_type, confidence = classify_edge_shape(edge_points_np, reference_line)
    
    # Calculate max deviation for compatibility with existing code
    x1, y1 = corner1
    x2, y2 = corner2
    
    # Line vector
    line_vec = (x2-x1, y2-y1)
    line_length = math.sqrt(line_vec[0]**2 + line_vec[1]**2)
    
    if line_length < 1:
        return primary_type, 0, sub_type, confidence
    
    # Normal vector
    normal_vec = (-line_vec[1]/line_length, line_vec[0]/line_length)
    
    # Use robust method to determine correct outward direction
    outward_normal = determine_outward_direction_robust(edge_points, corner1, corner2, centroid, normal_vec, 
                                                       piece_idx, edge_idx)
    
    # Calculate deviations for max_deviation value
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
            signed_deviation = sign * deviation
            deviations.append(signed_deviation)
    
    # Calculate max deviation based on edge type
    if deviations:
        if primary_type == "convex":
            max_deviation = max(deviations)
        elif primary_type == "concave":
            max_deviation = min(deviations)
        else:  # flat
            max_deviation = sum(deviations) / len(deviations)
    else:
        max_deviation = 0
    
    return primary_type, max_deviation, sub_type, confidence


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

