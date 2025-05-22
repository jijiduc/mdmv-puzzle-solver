"""Geometric calculations and transformations for puzzle pieces."""

import math
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import List, Tuple, Union


def debug_edge_classification(edge_points: List[Tuple[int, int]], corner1: Tuple[int, int], 
                             corner2: Tuple[int, int], centroid: Tuple[int, int], 
                             piece_idx: int, edge_idx: int, output_dir: str = "debug/05_geometry") -> None:
    """Create detailed debug visualization for edge classification algorithm.
    
    Args:
        edge_points: List of edge point coordinates
        corner1: First corner coordinates
        corner2: Second corner coordinates
        centroid: Centroid coordinates
        piece_idx: Piece index for naming
        edge_idx: Edge index for naming
        output_dir: Output directory for debug images
    """
    if len(edge_points) < 5:
        return
        
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Edge Classification Debug - Piece {piece_idx}, Edge {edge_idx}', fontsize=16)
    
    # Convert points to numpy array
    edge_points_np = np.array(edge_points)
    x1, y1 = corner1
    x2, y2 = corner2
    centroid_x, centroid_y = centroid
    
    # Calculate line vector and normal
    line_vec = (x2-x1, y2-y1)
    line_length = math.sqrt(line_vec[0]**2 + line_vec[1]**2)
    if line_length < 1:
        return
        
    normal_vec = (-line_vec[1]/line_length, line_vec[0]/line_length)
    
    # Calculate direction using robust method
    outward_normal = determine_outward_direction_robust(edge_points, corner1, corner2, centroid, normal_vec, 
                                                       piece_idx, edge_idx)
    
    # For comparison, also calculate old method
    mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
    centroid_to_mid = (mid_x - centroid_x, mid_y - centroid_y)
    normal_direction = centroid_to_mid[0]*normal_vec[0] + centroid_to_mid[1]*normal_vec[1]
    old_outward_normal = normal_vec if normal_direction > 0 else (-normal_vec[0], -normal_vec[1])
    
    # Calculate deviations
    deviations = []
    projected_points = []
    for x, y in edge_points:
        point_vec = (x-x1, y-y1)
        if line_length > 0:
            line_dot = (point_vec[0]*line_vec[0] + point_vec[1]*line_vec[1]) / line_length
            proj_x = x1 + line_dot * line_vec[0] / line_length
            proj_y = y1 + line_dot * line_vec[1] / line_length
            projected_points.append((proj_x, proj_y))
            
            dev_vec = (x-proj_x, y-proj_y)
            deviation = math.sqrt(dev_vec[0]**2 + dev_vec[1]**2)
            sign = 1 if (dev_vec[0]*outward_normal[0] + dev_vec[1]*outward_normal[1]) > 0 else -1
            deviations.append(sign * deviation)
    
    projected_points_np = np.array(projected_points)
    
    # Plot 1: Basic geometry
    ax1.set_title('Basic Edge Geometry')
    ax1.plot(edge_points_np[:, 0], edge_points_np[:, 1], 'b-', linewidth=2, label='Actual Edge')
    ax1.plot([x1, x2], [y1, y2], 'r-', linewidth=3, label='Reference Line')
    ax1.plot([x1, x2], [y1, y2], 'ro', markersize=8)
    ax1.plot(centroid_x, centroid_y, 'g*', markersize=15, label='Centroid')
    ax1.plot(projected_points_np[:, 0], projected_points_np[:, 1], 'r--', alpha=0.7, label='Projected Points')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # Plot 2: Normal vectors and directions
    ax2.set_title('Normal Vector Analysis')
    ax2.plot(edge_points_np[:, 0], edge_points_np[:, 1], 'b-', linewidth=2, label='Actual Edge')
    ax2.plot([x1, x2], [y1, y2], 'r-', linewidth=3, label='Reference Line')
    ax2.plot(centroid_x, centroid_y, 'g*', markersize=15, label='Centroid')
    
    # Draw normal vectors
    scale = line_length * 0.3
    ax2.arrow(mid_x, mid_y, normal_vec[0]*scale, normal_vec[1]*scale, 
              head_width=scale*0.1, head_length=scale*0.1, fc='orange', ec='orange', label='Normal Vector')
    ax2.arrow(mid_x, mid_y, old_outward_normal[0]*scale, old_outward_normal[1]*scale, 
              head_width=scale*0.1, head_length=scale*0.1, fc='red', ec='red', label='Old Outward Normal')
    ax2.arrow(mid_x, mid_y, outward_normal[0]*scale, outward_normal[1]*scale, 
              head_width=scale*0.1, head_length=scale*0.1, fc='purple', ec='purple', label='NEW Outward Normal')
    
    # Draw centroid to midpoint vector
    ax2.arrow(centroid_x, centroid_y, centroid_to_mid[0], centroid_to_mid[1], 
              head_width=scale*0.05, head_length=scale*0.05, fc='green', ec='green', label='Centroid→Mid')
    
    ax2.text(mid_x + scale*0.5, mid_y, f'Old Normal Dir: {normal_direction:.2f}', fontsize=10)
    
    # Show if direction changed
    direction_changed = not (np.allclose(outward_normal, old_outward_normal, atol=0.01))
    ax2.text(mid_x + scale*0.5, mid_y - scale*0.3, f'Direction Changed: {direction_changed}', 
             fontsize=10, color='red' if direction_changed else 'green')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    # Plot 3: Deviation analysis
    ax3.set_title('Deviation Analysis')
    colors = ['red' if d < 0 else 'blue' for d in deviations]
    ax3.scatter(range(len(deviations)), deviations, c=colors, alpha=0.7)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax3.set_xlabel('Point Index')
    ax3.set_ylabel('Signed Deviation (pixels)')
    ax3.grid(True, alpha=0.3)
    
    # Add statistics
    mean_dev = np.mean(deviations) if deviations else 0
    max_dev = np.max(np.abs(deviations)) if deviations else 0
    pos_count = sum(1 for d in deviations if d > 0)
    neg_count = sum(1 for d in deviations if d < 0)
    
    ax3.text(0.02, 0.98, f'Mean: {mean_dev:.2f}\nMax: {max_dev:.2f}\nPos: {pos_count}\nNeg: {neg_count}', 
             transform=ax3.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))
    
    # Plot 4: Color-coded edge points
    ax4.set_title('Deviation Visualization on Edge')
    norm_deviations = np.array(deviations)
    max_abs_dev = np.max(np.abs(norm_deviations)) if len(norm_deviations) > 0 else 1
    norm_deviations = norm_deviations / max_abs_dev if max_abs_dev > 0 else norm_deviations
    
    scatter = ax4.scatter(edge_points_np[:, 0], edge_points_np[:, 1], 
                         c=norm_deviations, cmap='RdBu', s=50, alpha=0.8)
    ax4.plot([x1, x2], [y1, y2], 'k-', linewidth=3, alpha=0.5, label='Reference Line')
    ax4.plot([x1, x2], [y1, y2], 'ko', markersize=8)
    ax4.plot(centroid_x, centroid_y, 'g*', markersize=15, label='Centroid')
    
    plt.colorbar(scatter, ax=ax4, label='Normalized Deviation (Red=Negative, Blue=Positive)')
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.axis('equal')
    
    # Save the debug image
    os.makedirs(output_dir, exist_ok=True)
    debug_path = os.path.join(output_dir, f'edge_classification_debug_piece_{piece_idx}_edge_{edge_idx}.png')
    plt.tight_layout()
    plt.savefig(debug_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Edge classification debug saved: {debug_path}")


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
                 piece_idx: int = -1, edge_idx: int = -1) -> Tuple[str, float]:
    """Classify an edge as straight, intrusion, or extrusion using improved multi-scale analysis.
    
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
    
    # Debug visualization for specific piece and edge
    # if piece_idx == 1 and edge_idx == 2:  # Piece 1, Edge 3 (0-indexed as edge 2)
    #     debug_edge_classification(edge_points, corner1, corner2, centroid, piece_idx, edge_idx)
    
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
    
    # Use robust method to determine correct outward direction
    outward_normal = determine_outward_direction_robust(edge_points, corner1, corner2, centroid, normal_vec, 
                                                       piece_idx, edge_idx)
    
    # Debug the direction calculation
    # if piece_idx == 1 and edge_idx == 2:  # Piece 1, Edge 3 (0-indexed as edge 2)
    #     debug_edge_classification(edge_points, corner1, corner2, centroid, piece_idx, edge_idx)
    
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
            signed_deviation = sign * deviation
            deviations.append(signed_deviation)
            
    
    # Multi-scale threshold analysis with balanced sensitivity
    fine_threshold = max(3, line_length * 0.025)     # 2.5% for fine details (slightly more permissive)
    coarse_threshold = max(8, line_length * 0.08)    # 8% for major features
    straight_tolerance = max(4, line_length * 0.035) # 3.5% tolerance for nearly-straight edges
    
    # Classification
    if deviations:
        # Statistical calculations
        mean_deviation = sum(deviations) / len(deviations)
        abs_deviations = [abs(d) for d in deviations]
        max_abs_deviation = max(abs_deviations)
        
        
        # Pattern consistency analysis
        max_consecutive_positive = count_consecutive_deviations(deviations, fine_threshold, 'positive')
        max_consecutive_negative = count_consecutive_deviations(deviations, fine_threshold, 'negative')
        
        # Consistency ratios
        total_points = len(deviations)
        positive_consistency_ratio = max_consecutive_positive / total_points
        negative_consistency_ratio = max_consecutive_negative / total_points
        
        # Count significant deviations at both scales
        significant_positive_fine = sum(1 for d in deviations if d > fine_threshold)
        significant_negative_fine = sum(1 for d in deviations if d < -fine_threshold)
        significant_positive_coarse = sum(1 for d in deviations if d > coarse_threshold)
        significant_negative_coarse = sum(1 for d in deviations if d < -coarse_threshold)
        
        # Deviation strength (magnitude of significant deviations)
        positive_deviations = [d for d in deviations if d > fine_threshold]
        negative_deviations = [d for d in deviations if d < -fine_threshold]
        
        positive_strength = np.mean(positive_deviations) if positive_deviations else 0
        negative_strength = abs(np.mean(negative_deviations)) if negative_deviations else 0
        
        # Curvature analysis
        curvature = calculate_edge_curvature(edge_points)
        high_curvature = curvature > 0.1  # Threshold for high curvature
        
        
        # Enhanced classification logic with balanced thresholds
        
        # 1. Very straight edges (using straight tolerance)
        if max_abs_deviation < straight_tolerance:
            edge_type = "straight"
            max_deviation = mean_deviation
            
        # 2. Clear intrusions with strong pattern consistency
        elif (negative_consistency_ratio > 0.25 and 
              negative_strength > fine_threshold and
              significant_negative_fine > significant_positive_fine * 1.5):
            edge_type = "intrusion"
            max_deviation = min(deviations)
            
        # 3. Clear extrusions with strong pattern consistency  
        elif (positive_consistency_ratio > 0.25 and 
              positive_strength > fine_threshold and
              significant_positive_fine > significant_negative_fine * 1.5):
            edge_type = "extrusion"
            max_deviation = max(deviations)
            
        # 4. Consistent deviation with sufficient magnitude (coarse scale)
        elif significant_negative_coarse > 0 and mean_deviation < -fine_threshold:
            edge_type = "intrusion"
            max_deviation = min(deviations)
            
        elif significant_positive_coarse > 0 and mean_deviation > fine_threshold:
            edge_type = "extrusion"
            max_deviation = max(deviations)
            
        # 5. High curvature analysis for subtle intrusions
        elif (high_curvature and 
              max_consecutive_negative >= 3 and 
              negative_strength > fine_threshold * 0.7):
            edge_type = "intrusion"
            max_deviation = min(deviations)
            
        elif (high_curvature and 
              max_consecutive_positive >= 3 and 
              positive_strength > fine_threshold * 0.7):
            edge_type = "extrusion"
            max_deviation = max(deviations)
            
        # 6. Nearly straight edges with minor deviations
        elif (max_abs_deviation < coarse_threshold and 
              abs(mean_deviation) < fine_threshold and
              max(positive_consistency_ratio, negative_consistency_ratio) < 0.3):
            edge_type = "straight"
            max_deviation = mean_deviation
            
        # 7. Fallback: check for any significant deviations
        elif max_abs_deviation > coarse_threshold:
            if abs(mean_deviation) < fine_threshold * 0.5:
                # Balanced but significant deviations - choose dominant direction
                if significant_negative_fine > significant_positive_fine:
                    edge_type = "intrusion"
                    max_deviation = min(deviations)
                elif significant_positive_fine > significant_negative_fine:
                    edge_type = "extrusion"
                    max_deviation = max(deviations)
                else:
                    edge_type = "straight"
                    max_deviation = mean_deviation
            elif mean_deviation > 0:
                edge_type = "extrusion"
                max_deviation = max(deviations)
            else:
                edge_type = "intrusion"
                max_deviation = min(deviations)
        
        # 8. Default to straight if no clear pattern
        else:
            edge_type = "straight"
            max_deviation = mean_deviation
            
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

