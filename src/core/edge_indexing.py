"""Edge indexing standardization for consistent piece orientation."""

import numpy as np
from typing import List, Tuple, Dict
import cv2

from .piece import Piece, EdgeSegment


def determine_piece_orientation(piece: Piece) -> Dict[str, any]:
    """Determine the orientation of a puzzle piece based on its edges.
    
    Returns a dictionary with:
    - piece_type: 'corner', 'edge', or 'center'
    - flat_edges: indices of flat edges
    - orientation: suggested standard orientation
    """
    flat_edges = []
    for i, edge in enumerate(piece.edges):
        if edge.edge_type == 'flat':
            flat_edges.append(i)
    
    piece_info = {
        'flat_edges': flat_edges,
        'flat_count': len(flat_edges)
    }
    
    if len(flat_edges) == 2:
        piece_info['piece_type'] = 'corner'
        # Determine which corner based on flat edge positions
        idx1, idx2 = flat_edges[0], flat_edges[1]
        diff = (idx2 - idx1) % 4
        
        if diff == 1:
            # Adjacent edges: idx1 is one edge, idx2 is next clockwise
            piece_info['corner_type'] = f"edges_{idx1}_{idx2}"
        elif diff == 3:
            # Adjacent edges: idx2 is one edge, idx1 is next clockwise
            piece_info['corner_type'] = f"edges_{idx2}_{idx1}"
        else:
            # Non-adjacent - shouldn't happen for valid corner
            piece_info['corner_type'] = 'invalid'
            
    elif len(flat_edges) == 1:
        piece_info['piece_type'] = 'edge'
        piece_info['flat_edge_idx'] = flat_edges[0]
    else:
        piece_info['piece_type'] = 'center'
    
    return piece_info


def get_edge_direction_vector(edge: EdgeSegment) -> np.ndarray:
    """Get the average direction vector of an edge from start to end."""
    if not edge.points or len(edge.points) < 2:
        return np.array([1, 0])  # Default to right
    
    points = np.array(edge.points)
    start_region = points[:max(1, len(points)//4)]
    end_region = points[-max(1, len(points)//4):]
    
    start_center = np.mean(start_region, axis=0)
    end_center = np.mean(end_region, axis=0)
    
    direction = end_center - start_center
    norm = np.linalg.norm(direction)
    if norm > 0:
        direction = direction / norm
    else:
        direction = np.array([1, 0])
    
    return direction


def determine_edge_side(edge: EdgeSegment, piece: Piece) -> str:
    """Determine which side of the piece an edge is on (top, right, bottom, left)."""
    # Get edge midpoint
    if not edge.points:
        return 'unknown'
    
    points = np.array(edge.points)
    midpoint = np.mean(points, axis=0)
    
    # Get vector from centroid to edge midpoint
    centroid = np.array(piece.centroid)
    to_edge = midpoint - centroid
    
    # Normalize
    norm = np.linalg.norm(to_edge)
    if norm > 0:
        to_edge = to_edge / norm
    
    # Determine primary direction
    angle = np.arctan2(to_edge[1], to_edge[0])
    
    # Convert to degrees and normalize to 0-360
    angle_deg = np.degrees(angle)
    if angle_deg < 0:
        angle_deg += 360
    
    # Classify into quadrants
    if 315 <= angle_deg or angle_deg < 45:
        return 'right'
    elif 45 <= angle_deg < 135:
        return 'bottom'
    elif 135 <= angle_deg < 225:
        return 'left'
    else:  # 225 <= angle_deg < 315
        return 'top'


def standardize_edge_indexing(piece: Piece) -> List[int]:
    """Return a mapping from current edge indices to standardized indices.
    
    Standard indexing:
    - 0: top edge
    - 1: right edge  
    - 2: bottom edge
    - 3: left edge
    
    Returns:
        List where list[old_idx] = new_idx
    """
    # Determine which side each edge is on
    edge_sides = []
    for edge in piece.edges:
        side = determine_edge_side(edge, piece)
        edge_sides.append(side)
    
    # Create mapping
    side_to_standard = {'top': 0, 'right': 1, 'bottom': 2, 'left': 3}
    mapping = [-1] * 4
    
    # For corner pieces with 2 flat edges, use them as reference
    piece_info = determine_piece_orientation(piece)
    
    if piece_info['piece_type'] == 'corner' and len(piece_info['flat_edges']) == 2:
        # Use flat edges to determine orientation
        flat_sides = [edge_sides[i] for i in piece_info['flat_edges']]
        
        # Determine which corner this is
        if set(flat_sides) == {'top', 'left'}:
            # Top-left corner
            for i, side in enumerate(edge_sides):
                if side in side_to_standard:
                    mapping[i] = side_to_standard[side]
        elif set(flat_sides) == {'top', 'right'}:
            # Top-right corner
            for i, side in enumerate(edge_sides):
                if side in side_to_standard:
                    mapping[i] = side_to_standard[side]
        elif set(flat_sides) == {'bottom', 'left'}:
            # Bottom-left corner
            for i, side in enumerate(edge_sides):
                if side in side_to_standard:
                    mapping[i] = side_to_standard[side]
        elif set(flat_sides) == {'bottom', 'right'}:
            # Bottom-right corner
            for i, side in enumerate(edge_sides):
                if side in side_to_standard:
                    mapping[i] = side_to_standard[side]
    else:
        # For edge and center pieces, use the detected sides directly
        for i, side in enumerate(edge_sides):
            if side in side_to_standard:
                mapping[i] = side_to_standard[side]
    
    # Handle any unmapped edges (shouldn't happen with 4 edges)
    used_indices = set(mapping)
    available_indices = set(range(4)) - used_indices - {-1}
    
    for i in range(4):
        if mapping[i] == -1 and available_indices:
            mapping[i] = available_indices.pop()
    
    return mapping


def reorder_edges(piece: Piece) -> None:
    """Reorder the edges of a piece to follow standard indexing in-place."""
    mapping = standardize_edge_indexing(piece)
    
    # Create new edge list in standard order
    new_edges = [None] * 4
    for old_idx, new_idx in enumerate(mapping):
        if new_idx >= 0:
            edge = piece.edges[old_idx]
            edge.edge_idx = new_idx  # Update edge's own index
            new_edges[new_idx] = edge
    
    # Replace edges
    piece.edges = new_edges
    
    # Also reorder corners to match
    new_corners = [None] * 4
    for old_idx, new_idx in enumerate(mapping):
        if new_idx >= 0 and old_idx < len(piece.corners):
            new_corners[new_idx] = piece.corners[old_idx]
    
    # Fill any None values with interpolated corners
    for i in range(4):
        if new_corners[i] is None and piece.edges[i] is not None:
            # Use edge endpoints as corners
            edge = piece.edges[i]
            if edge.points and len(edge.points) > 0:
                new_corners[i] = tuple(edge.points[0])
    
    piece.corners = new_corners


def visualize_edge_indexing(piece: Piece, output_path: str) -> None:
    """Create a visualization showing edge indices and types."""
    # Create a larger canvas
    img = piece.image.copy()
    h, w = img.shape[:2]
    canvas = np.ones((h + 200, w + 200, 3), dtype=np.uint8) * 255
    
    # Place piece image in center
    offset_x, offset_y = 100, 100
    canvas[offset_y:offset_y+h, offset_x:offset_x+w] = img
    
    # Draw edge indices and types
    for i, edge in enumerate(piece.edges):
        if edge and edge.points and len(edge.points) > 0:
            # Get edge midpoint
            points = np.array(edge.points)
            midpoint = np.mean(points, axis=0).astype(int)
            midpoint[0] += offset_x
            midpoint[1] += offset_y
            
            # Determine label position offset based on edge side
            side = determine_edge_side(edge, piece)
            if side == 'top':
                label_offset = (0, -30)
            elif side == 'right':
                label_offset = (30, 0)
            elif side == 'bottom':
                label_offset = (0, 30)
            else:  # left
                label_offset = (-30, 0)
            
            label_pos = (midpoint[0] + label_offset[0], midpoint[1] + label_offset[1])
            
            # Draw edge index
            cv2.putText(canvas, f"E{i}", tuple(label_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Draw edge type
            type_label = edge.edge_type[0].upper()  # F, C, or V
            cv2.putText(canvas, type_label, 
                       (label_pos[0], label_pos[1] + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            # Draw arrow showing edge direction
            if len(points) > 10:
                start_pt = points[len(points)//4].astype(int)
                end_pt = points[3*len(points)//4].astype(int)
                start_pt[0] += offset_x
                start_pt[1] += offset_y
                end_pt[0] += offset_x
                end_pt[1] += offset_y
                cv2.arrowedLine(canvas, tuple(start_pt), tuple(end_pt), 
                               (0, 255, 0), 2, tipLength=0.3)
    
    # Add legend
    cv2.putText(canvas, "Edge Index (E0-E3)", (10, 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
    cv2.putText(canvas, "Type: F=Flat, C=Concave, V=conVex", (10, 40), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)
    cv2.putText(canvas, "Green arrows show edge direction", (10, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    
    cv2.imwrite(output_path, canvas)


def standardize_all_pieces(pieces: List[Piece]) -> None:
    """Standardize edge indexing for all pieces."""
    print("\n=== Standardizing Edge Indexing ===")
    
    for piece in pieces:
        piece_info = determine_piece_orientation(piece)
        print(f"Piece {piece.index}: {piece_info['piece_type']} " +
              f"(flat edges: {piece_info['flat_edges']})")
        
        # Get current edge sides
        before_sides = [determine_edge_side(edge, piece) for edge in piece.edges]
        
        # Reorder edges
        reorder_edges(piece)
        
        # Get new edge sides
        after_sides = [determine_edge_side(edge, piece) for edge in piece.edges]
        
        print(f"  Edge reordering: {before_sides} -> {after_sides}")
        
        # Verify standard indexing
        expected = ['top', 'right', 'bottom', 'left']
        if after_sides != expected:
            print(f"  WARNING: Non-standard result: {after_sides}")