"""Rotation-aware edge matching algorithm."""

import numpy as np
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass
import cv2
from itertools import combinations

from ..core.piece import Piece, EdgeSegment
from .edge_matching import EdgeMatch, GlobalMatchRegistry, EdgeSpatialIndex
from .color_analysis import calculate_color_similarity, color_distance
from .shape_analysis import calculate_curvature_profile
from .edge_matching_improved import (
    calculate_improved_shape_compatibility,
    calculate_enhanced_color_continuity,
    validate_edge_classifications_improved
)


@dataclass
class PuzzleAssembly:
    """Represents the current state of puzzle assembly."""
    grid: Dict[Tuple[int, int], Tuple[Piece, int]]  # (row, col) -> (piece, rotation)
    placed_pieces: Set[int]  # Set of placed piece indices
    rows: int
    cols: int
    
    def __init__(self, rows: int, cols: int):
        self.grid = {}
        self.placed_pieces = set()
        self.rows = rows
        self.cols = cols
    
    def place_piece(self, piece: Piece, row: int, col: int, rotation: int):
        """Place a piece at given position with rotation (0, 90, 180, 270)."""
        self.grid[(row, col)] = (piece, rotation)
        self.placed_pieces.add(piece.index)
    
    def get_neighbor_edge(self, row: int, col: int, direction: str) -> Optional[Tuple[Piece, int, int]]:
        """Get the neighboring piece and its edge that faces the given position.
        
        Returns: (neighbor_piece, edge_index, rotation) or None
        """
        # Direction to position offset and which edge faces back
        dir_map = {
            'top': ((-1, 0), 2),     # neighbor above, their bottom edge faces us
            'right': ((0, 1), 3),    # neighbor right, their left edge faces us
            'bottom': ((1, 0), 0),   # neighbor below, their top edge faces us
            'left': ((0, -1), 1)     # neighbor left, their right edge faces us
        }
        
        if direction not in dir_map:
            return None
        
        (row_off, col_off), base_edge = dir_map[direction]
        neighbor_pos = (row + row_off, col + col_off)
        
        if neighbor_pos in self.grid:
            piece, rotation = self.grid[neighbor_pos]
            # Adjust edge index based on rotation
            actual_edge = (base_edge - rotation // 90) % 4
            return piece, actual_edge, rotation
        
        return None


def get_rotated_edge(piece: Piece, edge_idx: int, rotation: int) -> EdgeSegment:
    """Get the edge that would be at position edge_idx after rotation.
    
    Rotation is in degrees (0, 90, 180, 270).
    """
    # Calculate which original edge ends up at edge_idx after rotation
    rotation_steps = rotation // 90
    original_edge_idx = (edge_idx - rotation_steps) % 4
    return piece.edges[original_edge_idx]


def match_edges_with_rotation(edge1: EdgeSegment, edge2: EdgeSegment, 
                            piece1: Piece, piece2: Piece) -> Tuple[float, Dict[str, float]]:
    """Match two edges considering that edge2 might need to be flipped."""
    components = {}
    
    # Shape compatibility (already handles orientation)
    components['shape'] = calculate_improved_shape_compatibility(edge1, edge2)
    
    # Length compatibility
    length_ratio = min(edge1.length, edge2.length) / max(edge1.length, edge2.length)
    components['length'] = length_ratio
    
    # Color continuity
    components['continuity'] = calculate_enhanced_color_continuity(
        piece1, edge1, piece2, edge2
    )
    
    # Edge color similarity
    if edge1.color_sequence and edge2.color_sequence:
        # Try both orientations for edge2
        score_normal = calculate_color_similarity(edge1.color_sequence, edge2.color_sequence)
        score_reversed = calculate_color_similarity(edge1.color_sequence, 
                                                  list(reversed(edge2.color_sequence)))
        components['color'] = max(score_normal, score_reversed)
    else:
        components['color'] = 0.5
    
    # Calculate total score
    weights = {
        'shape': 0.40,
        'length': 0.20,
        'continuity': 0.25,
        'color': 0.15
    }
    
    total_score = sum(weights[k] * components[k] for k in weights)
    
    return total_score, components


def find_corner_pieces(pieces: List[Piece]) -> List[Piece]:
    """Find all corner pieces (2 flat edges)."""
    corners = []
    for piece in pieces:
        flat_count = sum(1 for edge in piece.edges if edge.edge_type == 'flat')
        if flat_count == 2:
            corners.append(piece)
    return corners


def find_edge_pieces(pieces: List[Piece]) -> List[Piece]:
    """Find all edge pieces (1 flat edge)."""
    edges = []
    for piece in pieces:
        flat_count = sum(1 for edge in piece.edges if edge.edge_type == 'flat')
        if flat_count == 1:
            edges.append(piece)
    return edges


def get_corner_position(piece: Piece) -> Tuple[str, int]:
    """Determine which corner type and rotation based on flat edge positions.
    
    Returns: (corner_type, rotation) where corner_type is 'TL', 'TR', 'BL', 'BR'
    and rotation is how much to rotate the piece to fit that corner.
    """
    flat_edges = [i for i, edge in enumerate(piece.edges) if edge.edge_type == 'flat']
    
    if len(flat_edges) != 2:
        return None, 0
    
    # Check which edges are flat
    edge1, edge2 = sorted(flat_edges)
    
    # Map flat edge pairs to corner types and required rotation
    # Format: (edge1, edge2) -> (corner_type, rotation_to_standard)
    corner_map = {
        (0, 1): ('TR', 0),   # Top-Right: flat on top and right
        (0, 3): ('TL', 0),   # Top-Left: flat on top and left
        (1, 2): ('BR', 0),   # Bottom-Right: flat on right and bottom
        (2, 3): ('BL', 0),   # Bottom-Left: flat on bottom and left
        # Non-adjacent cases (if piece is already rotated)
        (0, 2): ('TL', 90),  # Rotated piece
        (1, 3): ('TR', 90),  # Rotated piece
    }
    
    key = (edge1, edge2)
    if key in corner_map:
        return corner_map[key]
    
    # Default fallback
    return 'TL', 0


def try_place_corner(assembly: PuzzleAssembly, piece: Piece, 
                    corner_type: str, rotation: int) -> bool:
    """Try to place a corner piece in the assembly."""
    corner_positions = {
        'TL': (0, 0),
        'TR': (0, assembly.cols - 1),
        'BL': (assembly.rows - 1, 0),
        'BR': (assembly.rows - 1, assembly.cols - 1)
    }
    
    if corner_type not in corner_positions:
        return False
    
    row, col = corner_positions[corner_type]
    
    # Check if position is already occupied
    if (row, col) in assembly.grid:
        return False
    
    assembly.place_piece(piece, row, col, rotation)
    return True


def find_best_match_for_position(assembly: PuzzleAssembly, row: int, col: int,
                                 available_pieces: List[Piece], 
                                 pieces_dict: Dict[int, Piece]) -> Optional[Tuple[Piece, int, float]]:
    """Find the best piece and rotation for a given position.
    
    Returns: (piece, rotation, score) or None
    """
    # Get neighboring edges
    neighbors = []
    for direction in ['top', 'right', 'bottom', 'left']:
        neighbor_info = assembly.get_neighbor_edge(row, col, direction)
        if neighbor_info:
            neighbors.append((direction, neighbor_info))
    
    if not neighbors:
        return None
    
    best_match = None
    best_score = 0
    
    # Try each available piece
    for piece in available_pieces:
        # Skip if piece is already placed
        if piece.index in assembly.placed_pieces:
            continue
        
        # Try each rotation
        for rotation in [0, 90, 180, 270]:
            total_score = 0
            match_count = 0
            
            # Check compatibility with each neighbor
            for direction, (neighbor_piece, neighbor_edge_idx, neighbor_rotation) in neighbors:
                # Determine which edge of the current piece would face this neighbor
                dir_to_edge = {'top': 0, 'right': 1, 'bottom': 2, 'left': 3}
                our_edge_idx = dir_to_edge[direction]
                
                # Get the actual edges after rotation
                our_edge = get_rotated_edge(piece, our_edge_idx, rotation)
                neighbor_edge = neighbor_piece.edges[neighbor_edge_idx]
                
                # Check type compatibility
                if not ((our_edge.edge_type == 'convex' and neighbor_edge.edge_type == 'concave') or
                       (our_edge.edge_type == 'concave' and neighbor_edge.edge_type == 'convex')):
                    continue
                
                # Calculate match score
                score, _ = match_edges_with_rotation(our_edge, neighbor_edge, 
                                                    piece, neighbor_piece)
                total_score += score
                match_count += 1
            
            # Average score across all neighbor matches
            if match_count > 0:
                avg_score = total_score / match_count
                if avg_score > best_score:
                    best_score = avg_score
                    best_match = (piece, rotation, avg_score)
    
    return best_match


def solve_puzzle_progressive(pieces: List[Piece], rows: int = 2, cols: int = 3) -> PuzzleAssembly:
    """Solve puzzle by progressive assembly starting with corners."""
    print("\n=== Progressive Puzzle Assembly ===")
    
    assembly = PuzzleAssembly(rows, cols)
    pieces_dict = {p.index: p for p in pieces}
    
    # Step 1: Place corner pieces
    corners = find_corner_pieces(pieces)
    print(f"Found {len(corners)} corner pieces")
    
    # Try to place corners based on their flat edge configuration
    corner_types_needed = {'TL', 'TR', 'BL', 'BR'}
    
    for corner in corners:
        corner_type, rotation = get_corner_position(corner)
        if corner_type in corner_types_needed:
            if try_place_corner(assembly, corner, corner_type, rotation):
                corner_types_needed.remove(corner_type)
                print(f"  Placed piece {corner.index} at {corner_type} corner (rotation: {rotation}°)")
    
    # Step 2: Fill in edge pieces
    edge_pieces = find_edge_pieces(pieces)
    print(f"Found {len(edge_pieces)} edge pieces")
    
    # Edge positions to fill (between corners)
    edge_positions = []
    if rows == 2 and cols == 3:
        edge_positions = [(0, 1), (1, 1)]  # Top middle, bottom middle
    
    for row, col in edge_positions:
        best_match = find_best_match_for_position(assembly, row, col, 
                                                 edge_pieces, pieces_dict)
        if best_match:
            piece, rotation, score = best_match
            assembly.place_piece(piece, row, col, rotation)
            print(f"  Placed edge piece {piece.index} at ({row},{col}) " +
                  f"(rotation: {rotation}°, score: {score:.3f})")
    
    # Step 3: Fill in remaining pieces
    remaining_pieces = [p for p in pieces if p.index not in assembly.placed_pieces]
    print(f"{len(remaining_pieces)} pieces remaining")
    
    # Try to fill empty positions
    for row in range(rows):
        for col in range(cols):
            if (row, col) not in assembly.grid:
                best_match = find_best_match_for_position(assembly, row, col,
                                                         remaining_pieces, pieces_dict)
                if best_match:
                    piece, rotation, score = best_match
                    assembly.place_piece(piece, row, col, rotation)
                    print(f"  Placed piece {piece.index} at ({row},{col}) " +
                          f"(rotation: {rotation}°, score: {score:.3f})")
    
    return assembly


def visualize_assembly(assembly: PuzzleAssembly, pieces: List[Piece], output_path: str):
    """Create a visualization of the assembled puzzle."""
    # Calculate dimensions
    piece_size = 200  # Size for each piece in visualization
    canvas_width = assembly.cols * piece_size
    canvas_height = assembly.rows * piece_size
    
    # Create canvas
    canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 128  # Gray background
    
    # Place each piece
    for (row, col), (piece, rotation) in assembly.grid.items():
        # Get piece image
        piece_img = piece.image.copy()
        
        # Rotate piece image
        if rotation > 0:
            rows, cols = piece_img.shape[:2]
            M = cv2.getRotationMatrix2D((cols/2, rows/2), -rotation, 1)
            piece_img = cv2.warpAffine(piece_img, M, (cols, rows))
        
        # Resize to standard size
        piece_img = cv2.resize(piece_img, (piece_size, piece_size))
        
        # Place on canvas
        y_start = row * piece_size
        x_start = col * piece_size
        canvas[y_start:y_start+piece_size, x_start:x_start+piece_size] = piece_img
        
        # Add piece number
        cv2.putText(canvas, f"P{piece.index}", 
                   (x_start + 10, y_start + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Add rotation if not 0
        if rotation > 0:
            cv2.putText(canvas, f"{rotation}°", 
                       (x_start + 10, y_start + 55),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    # Draw grid lines
    for i in range(1, assembly.rows):
        cv2.line(canvas, (0, i * piece_size), (canvas_width, i * piece_size), (255, 255, 255), 2)
    for j in range(1, assembly.cols):
        cv2.line(canvas, (j * piece_size, 0), (j * piece_size, canvas_height), (255, 255, 255), 2)
    
    cv2.imwrite(output_path, canvas)
    print(f"Assembly visualization saved to {output_path}")


def perform_rotation_aware_matching(pieces: List[Piece]) -> Tuple[GlobalMatchRegistry, EdgeSpatialIndex, PuzzleAssembly]:
    """Perform rotation-aware edge matching."""
    # Validate edge classifications
    validate_edge_classifications_improved(pieces)
    
    # Create registry and spatial index
    registry = GlobalMatchRegistry()
    spatial_index = EdgeSpatialIndex()
    
    # Index all edges
    for piece in pieces:
        for edge_idx, edge in enumerate(piece.edges):
            spatial_index.add_edge(
                piece.index, edge_idx,
                edge.edge_type,
                edge.sub_type,
                edge.length,
                color_cluster_id=None
            )
    
    # Solve puzzle progressively
    assembly = solve_puzzle_progressive(pieces)
    
    # Create assembly visualization
    import os
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                             'debug', '08_matching')
    os.makedirs(output_dir, exist_ok=True)
    visualize_assembly(assembly, pieces, os.path.join(output_dir, 'puzzle_assembly.png'))
    
    # Convert assembly to edge matches for compatibility
    print("\n=== Assembly Results ===")
    for (row, col), (piece, rotation) in assembly.grid.items():
        print(f"Position ({row},{col}): Piece {piece.index} (rotation: {rotation}°)")
        
        # Check right neighbor
        if col < assembly.cols - 1 and (row, col + 1) in assembly.grid:
            neighbor, neighbor_rotation = assembly.grid[(row, col + 1)]
            our_edge = get_rotated_edge(piece, 1, rotation)  # Our right edge
            their_edge = get_rotated_edge(neighbor, 3, neighbor_rotation)  # Their left edge
            
            score, components = match_edges_with_rotation(our_edge, their_edge, piece, neighbor)
            
            # Add to registry
            match = EdgeMatch(
                piece_idx=neighbor.index,
                edge_idx=3,  # Their original left edge index
                similarity_score=score,
                shape_score=components['shape'],
                color_score=components['color'],
                confidence=min(components.values()),
                match_type='perfect' if score > 0.85 else 'good' if score > 0.7 else 'possible',
                validation_flags={
                    'type_compatible': True,
                    'length_compatible': components['length'] > 0.8,
                    'curvature_compatible': components['shape'] > 0.7,
                    'color_compatible': components['continuity'] > 0.6
                }
            )
            
            # Use original edge indices for registry
            our_original_edge = (1 - rotation // 90) % 4
            their_original_edge = (3 - neighbor_rotation // 90) % 4
            
            registry.add_match(piece.index, our_original_edge, 
                             neighbor.index, their_original_edge, match)
            registry.confirm_match(piece.index, our_original_edge,
                                 neighbor.index, their_original_edge)
            
            print(f"  Match: P{piece.index}E{our_original_edge} <-> " +
                  f"P{neighbor.index}E{their_original_edge} (score: {score:.3f})")
        
        # Check bottom neighbor
        if row < assembly.rows - 1 and (row + 1, col) in assembly.grid:
            neighbor, neighbor_rotation = assembly.grid[(row + 1, col)]
            our_edge = get_rotated_edge(piece, 2, rotation)  # Our bottom edge
            their_edge = get_rotated_edge(neighbor, 0, neighbor_rotation)  # Their top edge
            
            score, components = match_edges_with_rotation(our_edge, their_edge, piece, neighbor)
            
            # Add to registry
            match = EdgeMatch(
                piece_idx=neighbor.index,
                edge_idx=0,  # Their original top edge index
                similarity_score=score,
                shape_score=components['shape'],
                color_score=components['color'],
                confidence=min(components.values()),
                match_type='perfect' if score > 0.85 else 'good' if score > 0.7 else 'possible',
                validation_flags={
                    'type_compatible': True,
                    'length_compatible': components['length'] > 0.8,
                    'curvature_compatible': components['shape'] > 0.7,
                    'color_compatible': components['continuity'] > 0.6
                }
            )
            
            # Use original edge indices for registry
            our_original_edge = (2 - rotation // 90) % 4
            their_original_edge = (0 - neighbor_rotation // 90) % 4
            
            registry.add_match(piece.index, our_original_edge,
                             neighbor.index, their_original_edge, match)
            registry.confirm_match(piece.index, our_original_edge,
                                 neighbor.index, their_original_edge)
            
            print(f"  Match: P{piece.index}E{our_original_edge} <-> " +
                  f"P{neighbor.index}E{their_original_edge} (score: {score:.3f})")
    
    # Print statistics
    total_positions = assembly.rows * assembly.cols
    placed_pieces = len(assembly.placed_pieces)
    print(f"\nPlaced {placed_pieces}/{total_positions} pieces")
    
    return registry, spatial_index, assembly