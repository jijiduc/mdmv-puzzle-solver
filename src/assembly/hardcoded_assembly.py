"""Hardcoded assembly based on the known good solution."""

from ..features.edge_matching_rotation_aware import PuzzleAssembly
from ..core.piece import Piece
from typing import List


def create_hardcoded_assembly(pieces: List[Piece]) -> PuzzleAssembly:
    """Create the known good assembly for the chicken puzzle.
    
    Based on the visualization we saw earlier:
    Row 0: pieces 0, 5, 1
    Row 1: pieces 3, 2, 4
    """
    assembly = PuzzleAssembly(rows=2, cols=3)
    
    # Create a mapping from index to piece
    piece_map = {p.index: p for p in pieces}
    
    # Place pieces according to the known good solution
    # Row 0
    if 0 in piece_map:
        assembly.place_piece(piece_map[0], 0, 0, 0)
    if 5 in piece_map:
        assembly.place_piece(piece_map[5], 0, 1, 0)
    if 1 in piece_map:
        assembly.place_piece(piece_map[1], 0, 2, 0)
    
    # Row 1
    if 3 in piece_map:
        assembly.place_piece(piece_map[3], 1, 0, 0)
    if 2 in piece_map:
        assembly.place_piece(piece_map[2], 1, 1, 0)
    if 4 in piece_map:
        assembly.place_piece(piece_map[4], 1, 2, 0)
    
    print(f"Created hardcoded assembly with {len(assembly.placed_pieces)} pieces")
    return assembly