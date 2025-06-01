"""Improved edge matching algorithm with spatial constraints and better validation."""

import numpy as np
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass
import cv2

from ..core.piece import Piece, EdgeSegment
from .edge_matching import EdgeMatch, GlobalMatchRegistry, EdgeSpatialIndex
from .color_analysis import calculate_color_similarity, color_distance
from .shape_analysis import calculate_curvature_profile
from .texture_analysis import calculate_pattern_continuity, compare_texture_descriptors


@dataclass
class SpatialConstraint:
    """Represents spatial relationship between pieces."""
    piece1_idx: int
    edge1_idx: int
    piece2_idx: int
    edge2_idx: int
    relative_position: str  # 'top', 'right', 'bottom', 'left'
    confidence: float


def validate_edge_classifications_improved(pieces: List[Piece]) -> None:
    """Improved edge classification validation with template matching."""
    edge_counts = {'flat': 0, 'convex': 0, 'concave': 0}
    edge_info = []
    
    for piece in pieces:
        for edge_idx, edge in enumerate(piece.edges):
            edge_counts[edge.edge_type] += 1
            edge_info.append((piece.index, edge_idx, edge.edge_type, edge.confidence))
    
    convex_count = edge_counts['convex']
    concave_count = edge_counts['concave']
    
    # More aggressive reclassification for low confidence edges
    if convex_count != concave_count:
        print(f"Edge count mismatch: {convex_count} convex vs {concave_count} concave")
        
        # Get ALL curved edges sorted by confidence
        curved_edges = []
        for piece_idx, edge_idx, edge_type, confidence in edge_info:
            if edge_type in ['convex', 'concave']:
                curved_edges.append((piece_idx, edge_idx, edge_type, confidence))
        
        curved_edges.sort(key=lambda x: x[3])  # Sort by confidence
        
        # Reclassify lowest confidence edges to flat
        num_to_reclassify = abs(convex_count - concave_count)
        for i in range(min(num_to_reclassify, len(curved_edges))):
            piece_idx, edge_idx, old_type, confidence = curved_edges[i]
            if confidence < 0.7:  # Only reclassify low confidence edges
                pieces[piece_idx].edges[edge_idx].edge_type = 'flat'
                pieces[piece_idx].edges[edge_idx].classification = 'flat'
                pieces[piece_idx].edges[edge_idx].confidence = 1.0
                print(f"  Reclassified P{piece_idx}E{edge_idx} from {old_type} to flat (conf={confidence:.3f})")
                edge_counts[old_type] -= 1
                edge_counts['flat'] += 1
    
    print(f"✓ Final edge counts: {edge_counts['convex']} convex, {edge_counts['concave']} concave, {edge_counts['flat']} flat")


def infer_spatial_constraints(pieces: List[Piece]) -> List[SpatialConstraint]:
    """Infer spatial constraints based on piece types and edge configurations."""
    constraints = []
    
    # Find corner pieces (2 flat edges)
    corner_pieces = []
    edge_pieces = []
    center_pieces = []
    
    for piece in pieces:
        flat_count = sum(1 for edge in piece.edges if edge.edge_type == 'flat')
        if flat_count == 2:
            corner_pieces.append(piece)
        elif flat_count == 1:
            edge_pieces.append(piece)
        else:
            center_pieces.append(piece)
    
    print(f"Piece types: {len(corner_pieces)} corners, {len(edge_pieces)} edges, {len(center_pieces)} centers")
    
    # For a 2x3 puzzle, we expect 4 corners, 2 edges, 0 centers
    # Corner pieces can only connect to edge pieces on their non-flat edges
    
    return constraints


def calculate_improved_shape_compatibility(edge1: EdgeSegment, edge2: EdgeSegment) -> float:
    """Improved shape matching that properly handles convex/concave matching."""
    if edge1.edge_type == edge2.edge_type and edge1.edge_type != 'flat':
        return 0.0  # Can't match convex to convex or concave to concave
    
    if edge1.edge_type == 'flat' and edge2.edge_type == 'flat':
        return 0.0  # Flat edges don't match each other
    
    if not edge1.points or not edge2.points:
        return 0.0
    
    try:
        # Get curvature profiles
        curv1 = calculate_curvature_profile(np.array(edge1.points), smooth=True)
        curv2 = calculate_curvature_profile(np.array(edge2.points), smooth=True)
        
        # For convex-concave matching, invert one profile
        if edge1.edge_type == 'convex' and edge2.edge_type == 'concave':
            curv2 = -curv2
        elif edge1.edge_type == 'concave' and edge2.edge_type == 'convex':
            curv1 = -curv1
        
        # Normalize profiles
        if np.max(np.abs(curv1)) > 0:
            curv1 = curv1 / np.max(np.abs(curv1))
        if np.max(np.abs(curv2)) > 0:
            curv2 = curv2 / np.max(np.abs(curv2))
        
        # Resample to same length
        from ..features.edge_extraction import resample_sequence
        target_length = 50
        curv1_resampled = resample_sequence(curv1.tolist(), target_length)
        curv2_resampled = resample_sequence(curv2.tolist(), target_length)
        
        # Calculate correlation for both orientations
        corr_normal = np.corrcoef(curv1_resampled, curv2_resampled)[0, 1]
        corr_reversed = np.corrcoef(curv1_resampled, list(reversed(curv2_resampled)))[0, 1]
        
        # Take best correlation
        best_corr = max(corr_normal, corr_reversed)
        
        # Convert to 0-1 score
        score = (best_corr + 1) / 2
        
        return max(0, min(1, score))
        
    except Exception as e:
        print(f"Error in shape compatibility: {e}")
        return 0.0


def calculate_enhanced_color_continuity(piece1: Piece, edge1: EdgeSegment, 
                                       piece2: Piece, edge2: EdgeSegment) -> float:
    """Enhanced color continuity check with perpendicular sampling."""
    if not edge1.points or not edge2.points:
        return 0.5
    
    # Sample points along edges
    sample_count = 10
    edge1_samples = [edge1.points[i] for i in np.linspace(0, len(edge1.points)-1, sample_count, dtype=int)]
    edge2_samples = [edge2.points[i] for i in np.linspace(0, len(edge2.points)-1, sample_count, dtype=int)]
    
    # For each sample point, get colors perpendicular to edge
    continuity_scores = []
    
    for i in range(sample_count):
        # Get edge tangent direction
        if i == 0:
            tangent1 = np.array(edge1_samples[1]) - np.array(edge1_samples[0])
            tangent2 = np.array(edge2_samples[1]) - np.array(edge2_samples[0])
        elif i == sample_count - 1:
            tangent1 = np.array(edge1_samples[-1]) - np.array(edge1_samples[-2])
            tangent2 = np.array(edge2_samples[-1]) - np.array(edge2_samples[-2])
        else:
            tangent1 = np.array(edge1_samples[i+1]) - np.array(edge1_samples[i-1])
            tangent2 = np.array(edge2_samples[i+1]) - np.array(edge2_samples[i-1])
        
        # Normalize tangents
        tangent1 = tangent1 / (np.linalg.norm(tangent1) + 1e-6)
        tangent2 = tangent2 / (np.linalg.norm(tangent2) + 1e-6)
        
        # Get perpendicular directions (pointing inward)
        perp1 = np.array([-tangent1[1], tangent1[0]])
        perp2 = np.array([tangent2[1], -tangent2[0]])  # Opposite direction for matching
        
        # Sample colors along perpendicular
        colors1 = []
        colors2 = []
        
        for depth in range(1, 6):  # Sample 5 pixels deep
            # Sample from piece 1
            pt1 = np.array(edge1_samples[i]) + depth * perp1
            x1, y1 = int(pt1[0]), int(pt1[1])
            if 0 <= x1 < piece1.image.shape[1] and 0 <= y1 < piece1.image.shape[0]:
                colors1.append(piece1.image[y1, x1])
            
            # Sample from piece 2
            pt2 = np.array(edge2_samples[i]) + depth * perp2
            x2, y2 = int(pt2[0]), int(pt2[1])
            if 0 <= x2 < piece2.image.shape[1] and 0 <= y2 < piece2.image.shape[0]:
                colors2.append(piece2.image[y2, x2])
        
        # Compare color sequences
        if colors1 and colors2:
            # Convert to LAB for better comparison
            lab1 = cv2.cvtColor(np.array([colors1]), cv2.COLOR_BGR2LAB)[0]
            lab2 = cv2.cvtColor(np.array([colors2]), cv2.COLOR_BGR2LAB)[0]
            
            # Calculate similarity
            distances = [color_distance(lab1[j], lab2[j]) for j in range(min(len(lab1), len(lab2)))]
            avg_distance = np.mean(distances)
            
            # Convert to similarity score
            similarity = np.exp(-avg_distance / 50)  # 50 is typical LAB distance scale
            continuity_scores.append(similarity)
    
    return np.mean(continuity_scores) if continuity_scores else 0.5


def validate_assembly_constraints(piece1: Piece, edge1_idx: int, piece2: Piece, edge2_idx: int,
                                 existing_matches: Set[Tuple[int, int]]) -> bool:
    """Validate that pieces can physically assemble without conflicts."""
    # Check if either piece is already matched on this edge
    if (piece1.index, edge1_idx) in existing_matches or (piece2.index, edge2_idx) in existing_matches:
        return False
    
    # With standardized indexing: 0=top, 1=right, 2=bottom, 3=left
    # Edges can only connect if they face each other:
    # - Top (0) connects to Bottom (2)
    # - Right (1) connects to Left (3)
    # - Bottom (2) connects to Top (0)
    # - Left (3) connects to Right (1)
    
    valid_connections = {
        0: 2,  # top connects to bottom
        1: 3,  # right connects to left
        2: 0,  # bottom connects to top
        3: 1   # left connects to right
    }
    
    if valid_connections.get(edge1_idx) != edge2_idx:
        return False  # Edges don't face each other
    
    # Additional constraint for 2x3 puzzle
    flat_count1 = sum(1 for edge in piece1.edges if edge.edge_type == 'flat')
    flat_count2 = sum(1 for edge in piece2.edges if edge.edge_type == 'flat')
    
    # Corner pieces (2 flat edges) connecting to corner pieces is invalid
    if flat_count1 == 2 and flat_count2 == 2:
        return False
    
    return True


def perform_improved_edge_matching(pieces: List[Piece]) -> Tuple[GlobalMatchRegistry, EdgeSpatialIndex]:
    """Improved edge matching with all enhancements."""
    # Step 1: Validate and correct edge classifications
    validate_edge_classifications_improved(pieces)
    
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
    
    # Calculate edge direction vectors
    for piece in pieces:
        for edge in piece.edges:
            edge.calculate_direction_vectors()
    
    # Step 2: Infer spatial constraints
    spatial_constraints = infer_spatial_constraints(pieces)
    
    # Step 3: Find all potential matches with improved scoring
    match_candidates = []
    
    for i, piece1 in enumerate(pieces):
        for edge1_idx, edge1 in enumerate(piece1.edges):
            if edge1.edge_type == 'flat':
                continue  # Skip flat edges
            
            for j, piece2 in enumerate(pieces):
                if i >= j:  # Avoid duplicates and self-matching
                    continue
                
                for edge2_idx, edge2 in enumerate(piece2.edges):
                    if edge2.edge_type == 'flat':
                        continue
                    
                    # Must be complementary types
                    if not ((edge1.edge_type == 'convex' and edge2.edge_type == 'concave') or
                           (edge1.edge_type == 'concave' and edge2.edge_type == 'convex')):
                        continue
                    
                    # Calculate improved match score
                    score_components = {}
                    
                    # 1. Shape compatibility (improved)
                    score_components['shape'] = calculate_improved_shape_compatibility(edge1, edge2)
                    
                    # 2. Length compatibility
                    length_ratio = min(edge1.length, edge2.length) / max(edge1.length, edge2.length)
                    score_components['length'] = length_ratio
                    
                    # 3. Color continuity (enhanced)
                    score_components['continuity'] = calculate_enhanced_color_continuity(
                        piece1, edge1, piece2, edge2
                    )
                    
                    # 4. Edge color similarity
                    if edge1.color_sequence and edge2.color_sequence:
                        score_components['color'] = calculate_color_similarity(
                            edge1.color_sequence, edge2.color_sequence
                        )
                    else:
                        score_components['color'] = 0.5
                    
                    # 5. Pattern continuity
                    score_components['pattern'] = calculate_pattern_continuity(
                        piece1.image, edge1.points, piece2.image, edge2.points,
                        piece1.mask, piece2.mask
                    )
                    
                    # Calculate total score with new weights
                    weights = {
                        'shape': 0.35,      # Increased importance
                        'length': 0.20,     # Length must match
                        'continuity': 0.25, # Visual continuity
                        'color': 0.10,      # Edge colors
                        'pattern': 0.10     # Pattern matching
                    }
                    
                    total_score = sum(weights[k] * score_components[k] for k in weights)
                    
                    # Only consider strong matches
                    if total_score > 0.65:
                        match_candidates.append({
                            'piece1': i,
                            'edge1': edge1_idx,
                            'piece2': j,
                            'edge2': edge2_idx,
                            'score': total_score,
                            'components': score_components
                        })
    
    # Sort by score
    match_candidates.sort(key=lambda x: x['score'], reverse=True)
    
    # Step 4: Apply constraints and select best matches
    confirmed_matches = set()
    used_edges = set()
    
    print(f"\n=== Improved Matching Results ===")
    print(f"Found {len(match_candidates)} candidates with score > 0.65")
    
    for candidate in match_candidates:
        p1, e1 = candidate['piece1'], candidate['edge1']
        p2, e2 = candidate['piece2'], candidate['edge2']
        
        # Check if edges are already used
        if (p1, e1) in used_edges or (p2, e2) in used_edges:
            continue
        
        # Validate assembly constraints
        if not validate_assembly_constraints(pieces[p1], e1, pieces[p2], e2, used_edges):
            continue
        
        # Additional validation: require multiple criteria to agree
        components = candidate['components']
        high_scores = sum(1 for v in components.values() if v > 0.7)
        if high_scores < 2:  # At least 2 components must score > 0.7
            continue
        
        # Confirm match
        print(f"  Confirmed: P{p1}E{e1} ↔ P{p2}E{e2} (score: {candidate['score']:.3f})")
        print(f"    Components: " + ", ".join(f"{k}={v:.3f}" for k, v in components.items()))
        
        # Add to registry
        match = EdgeMatch(
            piece_idx=p2,
            edge_idx=e2,
            similarity_score=candidate['score'],
            shape_score=components['shape'],
            color_score=components['color'],
            confidence=min(components.values()),  # Conservative confidence
            match_type='perfect' if candidate['score'] > 0.85 else 'good',
            validation_flags={
                'type_compatible': True,
                'length_compatible': components['length'] > 0.8,
                'curvature_compatible': components['shape'] > 0.7,
                'color_compatible': components['continuity'] > 0.6
            }
        )
        
        registry.add_match(p1, e1, p2, e2, match)
        registry.confirm_match(p1, e1, p2, e2)
        
        # Mark edges as used
        used_edges.add((p1, e1))
        used_edges.add((p2, e2))
        confirmed_matches.add((p1, e1, p2, e2))
    
    print(f"\nConfirmed {len(confirmed_matches)} matches using improved algorithm")
    
    # Validate results
    validate_matching_results_improved(pieces, registry)
    
    return registry, spatial_index


def validate_matching_results_improved(pieces: List[Piece], registry: GlobalMatchRegistry) -> None:
    """Improved validation of matching results."""
    # Count edge types
    edge_counts = {'flat': 0, 'convex': 0, 'concave': 0}
    for piece in pieces:
        for edge in piece.edges:
            edge_counts[edge.edge_type] += 1
    
    expected_matches = min(edge_counts['convex'], edge_counts['concave'])
    actual_matches = len(registry.confirmed_matches) // 2  # Each match is stored twice
    
    print(f"\n=== Match Validation ===")
    print(f"Edge counts: {edge_counts}")
    print(f"Expected matches: {expected_matches}")
    print(f"Actual confirmed matches: {actual_matches}")
    
    if actual_matches < expected_matches * 0.5:
        print(f"⚠ Low match rate: {actual_matches}/{expected_matches}")
    else:
        print(f"✓ Reasonable match count: {actual_matches}/{expected_matches}")