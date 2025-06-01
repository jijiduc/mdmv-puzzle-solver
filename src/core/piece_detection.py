"""Individual puzzle piece processing and analysis."""

import cv2
import numpy as np
import math
import gc
import os
from typing import Dict, List, Tuple, Any

from .geometry import extract_edge_between_corners, classify_edge
from ..features.edge_extraction import extract_dtw_edge_features
from .piece import Piece, EdgeSegment
from .corner_detection_proper import find_puzzle_corners


def process_piece(piece: Piece, output_dirs: Tuple[str, ...]) -> Dict[str, Any]:
    """Process a single puzzle piece - optimized for performance.
    
    Args:
        piece: Piece object to process
        output_dirs: Tuple of output directory paths
        
    Returns:
        Dictionary containing processed piece data
    """
    piece_index = piece.index
    
    # Get output paths
    edges_dir, edge_types_dir, contours_dir = output_dirs
    
    # Get image and mask from piece
    piece_img = piece.image
    piece_mask = piece.mask
    
    # Get contour points directly from mask
    contours, _ = cv2.findContours(piece_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if not contours:
        return {
            'piece_idx': piece_index,
            'edge_types': [],
            'edge_deviations': [],
            'edge_sub_types': [],
            'edge_confidences': [],
            'edge_points': [],
            'edge_colors': [],
            'corners': [],
            'centroid': (0, 0),
            'img': piece_img.tolist(),
            'mask': piece_mask.tolist(),
            'piece_type': 'unknown'
        }
    
    # Use the largest contour (should be the only one)
    contour = max(contours, key=cv2.contourArea)
    
    # Extract edge coordinates from contour
    edge_coordinates = contour.squeeze()
    if len(edge_coordinates.shape) == 1:
        edge_coordinates = edge_coordinates.reshape(-1, 2)
    
    # Use piece's centroid
    centroid = piece.centroid
    centroid_x, centroid_y = centroid
    
    # Calculate distances and angles from centroid
    distances = []
    angles = []
    coords = []
    
    for x, y in edge_coordinates:
        # Distance from centroid
        dist = math.sqrt((x - centroid_x)**2 + (y - centroid_y)**2)
        distances.append(dist)
        
        # Angle from centroid
        angle = math.atan2(y - centroid_y, x - centroid_x)
        angles.append(angle)
        coords.append((x, y))
    
    # Sort by angle for proper ordering
    if angles:
        sorted_data = sorted(zip(angles, distances, coords))
        sorted_angles, sorted_distances, sorted_coords = zip(*sorted_data)
        sorted_angles = list(sorted_angles)
        sorted_distances = list(sorted_distances)
        sorted_coords = list(sorted_coords)
    else:
        sorted_angles, sorted_distances, sorted_coords = [], [], []
    
    # Find corners using proper polar distance profile analysis
    corners = find_puzzle_corners(sorted_distances, sorted_coords, sorted_angles)
    
    # Set corners on the piece
    piece.set_corners(corners)
    
    # Extract edges between consecutive corners
    edge_types = []
    edge_deviations = []
    edge_sub_types = []
    edge_confidences = []
    edge_points_list = []
    edge_colors = []
    
    for i in range(len(corners)):
        next_i = (i + 1) % len(corners)
        
        # Extract edge points between corners
        edge_points = extract_edge_between_corners(corners, i, next_i, 
                                                  np.array(sorted_coords), centroid)
        
        # Create EdgeSegment
        edge_segment = EdgeSegment(
            points=edge_points.tolist() if isinstance(edge_points, np.ndarray) else edge_points,
            corner1=corners[i],
            corner2=corners[next_i],
            piece_idx=piece_index,
            edge_idx=i
        )
        
        if len(edge_points) > 0:
            # Classify edge type (pass piece and edge indices for debug)
            edge_type, deviation, sub_type, confidence = classify_edge(edge_points, corners[i], corners[next_i], centroid, 
                                                                      piece_index, i)
            edge_segment.edge_type = edge_type
            edge_segment.deviation = deviation
            edge_segment.sub_type = sub_type
            edge_segment.confidence = confidence
            edge_types.append(edge_type)
            edge_deviations.append(deviation)
            edge_sub_types.append(sub_type)
            edge_confidences.append(confidence)
            edge_points_list.append(edge_points)
            
            # Extract edge features with improved color sampling
            edge_features = extract_dtw_edge_features(piece_img, edge_points, 
                                                    corners[i], corners[next_i], i, piece_mask)
            edge_colors.append(edge_features)
            
            # Update edge segment with features
            if 'edge_length' in edge_features:
                edge_segment.length = edge_features['edge_length']
            if 'curvature' in edge_features:
                edge_segment.curvature = edge_features['curvature']
            if 'color_sequence' in edge_features:
                edge_segment.color_sequence = edge_features['color_sequence']
            if 'confidence_sequence' in edge_features:
                edge_segment.confidence_sequence = edge_features['confidence_sequence']
            if 'texture_descriptor' in edge_features:
                edge_segment.texture_descriptor = edge_features['texture_descriptor']
        else:
            edge_types.append("unknown")
            edge_deviations.append(0)
            edge_sub_types.append(None)
            edge_confidences.append(0.0)
            edge_points_list.append([])
            edge_colors.append({})
        
        # Add edge to piece
        piece.add_edge(edge_segment)
    
    # Classify piece type based on edges
    piece._classify_piece_type()
    
    # Store color patterns for meaningful edges
    for edge_idx, edge in enumerate(piece.edges):
        if edge.edge_type != "flat" and edge.color_sequence:
            piece.edge_color_patterns[edge_idx] = {
                'color_sequence': edge.color_sequence,
                'confidence_sequence': edge.confidence_sequence,
                'edge_type': edge.edge_type,
                'sub_type': edge.sub_type
            }
    
    # Memory cleanup
    gc.collect()
    
    return {
        'piece_idx': piece_index,
        'edge_types': edge_types,
        'edge_deviations': edge_deviations,
        'edge_sub_types': edge_sub_types,
        'edge_confidences': edge_confidences,
        'edge_points': edge_points_list,
        'edge_colors': edge_colors,
        'corners': corners,
        'centroid': centroid,
        'img': piece_img.tolist(),  # Add piece image for visualization
        'mask': piece_mask.tolist(),  # Add piece mask
        'piece_type': piece.piece_type  # Add piece classification
    }


