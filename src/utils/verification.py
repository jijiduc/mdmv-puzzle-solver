"""
Verification utilities for puzzle piece detection.
This module provides functions for verifying and filtering detected puzzle pieces.
"""

import numpy as np
import logging
from typing import List, Dict, Any, Optional
import cv2


def final_area_verification(pieces, area_threshold: float = 2.0, expected_pieces: Optional[int] = None):
    """
    Perform final verification based on piece area.
    
    Args:
        pieces: List of PuzzlePiece objects
        area_threshold: Standard deviation threshold for filtering (default: 2.0)
        expected_pieces: Expected number of pieces (for recovery)
    
    Returns:
        List of verified pieces
    """
    # Rest of function remains the same...
        
    if not pieces or len(pieces) < 2:
        return pieces
    
    logger = logging.getLogger(__name__)
    logger.info(f"Performing final area verification with threshold {area_threshold}")
    
    # Extract areas from all pieces
    areas = [piece.features['area'] for piece in pieces]
    
    # Calculate statistics
    mean_area = np.mean(areas)
    std_area = np.std(areas)
    
    logger.info(f"Area statistics: mean={mean_area:.2f}, std={std_area:.2f}")
    
    # Define acceptable range
    min_acceptable = mean_area - area_threshold * std_area
    max_acceptable = mean_area + area_threshold * std_area
    
    logger.info(f"Acceptable area range: {min_acceptable:.2f} to {max_acceptable:.2f}")
    
    # Filter pieces
    verified_pieces = []
    rejected_pieces = []
    
    for piece in pieces:
        area = piece.features['area']
        if min_acceptable <= area <= max_acceptable:
            verified_pieces.append(piece)
        else:
            rejected_pieces.append(piece)
            # Update the piece's validation status
            piece.validation_status = f"area_outlier:{area:.2f}"
            piece.is_valid = False
    
    logger.info(f"Area verification: kept {len(verified_pieces)}/{len(pieces)} pieces, " +
               f"rejected {len(rejected_pieces)} outliers")
    
    # Log details about rejected pieces
    if rejected_pieces:
        rejected_areas = [p.features['area'] for p in rejected_pieces]
        logger.info(f"Rejected areas: {rejected_areas}")
    
     # Recovery step when we know the expected count and have too few pieces
    if expected_pieces and len(verified_pieces) < expected_pieces:
        # Sort rejected pieces by how close they are to the mean area
        rejected_by_distance = [(abs(p.features['area'] - mean_area), p) for p in rejected_pieces]
        rejected_by_distance.sort(key=lambda x: x[0])  # Sort by distance to mean
        
        # Add back just enough pieces to reach the expected count
        pieces_to_recover = rejected_by_distance[:expected_pieces - len(verified_pieces)]
        recovered_pieces = [p for _, p in pieces_to_recover]
        
        logger.info(f"Recovering {len(recovered_pieces)} pieces to match expected count of {expected_pieces}")
        verified_pieces.extend(recovered_pieces)
    
    return verified_pieces


def final_validation_check(pieces, 
                           expected_pieces: Optional[int] = None,
                           area_threshold: float = 2.0,
                           aspect_ratio_threshold: float = 3.0,
                           validation_score_threshold: float = 0.5):
    """
    Comprehensive final validation check that combines multiple criteria.
    
    Args:
        pieces: List of PuzzlePiece objects
        expected_pieces: Expected number of pieces (for adaptive thresholds)
        area_threshold: Standard deviation threshold for area filtering
        aspect_ratio_threshold: Maximum allowed aspect ratio
        validation_score_threshold: Minimum required validation score
    
    Returns:
        List of verified pieces
    """
    if not pieces or len(pieces) < 2:
        return pieces
    
    logger = logging.getLogger(__name__)
    logger.info("Performing comprehensive final validation check")
    
    # First, filter by area
    area_filtered = final_area_verification(pieces, area_threshold, expected_pieces)
    
    # Next, filter by aspect ratio
    ratio_filtered = []
    rejected_by_ratio = []
    
    for piece in area_filtered:
        x, y, w, h = piece.features['bbox']
        aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else float('inf')
        
        if aspect_ratio <= aspect_ratio_threshold:
            ratio_filtered.append(piece)
        else:
            piece.validation_status = f"aspect_ratio_outlier:{aspect_ratio:.2f}"
            piece.is_valid = False
            rejected_by_ratio.append(piece)
    
    logger.info(f"Aspect ratio filtering: rejected {len(rejected_by_ratio)} pieces")
    
    # Finally, filter by validation score
    score_filtered = []
    rejected_by_score = []
    
    for piece in ratio_filtered:
        validation_score = piece.validation_score if hasattr(piece, 'validation_score') else 0.0
        
        if validation_score >= validation_score_threshold:
            score_filtered.append(piece)
        else:
            piece.validation_status = f"low_validation_score:{validation_score:.2f}"
            piece.is_valid = False
            rejected_by_score.append(piece)
    
    logger.info(f"Validation score filtering: rejected {len(rejected_by_score)} pieces")
    
    # If we rejected too many pieces and have expected_pieces, consider relaxing criteria
    if expected_pieces and len(score_filtered) < expected_pieces * 0.8:
        logger.info(f"Too few pieces ({len(score_filtered)}/{expected_pieces}) after filtering. " +
                   f"Relaxing criteria.")
        
        # Relax area threshold
        relaxed_area_threshold = area_threshold * 1.5
        relaxed_pieces = final_area_verification(pieces, relaxed_area_threshold)
        
        logger.info(f"Relaxed criteria: found {len(relaxed_pieces)} pieces")
        return relaxed_pieces
    
    return score_filtered


def create_verification_visualization(image, verified_pieces, rejected_pieces):
    """
    Create a visualization showing verified and rejected pieces.
    
    Args:
        image: Original image
        verified_pieces: List of verified PuzzlePiece objects
        rejected_pieces: List of rejected PuzzlePiece objects
    
    Returns:
        Visualization image
    """
    vis = image.copy()
    
    # Draw verified pieces in green
    for piece in verified_pieces:
        cv2.drawContours(vis, [piece.contour], -1, (0, 255, 0), 2)
        
        # Add piece ID
        M = cv2.moments(piece.contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.putText(vis, f"#{piece.id}", (cx, cy),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Draw rejected pieces in red
    for piece in rejected_pieces:
        cv2.drawContours(vis, [piece.contour], -1, (0, 0, 255), 2)
        
        # Add rejection reason
        M = cv2.moments(piece.contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            reason = piece.validation_status.split(":")[0] if piece.validation_status else "rejected"
            cv2.putText(vis, reason, (cx, cy),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Add legend
    cv2.putText(vis, "Green: Verified Pieces", (20, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.putText(vis, "Red: Rejected Pieces", (20, 70),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    
    return vis