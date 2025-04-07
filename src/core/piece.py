"""
Enhanced puzzle piece representation and processing
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import sys
import os
import logging

# Add parent directory to path to allow imports from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.contour_utils import (
    calculate_contour_features, enhanced_find_corners, extract_borders, 
    enhanced_classify_border, validate_shape_as_puzzle_piece
)
from src.config.settings import Config


class PuzzlePiece:
    """
    Enhanced class representing a detected puzzle piece
    """
    
    def __init__(self, 
                image: np.ndarray, 
                contour: np.ndarray, 
                config: Config = None):
        """
        Initialize a puzzle piece from an image and contour
        
        Args:
            image: Source image
            contour: Contour of the piece
            config: Configuration parameters
        """
        self.config = config or Config()
        self.image = image
        self.contour = contour
        self.id = None  # Unique identifier for the piece
        self.logger = logging.getLogger(__name__)
        
        # Properties to be calculated
        self.corners = None
        self.borders = None
        self.border_types = None
        self.features = None
        self.extracted_image = None
        self.is_valid = False
        self.validation_status = None
        self.validation_score = 0.0
        self.neighbors = {}  # Dictionary of {neighbor_id: matching_score}
        
        # Process the piece
        self._extract_features()
        self._extract_piece_image()
        self._find_corners()
        
        # If corners are found, continue processing
        if self.corners is not None and len(self.corners) >= 4:
            self._process_borders()
            self._validate_piece()
        else:
            self.validation_status = "invalid_corners"
    
    def _extract_features(self) -> None:
        """Extract comprehensive features from the contour"""
        self.features = calculate_contour_features(self.contour)
    
    def _extract_piece_image(self) -> None:
        """Extract the piece image using the contour mask with enhanced visualization"""
        try:
            # Create mask from contour
            mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [self.contour], 0, 255, -1)
            
            # Apply mask to extract piece
            self.extracted_image = cv2.bitwise_and(self.image, self.image, mask=mask)
            
            # Crop to bounding rectangle with a small margin
            x, y, w, h = self.features['bbox']
            margin = 5  # Small margin to avoid cutting off edges
            x_min = max(0, x - margin)
            y_min = max(0, y - margin)
            x_max = min(self.image.shape[1], x + w + margin)
            y_max = min(self.image.shape[0], y + h + margin)
            
            self.extracted_image = self.extracted_image[y_min:y_max, x_min:x_max]
            
            # Create a cleaner white background version
            # First convert cropped mask to match the cropped image
            cropped_mask = mask[y_min:y_max, x_min:x_max]
            
            # Create white background
            white_bg = np.ones_like(self.extracted_image) * 255
            
            # Copy only the piece pixels over the white background
            mask_3ch = cv2.merge([cropped_mask, cropped_mask, cropped_mask])
            self.extracted_image = np.where(mask_3ch > 0, self.extracted_image, white_bg)
            
            self.validation_status = "valid_extraction"
        except Exception as e:
            self.logger.error(f"Error extracting piece image: {str(e)}")
            self.validation_status = f"extraction_error: {str(e)}"

    def get_extracted_image(self, clean_background: bool = True) -> np.ndarray:
        """
        Get the extracted piece image with options for background
        
        Args:
            clean_background: If True, return piece with clean white background
                            If False, return piece with original cropped background
        
        Returns:
            Image of just the piece
        """
        if self.extracted_image is None:
            # If extraction failed, create a simple visualization
            result = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.putText(result, "No extraction", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (0, 0, 255), 1)
            return result
        
        if clean_background:
            # Create a clean white background
            white_bg = np.ones_like(self.extracted_image) * 255
            
            # Create mask from the extracted image
            # Convert to grayscale if it's color
            if len(self.extracted_image.shape) == 3:
                gray = cv2.cvtColor(self.extracted_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = self.extracted_image
                
            # Threshold to get mask
            _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
            
            # Expand mask to 3 channels if needed
            if len(self.extracted_image.shape) == 3:
                mask_3ch = cv2.merge([mask, mask, mask])
            else:
                mask_3ch = mask
                
            # Combine piece and white background
            result = np.where(mask_3ch > 0, self.extracted_image, white_bg)
            return result
        
        return self.extracted_image.copy()
    
    def _find_corners(self) -> None:
        """Find corners of the piece using enhanced detection"""
        try:
            # Use enhanced corner detection with adaptive parameters
            self.corners = enhanced_find_corners(
                self.contour, 
                approx_epsilon=self.config.CORNER_APPROX_EPSILON,
                use_adaptive_epsilon=True,
                corner_refinement=True
            )
            
            # Check if we have a reasonable number of corners
            if self.corners is None or len(self.corners) < self.config.MIN_CORNERS:
                self.validation_status = f"too_few_corners: {len(self.corners) if self.corners is not None else 0}"
                self.corners = None
            elif len(self.corners) > self.config.MAX_CORNERS:
                self.validation_status = f"too_many_corners: {len(self.corners)}"
                
                # Instead of rejecting, try to filter corners to get a reasonable number
                # Sort corners by significance (how "sharp" they are)
                corner_significance = []
                for i, corner in enumerate(self.corners):
                    # Calculate significance based on angle
                    # Get preceding and following points in the contour
                    idx = np.argmin(np.sum((self.contour[:, 0, :] - corner) ** 2, axis=1))
                    prev_idx = (idx - 5) % len(self.contour)
                    next_idx = (idx + 5) % len(self.contour)
                    
                    # Calculate vectors
                    v1 = self.contour[prev_idx][0] - corner
                    v2 = self.contour[next_idx][0] - corner
                    
                    # Calculate angle
                    dot = np.dot(v1, v2)
                    mag1 = np.linalg.norm(v1)
                    mag2 = np.linalg.norm(v2)
                    cos_angle = dot / (mag1 * mag2) if mag1 * mag2 > 0 else 0
                    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
                    
                    # Sharper angles (closer to 0) are more significant
                    significance = np.pi - angle
                    corner_significance.append((i, significance))
                
                # Sort by significance (descending)
                corner_significance.sort(key=lambda x: x[1], reverse=True)
                
                # Keep only the most significant corners up to MAX_CORNERS
                significant_indices = [x[0] for x in corner_significance[:self.config.MAX_CORNERS]]
                self.corners = self.corners[significant_indices]
                
                self.validation_status = f"filtered_corners: {len(self.corners)}"
                
        except Exception as e:
            self.logger.error(f"Error finding corners: {str(e)}")
            self.validation_status = f"corner_detection_error: {str(e)}"
            self.corners = None
    
    def _process_borders(self) -> None:
        """Process and classify borders between corners with enhanced detection"""
        try:
            # Extract border segments
            self.borders = extract_borders(self.contour, self.corners)
            
            # Classify each border with adaptive thresholds
            self.border_types = []
            for border in self.borders:
                border_type = enhanced_classify_border(
                    border, 
                    complexity_threshold=self.config.TAB_COMPLEXITY_THRESHOLD,
                    deviation_threshold=self.config.TAB_DEVIATION_THRESHOLD,
                    use_adaptive_thresholds=True
                )
                self.border_types.append(border_type)
                
            # Calculate border statistics
            if self.border_types:
                # Count each type of border
                border_counts = {
                    "straight": self.border_types.count("straight"),
                    "tab": self.border_types.count("tab"),
                    "pocket": self.border_types.count("pocket")
                }
                
                self.features['border_counts'] = border_counts
                
        except Exception as e:
            self.logger.error(f"Error processing borders: {str(e)}")
            self.validation_status = f"border_processing_error: {str(e)}"
            self.borders = None
            self.border_types = None
    
    def _validate_piece(self) -> None:
        """Validate if this is a valid puzzle piece using multiple criteria"""
        # Start with default valid state
        self.is_valid = True
        score_components = []
        
        # Check if we have borders and border types
        if self.borders is None or self.border_types is None:
            self.is_valid = False
            self.validation_status = "missing_borders"
            return
        
        # 1. Check border type distribution (puzzle pieces usually have tabs and pockets)
        border_counts = {
            "straight": self.border_types.count("straight"),
            "tab": self.border_types.count("tab"),
            "pocket": self.border_types.count("pocket")
        }
        
        # Typical puzzle pieces have a mix of border types
        # They often have a similar number of tabs and pockets
        tab_pocket_diff = abs(border_counts["tab"] - border_counts["pocket"])
        tab_pocket_balance = 1.0 - (tab_pocket_diff / max(len(self.border_types), 1))
        
        # Non-edge puzzle pieces typically have few straight borders
        straight_ratio = border_counts["straight"] / max(len(self.border_types), 1)
        
        # Pieces on the edge of the puzzle will have more straight borders
        is_edge_piece = straight_ratio > 0.2
        
        # Score based on expected border distribution
        if is_edge_piece:
            # Edge pieces should have some straight borders but also some tabs/pockets
            border_score = 0.5 + 0.5 * (1.0 - abs(straight_ratio - 0.25))
        else:
            # Interior pieces should have good tab/pocket balance and few straight borders
            border_score = 0.7 * tab_pocket_balance + 0.3 * (1.0 - straight_ratio)
        
        score_components.append(("border_distribution", border_score))
        
        # 2. Check shape properties
        shape_score = 0.0
        
        # Puzzle pieces typically have a specific range of compactness values
        compactness = self.features.get('compactness', 0)
        if 2.0 <= compactness <= 8.0:
            # Ideal range: score peaks at around 5
            compactness_score = 1.0 - min(abs(compactness - 5.0) / 5.0, 1.0)
        else:
            compactness_score = max(0.0, 1.0 - abs(compactness - 5.0) / 10.0)
        
        # Solidity (area / convex hull area) is typically high for puzzle pieces
        solidity = self.features.get('solidity', 0)
        solidity_score = min(solidity / 0.85, 1.0) if solidity <= 0.95 else 2.0 - solidity / 0.95
        
        # Combine shape scores
        shape_score = 0.6 * compactness_score + 0.4 * solidity_score
        score_components.append(("shape_properties", shape_score))
        
        # 3. Check corner count and distribution
        corner_score = 0.0
        
        # Puzzle pieces typically have 4-6 corners
        num_corners = len(self.corners)
        if 4 <= num_corners <= 8:
            corner_count_score = 1.0
        elif num_corners < 4:
            corner_count_score = 0.5  # Too few corners
        else:
            corner_count_score = max(0.0, 1.0 - (num_corners - 8) / 10.0)  # Too many corners
        
        score_components.append(("corner_count", corner_count_score))
        
        # 4. Overall validation using general puzzle piece detector
        validation_score = 1.0 if validate_shape_as_puzzle_piece(self.contour) else 0.5
        score_components.append(("shape_validation", validation_score))
        
        # Calculate final validation score (weighted average)
        weights = {
            "border_distribution": 0.35,
            "shape_properties": 0.30,
            "corner_count": 0.15,
            "shape_validation": 0.20
        }
        
        self.validation_score = sum(weights[name] * score for name, score in score_components)
        
        # Piece is valid if it passes a threshold
        threshold = 0.65  # Fairly permissive
        self.is_valid = self.validation_score >= threshold
        
        # Set validation status
        if self.is_valid:
            if is_edge_piece:
                self.validation_status = f"valid_edge_piece:{self.validation_score:.2f}"
            else:
                self.validation_status = f"valid_interior_piece:{self.validation_score:.2f}"
        else:
            self.validation_status = f"invalid_piece:{self.validation_score:.2f}"
            
            # Add details about failure reasons
            failed_components = [name for name, score in score_components if score < 0.5]
            if failed_components:
                self.validation_status += f":failed:{','.join(failed_components)}"
    
    def draw(self, image: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Enhanced visualization of the piece with contour, corners, and borders
        
        Args:
            image: Optional image to draw on (if None, use original image)
        
        Returns:
            Image with visualization of the piece
        """
        if image is None:
            vis_img = self.image.copy()
        else:
            vis_img = image.copy()
        
        # Draw contour
        color = (0, 255, 0) if self.is_valid else (0, 0, 255)  # Green for valid, red for invalid
        cv2.drawContours(vis_img, [self.contour], -1, color, 2)
        
        # Draw piece ID if available
        if self.id is not None:
            # Calculate centroid for text placement
            M = cv2.moments(self.contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Draw ID with contrasting background for visibility
                text = f"#{self.id}"
                text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(vis_img, 
                             (cx - 5, cy - text_size[1] - 5), 
                             (cx + text_size[0] + 5, cy + 5), 
                             (0, 0, 0), 
                             -1)
                cv2.putText(vis_img, text, (cx, cy),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw validation score if available
        if hasattr(self, 'validation_score') and self.validation_score > 0:
            # Find top-left corner of piece for text placement
            x, y, _, _ = self.features['bbox']
            score_text = f"{self.validation_score:.2f}"
            cv2.putText(vis_img, score_text, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Draw corners if available
        if self.corners is not None:
            for i, corner in enumerate(self.corners):
                x, y = corner.astype(int)
                cv2.circle(vis_img, (x, y), self.config.CORNER_RADIUS, (0, 0, 255), -1)
                cv2.putText(vis_img, str(i), (x+10, y+10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Draw borders with color-coding if available
        if self.borders is not None and self.border_types is not None:
            for border, border_type in zip(self.borders, self.border_types):
                color = self.config.BORDER_COLORS.get(border_type, (255, 255, 255))
                pts = border.reshape((-1, 1, 2)).astype(np.int32)
                cv2.polylines(vis_img, [pts], False, color, self.config.CONTOUR_THICKNESS)
        
        return vis_img
    
    def calculate_match_score(self, other_piece: 'PuzzlePiece') -> Dict[str, float]:
        """
        Calculate how well this piece might match with another piece
        
        Args:
            other_piece: Another puzzle piece to compare with
        
        Returns:
            Dictionary of match scores for different criteria
        """
        if not self.is_valid or not other_piece.is_valid:
            return {'overall': 0.0}
        
        scores = {}
        
        # Check if the pieces have complementary borders (tab<->pocket)
        if self.border_types and other_piece.border_types:
            # Look for potential tab-pocket matches
            tab_pocket_matches = 0
            tab_count = self.border_types.count('tab')
            pocket_count = self.border_types.count('pocket')
            other_tab_count = other_piece.border_types.count('tab')
            other_pocket_count = other_piece.border_types.count('pocket')
            
            # Perfect matching would have this piece's tabs matching other's pockets
            # and this piece's pockets matching other's tabs
            potential_matches = min(tab_count, other_pocket_count) + min(pocket_count, other_tab_count)
            
            if potential_matches > 0:
                scores['border_complementarity'] = potential_matches / max(len(self.border_types), 1)
            else:
                scores['border_complementarity'] = 0.0
        else:
            scores['border_complementarity'] = 0.0
        
        # Check color similarity along edges (a simple approximation)
        if self.extracted_image is not None and other_piece.extracted_image is not None:
            # For now, a simplified color check
            # In a real implementation, we would compare colors along matching borders
            scores['color_similarity'] = 0.5  # Placeholder
        else:
            scores['color_similarity'] = 0.0
        
        # Calculate overall match score (weighted components)
        scores['overall'] = 0.8 * scores['border_complementarity'] + 0.2 * scores['color_similarity']
        
        return scores
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert piece to dictionary representation
        
        Args:
            Dictionary with piece data
        """
        # Create a serializable copy of features
        serializable_features = {}
        for key, value in self.features.items():
            if isinstance(value, np.ndarray):
                serializable_features[key] = value.tolist()
            elif key == 'min_area_rect':
                # Handle min_area_rect which has a special structure
                center, size, angle = value
                serializable_features[key] = {
                    'center': (float(center[0]), float(center[1])),
                    'size': (float(size[0]), float(size[1])),
                    'angle': float(angle)
                }
            elif key == 'moments':
                # Convert moments (OpenCV dictionary with numpy types)
                serializable_features[key] = {k: float(v) for k, v in value.items()}
            elif key == 'hu_moments':
                # Convert hu_moments (numpy array)
                serializable_features[key] = value.flatten().tolist()
            else:
                serializable_features[key] = value
        
        return {
            'id': self.id,
            'features': serializable_features,
            'corners': self.corners.tolist() if self.corners is not None else None,
            'borders': [b.tolist() for b in self.borders] if self.borders is not None else None,
            'border_types': self.border_types,
            'is_valid': self.is_valid,
            'validation_status': self.validation_status,
            'validation_score': self.validation_score if hasattr(self, 'validation_score') else 0.0
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], image: np.ndarray, contour: np.ndarray, config: Config = None) -> 'PuzzlePiece':
        """
        Create piece from dictionary representation
        
        Args:
            data: Dictionary with piece data
            image: Source image
            contour: Contour of the piece
            config: Configuration parameters
        
        Returns:
            PuzzlePiece object
        """
        piece = cls(image, contour, config)
        
        # Override with saved data
        piece.id = data.get('id')
        
        if data.get('corners') is not None:
            piece.corners = np.array(data['corners'])
            
        if data.get('borders') is not None:
            piece.borders = [np.array(b) for b in data['borders']]
            
        piece.border_types = data.get('border_types')
        piece.is_valid = data.get('is_valid', False)
        piece.validation_status = data.get('validation_status')
        
        if 'validation_score' in data:
            piece.validation_score = data['validation_score']
        
        return piece