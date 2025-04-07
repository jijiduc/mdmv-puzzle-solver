"""
Puzzle piece representation and processing
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import sys
import os

# Add parent directory to path to allow imports from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.contour_utils import (
    calculate_contour_features, find_corners, extract_borders, classify_border
)
from src.config.settings import Config


class PuzzlePiece:
    """
    Class representing a detected puzzle piece
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
        
        # Properties to be calculated
        self.corners = None
        self.borders = None
        self.border_types = None
        self.features = None
        self.extracted_image = None
        self.is_valid = False
        self.validation_status = None
        
        # Process the piece
        self._extract_features()
        self._extract_piece_image()
        self._find_corners()
        if self.corners is not None and len(self.corners) >= 4:
            self._process_borders()
            self.is_valid = True
    
    def _extract_features(self) -> None:
        """Extract features from the contour"""
        self.features = calculate_contour_features(self.contour)
    
    def _extract_piece_image(self) -> None:
        """Extract the piece image using the contour mask with enhanced visualization"""
        try:
            # Create mask from contour
            mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [self.contour], 0, 255, -1)
            
            # Apply mask to extract piece
            self.extracted_image = cv2.bitwise_and(self.image, self.image, mask=mask)
            
            # Crop to bounding rectangle
            x, y, w, h = self.features['bbox']
            self.extracted_image = self.extracted_image[y:y+h, x:x+w]
            
            # Create a cleaner white background version
            # First convert cropped mask to match the cropped image
            cropped_mask = mask[y:y+h, x:x+w]
            
            # Create white background
            white_bg = np.ones_like(self.extracted_image) * 255
            
            # Copy only the piece pixels over the white background
            mask_3ch = cv2.merge([cropped_mask, cropped_mask, cropped_mask])
            self.extracted_image = np.where(mask_3ch > 0, self.extracted_image, white_bg)
            
            self.validation_status = "valid"
        except Exception as e:
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
        """Find corners of the piece"""
        try:
            self.corners = find_corners(
                self.contour, 
                approx_epsilon=self.config.CORNER_APPROX_EPSILON
            )
            
            # Check if we have a reasonable number of corners
            if len(self.corners) < self.config.MIN_CORNERS:
                self.validation_status = f"too_few_corners: {len(self.corners)}"
                self.corners = None
            elif len(self.corners) > self.config.MAX_CORNERS:
                self.validation_status = f"too_many_corners: {len(self.corners)}"
                self.corners = None
                
        except Exception as e:
            self.validation_status = f"corner_detection_error: {str(e)}"
            self.corners = None
    
    def _process_borders(self) -> None:
        """Process and classify borders between corners"""
        try:
            # Extract border segments
            self.borders = extract_borders(self.contour, self.corners)
            
            # Classify each border
            self.border_types = []
            for border in self.borders:
                border_type = classify_border(
                    border, 
                    complexity_threshold=self.config.TAB_COMPLEXITY_THRESHOLD,
                    deviation_threshold=self.config.TAB_DEVIATION_THRESHOLD
                )
                self.border_types.append(border_type)
                
        except Exception as e:
            self.validation_status = f"border_processing_error: {str(e)}"
            self.borders = None
            self.border_types = None
    
    def draw(self, image: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Draw the piece with contour, corners, and borders
        
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
        cv2.drawContours(vis_img, [self.contour], -1, (0, 255, 0), 2)
        
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
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert piece to dictionary representation
        
        Returns:
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
            'validation_status': self.validation_status
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
        
        return piece