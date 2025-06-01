"""Puzzle piece representation and management."""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
from ..features.edge_matching import EdgeMatch


@dataclass
class EdgeSegment:
    """Represents a single edge of a puzzle piece."""
    points: List[Tuple[int, int]] = field(default_factory=list)
    corner1: Optional[Tuple[int, int]] = None
    corner2: Optional[Tuple[int, int]] = None
    edge_type: str = "unknown"  # flat/convex/concave
    sub_type: Optional[str] = None  # symmetric/asymmetric
    deviation: float = 0.0
    confidence: float = 0.0  # classification confidence
    length: float = 0.0
    curvature: float = 0.0
    color_sequence: List[List[float]] = field(default_factory=list)  # LAB color sequence
    confidence_sequence: List[float] = field(default_factory=list)
    normalized_points: List[Tuple[float, float]] = field(default_factory=list)
    texture_descriptor: Optional[Dict[str, np.ndarray]] = None  # Texture features
    piece_idx: int = -1
    edge_idx: int = -1
    
    # New fields for improved matching
    direction_vector: Optional[Tuple[float, float]] = None  # Start to end direction
    start_point: Optional[Tuple[int, int]] = None
    end_point: Optional[Tuple[int, int]] = None
    normal_vector: Optional[Tuple[float, float]] = None  # Perpendicular to edge
    
    def calculate_direction_vectors(self):
        """Calculate direction and normal vectors for this edge."""
        if not self.points or len(self.points) < 2:
            return
            
        # Set start and end points
        self.start_point = self.points[0]
        self.end_point = self.points[-1]
        
        # Calculate direction vector (normalized)
        dx = self.end_point[0] - self.start_point[0]
        dy = self.end_point[1] - self.start_point[1]
        length = np.sqrt(dx*dx + dy*dy)
        
        if length > 0:
            self.direction_vector = (dx/length, dy/length)
            # Normal vector (perpendicular, pointing "outward")
            self.normal_vector = (-dy/length, dx/length)
        else:
            self.direction_vector = (0.0, 0.0)
            self.normal_vector = (0.0, 0.0)


class Piece:
    """Represents a single puzzle piece with all its properties."""
    
    def __init__(self, index: int, image: np.ndarray, mask: np.ndarray, bbox: Optional[Tuple[int, int, int, int]] = None):
        """Initialize a puzzle piece.
        
        Args:
            index: Unique identifier for the piece
            image: Color image of the piece (BGR)
            mask: Binary mask of the piece
            bbox: Bounding box (x, y, width, height) if available
        """
        # Core data
        self.index = index
        self.image = image
        self.mask = mask
        self.bbox = bbox if bbox is not None else self._calculate_bbox()
        
        # Geometric properties
        self.corners: List[Tuple[int, int]] = []
        self.centroid: Optional[Tuple[int, int]] = None
        self.edges: List[EdgeSegment] = []
        
        # Classification
        self.piece_type: Optional[str] = None  # corner/edge/middle
        self.rotation: float = 0.0  # estimated rotation
        
        # Matching information
        self.neighbors: Dict[int, Optional['Piece']] = {0: None, 1: None, 2: None, 3: None}
        self.edge_matches: Dict[int, List[EdgeMatch]] = {}  # edge_idx -> [EdgeMatch objects]
        
        # Color patterns for meaningful edges (non-flat)
        self.edge_color_patterns: Dict[int, Dict[str, Any]] = {}  # edge_idx -> color pattern data
        
        # Calculate centroid
        self._calculate_centroid()
    
    def _calculate_bbox(self) -> Tuple[int, int, int, int]:
        """Calculate bounding box from mask."""
        if self.mask is None or self.mask.size == 0:
            return (0, 0, 0, 0)
        
        # Find non-zero pixels
        coords = np.column_stack(np.where(self.mask > 0))
        if len(coords) == 0:
            return (0, 0, 0, 0)
        
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        return (x_min, y_min, x_max - x_min + 1, y_max - y_min + 1)
    
    def _calculate_centroid(self) -> None:
        """Calculate the centroid of the piece from its mask."""
        import cv2
        
        if self.mask is None or self.mask.size == 0:
            self.centroid = (self.mask.shape[1] // 2, self.mask.shape[0] // 2)
            return
        
        moments = cv2.moments(self.mask)
        if moments["m00"] != 0:
            centroid_x = int(moments["m10"] / moments["m00"])
            centroid_y = int(moments["m01"] / moments["m00"])
            self.centroid = (centroid_x, centroid_y)
        else:
            self.centroid = (self.mask.shape[1] // 2, self.mask.shape[0] // 2)
    
    @property
    def area(self) -> int:
        """Calculate the area of the piece in pixels."""
        if self.mask is None:
            return 0
        return int(np.sum(self.mask > 0))
    
    @property
    def perimeter(self) -> float:
        """Calculate the perimeter of the piece."""
        import cv2
        
        if self.mask is None:
            return 0.0
        
        # Find contours
        contours, _ = cv2.findContours(self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if contours:
            return cv2.arcLength(contours[0], True)
        return 0.0
    
    def get_edge(self, edge_idx: int) -> Optional[EdgeSegment]:
        """Get a specific edge by index.
        
        Args:
            edge_idx: Index of the edge (0-3)
            
        Returns:
            EdgeSegment or None if not found
        """
        if 0 <= edge_idx < len(self.edges):
            return self.edges[edge_idx]
        return None
    
    def set_corners(self, corners: List[Tuple[int, int]]) -> None:
        """Set the corner points of the piece.
        
        Args:
            corners: List of 4 corner coordinates
        """
        if len(corners) != 4:
            raise ValueError(f"Expected 4 corners, got {len(corners)}")
        self.corners = corners
        self._classify_piece_type()
    
    def add_edge(self, edge: EdgeSegment) -> None:
        """Add an edge segment to the piece.
        
        Args:
            edge: EdgeSegment to add
        """
        edge.piece_idx = self.index
        edge.edge_idx = len(self.edges)
        self.edges.append(edge)
    
    def _classify_piece_type(self) -> None:
        """Classify the piece as corner, edge, or middle based on edge types."""
        if not self.edges:
            self.piece_type = None
            return
        
        # Count edge types
        straight_count = sum(1 for edge in self.edges if edge.edge_type == "flat")
        
        if straight_count == 2:
            # Check if straight edges are adjacent (corner piece)
            straight_indices = [i for i, edge in enumerate(self.edges) if edge.edge_type == "flat"]
            if len(straight_indices) == 2:
                diff = abs(straight_indices[1] - straight_indices[0])
                if diff == 1 or diff == 3:  # Adjacent edges
                    self.piece_type = "corner"
                else:
                    self.piece_type = "middle"  # Opposite edges (shouldn't happen in standard puzzles)
        elif straight_count == 1:
            self.piece_type = "edge"
        elif straight_count == 0:
            self.piece_type = "middle"
        else:
            self.piece_type = "unknown"  # More than 2 straight edges (unusual)
    
    def rotate(self, angle: float) -> None:
        """Rotate the piece by a given angle.
        
        Args:
            angle: Rotation angle in degrees
        """
        import cv2
        
        # Update rotation tracking
        self.rotation = (self.rotation + angle) % 360
        
        # Get rotation matrix
        center = self.centroid if self.centroid else (self.image.shape[1] // 2, self.image.shape[0] // 2)
        rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Rotate image and mask
        self.image = cv2.warpAffine(self.image, rot_matrix, (self.image.shape[1], self.image.shape[0]))
        self.mask = cv2.warpAffine(self.mask, rot_matrix, (self.mask.shape[1], self.mask.shape[0]))
        
        # Update geometric properties
        self._calculate_centroid()
        self.bbox = self._calculate_bbox()
        
        # Rotate corners and edge points if they exist
        if self.corners:
            self.corners = self._rotate_points(self.corners, rot_matrix)
        
        for edge in self.edges:
            if edge.points:
                edge.points = self._rotate_points(edge.points, rot_matrix)
            if edge.corner1:
                edge.corner1 = self._rotate_point(edge.corner1, rot_matrix)
            if edge.corner2:
                edge.corner2 = self._rotate_point(edge.corner2, rot_matrix)
    
    def _rotate_points(self, points: List[Tuple[int, int]], rot_matrix: np.ndarray) -> List[Tuple[int, int]]:
        """Rotate a list of points using the rotation matrix."""
        rotated_points = []
        for x, y in points:
            new_point = rot_matrix @ np.array([x, y, 1])
            rotated_points.append((int(new_point[0]), int(new_point[1])))
        return rotated_points
    
    def _rotate_point(self, point: Tuple[int, int], rot_matrix: np.ndarray) -> Tuple[int, int]:
        """Rotate a single point using the rotation matrix."""
        x, y = point
        new_point = rot_matrix @ np.array([x, y, 1])
        return (int(new_point[0]), int(new_point[1]))
    
    def add_edge_match(self, edge_idx: int, match: EdgeMatch) -> None:
        """Add a match for a specific edge.
        
        Args:
            edge_idx: Index of the edge on this piece
            match: EdgeMatch object containing match details
        """
        if edge_idx not in self.edge_matches:
            self.edge_matches[edge_idx] = []
        self.edge_matches[edge_idx].append(match)
        # Sort by similarity score (descending)
        self.edge_matches[edge_idx].sort(key=lambda m: m.similarity_score, reverse=True)
    
    def get_best_match(self, edge_idx: int) -> Optional[EdgeMatch]:
        """Get the best match for a specific edge.
        
        Args:
            edge_idx: Index of the edge
            
        Returns:
            Best EdgeMatch or None if no matches
        """
        if edge_idx in self.edge_matches and self.edge_matches[edge_idx]:
            return self.edge_matches[edge_idx][0]
        return None
    
    def get_edge_matches(self, edge_idx: int, min_score: float = 0.0) -> List[EdgeMatch]:
        """Get all matches for an edge above a minimum score.
        
        Args:
            edge_idx: Index of the edge
            min_score: Minimum similarity score threshold
            
        Returns:
            List of EdgeMatch objects above the threshold
        """
        if edge_idx not in self.edge_matches:
            return []
        return [m for m in self.edge_matches[edge_idx] if m.similarity_score >= min_score]
    
    def clear_edge_matches(self, edge_idx: Optional[int] = None) -> None:
        """Clear matches for a specific edge or all edges.
        
        Args:
            edge_idx: Index of edge to clear, or None to clear all
        """
        if edge_idx is None:
            self.edge_matches.clear()
        elif edge_idx in self.edge_matches:
            del self.edge_matches[edge_idx]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert piece to dictionary for serialization."""
        return {
            'index': self.index,
            'img': self.image.tolist(),
            'mask': self.mask.tolist(),
            'bbox': self.bbox,
            'corners': self.corners,
            'centroid': self.centroid,
            'piece_type': self.piece_type,
            'rotation': self.rotation,
            'edges': [
                {
                    'edge_type': edge.edge_type,
                    'sub_type': edge.sub_type,
                    'deviation': edge.deviation,
                    'confidence': edge.confidence,
                    'length': edge.length,
                    'curvature': edge.curvature,
                    'edge_idx': edge.edge_idx
                } for edge in self.edges
            ],
            'edge_matches': {
                str(edge_idx): [match.to_dict() for match in matches]
                for edge_idx, matches in self.edge_matches.items()
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Piece':
        """Create piece from dictionary."""
        piece = cls(
            index=data['index'],
            image=np.array(data['img'], dtype=np.uint8),
            mask=np.array(data['mask'], dtype=np.uint8),
            bbox=tuple(data.get('bbox', []))
        )
        
        if 'corners' in data:
            piece.corners = [tuple(c) for c in data['corners']]
        if 'centroid' in data and data['centroid']:
            piece.centroid = tuple(data['centroid'])
        if 'piece_type' in data:
            piece.piece_type = data['piece_type']
        if 'rotation' in data:
            piece.rotation = data['rotation']
        
        return piece