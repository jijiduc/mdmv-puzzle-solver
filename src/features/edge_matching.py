"""Edge matching data structures and utilities for puzzle assembly."""

import heapq
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set, Optional, Any
from collections import defaultdict
import time


@dataclass
class EdgeMatch:
    """Represents a potential match between two puzzle piece edges."""
    piece_idx: int
    edge_idx: int
    similarity_score: float  # Combined score (0-1)
    shape_score: float      # Shape similarity (0-1)
    color_score: float      # Color similarity (0-1)
    confidence: float       # Match confidence
    match_type: str         # "perfect", "good", "possible"
    validation_flags: Dict[str, bool] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate match type based on scores."""
        if self.similarity_score >= 0.9:
            self.match_type = "perfect"
        elif self.similarity_score >= 0.7:
            self.match_type = "good"
        else:
            self.match_type = "possible"
    
    def is_valid(self) -> bool:
        """Check if all validation flags are True."""
        return all(self.validation_flags.values()) if self.validation_flags else True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'piece_idx': self.piece_idx,
            'edge_idx': self.edge_idx,
            'similarity_score': self.similarity_score,
            'shape_score': self.shape_score,
            'color_score': self.color_score,
            'confidence': self.confidence,
            'match_type': self.match_type,
            'validation_flags': self.validation_flags
        }


@dataclass
class GlobalMatchRegistry:
    """Central registry for all edge matches in the puzzle."""
    
    def __init__(self):
        # Bidirectional match lookup
        # Key: (piece1_idx, edge1_idx), Value: {(piece2_idx, edge2_idx): EdgeMatch}
        self.matches: Dict[Tuple[int, int], Dict[Tuple[int, int], EdgeMatch]] = defaultdict(dict)
        
        # Quick access to confirmed matches
        self.confirmed_matches: Set[Tuple[int, int, int, int]] = set()
        
        # Match statistics for optimization
        self.match_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        
        # Track match history
        self.match_history: List[Tuple[float, Tuple[int, int, int, int]]] = []
    
    def add_match(self, piece1_idx: int, edge1_idx: int, 
                  piece2_idx: int, edge2_idx: int, match: EdgeMatch) -> None:
        """Add a match to the registry."""
        # Add bidirectional entries
        self.matches[(piece1_idx, edge1_idx)][(piece2_idx, edge2_idx)] = match
        
        # Create reverse match
        reverse_match = EdgeMatch(
            piece_idx=piece1_idx,
            edge_idx=edge1_idx,
            similarity_score=match.similarity_score,
            shape_score=match.shape_score,
            color_score=match.color_score,
            confidence=match.confidence,
            match_type=match.match_type,
            validation_flags=match.validation_flags
        )
        self.matches[(piece2_idx, edge2_idx)][(piece1_idx, edge1_idx)] = reverse_match
        
        # Record timestamp
        self.match_history.append((time.time(), (piece1_idx, edge1_idx, piece2_idx, edge2_idx)))
    
    def confirm_match(self, piece1_idx: int, edge1_idx: int, 
                     piece2_idx: int, edge2_idx: int) -> bool:
        """Confirm a match as final."""
        # Check if match exists
        if (piece2_idx, edge2_idx) not in self.matches.get((piece1_idx, edge1_idx), {}):
            return False
        
        # Add to confirmed set
        self.confirmed_matches.add((piece1_idx, edge1_idx, piece2_idx, edge2_idx))
        self.confirmed_matches.add((piece2_idx, edge2_idx, piece1_idx, edge1_idx))
        
        return True
    
    def get_match(self, piece1_idx: int, edge1_idx: int, 
                  piece2_idx: int, edge2_idx: int) -> Optional[EdgeMatch]:
        """Get a specific match."""
        return self.matches.get((piece1_idx, edge1_idx), {}).get((piece2_idx, edge2_idx))
    
    def get_best_matches(self, piece_idx: int, edge_idx: int, 
                        n: int = 5) -> List[Tuple[Tuple[int, int], EdgeMatch]]:
        """Get top N matches for a given edge."""
        edge_matches = self.matches.get((piece_idx, edge_idx), {})
        sorted_matches = sorted(edge_matches.items(), 
                              key=lambda x: x[1].similarity_score, 
                              reverse=True)
        return sorted_matches[:n]
    
    def update_statistics(self, edge_type1: str, edge_type2: str) -> None:
        """Update match statistics for edge type combinations."""
        self.match_stats[edge_type1][edge_type2] += 1
        self.match_stats[edge_type2][edge_type1] += 1
    
    def get_match_probability(self, edge_type1: str, edge_type2: str) -> float:
        """Get historical match probability for edge type combination."""
        total_matches = sum(self.match_stats[edge_type1].values())
        if total_matches == 0:
            return 0.0
        return self.match_stats[edge_type1][edge_type2] / total_matches


@dataclass
class MatchEvaluationCache:
    """Cache for expensive match computations."""
    
    def __init__(self, cache_size: int = 10000):
        self.cache_size = cache_size
        
        # Cache computed similarities
        self.shape_similarities: Dict[Tuple[int, int, int, int], float] = {}
        self.color_similarities: Dict[Tuple[int, int, int, int], float] = {}
        
        # Cache expensive computations (reserved for future use)
        
        # Invalidation tracking
        self.last_modified: Dict[Tuple[int, int], float] = {}
        
        # Access tracking for LRU
        self.access_times: Dict[Any, float] = {}
    
    def _make_key(self, piece1_idx: int, edge1_idx: int, 
                  piece2_idx: int, edge2_idx: int) -> Tuple[int, int, int, int]:
        """Create canonical key for bidirectional lookup."""
        # Always use smaller indices first for consistency
        if (piece1_idx, edge1_idx) < (piece2_idx, edge2_idx):
            return (piece1_idx, edge1_idx, piece2_idx, edge2_idx)
        return (piece2_idx, edge2_idx, piece1_idx, edge1_idx)
    
    def get_shape_similarity(self, piece1_idx: int, edge1_idx: int,
                            piece2_idx: int, edge2_idx: int) -> Optional[float]:
        """Get cached shape similarity."""
        key = self._make_key(piece1_idx, edge1_idx, piece2_idx, edge2_idx)
        if key in self.shape_similarities:
            self.access_times[('shape', key)] = time.time()
            return self.shape_similarities[key]
        return None
    
    def set_shape_similarity(self, piece1_idx: int, edge1_idx: int,
                            piece2_idx: int, edge2_idx: int, similarity: float) -> None:
        """Cache shape similarity."""
        key = self._make_key(piece1_idx, edge1_idx, piece2_idx, edge2_idx)
        self.shape_similarities[key] = similarity
        self.access_times[('shape', key)] = time.time()
        self._enforce_cache_limit()
    
    def get_color_similarity(self, piece1_idx: int, edge1_idx: int,
                            piece2_idx: int, edge2_idx: int) -> Optional[float]:
        """Get cached color similarity."""
        key = self._make_key(piece1_idx, edge1_idx, piece2_idx, edge2_idx)
        if key in self.color_similarities:
            self.access_times[('color', key)] = time.time()
            return self.color_similarities[key]
        return None
    
    def set_color_similarity(self, piece1_idx: int, edge1_idx: int,
                            piece2_idx: int, edge2_idx: int, similarity: float) -> None:
        """Cache color similarity."""
        key = self._make_key(piece1_idx, edge1_idx, piece2_idx, edge2_idx)
        self.color_similarities[key] = similarity
        self.access_times[('color', key)] = time.time()
        self._enforce_cache_limit()
    
    def invalidate_edge(self, piece_idx: int, edge_idx: int) -> None:
        """Invalidate all cached data for a specific edge."""
        self.last_modified[(piece_idx, edge_idx)] = time.time()
        
        # Remove from all caches
        keys_to_remove = []
        for key in self.shape_similarities:
            if (piece_idx, edge_idx) in [(key[0], key[1]), (key[2], key[3])]:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            self.shape_similarities.pop(key, None)
            self.color_similarities.pop(key, None)
            self.access_times.pop(('shape', key), None)
            self.access_times.pop(('color', key), None)
    
    def _enforce_cache_limit(self) -> None:
        """Remove least recently used items if cache is too large."""
        total_items = len(self.shape_similarities) + len(self.color_similarities)
        
        if total_items > self.cache_size:
            # Sort by access time
            sorted_items = sorted(self.access_times.items(), key=lambda x: x[1])
            
            # Remove oldest 10%
            items_to_remove = int(self.cache_size * 0.1)
            for (cache_type, key), _ in sorted_items[:items_to_remove]:
                if cache_type == 'shape':
                    self.shape_similarities.pop(key, None)
                elif cache_type == 'color':
                    self.color_similarities.pop(key, None)
                self.access_times.pop((cache_type, key), None)


class MatchPriorityQueue:
    """Priority queue for iterative puzzle assembly."""
    
    def __init__(self):
        self.heap: List[Tuple[float, Tuple[int, int, int, int], EdgeMatch]] = []
        self.processed: Set[Tuple[int, int, int, int]] = set()
        self.edge_usage: Dict[Tuple[int, int], bool] = {}  # Track used edges
    
    def push(self, match: EdgeMatch, piece1_idx: int, edge1_idx: int) -> None:
        """Add a match to the priority queue."""
        key = (piece1_idx, edge1_idx, match.piece_idx, match.edge_idx)
        
        # Skip if already processed or edges are used
        if key in self.processed:
            return
        if self.edge_usage.get((piece1_idx, edge1_idx), False):
            return
        if self.edge_usage.get((match.piece_idx, match.edge_idx), False):
            return
        
        # Use negative score for max heap behavior
        heapq.heappush(self.heap, (-match.similarity_score, key, match))
    
    def pop(self) -> Optional[Tuple[Tuple[int, int, int, int], EdgeMatch]]:
        """Get the best available match."""
        while self.heap:
            _, key, match = heapq.heappop(self.heap)
            
            # Check if edges are still available
            piece1_idx, edge1_idx, piece2_idx, edge2_idx = key
            if (not self.edge_usage.get((piece1_idx, edge1_idx), False) and
                not self.edge_usage.get((piece2_idx, edge2_idx), False)):
                
                # Mark edges as used
                self.edge_usage[(piece1_idx, edge1_idx)] = True
                self.edge_usage[(piece2_idx, edge2_idx)] = True
                self.processed.add(key)
                
                return key, match
        
        return None
    
    def mark_edge_used(self, piece_idx: int, edge_idx: int) -> None:
        """Mark an edge as used."""
        self.edge_usage[(piece_idx, edge_idx)] = True
    
    def is_edge_available(self, piece_idx: int, edge_idx: int) -> bool:
        """Check if an edge is available for matching."""
        return not self.edge_usage.get((piece_idx, edge_idx), False)
    
    def size(self) -> int:
        """Get number of matches in queue."""
        return len(self.heap)


@dataclass
class EdgeSpatialIndex:
    """Spatial index for efficient edge candidate filtering."""
    
    def __init__(self):
        # Group edges by type and characteristics
        self.by_type: Dict[str, Dict[str, List[Tuple[int, int]]]] = defaultdict(lambda: defaultdict(list))
        self.by_length_range: Dict[Tuple[float, float], List[Tuple[int, int]]] = defaultdict(list)
        self.by_color_cluster: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        
        # Configuration
        self.length_bin_size = 50.0  # pixels
        self.length_tolerance = 0.2  # 20% tolerance
    
    def add_edge(self, piece_idx: int, edge_idx: int, edge_type: str, 
                 sub_type: Optional[str], length: float, 
                 color_cluster_id: Optional[int] = None) -> None:
        """Add an edge to the spatial index."""
        # Index by type
        self.by_type[edge_type][sub_type or "none"].append((piece_idx, edge_idx))
        
        # Index by length range
        length_bin = self._get_length_bin(length)
        self.by_length_range[length_bin].append((piece_idx, edge_idx))
        
        # Index by color cluster if provided
        if color_cluster_id is not None:
            self.by_color_cluster[color_cluster_id].append((piece_idx, edge_idx))
    
    def get_compatible_edges(self, edge_type: str, sub_type: Optional[str], 
                           length: float) -> List[Tuple[int, int]]:
        """Get edges that could potentially match based on type and length."""
        candidates = []
        
        # Get compatible edge types
        compatible_types = self._get_compatible_types(edge_type)
        
        for comp_type in compatible_types:
            # For sub-types, prefer matching symmetric with symmetric
            if sub_type == "symmetric":
                candidates.extend(self.by_type[comp_type].get("symmetric", []))
            elif sub_type == "asymmetric":
                candidates.extend(self.by_type[comp_type].get("asymmetric", []))
            else:
                # Include all sub-types if no sub-type specified
                for edges in self.by_type[comp_type].values():
                    candidates.extend(edges)
        
        # Filter by length compatibility
        length_compatible = self._get_length_compatible_edges(length)
        
        # Intersection of type-compatible and length-compatible
        candidates_set = set(candidates)
        length_set = set(length_compatible)
        
        return list(candidates_set & length_set)
    
    def _get_compatible_types(self, edge_type: str) -> List[str]:
        """Get edge types that can match with given type."""
        if edge_type == "flat":
            return []  # Flat edges are not matched (puzzle border)
        elif edge_type == "convex":
            return ["concave"]
        elif edge_type == "concave":
            return ["convex"]
        else:
            return []
    
    def _get_length_bin(self, length: float) -> Tuple[float, float]:
        """Get length bin for given edge length."""
        bin_index = int(length / self.length_bin_size)
        return (bin_index * self.length_bin_size, 
                (bin_index + 1) * self.length_bin_size)
    
    def _get_length_compatible_edges(self, length: float) -> List[Tuple[int, int]]:
        """Get edges with compatible lengths."""
        min_length = length * (1 - self.length_tolerance)
        max_length = length * (1 + self.length_tolerance)
        
        compatible_edges = []
        
        # Check all length bins that could contain compatible edges
        for (bin_min, bin_max), edges in self.by_length_range.items():
            if bin_max >= min_length and bin_min <= max_length:
                compatible_edges.extend(edges)
        
        return compatible_edges
    
    def get_edges_by_color_cluster(self, cluster_id: int) -> List[Tuple[int, int]]:
        """Get all edges in a specific color cluster."""
        return self.by_color_cluster.get(cluster_id, [])
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get index statistics for debugging."""
        stats = {
            'total_edges': sum(len(edges) for subtype_dict in self.by_type.values() 
                             for edges in subtype_dict.values()),
            'edge_types': {edge_type: sum(len(edges) for edges in subtype_dict.values())
                          for edge_type, subtype_dict in self.by_type.items()},
            'length_bins': len(self.by_length_range),
            'color_clusters': len(self.by_color_cluster)
        }
        return stats