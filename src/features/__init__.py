"""Feature extraction modules for puzzle pieces."""

from .edge_matching import (
    EdgeMatch,
    GlobalMatchRegistry,
    MatchEvaluationCache,
    MatchPriorityQueue,
    EdgeSpatialIndex
)

__all__ = [
    'EdgeMatch',
    'GlobalMatchRegistry',
    'MatchEvaluationCache',
    'MatchPriorityQueue',
    'EdgeSpatialIndex'
]