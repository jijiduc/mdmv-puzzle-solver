"""Configuration settings for the puzzle solver."""

# Input configuration
INPUT_PATH = "picture/puzzle_6/chickens.png"

# Image processing parameters
THRESHOLD_VALUE = 135
MIN_CONTOUR_AREA = 150

# Processing parameters
DEFAULT_MAX_WORKERS = None  # Use all available cores
DEFAULT_COLOR_RADIUS = 2
DEFAULT_TARGET_EDGE_POINTS = 50

# Output directories - organized by analysis workflow
DEBUG_DIRS = {
    'base': 'debug',
    # Step 1: Input and preprocessing
    'input': 'debug/01_input',
    'preprocessing': 'debug/02_preprocessing',
    # Step 2: Piece detection and extraction
    'detection': 'debug/03_detection',
    'pieces': 'debug/04_pieces',
    # Step 3: Geometric analysis
    'geometry': 'debug/05_geometry',
    # Step 4: Feature analysis (merged colors + edges)
    'features': 'debug/06_features',
    'features_shape': 'debug/06_features/shape',
    'features_color': 'debug/06_features/color',
    # Step 5: Piece classification and matching
    'classification': 'debug/07_piece_classification',
    'matching': 'debug/08_matching',
    # Step 6: Assembly results
    'assembly': 'debug/09_assembly',
    # Legacy mappings for compatibility
    'masks': 'debug/02_preprocessing/masks',
    'contours': 'debug/03_detection/contours',
    'edge_types': 'debug/07_piece_classification/edge_types',
    'edges': 'debug/06_features/shape'
}

# Cache configuration

# Performance settings
HIGH_PRIORITY_PROCESS = True
ENABLE_NUMBA = True