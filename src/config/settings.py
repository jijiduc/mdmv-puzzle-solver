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
    # Step 3: Feature analysis
    'geometry': 'debug/05_geometry',
    'colors': 'debug/06_colors', 
    'edges': 'debug/07_edges',
    # Step 4: Classification and matching
    'classification': 'debug/08_classification',
    'matching': 'debug/09_matching',
    # Step 5: Assembly results
    'assembly': 'debug/10_assembly',
    # Legacy mappings for compatibility
    'masks': 'debug/02_preprocessing/masks',
    'corners': 'debug/05_geometry/corners',
    'contours': 'debug/03_detection/contours',
    'edge_types': 'debug/08_classification/edge_types',
    'color_features': 'debug/06_colors/features'
}

# Cache configuration
CACHE_DIR = '.cache'
ENABLE_CACHING = True

# Performance settings
HIGH_PRIORITY_PROCESS = True
ENABLE_NUMBA = True