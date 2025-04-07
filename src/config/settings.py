"""
Configuration settings for the puzzle piece detection system
"""

class Config:
    """Global configuration parameters"""
    
    # Debug settings
    DEBUG = True
    DEBUG_DIR = "debug"
    
    # Pipeline selection
    USE_SOBEL_PIPELINE = False  # When True, use the Sobel edge detection pipeline
    
    # Image preprocessing - no resizing
    BLUR_KERNEL_SIZE = (5, 5)
    CLAHE_CLIP_LIMIT = 2.0
    CLAHE_GRID_SIZE = (8, 8)
    
    # Sobel pipeline parameters
    SOBEL_KSIZE = 3  # Increased from 3 for better edge detection 
    MORPH_KERNEL_SIZE_SOBEL = 3
    DILATE_ITERATIONS = 2  # Increased from 1 to better close contours
    ERODE_ITERATIONS = 2
    
    # Thresholding options
    USE_AUTO_THRESHOLD = False  # Auto-select between Otsu and adaptive thresholding
    ADAPTIVE_BLOCK_SIZE = 35
    ADAPTIVE_C = 10
    
    # Morphological operations
    MORPH_KERNEL_SIZE = 5
    MORPH_ITERATIONS = 2
    
    # Canny edge detection
    CANNY_LOW_THRESHOLD = 50
    CANNY_HIGH_THRESHOLD = 150
    
    # Contour filtering - adjusted for full-sized images
    MIN_CONTOUR_AREA = 3000
    MAX_CONTOUR_AREA_RATIO = 0.3  # Percentage of image area
    MIN_CONTOUR_PERIMETER = 50
    SOLIDITY_RANGE = (0.7, 0.99)  # solidity is the ratio of contour area to convex hull area
    ASPECT_RATIO_RANGE = (0.25, 4.0)  # More permissive (was 0.2, 5.0)
    
    # Mean-based contour filtering
    USE_MEAN_FILTERING = True
    MEAN_DEVIATION_THRESHOLD = 1.5  # Increased from 2.0 for more flexibility
    
    # Piece validation
    CORNER_APPROX_EPSILON = 0.03  # Increased from 0.02 for better corner detection
    MIN_CORNERS = 3  # Reduced from 4 to catch more pieces
    MAX_CORNERS = 12  # Increased from 10 to handle more complex pieces
    
    # Border classification
    BORDER_TYPES = ["straight", "tab", "pocket"]
    TAB_DEVIATION_THRESHOLD = 12  # Increased from 10 for better tab detection
    TAB_COMPLEXITY_THRESHOLD = 1.3  # Increased from 1.2 for better complexity measurement
    BORDER_SMOOTHING_KERNEL = (5, 5)
    
    # Visualization
    CONTOUR_THICKNESS = 2
    CORNER_RADIUS = 5
    BORDER_COLORS = {
        "straight": (0, 255, 0),  # Green
        "tab": (0, 0, 255),       # Red
        "pocket": (255, 0, 0)     # Blue
    }
    
    # Performance
    USE_MULTIPROCESSING = True
    NUM_PROCESSES = 4  # Set to number of CPU cores for optimal performance