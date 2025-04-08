"""
Enhanced configuration settings for the puzzle piece detection system
with adaptive parameter optimization
"""

class Config:
    """Global configuration parameters with adaptive capabilities"""
    
    # Debug settings
    DEBUG = False
    DEBUG_DIR = "debug"
    
    # Pipeline selection
    USE_SOBEL_PIPELINE = False  # When True, use the Sobel edge detection pipeline
    USE_ADAPTIVE_PREPROCESSING = True  # When True, use adaptive preprocessing for best channel
    USE_PARAMETER_OPTIMIZATION = True  # When True, try multiple parameter sets
    
    # Image preprocessing - no resizing
    BLUR_KERNEL_SIZE = (5, 5)
    CLAHE_CLIP_LIMIT = 2.0
    CLAHE_GRID_SIZE = (8, 8)
    
    # Sobel pipeline parameters
    SOBEL_KSIZE = 3  # Kernel size for Sobel operator
    MORPH_KERNEL_SIZE_SOBEL = 3
    DILATE_ITERATIONS = 2  # Increased from 1 to better close contours
    ERODE_ITERATIONS = 2
    
    # Thresholding options
    USE_AUTO_THRESHOLD = True  # Auto-select between Otsu and adaptive thresholding
    ADAPTIVE_BLOCK_SIZE = 35
    ADAPTIVE_C = 10
    
    # Morphological operations
    MORPH_KERNEL_SIZE = 5
    MORPH_ITERATIONS = 2
    
    # Canny edge detection
    CANNY_LOW_THRESHOLD = 50
    CANNY_HIGH_THRESHOLD = 150
    
    # Multi-pass detection
    USE_MULTI_PASS_DETECTION = True  # When True, try multiple detection passes
    
    # Contour filtering - adjusted for full-sized images
    MIN_CONTOUR_AREA = 3000
    MAX_CONTOUR_AREA_RATIO = 0.3  # Percentage of image area
    MIN_CONTOUR_PERIMETER = 50
    SOLIDITY_RANGE = (0.7, 0.99)  # solidity is the ratio of contour area to convex hull area
    ASPECT_RATIO_RANGE = (0.25, 4.0)  # More permissive (was 0.2, 5.0)
    
    # Mean-based contour filtering
    USE_MEAN_FILTERING = True
    MEAN_DEVIATION_THRESHOLD = 1.5  # Standard deviation multiplier for area filtering
    
    # Piece validation
    CORNER_APPROX_EPSILON = 0.03  # Polygon approximation accuracy
    MIN_CORNERS = 3  # Minimum number of corners for a valid piece
    MAX_CORNERS = 12  # Maximum number of corners for a valid piece
    VALIDATION_THRESHOLD = 0.65  # Minimum validation score for a piece to be considered valid
    
    # Border classification
    BORDER_TYPES = ["straight", "tab", "pocket"]
    TAB_DEVIATION_THRESHOLD = 12  # Threshold for tab/pocket detection
    TAB_COMPLEXITY_THRESHOLD = 1.3  # Threshold for border complexity
    BORDER_SMOOTHING_KERNEL = (5, 5)
    
    # Recovery parameters
    USE_PIECE_RECOVERY = True  # Try to recover missed pieces
    RECOVERY_MIN_AREA_FACTOR = 0.7  # Factor to reduce min area for recovery
    
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
    NUM_PROCESSES = 8  # Set to number of CPU cores for optimal performance
    
    @classmethod
    def from_dict(cls, param_dict):
        """
        Create a config instance from a dictionary of parameters
        
        Args:
            param_dict: Dictionary of parameter values
            
        Returns:
            Config instance with updated parameters
        """
        config = cls()
        for key, value in param_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config
    
    def to_dict(self):
        """
        Convert config to dictionary
        
        Returns:
            Dictionary of parameter values
        """
        result = {}
        for key in dir(self):
            # Skip private and method attributes
            if not key.startswith('_') and not callable(getattr(self, key)):
                result[key] = getattr(self, key)
        return result
    
    def update(self, **kwargs):
        """
        Update config parameters
        
        Args:
            **kwargs: Parameter name-value pairs
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                
    def optimize_for_image_characteristics(self, image_analysis):
        """
        Update parameters based on image analysis
        
        Args:
            image_analysis: Dictionary with image analysis results
        """
        # Adaptive preprocessing for low contrast images
        if image_analysis.get('contrast', 1.0) < 0.4:
            self.USE_ADAPTIVE_PREPROCESSING = True
            self.CLAHE_CLIP_LIMIT = 3.0  # More aggressive contrast enhancement
        
        # Adjust mean filtering threshold based on histogram characteristics
        if image_analysis.get('is_bimodal', False):
            # More strict filtering for clear bimodal images
            self.MEAN_DEVIATION_THRESHOLD = 1.2
        else:
            # More permissive for complex histograms
            self.MEAN_DEVIATION_THRESHOLD = 2.0
        
        # Adjust contour area threshold based on image size and background
        if image_analysis.get('is_dark_background', True):
            # Dark backgrounds typically need less aggressive filtering
            self.MIN_CONTOUR_AREA = max(1000, self.MIN_CONTOUR_AREA * 0.8)
        else:
            # Light or variable backgrounds may need more filtering
            self.MIN_CONTOUR_AREA = min(5000, self.MIN_CONTOUR_AREA * 1.2)
        
        # Adjust corner detection based on edge density
        edge_density = image_analysis.get('edge_density', 0.1)
        if edge_density > 0.2:
            # Images with high edge density need more precise corner detection
            self.CORNER_APPROX_EPSILON = 0.02
        elif edge_density < 0.05:
            # Images with low edge density need more lenient corner detection
            self.CORNER_APPROX_EPSILON = 0.04