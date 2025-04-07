"""
Utility functions for image processing
"""

import cv2
import numpy as np
import os
from typing import Tuple, Optional, Union, List


def read_image(path: str) -> np.ndarray:
    """
    Read an image from file
    
    Args:
        path: Path to the image file
    
    Returns:
        Image as numpy array
    
    Raises:
        FileNotFoundError: If the image file doesn't exist
        ValueError: If the image couldn't be read
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image file not found: {path}")
    
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Unable to read image from {path}")
    
    return image


def save_image(image: np.ndarray, path: str) -> None:
    """
    Save an image to file
    
    Args:
        image: Image as numpy array
        path: Destination path
    
    Returns:
        None
    """
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    
    cv2.imwrite(path, image)


def resize_image(image: np.ndarray, 
                 width: Optional[int] = None, 
                 height: Optional[int] = None, 
                 scale: Optional[float] = None) -> np.ndarray:
    """
    Resize an image with various options
    
    Args:
        image: Input image
        width: Target width (if None, calculated from height and aspect ratio)
        height: Target height (if None, calculated from width and aspect ratio)
        scale: Scale factor (overrides width and height if provided)
    
    Returns:
        Resized image
    """
    if scale is not None:
        return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    
    h, w = image.shape[:2]
    
    if width is None and height is None:
        return image
    
    if width is None:
        aspect_ratio = w / h
        width = int(height * aspect_ratio)
    elif height is None:
        aspect_ratio = h / w
        height = int(width * aspect_ratio)
    
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def enhance_contrast(image: np.ndarray) -> np.ndarray:
    """
    Enhance image contrast using CLAHE
    
    Args:
        image: Input image (color or grayscale)
    
    Returns:
        Contrast-enhanced image
    """
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Convert to LAB color space for better contrast enhancement
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_l = clahe.apply(l)
        
        # Merge channels and convert back to BGR
        enhanced_lab = cv2.merge([enhanced_l, a, b])
        return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    else:
        # Grayscale image
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image)


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Preprocess image for puzzle piece detection - simplified version
    
    Args:
        image: Input color image
    
    Returns:
        Preprocessed grayscale image
    """
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    # Enhanced version with blurring for better noise reduction
    return clahe.apply(blurred)


def adaptive_threshold(image: np.ndarray, 
                       block_size: int = 19, 
                       c: int = 7,
                       invert: bool = True) -> np.ndarray:
    """
    Apply adaptive thresholding to an image
    
    Args:
        image: Input grayscale image
        block_size: Size of pixel neighborhood for thresholding
        c: Constant subtracted from mean
        invert: Whether to invert the threshold result
    
    Returns:
        Binary image
    """
    if invert:
        return cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, block_size, c
        )
    else:
        return cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, block_size, c
        )


def apply_morphology(image: np.ndarray, 
                    operation: int = cv2.MORPH_CLOSE, 
                    kernel_size: int = 5,
                    iterations: int = 1) -> np.ndarray:
    """
    Apply morphological operations to an image
    
    Args:
        image: Binary input image
        operation: Morphological operation (cv2.MORPH_*)
        kernel_size: Size of the structuring element
        iterations: Number of times to apply the operation
    
    Returns:
        Processed binary image
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.morphologyEx(image, operation, kernel, iterations=iterations)


def detect_edges(image: np.ndarray, 
                low_threshold: int = 50, 
                high_threshold: int = 150) -> np.ndarray:
    """
    Detect edges in an image using Canny edge detector
    
    Args:
        image: Input grayscale image
        low_threshold: Lower threshold for hysteresis
        high_threshold: Upper threshold for hysteresis
    
    Returns:
        Binary edge image
    """
    return cv2.Canny(image, low_threshold, high_threshold)


def create_blank_image(shape: Tuple[int, int], 
                       color: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    """
    Create a blank image with specified shape and color
    
    Args:
        shape: (height, width) tuple
        color: BGR color tuple
    
    Returns:
        Blank image
    """
    if len(shape) == 2:
        height, width = shape
        channels = 3  # Default to BGR
    else:
        height, width, channels = shape
    
    if channels == 1:
        return np.ones((height, width), dtype=np.uint8) * color[0]
    else:
        return np.ones((height, width, channels), dtype=np.uint8) * color


def overlay_mask(image: np.ndarray, 
                mask: np.ndarray, 
                color: Tuple[int, int, int] = (0, 0, 255),
                alpha: float = 0.5) -> np.ndarray:
    """
    Overlay a colored mask on an image
    
    Args:
        image: Input color image
        mask: Binary mask
        color: BGR color for the overlay
        alpha: Transparency factor (0.0 to 1.0)
    
    Returns:
        Image with colored overlay
    """
    colored_mask = np.zeros_like(image)
    colored_mask[mask > 0] = color
    
    return cv2.addWeighted(image, 1, colored_mask, alpha, 0)


def find_contours_improved(image: np.ndarray) -> List[np.ndarray]:
    """
    Improved contour finding with Otsu thresholding and morphological operations
    
    Args:
        image: Input color image
    
    Returns:
        List of detected contours
    """
    # Pre-process the image
    preprocessed = preprocess_image(image)
    
    # Apply Otsu thresholding
    _, threshold = cv2.threshold(preprocessed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphological operations to reduce noise and consolidate contours
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morphed = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)
    
    # Find contours with less sensitive parameters
    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter out very small contours
    min_contour_area = 500
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
    
    return filtered_contours

def compare_threshold_methods(preprocessed_image: np.ndarray, 
                          adaptive_block_size: int = 35, 
                          adaptive_c: int = 10,
                          area_threshold: float = 200) -> Tuple[np.ndarray, str, dict[str, any]]:
    """
    Compare Otsu and adaptive thresholding methods and return the best one
    
    Args:
        preprocessed_image: Preprocessed grayscale image
        adaptive_block_size: Block size for adaptive thresholding
        adaptive_c: Constant for adaptive thresholding
        area_threshold: Minimum area to consider a contour significant
    
    Returns:
        Tuple of (best binary image, method name, metrics dictionary)
    """
    # Apply Otsu thresholding
    _, otsu_binary = cv2.threshold(preprocessed_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Apply adaptive thresholding
    adaptive_binary = cv2.adaptiveThreshold(
        preprocessed_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, adaptive_block_size, adaptive_c
    )
    
    # Evaluate each method
    methods = {
        'otsu': otsu_binary,
        'adaptive': adaptive_binary
    }
    
    results = {}
    
    for method_name, binary in methods.items():
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area
        significant_contours = [c for c in contours if cv2.contourArea(c) > area_threshold]
        
        # Calculate additional metrics
        avg_area = np.mean([cv2.contourArea(c) for c in significant_contours]) if significant_contours else 0
        avg_complexity = np.mean([cv2.arcLength(c, True) / (4 * np.sqrt(cv2.contourArea(c))) 
                               for c in significant_contours]) if significant_contours else 0
        
        results[method_name] = {
            'contour_count': len(significant_contours),
            'avg_area': avg_area,
            'avg_complexity': avg_complexity
        }
    
    # Determine the best method
    # For now, prioritize the method that finds more contours
    if results['otsu']['contour_count'] >= results['adaptive']['contour_count']:
        best_method = 'otsu'
        best_binary = otsu_binary
    else:
        best_method = 'adaptive'
        best_binary = adaptive_binary
    
    return best_binary, best_method, results