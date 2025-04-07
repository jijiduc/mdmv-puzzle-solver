"""
Enhanced utility functions for image processing with adaptive techniques
"""

import cv2
import numpy as np
import os
from typing import Tuple, Optional, Union, List, Dict, Any
from scipy import ndimage


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


def analyze_image(image: np.ndarray) -> Dict[str, Any]:
    """
    Analyze image characteristics to determine optimal processing parameters
    
    Args:
        image: Input color image
    
    Returns:
        Dictionary with image analysis results
    """
    results = {}
    
    # Convert to grayscale for certain analyses
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Calculate histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    
    # Calculate basic statistics
    results['mean'] = np.mean(gray)
    results['std'] = np.std(gray)
    results['median'] = np.median(gray)
    
    # Calculate histogram peaks (modes)
    peaks = []
    for i in range(1, 255):
        if hist[i] > hist[i-1] and hist[i] > hist[i+1] and hist[i] > 0.01 * gray.size:
            peaks.append((i, hist[i][0]))
    
    # Sort peaks by height (frequency)
    peaks.sort(key=lambda x: x[1], reverse=True)
    results['histogram_peaks'] = peaks[:5]  # Store top 5 peaks
    
    # Check for bimodal histogram (typical for puzzle pieces on dark background)
    is_bimodal = False
    if len(peaks) >= 2:
        peak_values = [p[0] for p in peaks[:2]]
        peak_values.sort()
        
        # Check if peaks are far apart (suggesting foreground and background)
        if peak_values[1] - peak_values[0] > 50:
            is_bimodal = True
    
    results['is_bimodal'] = is_bimodal
    
    # Calculate otsu threshold - useful to separate background and foreground
    otsu_thresh, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    results['otsu_threshold'] = otsu_thresh
    
    # Estimate background color
    # For black background, the first peak is usually the background
    if is_bimodal and peaks[0][0] < 100:  # If darkest peak is dark and prominent
        results['background_is_dark'] = True
        results['background_value'] = peaks[0][0]
    else:
        # Try to determine if background is dark or light
        dark_ratio = np.sum(gray < 100) / gray.size
        if dark_ratio > 0.5:  # If more than 50% of image is dark
            results['background_is_dark'] = True
            results['background_value'] = np.mean(gray[gray < 100])
        else:
            results['background_is_dark'] = False
            results['background_value'] = np.mean(gray[gray > 155])
    
    # Calculate noise estimation
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    noise = gray.astype(np.float32) - blurred.astype(np.float32)
    results['noise_level'] = np.std(noise)
    
    # Image contrast estimation
    p5 = np.percentile(gray, 5)
    p95 = np.percentile(gray, 95)
    results['contrast'] = (p95 - p5) / 255.0
    
    # Color analysis if it's a color image
    if len(image.shape) == 3:
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        results['saturation_mean'] = np.mean(hsv[:,:,1])
        results['value_mean'] = np.mean(hsv[:,:,2])
        
        # Calculate color variance (useful for determining if pieces are varied in color)
        results['color_variance'] = np.mean(np.var(image, axis=(0, 1)))
    
    return results


def enhance_contrast(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    """
    Enhanced contrast adjustment using CLAHE with adaptive parameters
    
    Args:
        image: Input image (color or grayscale)
        clip_limit: Threshold for contrast limiting (adaptive)
        tile_grid_size: Size of grid for histogram equalization
    
    Returns:
        Contrast-enhanced image
    """
    # Analyze image to determine optimal parameters
    img_analysis = analyze_image(image)
    
    # Adapt clip limit based on image contrast
    adaptive_clip_limit = clip_limit
    if img_analysis.get('contrast', 0) < 0.3:  # Low contrast image
        adaptive_clip_limit = clip_limit * 2  # More aggressive enhancement
    elif img_analysis.get('contrast', 0) > 0.7:  # High contrast image
        adaptive_clip_limit = clip_limit * 0.7  # Less aggressive enhancement
    
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Convert to LAB color space for better contrast enhancement
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel with adaptive parameters
        clahe = cv2.createCLAHE(clipLimit=adaptive_clip_limit, tileGridSize=tile_grid_size)
        enhanced_l = clahe.apply(l)
        
        # Merge channels and convert back to BGR
        enhanced_lab = cv2.merge([enhanced_l, a, b])
        return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    else:
        # Grayscale image
        clahe = cv2.createCLAHE(clipLimit=adaptive_clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(image)


def multi_channel_preprocess(image: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Advanced preprocessing using multiple color spaces and channels
    
    Args:
        image: Input color image
    
    Returns:
        Tuple of (best preprocessed image, dictionary of all channel representations)
    """
    # Analyze image first
    analysis = analyze_image(image)
    
    # Create dictionary to store all processed channels
    channels = {}
    
    # Basic grayscale conversion
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    channels['gray'] = gray.copy()
    
    # Apply basic blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    channels['blurred_gray'] = blurred
    
    # Enhanced contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)
    channels['enhanced_gray'] = enhanced
    
    # HSV color space processing (good for color segmentation)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Store individual HSV channels
    channels['hsv_hue'] = h
    channels['hsv_saturation'] = s
    channels['hsv_value'] = v
    
    # Enhanced value channel (often good for separating puzzle pieces)
    enhanced_v = clahe.apply(v)
    channels['enhanced_value'] = enhanced_v
    
    # LAB color space processing
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Store individual LAB channels
    channels['lab_lightness'] = l
    channels['lab_a'] = a
    channels['lab_b'] = b
    
    # Enhanced lightness channel
    enhanced_l = clahe.apply(l)
    channels['enhanced_lightness'] = enhanced_l
    
    # Create specialized feature channels to highlight edges
    # Sobel gradient magnitude
    sobelx = cv2.Sobel(enhanced, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(enhanced, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = cv2.magnitude(sobelx, sobely)
    sobel_mag = cv2.normalize(sobel_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    channels['sobel_magnitude'] = sobel_mag
    
    # Background removal based on analysis
    if analysis['background_is_dark']:
        # For dark background, we can enhance the contrast between background and pieces
        background_mask = (gray < analysis['background_value'] + 30)
        foreground = gray.copy()
        foreground[background_mask] = 0
        foreground = clahe.apply(foreground)
        channels['background_removed'] = foreground
    
    # Determine the best channel to return based on image analysis
    # For puzzle pieces on dark background, enhanced value channel or background removal often works best
    if analysis['background_is_dark']:
        if analysis['contrast'] < 0.4:
            best_channel = channels['enhanced_value']  # For low contrast images
        else:
            best_channel = channels['background_removed']  # For normal contrast images
    else:
        best_channel = channels['enhanced_gray']  # Default
    
    return best_channel, channels


def adaptive_preprocess(image: np.ndarray) -> np.ndarray:
    """
    Smart preprocessing that adapts to image characteristics
    
    Args:
        image: Input color image
    
    Returns:
        Preprocessed grayscale image optimized for puzzle piece detection
    """
    # Analyze image properties
    analysis = analyze_image(image)
    
    # Convert to grayscale if not already
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply appropriate blur based on noise level
    blur_size = 5  # Default
    if analysis['noise_level'] > 10:
        blur_size = 7  # More aggressive blur for noisy images
    elif analysis['noise_level'] < 5:
        blur_size = 3  # Less aggressive blur for clean images
    
    blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
    
    # Apply adaptive contrast enhancement
    clip_limit = 2.0  # Default
    
    if analysis['contrast'] < 0.3:
        clip_limit = 3.0  # Stronger enhancement for low contrast
    elif analysis['contrast'] > 0.7:
        clip_limit = 1.0  # Weaker enhancement for high contrast
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)
    
    # For very dark or uneven backgrounds, apply additional processing
    if analysis['background_is_dark'] and analysis['contrast'] < 0.4:
        # Apply morphological closing to reduce background texture
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
    
    return enhanced


def select_best_channel(channels: Dict[str, np.ndarray]) -> Tuple[np.ndarray, str]:
    """
    Select the best channel for puzzle piece detection from multiple channels
    
    Args:
        channels: Dictionary of processed image channels
        
    Returns:
        Tuple of (best channel image, channel name)
    """
    # Score each channel based on metrics relevant to puzzle piece detection
    scores = {}
    
    for name, channel in channels.items():
        # Skip non-grayscale channels (like color components)
        if len(channel.shape) > 2:
            continue
        
        # Calculate histogram
        hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()  # Normalize
        
        # Calculate entropy (higher entropy = more information = better)
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        
        # Calculate number of edges
        edges = cv2.Canny(channel, 50, 150)
        edge_count = np.count_nonzero(edges) / channel.size
        
        # Detect contours (more contours with reasonable areas = better)
        contours, _ = cv2.findContours(
            cv2.threshold(channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Filter contours by area
        reasonable_contours = sum(1 for c in contours if 1000 < cv2.contourArea(c) < channel.size / 10)
        
        # Calculate final score (custom weighted metric)
        # We want high entropy, moderate edge count, and a reasonable number of contours
        scores[name] = (
            0.4 * entropy / 8.0 +  # Normalize by max possible 8-bit entropy
            0.3 * min(edge_count * 100, 1.0) +  # Cap edge percentage contribution
            0.3 * min(reasonable_contours / 30, 1.0)  # Cap contour contribution
        )
    
    # Get the channel with the highest score
    best_channel_name = max(scores, key=scores.get)
    return channels[best_channel_name], best_channel_name


def preprocess_image(image: np.ndarray, advanced: bool = True) -> np.ndarray:
    """
    Preprocess image for puzzle piece detection - enhanced version
    
    Args:
        image: Input color image
        advanced: Whether to use advanced multi-channel processing
    
    Returns:
        Preprocessed grayscale image
    """
    if advanced:
        # Use advanced preprocessing with multi-channel analysis
        best_channel, _ = multi_channel_preprocess(image)
        return best_channel
    else:
        # Use simpler adaptive preprocessing
        return adaptive_preprocess(image)


def find_optimal_threshold_parameters(preprocessed: np.ndarray) -> Dict[str, Any]:
    """
    Find optimal thresholding parameters for a preprocessed image
    
    Args:
        preprocessed: Preprocessed grayscale image
        
    Returns:
        Dictionary with optimal threshold parameters
    """
    # Image analysis
    analysis = analyze_image(preprocessed)
    
    # Test various adaptive threshold parameters
    block_sizes = [15, 25, 35, 45, 55]
    c_values = [2, 5, 10, 15]
    
    best_score = -1
    best_params = {'method': 'otsu', 'block_size': 35, 'c': 10}
    
    # Create Otsu threshold for comparison
    _, otsu_binary = cv2.threshold(preprocessed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_score = evaluate_binary_image(otsu_binary)
    
    if otsu_score > best_score:
        best_score = otsu_score
        best_params = {'method': 'otsu'}
    
    # Test adaptive thresholds
    for block_size in block_sizes:
        for c in c_values:
            adaptive_binary = cv2.adaptiveThreshold(
                preprocessed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, block_size, c
            )
            
            score = evaluate_binary_image(adaptive_binary)
            
            if score > best_score:
                best_score = score
                best_params = {'method': 'adaptive', 'block_size': block_size, 'c': c}
    
    # Also test hybrid approach for low contrast images
    if analysis['contrast'] < 0.4:
        # Create a hybrid binary image that combines Otsu and adaptive results
        adaptive_binary = cv2.adaptiveThreshold(
            preprocessed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 35, 10
        )
        
        hybrid_binary = cv2.bitwise_or(otsu_binary, adaptive_binary)
        hybrid_score = evaluate_binary_image(hybrid_binary)
        
        if hybrid_score > best_score:
            best_score = hybrid_score
            best_params = {'method': 'hybrid'}
    
    return best_params


def evaluate_binary_image(binary: np.ndarray) -> float:
    """
    Evaluate the quality of a binary image for puzzle piece detection
    
    Args:
        binary: Binary image
        
    Returns:
        Quality score (higher is better)
    """
    # Find contours in the binary image
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # No contours is bad
    if not contours:
        return 0.0
    
    # Calculate contour metrics
    areas = [cv2.contourArea(c) for c in contours]
    perimeters = [cv2.arcLength(c, True) for c in contours]
    
    # Filter out tiny contours
    valid_contours = [(a, p) for a, p in zip(areas, perimeters) if a > 1000]
    
    if not valid_contours:
        return 0.0
    
    areas, perimeters = zip(*valid_contours)
    
    # Calculate statistics
    mean_area = np.mean(areas)
    std_area = np.std(areas)
    compactness = [p**2 / (4 * np.pi * a) if a > 0 else 0 for a, p in zip(areas, perimeters)]
    
    # Calculate score components
    
    # 1. Number of contours: We want a reasonable number (neither too few nor too many)
    # Assuming 20-30 is a good range for puzzle pieces
    count_score = min(len(valid_contours) / 25.0, 1.0)
    
    # 2. Area consistency: Lower standard deviation relative to mean is better
    # (puzzle pieces tend to have similar sizes)
    area_consistency = 1.0 - min(std_area / (mean_area + 1e-5), 1.0)
    
    # 3. Shape complexity: Puzzle pieces should have complex shapes (not too circular)
    # Average compactness around 2-3 is typical for puzzle pieces
    shape_score = min(np.mean(compactness) / 2.5, 1.0)
    
    # Weight the components (customize weights as needed)
    final_score = (
        0.4 * count_score +
        0.4 * area_consistency +
        0.2 * shape_score
    )
    
    return final_score


def adaptive_threshold(image: np.ndarray, 
                      block_size: int = 35, 
                      c: int = 10,
                      find_optimal: bool = True,
                      invert: bool = False) -> np.ndarray:
    """
    Apply adaptive thresholding with parameter optimization
    
    Args:
        image: Input grayscale image
        block_size: Size of pixel neighborhood for thresholding (odd number)
        c: Constant subtracted from mean
        find_optimal: Whether to find optimal parameters automatically
        invert: Whether to invert the threshold result
    
    Returns:
        Binary image
    """
    if find_optimal:
        # Find the best parameters for this specific image
        params = find_optimal_threshold_parameters(image)
        
        if params['method'] == 'otsu':
            _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif params['method'] == 'adaptive':
            binary = cv2.adaptiveThreshold(
                image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, params['block_size'], params['c']
            )
        elif params['method'] == 'hybrid':
            # Hybrid method: combine Otsu and adaptive thresholds
            _, otsu_binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            adaptive_binary = cv2.adaptiveThreshold(
                image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 35, 10
            )
            binary = cv2.bitwise_or(otsu_binary, adaptive_binary)
    else:
        # Use provided parameters
        binary = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY, 
            block_size, c
        )
    
    # Invert if needed
    if invert and find_optimal:
        binary = cv2.bitwise_not(binary)
    
    return binary


def fuse_binary_images(binaries: List[np.ndarray], method: str = 'weighted') -> np.ndarray:
    """
    Combine multiple binary images to create an improved result
    
    Args:
        binaries: List of binary images
        method: Fusion method ('weighted', 'union', or 'intersection')
    
    Returns:
        Fused binary image
    """
    if not binaries:
        return np.zeros((100, 100), dtype=np.uint8)
    
    if len(binaries) == 1:
        return binaries[0]
    
    # Ensure all images have the same dimensions
    shape = binaries[0].shape
    resized_binaries = [cv2.resize(img, (shape[1], shape[0])) for img in binaries]
    
    if method == 'union':
        # Logical OR of all images
        result = np.zeros_like(resized_binaries[0])
        for binary in resized_binaries:
            result = cv2.bitwise_or(result, binary)
    
    elif method == 'intersection':
        # Logical AND of all images
        result = np.ones_like(resized_binaries[0]) * 255
        for binary in resized_binaries:
            result = cv2.bitwise_and(result, binary)
    
    elif method == 'weighted':
        # Convert binary images to float for weighted sum
        float_images = [img.astype(np.float32) / 255.0 for img in resized_binaries]
        
        # Calculate quality scores for each binary image
        scores = [evaluate_binary_image(binary) for binary in resized_binaries]
        total_score = sum(scores) + 1e-10  # Avoid division by zero
        
        # Normalize scores to create weights
        weights = [score / total_score for score in scores]
        
        # Apply weighted sum
        weighted_sum = np.zeros_like(float_images[0])
        for img, weight in zip(float_images, weights):
            weighted_sum += img * weight
        
        # Threshold to get final binary image
        result = (weighted_sum > 0.5).astype(np.uint8) * 255
    
    else:
        # Default to union
        result = np.zeros_like(resized_binaries[0])
        for binary in resized_binaries:
            result = cv2.bitwise_or(result, binary)
    
    return result


def apply_morphology(image: np.ndarray, 
                    operation: int = cv2.MORPH_CLOSE, 
                    kernel_size: int = 5,
                    iterations: int = 1,
                    kernel_shape: int = cv2.MORPH_ELLIPSE) -> np.ndarray:
    """
    Apply morphological operations to an image with enhanced options
    
    Args:
        image: Binary input image
        operation: Morphological operation (cv2.MORPH_*)
        kernel_size: Size of the structuring element
        iterations: Number of times to apply the operation
        kernel_shape: Shape of the kernel (cv2.MORPH_RECT, cv2.MORPH_ELLIPSE, cv2.MORPH_CROSS)
    
    Returns:
        Processed binary image
    """
    kernel = cv2.getStructuringElement(kernel_shape, (kernel_size, kernel_size))
    return cv2.morphologyEx(image, operation, kernel, iterations=iterations)


def detect_edges(image: np.ndarray, 
                low_threshold: int = 50, 
                high_threshold: int = 150,
                use_auto_thresholds: bool = True) -> np.ndarray:
    """
    Detect edges in an image using Canny edge detector with adaptive thresholds
    
    Args:
        image: Input grayscale image
        low_threshold: Lower threshold for hysteresis
        high_threshold: Upper threshold for hysteresis
        use_auto_thresholds: Whether to automatically determine optimal thresholds
    
    Returns:
        Binary edge image
    """
    if use_auto_thresholds:
        # Calculate the median of the image
        v = np.median(image)
        
        # Use median-based thresholds (Canny's recommended approach)
        low_threshold = int(max(0, (1.0 - 0.33) * v))
        high_threshold = int(min(255, (1.0 + 0.33) * v))
    
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


def multi_scale_edge_detection(image: np.ndarray) -> np.ndarray:
    """
    Perform edge detection at multiple scales and combine results
    
    Args:
        image: Input grayscale image
    
    Returns:
        Binary edge image
    """
    # Apply Canny edge detection at different scales
    canny_default = cv2.Canny(image, 50, 150)
    
    # Downscale by 2x, detect edges, then upscale
    h, w = image.shape[:2]
    small_image = cv2.resize(image, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    canny_small = cv2.Canny(small_image, 50, 150)
    canny_small = cv2.resize(canny_small, (w, h), interpolation=cv2.INTER_NEAREST)
    
    # Combine edge maps (union)
    return cv2.bitwise_or(canny_default, canny_small)


def compare_threshold_methods(preprocessed_image: np.ndarray, 
                          adaptive_block_size: int = 35, 
                          adaptive_c: int = 10,
                          area_threshold: float = 200) -> Tuple[np.ndarray, str, Dict[str, Any]]:
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
    
    # Create a hybrid (combination) of both methods
    hybrid_binary = cv2.bitwise_or(otsu_binary, adaptive_binary)
    
    # Apply triangle thresholding (often good for bimodal histograms like puzzle pieces)
    _, triangle_binary = cv2.threshold(
        preprocessed_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE
    )
    
    # Evaluate each method
    methods = {
        'otsu': otsu_binary,
        'adaptive': adaptive_binary,
        'hybrid': hybrid_binary,
        'triangle': triangle_binary
    }
    
    results = {}
    best_score = -1
    best_method = 'otsu'  # Default
    
    for method_name, binary in methods.items():
        score = evaluate_binary_image(binary)
        results[method_name] = {
            'score': score
        }
        
        if score > best_score:
            best_score = score
            best_method = method_name
    
    return methods[best_method], best_method, results