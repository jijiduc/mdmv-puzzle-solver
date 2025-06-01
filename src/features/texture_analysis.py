"""Texture and pattern analysis for edge matching."""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from scipy import signal
from skimage.feature import local_binary_pattern


def extract_lbp_features(image: np.ndarray, points: List[Tuple[int, int]], 
                        radius: int = 2, n_points: int = 8,
                        sample_width: int = 10) -> np.ndarray:
    """Extract Local Binary Pattern features along edge.
    
    Args:
        image: Grayscale image
        points: Edge points
        radius: LBP radius (reduced to 2 to avoid edge issues)
        n_points: Number of points in LBP
        sample_width: Width of region to sample perpendicular to edge
        
    Returns:
        LBP histogram features
    """
    if len(points) < 3:
        return np.zeros(n_points + 2)
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Ensure image is large enough for LBP
    if gray.shape[0] < 2*radius+1 or gray.shape[1] < 2*radius+1:
        return np.zeros(n_points + 2)
    
    try:
        # Compute LBP
        lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
        
        # Sample LBP values along edge
        lbp_values = []
        for i in range(0, len(points), max(1, len(points) // 50)):  # Sample 50 points
            x, y = int(points[i][0]), int(points[i][1])
            
            # Sample in a small region around the point
            y1 = max(0, y - sample_width // 2)
            y2 = min(lbp.shape[0], y + sample_width // 2)
            x1 = max(0, x - sample_width // 2)
            x2 = min(lbp.shape[1], x + sample_width // 2)
            
            if y2 > y1 and x2 > x1:
                region_lbp = lbp[y1:y2, x1:x2]
                lbp_values.extend(region_lbp.flatten())
        
        if not lbp_values:
            return np.zeros(n_points + 2)
        
        # Create histogram of LBP values
        hist, _ = np.histogram(lbp_values, bins=n_points + 2, range=(0, n_points + 2))
        hist = hist.astype(float) / (len(lbp_values) + 1e-6)  # Normalize
        
        return hist
    except Exception as e:
        # Return default histogram on error
        return np.zeros(n_points + 2)


def extract_gabor_features(image: np.ndarray, points: List[Tuple[int, int]],
                          orientations: int = 4, frequencies: List[float] = None) -> np.ndarray:
    """Extract Gabor filter responses along edge.
    
    Args:
        image: Grayscale image
        points: Edge points
        orientations: Number of orientations
        frequencies: List of frequencies (default: [0.1, 0.2, 0.3])
        
    Returns:
        Gabor feature vector
    """
    if frequencies is None:
        frequencies = [0.1, 0.2, 0.3]
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    features = []
    
    for freq in frequencies:
        for theta_idx in range(orientations):
            theta = theta_idx * np.pi / orientations
            
            # Create Gabor kernel
            kernel = cv2.getGaborKernel((21, 21), 4.0, theta, 10.0/freq, 0.5, 0, ktype=cv2.CV_32F)
            
            # Apply filter
            filtered = cv2.filter2D(gray, cv2.CV_32F, kernel)
            
            # Sample along edge
            edge_responses = []
            for i in range(0, len(points), max(1, len(points) // 30)):
                x, y = int(points[i][0]), int(points[i][1])
                if 0 <= y < filtered.shape[0] and 0 <= x < filtered.shape[1]:
                    edge_responses.append(filtered[y, x])
            
            if edge_responses:
                # Store mean and std of responses
                features.extend([np.mean(edge_responses), np.std(edge_responses)])
            else:
                features.extend([0.0, 0.0])
    
    return np.array(features)


def detect_pattern_direction(image: np.ndarray, points: List[Tuple[int, int]],
                           window_size: int = 15) -> Tuple[float, float]:
    """Detect dominant pattern direction at edge endpoints.
    
    Args:
        image: Source image
        points: Edge points
        window_size: Size of analysis window
        
    Returns:
        Tuple of (start_direction, end_direction) in radians
    """
    if len(points) < 2:
        return (0.0, 0.0)
    
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    def get_gradient_direction(point, window_size):
        x, y = int(point[0]), int(point[1])
        
        # Extract window
        half_win = window_size // 2
        y1 = max(0, y - half_win)
        y2 = min(gray.shape[0], y + half_win + 1)
        x1 = max(0, x - half_win)
        x2 = min(gray.shape[1], x + half_win + 1)
        
        if y2 <= y1 or x2 <= x1:
            return 0.0
        
        window = gray[y1:y2, x1:x2].astype(np.float32)
        
        # Compute gradients
        grad_x = cv2.Sobel(window, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(window, cv2.CV_32F, 0, 1, ksize=3)
        
        # Compute dominant direction using structure tensor
        Ixx = grad_x * grad_x
        Iyy = grad_y * grad_y
        Ixy = grad_x * grad_y
        
        # Gaussian weighting - ensure proper shape
        gauss_size = min(window.shape[0], window.shape[1])
        if gauss_size % 2 == 0:
            gauss_size -= 1  # Ensure odd size
        gauss_size = max(3, gauss_size)  # Minimum size of 3
        
        gaussian_x = cv2.getGaussianKernel(window.shape[1], window.shape[1]/4)
        gaussian_y = cv2.getGaussianKernel(window.shape[0], window.shape[0]/4)
        gaussian_2d = gaussian_y @ gaussian_x.T
        
        # Ensure shapes match
        if gaussian_2d.shape != window.shape:
            # Simple uniform weighting as fallback
            Ixx_sum = np.sum(Ixx)
            Iyy_sum = np.sum(Iyy)
            Ixy_sum = np.sum(Ixy)
        else:
            Ixx_sum = np.sum(Ixx * gaussian_2d)
            Iyy_sum = np.sum(Iyy * gaussian_2d)
            Ixy_sum = np.sum(Ixy * gaussian_2d)
        
        # Compute eigenvalues and eigenvectors
        trace = Ixx_sum + Iyy_sum
        det = Ixx_sum * Iyy_sum - Ixy_sum * Ixy_sum
        
        if trace > 0:
            # Dominant direction is eigenvector of larger eigenvalue
            angle = 0.5 * np.arctan2(2 * Ixy_sum, Ixx_sum - Iyy_sum)
            return angle
        
        return 0.0
    
    # Get directions at start and end
    start_dir = get_gradient_direction(points[0], window_size)
    end_dir = get_gradient_direction(points[-1], window_size)
    
    return (start_dir, end_dir)


def calculate_pattern_continuity(image1: np.ndarray, edge1_points: List[Tuple[int, int]],
                               image2: np.ndarray, edge2_points: List[Tuple[int, int]],
                               mask1: Optional[np.ndarray] = None,
                               mask2: Optional[np.ndarray] = None) -> float:
    """Calculate pattern continuity score between two edges with bidirectional checking.
    
    Args:
        image1, image2: Source images
        edge1_points, edge2_points: Edge points
        mask1, mask2: Optional masks
        
    Returns:
        Continuity score (0-1)
    """
    if len(edge1_points) < 10 or len(edge2_points) < 10:
        return 0.5
    
    # Get gradient directions at junction points
    dir1_start, dir1_end = detect_pattern_direction(image1, edge1_points)
    dir2_start, dir2_end = detect_pattern_direction(image2, edge2_points)
    
    # Test all possible orientations
    direction_scores = []
    
    # Orientation 1: edge1_end matches edge2_start
    angle_diff = abs(dir1_end - dir2_start)
    angle_diff = min(angle_diff, np.pi - angle_diff)
    direction_scores.append(1.0 - (angle_diff / (np.pi / 2)))
    
    # Orientation 2: edge1_end matches edge2_end
    angle_diff = abs(dir1_end - dir2_end)
    angle_diff = min(angle_diff, np.pi - angle_diff)
    direction_scores.append(1.0 - (angle_diff / (np.pi / 2)))
    
    # Orientation 3: edge1_start matches edge2_start
    angle_diff = abs(dir1_start - dir2_start)
    angle_diff = min(angle_diff, np.pi - angle_diff)
    direction_scores.append(1.0 - (angle_diff / (np.pi / 2)))
    
    # Orientation 4: edge1_start matches edge2_end
    angle_diff = abs(dir1_start - dir2_end)
    angle_diff = min(angle_diff, np.pi - angle_diff)
    direction_scores.append(1.0 - (angle_diff / (np.pi / 2)))
    
    best_direction_score = max(direction_scores)
    
    # Extract texture features for different edge segments
    lbp_scores = []
    
    # Test different LBP combinations
    # Edge1 end vs Edge2 start
    lbp1_end = extract_lbp_features(image1, edge1_points[-30:])
    lbp2_start = extract_lbp_features(image2, edge2_points[:30])
    lbp_scores.append(1.0 - np.sum(np.abs(lbp1_end - lbp2_start)) / 2.0)
    
    # Edge1 end vs Edge2 end
    lbp2_end = extract_lbp_features(image2, edge2_points[-30:])
    lbp_scores.append(1.0 - np.sum(np.abs(lbp1_end - lbp2_end)) / 2.0)
    
    # Edge1 start vs Edge2 start
    lbp1_start = extract_lbp_features(image1, edge1_points[:30])
    lbp_scores.append(1.0 - np.sum(np.abs(lbp1_start - lbp2_start)) / 2.0)
    
    # Edge1 start vs Edge2 end
    lbp_scores.append(1.0 - np.sum(np.abs(lbp1_start - lbp2_end)) / 2.0)
    
    best_lbp_score = max(lbp_scores)
    
    # Combine scores
    continuity_score = 0.6 * best_direction_score + 0.4 * best_lbp_score
    
    return max(0, min(1, continuity_score))


def extract_edge_texture_descriptor(image: np.ndarray, edge_points: List[Tuple[int, int]],
                                   mask: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """Extract comprehensive texture descriptor for an edge.
    
    Args:
        image: Source image
        edge_points: Edge points
        mask: Optional mask
        
    Returns:
        Dictionary of texture features
    """
    features = {}
    
    try:
        # LBP features
        features['lbp'] = extract_lbp_features(image, edge_points)
        
        # Gabor features
        features['gabor'] = extract_gabor_features(image, edge_points)
        
        # Gradient directions
        start_dir, end_dir = detect_pattern_direction(image, edge_points)
        features['gradient_dirs'] = np.array([start_dir, end_dir])
        
        # Edge statistics
        if len(edge_points) > 0:
            # Sample intensities along edge
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            intensities = []
            for i in range(0, len(edge_points), max(1, len(edge_points) // 50)):
                x, y = int(edge_points[i][0]), int(edge_points[i][1])
                if 0 <= y < gray.shape[0] and 0 <= x < gray.shape[1]:
                    intensities.append(gray[y, x])
            
            if intensities:
                features['intensity_stats'] = np.array([
                    np.mean(intensities),
                    np.std(intensities),
                    np.percentile(intensities, 25),
                    np.percentile(intensities, 75)
                ])
            else:
                features['intensity_stats'] = np.zeros(4)
        else:
            features['intensity_stats'] = np.zeros(4)
            
    except Exception as e:
        # Return default features on error
        features = {
            'lbp': np.zeros(10),  # n_points + 2 with default n_points=8
            'gabor': np.zeros(24),  # 3 frequencies * 4 orientations * 2 stats
            'gradient_dirs': np.zeros(2),
            'intensity_stats': np.zeros(4)
        }
    
    return features


def compare_texture_descriptors(desc1: Dict[str, np.ndarray], 
                               desc2: Dict[str, np.ndarray]) -> float:
    """Compare two texture descriptors.
    
    Args:
        desc1, desc2: Texture descriptors
        
    Returns:
        Similarity score (0-1)
    """
    scores = []
    
    # Compare LBP
    if 'lbp' in desc1 and 'lbp' in desc2:
        lbp_sim = 1.0 - np.sum(np.abs(desc1['lbp'] - desc2['lbp'])) / 2.0
        scores.append(('lbp', lbp_sim, 0.3))
    
    # Compare Gabor
    if 'gabor' in desc1 and 'gabor' in desc2:
        gabor_diff = np.linalg.norm(desc1['gabor'] - desc2['gabor'])
        gabor_sim = np.exp(-gabor_diff / 10.0)  # Exponential similarity
        scores.append(('gabor', gabor_sim, 0.3))
    
    # Compare gradient directions (for endpoints that would connect)
    if 'gradient_dirs' in desc1 and 'gradient_dirs' in desc2:
        # Test all possible orientations
        dir_sims = []
        
        # Edge1 end vs Edge2 start
        angle_diff = abs(desc1['gradient_dirs'][1] - desc2['gradient_dirs'][0])
        angle_diff = min(angle_diff, np.pi - angle_diff)
        dir_sims.append(1.0 - (angle_diff / (np.pi / 2)))
        
        # Edge1 end vs Edge2 end
        angle_diff = abs(desc1['gradient_dirs'][1] - desc2['gradient_dirs'][1])
        angle_diff = min(angle_diff, np.pi - angle_diff)
        dir_sims.append(1.0 - (angle_diff / (np.pi / 2)))
        
        # Edge1 start vs Edge2 start
        angle_diff = abs(desc1['gradient_dirs'][0] - desc2['gradient_dirs'][0])
        angle_diff = min(angle_diff, np.pi - angle_diff)
        dir_sims.append(1.0 - (angle_diff / (np.pi / 2)))
        
        # Edge1 start vs Edge2 end
        angle_diff = abs(desc1['gradient_dirs'][0] - desc2['gradient_dirs'][1])
        angle_diff = min(angle_diff, np.pi - angle_diff)
        dir_sims.append(1.0 - (angle_diff / (np.pi / 2)))
        
        dir_sim = max(dir_sims)
        scores.append(('gradient', dir_sim, 0.2))
    
    # Compare intensity statistics
    if 'intensity_stats' in desc1 and 'intensity_stats' in desc2:
        intensity_diff = np.abs(desc1['intensity_stats'] - desc2['intensity_stats'])
        # Normalize by expected ranges
        intensity_diff[0] /= 255.0  # Mean
        intensity_diff[1] /= 128.0  # Std
        intensity_diff[2:] /= 255.0  # Percentiles
        intensity_sim = 1.0 - np.mean(intensity_diff)
        scores.append(('intensity', intensity_sim, 0.2))
    
    # Weighted combination
    if scores:
        total_weight = sum(weight for _, _, weight in scores)
        weighted_sum = sum(score * weight for _, score, weight in scores)
        return weighted_sum / total_weight
    
    return 0.5  # Default if no features available