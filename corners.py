import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import find_peaks, savgol_filter
from scipy.spatial.distance import euclidean
import math
from itertools import combinations
import concurrent.futures
import multiprocessing
import time
import gc
import sys
import psutil
import functools
import json
import hashlib
from typing import Dict, List, Tuple, Any
try:
    import numba
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    print("Numba not found. Install with 'pip install numba' for faster processing")

# ========= CONFIGURATION =========
INPUT_PATH = "picture/puzzle_24-1/b-2.jpg"
THRESHOLD_VALUE = 135
MIN_CONTOUR_AREA = 150

# ========= DTW COLOR MATCHING =========

def extract_robust_color(image, x, y, radius=2):
    """
    Extract average color from a small region to reduce noise.
    
    Args:
        image: Source image (BGR)
        x, y: Center coordinates
        radius: Radius of sampling region
        
    Returns:
        Average color of the region (BGR)
    """
    x, y = int(x), int(y)
    # Ensure coordinates are within image bounds
    if not (0 <= y < image.shape[0] and 0 <= x < image.shape[1]):
        return np.array([0, 0, 0], dtype=np.uint8)
        
    # Extract region
    region = image[max(0, y-radius):min(image.shape[0], y+radius+1), 
                  max(0, x-radius):min(image.shape[1], x+radius+1)]
    
    if region.size > 0:
        return np.mean(region, axis=(0, 1)).astype(np.uint8)
    return image[y, x]  # Fallback to single pixel

def color_confidence(image, x, y, radius=2):
    """
    Calculate confidence based on color variance in local region.
    
    Args:
        image: Source image
        x, y: Center coordinates
        radius: Radius of sampling region
        
    Returns:
        Confidence score between 0 and 1
    """
    x, y = int(x), int(y)
    # Ensure coordinates are within image bounds
    if not (0 <= y < image.shape[0] and 0 <= x < image.shape[1]):
        return 0.5
        
    # Extract region
    region = image[max(0, y-radius):min(image.shape[0], y+radius+1), 
                  max(0, x-radius):min(image.shape[1], x+radius+1)]
    
    if region.size > 0:
        # Calculate variance in each channel
        std_dev = np.std(region, axis=(0, 1))
        # Lower variance = higher confidence
        return 1.0 / (1.0 + np.mean(std_dev))
    return 0.5  # Default confidence

def normalize_edge_colors(colors):
    """
    Apply color normalization to make matching more robust.
    
    Args:
        colors: List of color values (any color space)
        
    Returns:
        Normalized color array
    """
    if len(colors) == 0:
        return np.array([])
        
    colors_array = np.array(colors)
    
    # Skip normalization if too few colors
    if len(colors_array) < 3:
        return colors_array
    
    # Simple normalization: scale to use full range
    normalized = np.zeros_like(colors_array, dtype=np.float32)
    
    # Normalize each channel independently
    for channel in range(colors_array.shape[1]):
        channel_data = colors_array[:, channel].astype(np.float32)
        channel_min = np.min(channel_data)
        channel_max = np.max(channel_data)
        
        # Avoid division by zero
        if channel_max > channel_min:
            # Scale to [0, 255]
            normalized[:, channel] = ((channel_data - channel_min) * 255.0 / 
                                     (channel_max - channel_min))
        else:
            normalized[:, channel] = channel_data
    
    return normalized

def color_distance(color1, color2):
    """
    Calculate perceptual distance between two colors in LAB space.
    
    Args:
        color1: First color in LAB space
        color2: Second color in LAB space
        
    Returns:
        Perceptual distance
    """
    # Simple Euclidean distance in LAB space is a good approximation of perceptual distance
    return np.sqrt(np.sum((color1 - color2)**2))

def sort_edge_points(edge_points, corner1, corner2):
    """
    Sort edge points from one corner to another.
    
    Args:
        edge_points: List of (x, y) coordinates
        corner1: Start corner coordinates
        corner2: End corner coordinates
        
    Returns:
        Sorted list of edge points
    """
    if len(edge_points) < 2:
        return edge_points
    
    # Create vector from corner1 to corner2
    corner_vec = (corner2[0] - corner1[0], corner2[1] - corner1[1])
    corner_length = np.sqrt(corner_vec[0]**2 + corner_vec[1]**2)
    
    if corner_length == 0:
        return edge_points
    
    # Project each point onto line connecting corners
    projections = []
    for x, y in edge_points:
        # Vector from corner1 to point
        point_vec = (x - corner1[0], y - corner1[1])
        # Dot product divided by line length = distance along line
        proj = (point_vec[0]*corner_vec[0] + point_vec[1]*corner_vec[1]) / corner_length
        projections.append((proj, (x, y)))
    
    # Sort by projection
    sorted_points = [p[1] for p in sorted(projections)]
    return sorted_points

def extract_edge_color_sequence(piece_img, edge_points, corner1, corner2):
    """
    Extract color sequence and confidence values along an edge.
    
    Args:
        piece_img: Source image containing the puzzle piece
        edge_points: List of (x, y) coordinates along the edge
        corner1: First corner coordinates
        corner2: Second corner coordinates
        
    Returns:
        Tuple of (color_sequence, confidence_sequence)
    """
    if len(edge_points) == 0:
        return [], []
    
    # Sort points along edge path
    sorted_points = sort_edge_points(edge_points, corner1, corner2)
    
    # Extract color and confidence for each point
    bgr_sequence = []
    confidence_sequence = []
    
    for x, y in sorted_points:
        robust_color = extract_robust_color(piece_img, x, y)
        bgr_sequence.append(robust_color)
        confidence_sequence.append(color_confidence(piece_img, x, y))
    
    # Convert to LAB color space for better perceptual matching
    lab_sequence = []
    for color in bgr_sequence:
        # Reshape for cv2.cvtColor
        color_rgb = np.array([[color]], dtype=np.uint8)
        color_lab = cv2.cvtColor(color_rgb, cv2.COLOR_BGR2Lab)
        lab_sequence.append(color_lab[0, 0])
    
    # Normalize colors
    if len(lab_sequence) > 0:
        lab_sequence = normalize_edge_colors(lab_sequence)
    
    return lab_sequence, confidence_sequence

def resample_sequence(sequence, target_length):
    """
    Resample a sequence to a target length using linear interpolation.
    
    Args:
        sequence: Source sequence
        target_length: Desired length
        
    Returns:
        Resampled sequence
    """
    if len(sequence) == 0:
        return []
    
    if len(sequence) == target_length:
        return sequence
    
    sequence = np.array(sequence)
    # Create indices for interpolation
    orig_indices = np.arange(len(sequence))
    target_indices = np.linspace(0, len(sequence) - 1, target_length)
    
    # Interpolate each channel separately
    result = np.zeros((target_length, sequence.shape[1]), dtype=sequence.dtype)
    
    for channel in range(sequence.shape[1]):
        result[:, channel] = np.interp(
            target_indices, orig_indices, sequence[:, channel])
    
    return result

def dtw_color_matching(sequence1, sequence2, confidence1=None, confidence2=None):
    """
    Match color sequences using Dynamic Time Warping with confidence weighting.
    
    Args:
        sequence1: First color sequence in LAB space
        sequence2: Second color sequence in LAB space
        confidence1: Optional confidence values for first sequence
        confidence2: Optional confidence values for second sequence
        
    Returns:
        Similarity score between 0 and 1
    """
    if len(sequence1) == 0 or len(sequence2) == 0:
        return 0.0
    
    # Convert to numpy arrays
    sequence1 = np.array(sequence1)
    sequence2 = np.array(sequence2)
    
    # Default confidence if not provided
    if confidence1 is None:
        confidence1 = np.ones(len(sequence1))
    if confidence2 is None:
        confidence2 = np.ones(len(sequence2))
    
    # Resample to manageable length if too long
    target_length = 50  # Balance between accuracy and performance
    if len(sequence1) > target_length:
        # Also resample confidence values
        indices = np.linspace(0, len(confidence1)-1, target_length).astype(int)
        confidence1 = np.array([confidence1[i] for i in indices])
        sequence1 = resample_sequence(sequence1, target_length)
    
    if len(sequence2) > target_length:
        indices = np.linspace(0, len(confidence2)-1, target_length).astype(int)
        confidence2 = np.array([confidence2[i] for i in indices])
        sequence2 = resample_sequence(sequence2, target_length)
    
    n, m = len(sequence1), len(sequence2)
    
    # Initialize DTW matrix
    dtw_matrix = np.ones((n+1, m+1)) * float('inf')
    dtw_matrix[0, 0] = 0
    
    # Allow partial matching at ends
    for i in range(1, n+1):
        dtw_matrix[i, 0] = 0
    for j in range(1, m+1):
        dtw_matrix[0, j] = 0
    
    # Define warping band with 30% of the longer sequence
    band_width = int(max(n, m) * 0.3)
    
    # Fill DTW matrix
    for i in range(1, n+1):
        # Define band boundaries
        j_start = max(1, i - band_width)
        j_end = min(m+1, i + band_width)
        
        for j in range(j_start, j_end):
            # Get colors and confidence values
            color1 = sequence1[i-1]
            color2 = sequence2[j-1]
            conf1 = confidence1[i-1]
            conf2 = confidence2[j-1]
            
            # Weight cost by confidence (lower confidence = higher cost)
            conf_weight = (conf1 + conf2) / 2.0
            base_cost = color_distance(color1, color2)
            weighted_cost = base_cost * (2.0 - conf_weight)
            
            # Step pattern with penalty for insertions/deletions
            diag_cost = dtw_matrix[i-1, j-1]
            horiz_cost = dtw_matrix[i, j-1] + 0.5  # Penalty
            vert_cost = dtw_matrix[i-1, j] + 0.5   # Penalty
            
            # Find minimum cost path
            dtw_matrix[i, j] = weighted_cost + min(diag_cost, horiz_cost, vert_cost)
    
    # Find minimum cost in last row or column (for subsequence matching)
    last_row = dtw_matrix[n, 1:]
    last_col = dtw_matrix[1:, m]
    best_cost = min(np.min(last_row), np.min(last_col))
    
    # Normalize to similarity score
    max_possible_cost = 400 * min(n, m)  # Maximum LAB distance is around 400
    similarity = 1.0 - min(1.0, best_cost / max_possible_cost)
    
    return similarity

# Options de performance
USE_CACHE = True               # Activer le cache pour éviter de recalculer
CACHE_DIR = ".cache"           # Dossier pour le cache
SKIP_VISUALIZATION = False     # Désactiver les visualisations
DISABLE_DEBUG_FILES = False    # Désactiver les fichiers de débogage
DEBUG_PERFORMANCE = True       # Montrer les infos de performance
USE_NUMBA = HAS_NUMBA          # Utiliser Numba si disponible
MAX_WORKERS = None             # Nombre de processus (None = CPU count - 1)

# Créer le dossier cache si activé
if USE_CACHE:
    os.makedirs(CACHE_DIR, exist_ok=True)

# ========= UTILITAIRES =========
class Timer:
    """Simple timer pour le debugging de performance."""
    def __init__(self, name="Operation"):
        self.name = name
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, *args):
        elapsed = time.time() - self.start_time
        if DEBUG_PERFORMANCE:
            print(f"{self.name} completed in {elapsed:.3f}s")

def generate_cache_key(*args):
    """Génère une clé de cache basée sur les arguments."""
    if not USE_CACHE:
        return None
        
    key_parts = []
    for arg in args:
        if isinstance(arg, np.ndarray):
            # Hash pour les tableaux NumPy
            key_parts.append(hashlib.md5(arg.tobytes()).hexdigest())
        elif isinstance(arg, (list, tuple, dict)):
            # Hash pour les structures de données
            key_parts.append(hashlib.md5(str(arg).encode()).hexdigest())
        else:
            # Autres types
            key_parts.append(str(arg))
    
    # Combiner en une seule clé
    return hashlib.md5("_".join(key_parts).encode()).hexdigest()

def cache_result(func):
    """Décorateur pour mettre en cache les résultats de fonctions."""
    if not USE_CACHE:
        return func
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Générer une clé unique basée sur la fonction et ses arguments
        cache_key = func.__name__ + "_" + generate_cache_key(*args, *kwargs.values())
        cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")
        
        # Vérifier si le résultat est en cache
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    result = json.loads(f.read())
                if DEBUG_PERFORMANCE:
                    print(f"Cache hit for {func.__name__}")
                return result
            except Exception as e:
                print(f"Cache read error: {e}")
        
        # Calculer le résultat
        result = func(*args, **kwargs)
        
        # Mettre en cache le résultat (seulement s'il est sérialisable)
        try:
            with open(cache_file, 'w') as f:
                json.dump(result, f)
        except Exception as e:
            print(f"Cache write error for {func.__name__}: {e}")
        
        return result
    
    return wrapper

# ========= DÉTECTION DE PUZZLE ET TRAITEMENT =========
@cache_result
def detect_puzzle_pieces(img_path, threshold_value, min_area):
    """Détecte et extrait les pièces de puzzle d'une image."""
    with Timer("Image loading and processing"):
        # Lire l'image
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Could not read image from {img_path}")
        
        # Conversion en niveaux de gris et seuillage
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary_mask = cv2.threshold(gray_img, threshold_value, 255, cv2.THRESH_BINARY)
        binary_mask = np.uint8(binary_mask)
        
        # Morphologie
        closing_kernel = np.ones((9, 9), np.uint8)
        dilation_kernel = np.ones((3, 3), np.uint8)
        
        closed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, closing_kernel)
        processed_mask = cv2.dilate(closed_mask, dilation_kernel, iterations=1)
        
        # Trouver et filtrer les contours
        contours, _ = cv2.findContours(processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
        
        # Créer le masque final
        filled_mask = np.zeros_like(processed_mask)
        cv2.drawContours(filled_mask, valid_contours, -1, 255, -1)
        
        # Pièces de puzzle
        pieces = []
        padding = 5  # Petit padding autour de chaque pièce
        
        for i, contour in enumerate(valid_contours):
            # Obtenir la boîte englobante
            x, y, w, h = cv2.boundingRect(contour)
            
            # Ajouter un padding (tout en restant dans les limites)
            x1, y1 = max(0, x-padding), max(0, y-padding)
            x2, y2 = min(img.shape[1], x+w+padding), min(img.shape[0], y+h+padding)
            
            # Extraire la pièce
            piece_img = img[y1:y2, x1:x2].copy()
            piece_mask = filled_mask[y1:y2, x1:x2].copy()
            
            # Appliquer le masque à la pièce (pour l'isoler du fond)
            masked_piece = cv2.bitwise_and(piece_img, piece_img, mask=piece_mask)
            
            # Ajouter à la liste
            pieces.append({
                'index': i,
                'img': masked_piece.tolist(),  # Convertir en liste pour JSON
                'mask': piece_mask.tolist(),
                'bbox': (x1, y1, x2, y2)
            })
    
    return {
        'count': len(pieces),
        'pieces': pieces
    }

def process_piece(piece_data, output_dirs):
    """Traite une seule pièce - optimisé pour la performance."""
    piece_index = piece_data['index']
    
    # Récupérer les chemins de sortie
    edges_dir, edge_types_dir, corners_dir, contours_dir = output_dirs
    
    color_features_dir = os.path.join(os.path.dirname(edges_dir), "color_features")
    os.makedirs(color_features_dir, exist_ok=True)
    
    # Convertir les listes en tableaux NumPy
    piece_img = np.array(piece_data['img'], dtype=np.uint8)
    piece_mask = np.array(piece_data['mask'], dtype=np.uint8)
    
    # Détecter les contours et le centroïde
    edges = cv2.Canny(piece_mask, 50, 150)
    
    # Trouver les points de contour
    edge_points = np.where(edges > 0)
    y_edge, x_edge = edge_points[0], edge_points[1]
    edge_coordinates = np.column_stack((x_edge, y_edge))
    
    # Calculer le centroïde
    moments = cv2.moments(piece_mask)
    if moments["m00"] != 0:
        centroid_x = int(moments["m10"] / moments["m00"])
        centroid_y = int(moments["m01"] / moments["m00"])
    else:
        centroid_x = piece_mask.shape[1] // 2
        centroid_y = piece_mask.shape[0] // 2
    centroid = (centroid_x, centroid_y)
    
    # Calculer distances et angles depuis le centroïde
    distances = []
    angles = []
    coords = []
    
    if USE_NUMBA and HAS_NUMBA:
        # Optimisation Numba
        @numba.jit(nopython=True)
        def calc_dist_angle(points, cx, cy):
            dists = np.zeros(len(points), dtype=np.float64)
            angs = np.zeros(len(points), dtype=np.float64)
            for i in range(len(points)):
                x, y = points[i]
                dists[i] = np.sqrt((x - cx)**2 + (y - cy)**2)
                angs[i] = np.arctan2(y - cy, x - cx)
            return dists, angs
        
        distances, angles = calc_dist_angle(edge_coordinates, centroid_x, centroid_y)
        coords = edge_coordinates
    else:
        # Version standard
        for x, y in edge_coordinates:
            dist = euclidean((centroid_x, centroid_y), (x, y))
            angle = math.atan2(y - centroid_y, x - centroid_x)
            distances.append(dist)
            angles.append(angle)
            coords.append((x, y))
        distances = np.array(distances)
        angles = np.array(angles)
    
    # Trier par angle
    sort_idx = np.argsort(angles)
    sorted_angles = angles[sort_idx]
    sorted_distances = distances[sort_idx]
    sorted_coords = np.array(coords)[sort_idx]
    
    # Lisser la courbe
    window_length = min(51, len(sorted_distances) // 5 * 2 + 1)
    if window_length >= 3:
        if window_length % 2 == 0:  # S'assurer que window_length est impair
            window_length += 1
        sorted_distances_smooth = savgol_filter(sorted_distances, window_length, 3)
    else:
        sorted_distances_smooth = sorted_distances
    
    # Détecter les pics
    peaks, _ = find_peaks(
        sorted_distances_smooth, 
        prominence=5,
        distance=len(sorted_distances_smooth)/15
    )
    
    # Si pas assez de pics, essayer avec une prominence plus faible
    if len(peaks) < 6:
        peaks, _ = find_peaks(
            sorted_distances_smooth, 
            prominence=3,
            distance=len(sorted_distances_smooth)/20
        )
    
    # Si toujours pas assez, ajouter les points les plus éloignés
    if len(peaks) < 6:
        highest_dist_indices = np.argsort(sorted_distances_smooth)[-10:]
        peaks = np.unique(np.concatenate([peaks, highest_dist_indices]))
    
    # Obtenir les coordonnées des pics
    peak_coords = sorted_coords[peaks]
    
    # Sélectionner les 4 meilleurs coins pour former un rectangle
    if len(peaks) >= 6:
        # Fonction pour calculer l'aire d'un quadrilatère
        def quad_area(points):
            x = [p[0] for p in points]
            y = [p[1] for p in points]
            return 0.5 * abs(sum(x[i] * y[i+1] - x[i+1] * y[i] 
                           for i in range(len(points)-1)) + 
                       x[-1] * y[0] - x[0] * y[-1])
        
        # Fonction pour calculer la "rectangularité"
        def rectangle_score(points):
            # Calculer l'aire - plus grande est meilleure
            area = quad_area(points)
            
            # Calculer les angles entre les côtés
            angles_deg = []
            for j in range(4):
                p1 = points[j]
                p2 = points[(j+1) % 4]
                p3 = points[(j+2) % 4]
                
                # Vecteurs pour les côtés adjacents
                v1 = (p2[0] - p1[0], p2[1] - p1[1])
                v2 = (p3[0] - p2[0], p3[1] - p2[1])
                
                # Calculer l'angle en degrés
                dot_product = v1[0] * v2[0] + v1[1] * v2[1]
                mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
                mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
                
                if mag1 * mag2 == 0:
                    angle_deg = 0
                else:
                    cos_angle = dot_product / (mag1 * mag2)
                    # Limiter pour éviter les erreurs numériques
                    cos_angle = max(-1, min(1, cos_angle))
                    angle_deg = math.degrees(math.acos(cos_angle))
                
                angles_deg.append(angle_deg)
            
            # Pénalité pour déviation de 90 degrés
            angle_penalty = sum(abs(angle - 90) for angle in angles_deg)
            
            # Calculer périmètre
            perimeter = sum(math.sqrt((points[j][0] - points[(j+1)%4][0])**2 + 
                                    (points[j][1] - points[(j+1)%4][1])**2)
                           for j in range(4))
            
            # Forme plus compacte (rapport aire/périmètre plus élevé) est meilleure
            compactness = area / (perimeter**2 + 1e-10)  # Éviter division par zéro
            
            # Score combiné
            return area * compactness * (1000 / (angle_penalty + 1))
        
        # Chercher la meilleure combinaison de 4 coins
        best_score = -1
        best_corners = None
        
        # Limiter le nombre de combinaisons
        max_combs = min(100, len(peak_coords) * (len(peak_coords) - 1) * (len(peak_coords) - 2) * (len(peak_coords) - 3) // 24)
        peak_distances = sorted_distances_smooth[peaks]
        
        # Prendre les 8-10 pics les plus lointains
        top_peaks_idx = np.argsort(peak_distances)[-min(10, len(peak_distances)):]
        for i in top_peaks_idx:
            for j in top_peaks_idx:
                if i == j:
                    continue
                for k in top_peaks_idx:
                    if k == i or k == j:
                        continue
                    for l in top_peaks_idx:
                        if l == i or l == j or l == k:
                            continue
                        
                        points = [peak_coords[i], peak_coords[j], peak_coords[k], peak_coords[l]]
                        
                        # Trier pour former un quadrilatère
                        cx = sum(p[0] for p in points) / 4
                        cy = sum(p[1] for p in points) / 4
                        points.sort(key=lambda p: math.atan2(p[1] - cy, p[0] - cx))
                        
                        score = rectangle_score(points)
                        if score > best_score:
                            best_score = score
                            best_corners = points
        
        if best_corners is not None:
            corner_points = best_corners
        else:
            corner_points = peak_coords[:4]
    else:
        corner_points = peak_coords[:min(4, len(peak_coords))]
    
    # S'assurer que les coins sont arrangés dans le sens des aiguilles d'une montre
    cx = sum(p[0] for p in corner_points) / len(corner_points)
    cy = sum(p[1] for p in corner_points) / len(corner_points)
    corner_points = sorted(corner_points, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))
    
    # Créer l'image avec coins détectés
    corner_img = piece_img.copy()
    
    # Dessiner et étiqueter les coins
    for j, (x, y) in enumerate(corner_points):
        cv2.circle(corner_img, (int(x), int(y)), 5, (255, 0, 0), -1)
        cv2.putText(corner_img, str(j), (int(x)+5, int(y)+5), 
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Dessiner le centroïde
    cv2.circle(corner_img, centroid, 3, (0, 0, 255), -1)
    
    # Créer la visualisation combinée
    if not SKIP_VISUALIZATION and not DISABLE_DEBUG_FILES:
        # Visualisation avec Matplotlib
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # Image avec coins
        axes[0].imshow(cv2.cvtColor(corner_img, cv2.COLOR_BGR2RGB))
        axes[0].set_title(f"Corners in Piece {piece_index+1}")
        axes[0].axis('off')
        
        # Graphique distances-angles
        axes[1].plot(sorted_angles, sorted_distances_smooth, 'orange', linewidth=2)
        
        # Marquer les pics
        peak_angles = [sorted_angles[p] for p in peaks]
        peak_distances = [sorted_distances_smooth[p] for p in peaks]
        axes[1].scatter(peak_angles, peak_distances, c='green', marker='x', s=50)
        
        # Configurer le graphique
        axes[1].set_xlabel('Angles')
        axes[1].set_ylabel('Dist')
        axes[1].set_title(f'Corner Detection in Piece {piece_index+1}')
        axes[1].grid(True, linestyle='--', alpha=0.7)
        
        # Utiliser l'approche direct_to_file, plus fiable avec multiprocessing
        plt.tight_layout()
        
        # Enregistrer directement dans le fichier final
        os.makedirs(corners_dir, exist_ok=True)
        corner_path = os.path.join(corners_dir, f"corner_piece_{piece_index+1}.png")
        fig.savefig(corner_path, dpi=120)
        plt.close(fig)
    
    # Extraire et classifier les quatre bords
    edges = []
    edge_types = []
    edge_deviations = []
    edge_colors = []
    color_vis_data = []
    
    for j in range(4):
        next_j = (j + 1) % 4
        edge_points = extract_edge_between_corners(corner_points, j, next_j, edge_coordinates, centroid)
        
        if len(edge_points) > 0:
            edge_type, deviation = classify_edge(edge_points, corner_points[j], corner_points[next_j], centroid)
            color_feature, vis_data = extract_edge_color_features(piece_img, edge_points, corner_points[j], corner_points[next_j], j)
        else:
            edge_type, deviation = "unknown", 0
            color_feature = None
        
        edges.append(edge_points)
        edge_types.append(edge_type)
        edge_deviations.append(deviation)
        edge_colors.append(color_feature)
        color_vis_data.append(vis_data)
        
        # Créer image pour chaque bord
        if not DISABLE_DEBUG_FILES:
            color_vis_path = os.path.join(color_features_dir, f"piece_{piece_index+1}_color_features.png")
            create_color_feature_visualization(piece_img, color_vis_data, piece_index, color_vis_path)
            
            os.makedirs(edges_dir, exist_ok=True)
            if len(edge_points) > 0:
                # Calculer la boîte englobante
                min_x = max(0, int(min(x for x, y in edge_points)) - 20)
                min_y = max(0, int(min(y for x, y in edge_points)) - 20)
                max_x = min(piece_mask.shape[1], int(max(x for x, y in edge_points)) + 20)
                max_y = min(piece_mask.shape[0], int(max(y for x, y in edge_points)) + 20)
                
                # Inclure les coins
                min_x = min(min_x, int(corner_points[j][0]) - 10, int(corner_points[next_j][0]) - 10)
                min_y = min(min_y, int(corner_points[j][1]) - 10, int(corner_points[next_j][1]) - 10)
                max_x = max(max_x, int(corner_points[j][0]) + 10, int(corner_points[next_j][0]) + 10)
                max_y = max(max_y, int(corner_points[j][1]) + 10, int(corner_points[next_j][1]) + 10)
                
                # Dimensions finales
                width = max_x - min_x
                height = max_y - min_y
                
                # Assurer des dimensions minimales
                if width < 100:
                    padding = (100 - width) // 2
                    min_x = max(0, min_x - padding)
                    max_x = min(piece_mask.shape[1], max_x + padding)
                    width = max_x - min_x
                    
                if height < 100:
                    padding = (100 - height) // 2
                    min_y = max(0, min_y - padding)
                    max_y = min(piece_mask.shape[0], max_y + padding)
                    height = max_y - min_y
                
                edge_img = np.zeros((height, width, 3), dtype=np.uint8)
                
                # Dessiner les points d'arête
                for x, y in edge_points:
                    new_x = int(x - min_x)
                    new_y = int(y - min_y)
                    if 0 <= new_y < edge_img.shape[0] and 0 <= new_x < edge_img.shape[1]:
                        edge_img[new_y, new_x] = [0, 255, 0]
                
                # Dessiner les coins
                for k, corner in enumerate(corner_points):
                    if k == j or k == next_j:
                        new_x = int(corner[0] - min_x)
                        new_y = int(corner[1] - min_y)
                        if 0 <= new_y < edge_img.shape[0] and 0 <= new_x < edge_img.shape[1]:
                            cv2.circle(edge_img, (new_x, new_y), 5, [255, 0, 0], -1)
                            cv2.putText(edge_img, str(k), (new_x+7, new_y+7), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, [255, 0, 0], 2)
                
                # Sauvegarder directement le fichier
                edge_path = os.path.join(edges_dir, f"piece_{piece_index+1}_edge_{j+1}.png")
                cv2.imwrite(edge_path, edge_img)
            else:
                # Image vide
                edge_img = np.zeros((100, 100, 3), dtype=np.uint8)
                edge_path = os.path.join(edges_dir, f"piece_{piece_index+1}_edge_{j+1}.png")
                cv2.imwrite(edge_path, edge_img)
    
    # Créer la visualisation des types d'arêtes
    if not SKIP_VISUALIZATION and not DISABLE_DEBUG_FILES:
        os.makedirs(edge_types_dir, exist_ok=True)
        
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        fig.suptitle(f"Piece {piece_index+1} Edge Classification", fontsize=16)
        
        for j in range(4):
            row = j // 2
            col = j % 2
            
            # Créer la visualisation de cette arête
            edge_vis = (piece_img.copy() * 0.5).astype(np.uint8)
            
            # Dessiner les points
            if len(edges[j]) > 0:
                for x, y in edges[j]:
                    if 0 <= y < edge_vis.shape[0] and 0 <= x < edge_vis.shape[1]:
                        edge_vis[int(y), int(x)] = [0, 255, 0]
            
            # Dessiner les coins
            for k, corner in enumerate(corner_points):
                if k == j or k == (j+1)%4:
                    cv2.circle(edge_vis, (int(corner[0]), int(corner[1])), 5, [255, 0, 0], -1)
                    cv2.putText(edge_vis, str(k), (int(corner[0])+7, int(corner[1])+7), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, [255, 0, 0], 2)
                else:
                    cv2.circle(edge_vis, (int(corner[0]), int(corner[1])), 3, [0, 0, 255], -1)
                    cv2.putText(edge_vis, str(k), (int(corner[0])+7, int(corner[1])+7), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 0, 255], 2)
            
            # Afficher
            axes[row, col].imshow(cv2.cvtColor(edge_vis, cv2.COLOR_BGR2RGB))
            axes[row, col].set_title(f"Edge {j+1}: {edge_types[j]}")
            axes[row, col].axis('off')
        
        # Montrer la pièce originale
        axes[0, 2].imshow(cv2.cvtColor(piece_img, cv2.COLOR_BGR2RGB))
        axes[0, 2].set_title("Original Piece")
        axes[0, 2].axis('off')
        
        # Ajouter un résumé
        edge_summary = "\n".join([f"Edge {j+1}: {edge_types[j]} ({edge_deviations[j]:.1f}px)" for j in range(4)])
        axes[1, 2].axis('off')
        axes[1, 2].text(0.1, 0.5, edge_summary, fontsize=12, va='center')
        
        plt.tight_layout()
        
        # Sauvegarder directement
        vis_path = os.path.join(edge_types_dir, f"piece_{piece_index+1}_edge_types.png")
        plt.savefig(vis_path)
        plt.close(fig)
    
    # Nettoyage
    gc.collect()
    
    return {
        'piece_idx': piece_index,
        'edge_types': edge_types,
        'edge_deviations': edge_deviations,
        'edge_colors': edge_colors
    }

def extract_edge_between_corners(corners, corner_idx1, corner_idx2, edge_coords, centroid):
    """Extrait les points de bord entre deux coins."""
    corner1 = corners[corner_idx1]
    corner2 = corners[corner_idx2]
    centroid_x, centroid_y = centroid
    
    if len(edge_coords) == 0:
        return []
    
    # Calculer les angles des coins par rapport au centroïde
    angle1 = math.atan2(corner1[1] - centroid_y, corner1[0] - centroid_x)
    angle2 = math.atan2(corner2[1] - centroid_y, corner2[0] - centroid_x)
    
    # S'assurer que angle2 > angle1 pour la vérification de plage
    if angle2 < angle1:
        angle2 += 2 * math.pi
    
    # Calculer les angles pour tous les points
    all_angles = np.array([math.atan2(y - centroid_y, x - centroid_x) for x, y in edge_coords])
    all_angles_normalized = all_angles.copy()
    all_angles_normalized[all_angles_normalized < angle1] += 2 * math.pi
    
    # Masque pour les points dans la plage angulaire
    angle_mask = (all_angles_normalized >= angle1) & (all_angles_normalized <= angle2)
    filtered_points = edge_coords[angle_mask]
    
    # Si pas de points filtrés, retourner une liste vide
    if len(filtered_points) == 0:
        return []
    
    # Trier par angle
    sorted_indices = np.argsort([math.atan2(y - centroid_y, x - centroid_x) for x, y in filtered_points])
    sorted_points = filtered_points[sorted_indices]
    # If there's any empty check before returning, make sure it's:
    if len(filtered_points) == 0:
        return []
    
    return sorted_points

def classify_edge(edge_points, corner1, corner2, centroid):
    """Classifie un bord comme droit, intrusion ou extrusion."""
    if len(edge_points) == 0 or len(edge_points) < 5:
        return "unknown", 0
    
    # Créer une ligne droite entre les coins
    x1, y1 = corner1
    x2, y2 = corner2
    centroid_x, centroid_y = centroid
    
    # Vecteur de ligne
    line_vec = (x2-x1, y2-y1)
    line_length = math.sqrt(line_vec[0]**2 + line_vec[1]**2)
    if line_length < 1:
        return "unknown", 0
    
    # Vecteur normal
    normal_vec = (-line_vec[1]/line_length, line_vec[0]/line_length)
    
    # Vecteur du centroïde au milieu de la ligne
    mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
    centroid_to_mid = (mid_x - centroid_x, mid_y - centroid_y)
    
    # Direction du vecteur normal (vers l'intérieur ou l'extérieur)
    normal_direction = centroid_to_mid[0]*normal_vec[0] + centroid_to_mid[1]*normal_vec[1]
    
    # S'assurer que le vecteur normal pointe vers l'extérieur
    outward_normal = normal_vec if normal_direction > 0 else (-normal_vec[0], -normal_vec[1])
    
    # Calculer les déviations pour tous les points du bord
    deviations = []
    for x, y in edge_points:
        # Vecteur du premier coin au point
        point_vec = (x-x1, y-y1)
        
        # Projection du vecteur point sur le vecteur ligne
        if line_length > 0:
            line_dot = (point_vec[0]*line_vec[0] + point_vec[1]*line_vec[1]) / line_length
            
            # Coordonnées du point projeté
            proj_x = x1 + line_dot * line_vec[0] / line_length
            proj_y = y1 + line_dot * line_vec[1] / line_length
            
            # Vecteur de déviation
            dev_vec = (x-proj_x, y-proj_y)
            
            # Magnitude de déviation
            deviation = math.sqrt(dev_vec[0]**2 + dev_vec[1]**2)
            
            # Signe de déviation (positif si dans la direction du vecteur normal extérieur)
            sign = 1 if (dev_vec[0]*outward_normal[0] + dev_vec[1]*outward_normal[1]) > 0 else -1
            
            # Déviation signée
            deviations.append(sign * deviation)
    
    # Calculer le seuil adaptatif
    straight_threshold = max(5, line_length * 0.05)  # Au moins 5px ou 5% de la longueur
    
    # Classification
    if deviations:
        # Calculs statistiques
        mean_deviation = sum(deviations) / len(deviations)
        abs_deviations = [abs(d) for d in deviations]
        max_abs_deviation = max(abs_deviations)
        
        # Compter les déviations significatives
        significant_positive = sum(1 for d in deviations if d > straight_threshold)
        significant_negative = sum(1 for d in deviations if d < -straight_threshold)
        
        # Portion du bord avec déviations significatives
        portion_significant = (significant_positive + significant_negative) / len(deviations)
        
        # Logique de classification
        if max_abs_deviation < straight_threshold or portion_significant < 0.2:
            edge_type = "straight"
            max_deviation = mean_deviation
        elif abs(mean_deviation) < straight_threshold * 0.5:
            # Si la moyenne est proche de zéro mais le max est élevé
            if significant_positive > significant_negative * 2:
                edge_type = "extrusion"
                max_deviation = max([d for d in deviations if d > 0], default=0)
            elif significant_negative > significant_positive * 2:
                edge_type = "intrusion"
                max_deviation = min([d for d in deviations if d < 0], default=0)
            else:
                edge_type = "straight"  # Déviations équilibrées
                max_deviation = mean_deviation
        elif mean_deviation > 0:
            edge_type = "extrusion"
            max_deviation = max(deviations)
        else:
            edge_type = "intrusion"
            max_deviation = min(deviations)
    else:
        edge_type = "unknown"
        max_deviation = 0
    
    return edge_type, max_deviation

def setup_output_directories():
    """Crée les répertoires de sortie nécessaires."""
    # Répertoire debug principal
    debug_dir = "debug_corners"
    os.makedirs(debug_dir, exist_ok=True)
    
    # Sous-répertoires
    dirs = {
        'masks': os.path.join(debug_dir, "masks"),
        'pieces': os.path.join(debug_dir, "pieces"),
        'transforms': os.path.join(debug_dir, "transforms"),
        'contours': os.path.join(debug_dir, "contours"),
        'corners': os.path.join(debug_dir, "corners"),
        'edges': os.path.join(debug_dir, "edges"),
        'edge_types': os.path.join(debug_dir, "edge_types")
    }
    
    # Créer tous les sous-répertoires
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return dirs

def init_worker():
    """Initialisation pour les processus workers."""
    # Désactiver GC automatique
    gc.disable()
    
    # Priorité élevée
    try:
        if sys.platform == 'win32':
            psutil.Process().nice(psutil.HIGH_PRIORITY_CLASS)
        else:
            os.nice(-5)
    except:
        pass

def parallel_process_pieces(piece_data, output_dirs, max_workers=None):
    """Traite les pièces en parallèle avec une gestion efficace des erreurs."""
    # Nombre de cœurs à utiliser
    if max_workers is None:
        max_workers = max(1, multiprocessing.cpu_count() - 1)
    
    print(f"Processing {len(piece_data)} pieces using {max_workers} cores...")
    
    # Les résultats finaux
    results = []
    
    with Timer("Parallel processing"):
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=init_worker
        ) as executor:
            # Soumettre toutes les tâches
            futures = [executor.submit(process_piece, piece, output_dirs) 
                     for piece in piece_data]
            
            # Traiter les résultats au fur et à mesure
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                try:
                    result = future.result()
                    results.append(result)
                    piece_idx = result['piece_idx']
                    print(f"Completed piece {piece_idx+1}/{len(piece_data)}")
                except Exception as e:
                    print(f"Error processing piece: {str(e)}")
                    import traceback
                    traceback.print_exc()
    
    return results

def save_masks(img, filled_mask, valid_contours, dirs):
    """Sauvegarde les masques et images de base."""
    if DISABLE_DEBUG_FILES:
        return
        
    # Masque appliqué à l'image
    masked_img = cv2.bitwise_and(img, img, mask=filled_mask)
    
    # Image avec contours
    contour_img = masked_img.copy()
    for contour in valid_contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(contour_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Sauvegarder
    cv2.imwrite(os.path.join(dirs['masks'], "binary_mask.png"), filled_mask)
    cv2.imwrite(os.path.join(dirs['masks'], "masked_img.png"), masked_img)
    cv2.imwrite(os.path.join(dirs['masks'], "contour_img.png"), contour_img)

def save_pieces(pieces, img, filled_mask, dirs):
    """Sauvegarde les pièces individuelles et leurs transformations."""
    if DISABLE_DEBUG_FILES:
        return
    
    # Extraire les pièces
    for i, piece_data in enumerate(pieces):
        # Récupérer les données de la pièce
        index = piece_data['index']
        bbox = piece_data['bbox']
        x1, y1, x2, y2 = bbox
        
        # Extraire la pièce
        piece_img = img[y1:y2, x1:x2].copy()
        piece_mask = filled_mask[y1:y2, x1:x2].copy()
        
        # Masquer la pièce
        masked_piece = cv2.bitwise_and(piece_img, piece_img, mask=piece_mask)
        
        # Calculer la transformation de distance
        dist_transform = cv2.distanceTransform(piece_mask, cv2.DIST_L2, 5)
        cv2.normalize(dist_transform, dist_transform, 0, 1.0, cv2.NORM_MINMAX)
        
        # Sauvegarder
        cv2.imwrite(os.path.join(dirs['pieces'], f"piece_{index+1}.png"), masked_piece)
        plt.imsave(os.path.join(dirs['transforms'], f"distance_transform_{index+1}.png"), dist_transform, cmap='gray')
        
def extract_dtw_edge_features(piece_img, edge_points, corner1, corner2, edge_index):
    """
    Extract DTW-compatible color features for an edge and return visualization data.
    Uses precise color sequences along the edge for DTW-based matching.
    
    Args:
        piece_img: Source image containing the puzzle piece
        edge_points: List of (x, y) coordinates along the edge
        corner1: First corner coordinates
        corner2: Second corner coordinates
        edge_index: Index of the edge (for visualization purposes)
        
    Returns:
        Tuple of (color_feature, visualization_data)
    """
    if len(edge_points) == 0:
        return None, None
    
    # Sort edge points from corner1 to corner2
    sorted_points = sort_edge_points(edge_points, corner1, corner2)
    
    # Extract color sequence and confidence values
    lab_sequence, confidence_sequence = extract_edge_color_sequence(piece_img, sorted_points, corner1, corner2)
    
    if len(lab_sequence) == 0:
        return None, None
    
    # Calculate basic statistics for the color sequence
    mean_color = np.mean(lab_sequence, axis=0).tolist() if len(lab_sequence) > 0 else [0, 0, 0]
    std_color = np.std(lab_sequence, axis=0).tolist() if len(lab_sequence) > 0 else [0, 0, 0]
    
    # Calculate gradients (differences between adjacent points)
    gradients = []
    if len(lab_sequence) > 1:
        diffs = np.abs(np.diff(lab_sequence, axis=0))
        mean_gradient = np.mean(diffs, axis=0).tolist()
        max_gradient = np.max(diffs, axis=0).tolist()
        # Count significant color changes
        significant_changes = np.sum(np.sqrt(np.sum(diffs**2, axis=1)) > 10)
        gradients = {
            'mean_gradient': mean_gradient,
            'max_gradient': max_gradient,
            'significant_changes': int(significant_changes)
        }
    else:
        gradients = {
            'mean_gradient': [0, 0, 0],
            'max_gradient': [0, 0, 0],
            'significant_changes': 0
        }
    
    # Create feature vector for DTW matching
    color_feature = {
        'method': 'dtw',
        'lab_sequence': lab_sequence.tolist() if isinstance(lab_sequence, np.ndarray) else lab_sequence,
        'confidence_sequence': confidence_sequence,
        'mean_color': mean_color,
        'std_color': std_color,
        'gradients': gradients,
        'sequence_length': len(lab_sequence)
    }
    
    # Create visualization data
    # Create a mask for visualization
    mask = np.zeros(piece_img.shape[:2], dtype=np.uint8)
    
    # Draw the edge points with a small buffer for visualization
    for x, y in sorted_points:
        if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
            cv2.circle(mask, (int(x), int(y)), 3, 255, -1)
    
    edge_img = cv2.bitwise_and(piece_img, piece_img, mask=mask)
    
    # Convert sequences to colorful visualization
    color_vis = np.zeros((50, len(lab_sequence), 3), dtype=np.uint8)
    if len(lab_sequence) > 0:
        for i, lab_color in enumerate(lab_sequence):
            # Convert LAB to BGR for visualization
            lab_color_single = np.array([[lab_color]], dtype=np.uint8)
            bgr_color = cv2.cvtColor(lab_color_single, cv2.COLOR_Lab2BGR)[0, 0]
            
            # Draw a column with this color
            conf = confidence_sequence[i] if i < len(confidence_sequence) else 1.0
            column_height = int(50 * conf)
            color_vis[:column_height, i] = bgr_color
    
    # Create visualization data
    vis_data = {
        'method': 'dtw',
        'edge_img': edge_img,
        'edge_index': edge_index,
        'color_sequence_vis': color_vis,
        'lab_sequence': lab_sequence,
        'confidence_sequence': confidence_sequence,
        'gradients': gradients
    }
    
    return color_feature, vis_data

def extract_edge_color_features(piece_img, edge_points, corner1, corner2, edge_index):
    """
    Extract DTW-based color features for an edge and return visualization data.
    
    Args:
        piece_img: Source image containing the puzzle piece
        edge_points: List of (x, y) coordinates along the edge
        corner1: First corner coordinates
        corner2: Second corner coordinates
        edge_index: Index of the edge (for visualization purposes)
        
    Returns:
        Tuple of (color_feature, visualization_data)
    """
    return extract_dtw_edge_features(piece_img, edge_points, corner1, corner2, edge_index)

def create_color_feature_visualization(piece_img, vis_data_list, piece_index, output_path):
    """
    Create a visualization of DTW color features for all edges of a piece.
    """
    if not vis_data_list or all(v is None for v in vis_data_list):
        # Create empty visualization if no data
        vis_img = np.ones((400, 600, 3), dtype=np.uint8) * 255
        cv2.putText(vis_img, f"No color data for piece {piece_index+1}", 
                   (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.imwrite(output_path, vis_img)
        return
    
    # Use the Agg backend which is thread-safe
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    
    # Create a larger figure with more complex layout for enhanced visualization
    plt.figure(figsize=(20, 30))
    gs = GridSpec(12, 4, figure=plt.gcf())  # Increased from 8 to 12 rows to accommodate more edges
    
    plt.suptitle(f"Piece {piece_index+1} - Edge Color Features (DTW)", fontsize=20)
    
    # Show original piece image at the top
    ax_piece = plt.subplot(gs[0, 1:3])
    if piece_img is not None:
        ax_piece.imshow(cv2.cvtColor(piece_img, cv2.COLOR_BGR2RGB))
        ax_piece.set_title("Original Piece", fontsize=16)
    else:
        ax_piece.text(0.5, 0.5, "Original piece image not available", 
                     ha='center', va='center', transform=ax_piece.transAxes)
    ax_piece.axis('off')
    
    valid_edges = 0
    for i, vis_data in enumerate(vis_data_list):
        if vis_data is None:
            continue
            
        valid_edges += 1
        edge_index = vis_data['edge_index']
        row_start = 1 + i*2  # Each edge gets 2 rows
        
        # 1. Edge Image - Show the edge with its color samples
        ax_edge = plt.subplot(gs[row_start, 0])
        edge_img = cv2.cvtColor(vis_data['edge_img'], cv2.COLOR_BGR2RGB)
        ax_edge.imshow(edge_img)
        ax_edge.set_title(f"Edge {edge_index+1}", fontsize=14)
        ax_edge.axis('off')
        
        # 2. Color Sequence Visualization
        if 'color_sequence_vis' in vis_data and vis_data['color_sequence_vis'] is not None:
            ax_seq = plt.subplot(gs[row_start, 1:4])
            color_vis = vis_data['color_sequence_vis']
            
            # Display the color sequence
            ax_seq.imshow(cv2.cvtColor(color_vis, cv2.COLOR_BGR2RGB))
            ax_seq.set_title(f"Edge {edge_index+1} Color Sequence", fontsize=12)
            ax_seq.set_xlabel("Position along edge (corner1 to corner2)")
            ax_seq.set_ylabel("Confidence")
            ax_seq.set_yticks([])  # Hide y-axis ticks
            
            # Add markers at regular intervals
            seq_length = color_vis.shape[1]
            for x in range(0, seq_length, max(1, seq_length // 10)):
                ax_seq.axvline(x, color='gray', linestyle='--', alpha=0.3)
        
        # 3. Gradient information
        if 'gradients' in vis_data and vis_data['gradients'] is not None:
            ax_grad = plt.subplot(gs[row_start+1, 0:2])
            
            grad_info = vis_data['gradients']
            
            # Create text-based gradient info
            grad_text = f"Color Gradient Information:\n\n"
            
            mean_grad = grad_info.get('mean_gradient', [0, 0, 0])
            max_grad = grad_info.get('max_gradient', [0, 0, 0])
            sig_changes = grad_info.get('significant_changes', 0)
            
            grad_text += f"Mean Gradient (L/A/B): {mean_grad[0]:.1f} / {mean_grad[1]:.1f} / {mean_grad[2]:.1f}\n"
            grad_text += f"Max Gradient (L/A/B): {max_grad[0]:.1f} / {max_grad[1]:.1f} / {max_grad[2]:.1f}\n"
            grad_text += f"Significant Color Changes: {sig_changes}\n\n"
            
            # Characterize the edge color pattern
            if sig_changes > 8:
                char_text = "High color variation - distinctive edge"
            elif sig_changes > 4:
                char_text = "Moderate color variation - somewhat distinctive"
            else:
                char_text = "Low color variation - uniform edge color"
            
            grad_text += f"Characterization: {char_text}"
            
            # Display gradient information
            ax_grad.text(0.05, 0.5, grad_text, fontsize=12, va='center', transform=ax_grad.transAxes)
            ax_grad.axis('off')
        
        # 4. Confidence information
        if 'confidence_sequence' in vis_data and vis_data['confidence_sequence'] is not None:
            ax_conf = plt.subplot(gs[row_start+1, 2:4])
            
            confidence = vis_data['confidence_sequence']
            
            if len(confidence) > 0:
                # Plot confidence values
                ax_conf.plot(confidence, 'b-', linewidth=2)
                ax_conf.set_title(f"Edge {edge_index+1} Color Confidence", fontsize=12)
                ax_conf.set_xlabel("Position along edge")
                ax_conf.set_ylabel("Confidence")
                ax_conf.set_ylim([0, 1.05])
                ax_conf.grid(True, linestyle='--', alpha=0.5)
                
                # Add mean confidence line
                mean_conf = np.mean(confidence)
                ax_conf.axhline(mean_conf, color='r', linestyle='--', 
                               label=f"Mean: {mean_conf:.2f}")
                ax_conf.legend()
            else:
                ax_conf.text(0.5, 0.5, "No confidence data available", 
                           ha='center', va='center', transform=ax_conf.transAxes)
                ax_conf.axis('off')
    
    if valid_edges == 0:
        plt.figtext(0.5, 0.5, "No color data available", 
                  ha='center', va='center', fontsize=16)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95, hspace=0.4)
    
    # Save figure directly to file instead of converting to OpenCV image
    plt.savefig(output_path, dpi=120)
    plt.close()

# ========= ADVANCED PIECE CLASSIFICATION FUNCTIONS =========

def calculate_edge_straightness(edge_points):
    """
    Calculate how straight an edge is (0.0 to 1.0).

    Args:
        edge_points: Liste of (x, y) edge points

    Returns:
        Straightness score between 0.0 (not straight) and 1.0 (perfectly straight)
    """
    # Check if we have enough points
    if len(edge_points) < 3:
        return 0.5  # Not enough points to determine straightness

    # Convert to numpy array if it's not already
    edge_points = np.array(edge_points)

    # Fit a line to the edge points
    vx, vy, x0, y0 = cv2.fitLine(edge_points, cv2.DIST_L2, 0, 0.01, 0.01)

    # Calculate distance of each point from the line
    distances = []
    for point in edge_points:
        # Calculate distance from point to line
        distance = abs((vy * (point[0] - x0) - vx * (point[1] - y0)) /
                       np.sqrt(vx*vx + vy*vy))
        distances.append(distance)

    # Calculate statistics
    max_distance = max(distances) if distances else 0
    avg_distance = sum(distances) / len(distances) if distances else 0

    # Calculate edge length to normalize distances
    if len(edge_points) > 1:
        # Use the distance between first and last points as approximation
        edge_length = np.linalg.norm(edge_points[-1] - edge_points[0])
    else:
        edge_length = 1.0  # Avoid division by zero

    # Normalize by edge length to get a relative measure
    relative_max_distance = max_distance / edge_length if edge_length > 0 else 1.0
    relative_avg_distance = avg_distance / edge_length if edge_length > 0 else 1.0

    # Combine into a single score (0 = very curved, 1 = perfectly straight)
    straightness = 1.0 - min(1.0, (relative_max_distance * 0.7 + relative_avg_distance * 0.3))

    return straightness

def validate_corner_angle(edge1_points, edge2_points):
    """
    Validate that two edges form a corner with approximately right angle.

    Args:
        edge1_points: Liste of (x, y) points for first edge
        edge2_points: Liste of (x, y) points for second edge

    Returns:
        Tuple (is_right_angle, angle_degrees)
    """
    # Check if we have enough points
    if len(edge1_points) < 2 or len(edge2_points) < 2:
        return False, 0.0

    # Convert to numpy arrays if they're not already
    edge1_points = np.array(edge1_points)
    edge2_points = np.array(edge2_points)

    # Calculate direction vectors for the edges
    vec1 = edge1_points[-1] - edge1_points[0]
    vec2 = edge2_points[-1] - edge2_points[0]

    # Normalize vectors
    vec1_norm = np.linalg.norm(vec1)
    vec2_norm = np.linalg.norm(vec2)

    if vec1_norm == 0 or vec2_norm == 0:
        return False, 0.0

    vec1 = vec1 / vec1_norm
    vec2 = vec2 / vec2_norm

    # Calculate the angle between the vectors (dot product)
    dot_product = np.dot(vec1, vec2)
    angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
    angle_degrees = np.degrees(angle)

    # Check if angle is approximately 90 degrees (with tolerance)
    is_right_angle = abs(angle_degrees - 90) < 20

    return is_right_angle, angle_degrees

def classify_corner_type(piece_data):
    """
    Classify the corner type (top-left, top-right, bottom-left, bottom-right).

    Args:
        piece_data: Piece data dictionary containing edge information

    Returns:
        Corner type as string: "top_left", "top_right", "bottom_left", "bottom_right" or "unknown"
    """
    if 'edge_types' not in piece_data or 'edge_points' not in piece_data:
        return "unknown"

    # Find the indices of straight edges
    straight_edge_indices = [i for i, edge_type in enumerate(piece_data['edge_types'])
                           if edge_type == "straight"]

    if len(straight_edge_indices) < 2:
        return "unknown"  # Not a corner

    # We need to determine the orientation of straight edges
    # This is a simplified approach; in a real puzzle, we would need
    # to consider the global orientation of the puzzle

    # Get direction vectors for the straight edges
    directions = []
    for edge_idx in straight_edge_indices:
        edge_points = piece_data['edge_points'][edge_idx]
        if len(edge_points) < 2:
            continue

        # Get normalized direction vector
        start, end = edge_points[0], edge_points[-1]
        direction = np.array([end[0] - start[0], end[1] - start[1]])
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction = direction / norm
        directions.append(direction)

    if len(directions) < 2:
        return "unknown"

    # Calculate angles of each direction with respect to the positive x-axis
    angles = []
    for direction in directions:
        angle = np.arctan2(direction[1], direction[0])
        angles.append(angle)

    # Convert to degrees for easier interpretation
    angles_deg = [np.degrees(angle) % 360 for angle in angles]

    # Determine corner type based on the angles
    # This is a simplified heuristic and can be improved
    sorted_angles = sorted(angles_deg)
    angle_diff = (sorted_angles[1] - sorted_angles[0]) % 360

    # Check if the angle difference is approximately 90 degrees
    if abs(angle_diff - 90) > 20:
        return "unknown"  # Not a 90-degree corner

    # Classify corner type based on angle combinations
    # These thresholds can be tuned based on your specific puzzle
    if (0 <= sorted_angles[0] <= 45 and 45 <= sorted_angles[1] <= 135):
        return "top_left"
    elif (45 <= sorted_angles[0] <= 135 and 135 <= sorted_angles[1] <= 225):
        return "top_right"
    elif (135 <= sorted_angles[0] <= 225 and 225 <= sorted_angles[1] <= 315):
        return "bottom_right"
    elif ((225 <= sorted_angles[0] <= 315 and 315 <= sorted_angles[1] <= 360) or
          (225 <= sorted_angles[0] <= 315 and 0 <= sorted_angles[1] <= 45)):
        return "bottom_left"
    else:
        return "unknown"

def classify_puzzle_pieces_refined(piece_results):
    """
    Refined classification of puzzle pieces with confidence scores and corner types.

    Args:
        piece_results: List of processed piece data with edge information

    Returns:
        Dictionary {piece_idx: {"category": category, "confidence": confidence, "type": corner_type}}
    """
    piece_categories = {}

    for piece_idx, piece_data in enumerate(piece_results):
        if 'edge_types' not in piece_data or 'edge_points' not in piece_data:
            piece_categories[piece_idx] = {
                "category": "regular",
                "confidence": 0.5,
                "type": "unknown"
            }
            continue

        # Calculate straightness scores for all edges
        edge_straightness = []
        for edge_points in piece_data['edge_points']:
            straightness = calculate_edge_straightness(edge_points)
            edge_straightness.append(straightness)

        # Find edges with high straightness scores
        straight_edge_indices = [i for i, score in enumerate(edge_straightness) if score > 0.8]
        straight_edges_count = len(straight_edge_indices)

        # Calculate average straightness of straight edges as confidence
        avg_straightness = np.mean([edge_straightness[i] for i in straight_edge_indices]) if straight_edge_indices else 0

        # Validate corners by checking angles between straight edges
        is_valid_corner = False
        corner_angle = 0

        if straight_edges_count >= 2:
            # Check pairs of straight edges to find a valid corner
            from itertools import combinations
            for i, j in combinations(straight_edge_indices, 2):
                is_right, angle = validate_corner_angle(
                    piece_data['edge_points'][i],
                    piece_data['edge_points'][j]
                )
                if is_right:
                    is_valid_corner = True
                    corner_angle = angle
                    break

        # Classify with confidence
        if straight_edges_count > 1 and is_valid_corner:
            category = "corner"
            corner_type = classify_corner_type(piece_data)
            confidence = avg_straightness * 0.7 + (1.0 - abs(corner_angle - 90) / 90) * 0.3
        elif straight_edges_count == 1:
            category = "edge"
            corner_type = "unknown"
            confidence = avg_straightness
        else:
            category = "regular"
            corner_type = "unknown"
            confidence = 1.0 - (max(edge_straightness) if edge_straightness else 0.5)

        piece_categories[piece_idx] = {
            "category": category,
            "confidence": confidence,
            "type": corner_type
        }

    return piece_categories

def classify_puzzle_pieces(piece_results):
    """
    Classifie les pièces du puzzle en trois catégories : coins, bords et régulières.

    Une pièce est:
    - un coin si elle a plus de 1 bord droit
    - un bord si elle a exactement 1 bord droit
    - régulière autrement

    Args:
        piece_results: Liste des résultats de traitement des pièces avec leurs types de bords

    Returns:
        Dictionnaire de classification {piece_idx: "corner"|"edge"|"regular"}
    """
    piece_categories = {}

    for piece_idx, piece_data in enumerate(piece_results):
        if 'edge_types' not in piece_data:
            # Si les types de bords ne sont pas disponibles, on considère la pièce comme régulière
            piece_categories[piece_idx] = "regular"
            continue

        # Compter les bords droits
        edge_types = piece_data['edge_types']
        straight_edges_count = sum(1 for edge_type in edge_types if edge_type == "straight")

        # Classer selon le nombre de bords droits
        if straight_edges_count > 1:
            piece_categories[piece_idx] = "corner"
        elif straight_edges_count == 1:
            piece_categories[piece_idx] = "edge"
        else:
            piece_categories[piece_idx] = "regular"

    return piece_categories

def calculate_puzzle_dimensions(piece_categories, total_pieces):
    """
    Calcule les dimensions du puzzle en utilisant la formule:
    - Total edges = (nb de pièces bord) + 2 x (nb de pièces coin)
    - P = total edges / 4
    - Delta = sqrt(P^2 - (total no. of pieces))
    - Puzzle width = P + delta
    - Puzzle height = P - delta

    Args:
        piece_categories: Dictionnaire de classification des pièces
        total_pieces: Nombre total de pièces dans le puzzle

    Returns:
        Tuple (width, height) représentant les dimensions calculées du puzzle
    """
    # Compter les pièces dans chaque catégorie
    corner_count = sum(1 for cat in piece_categories.values() if cat == "corner")
    edge_count = sum(1 for cat in piece_categories.values() if cat == "edge")

    # Calculer les dimensions selon la formule
    total_edges = edge_count + (2 * corner_count)
    P = total_edges / 4

    # Gérer le cas où P^2 < total_pieces (problème de racine négative)
    delta_squared = P**2 - total_pieces
    if delta_squared < 0:
        # En cas d'inconsistance, on utilise une valeur par défaut
        print(f"ATTENTION: Problème avec la formule (P^2 < total_pieces). Utilisation d'un delta par défaut.")
        delta = 0
        width = height = math.ceil(math.sqrt(total_pieces))
    else:
        delta = math.sqrt(delta_squared)
        width = math.ceil(P + delta)
        height = math.ceil(P - delta)

    return width, height

def create_piece_category_visualization(piece_categories, piece_images, output_path):
    """
    Crée une visualisation des catégories de pièces (coins, bords, régulières).

    Args:
        piece_categories: Dictionnaire de classification des pièces
        piece_images: Dictionnaire d'images des pièces
        output_path: Chemin pour sauvegarder la visualisation
    """
    if not piece_categories or not piece_images:
        return

    # Compter les pièces dans chaque catégorie
    corner_count = sum(1 for cat in piece_categories.values() if cat == "corner")
    edge_count = sum(1 for cat in piece_categories.values() if cat == "edge")
    regular_count = sum(1 for cat in piece_categories.values() if cat == "regular")

    # Nombre total de pièces
    total_pieces = len(piece_categories)

    # Créer une grande image
    max_pieces_per_row = 8
    rows_needed = math.ceil(total_pieces / max_pieces_per_row)

    # Obtenir une taille d'image de pièce typique
    sample_img = next(iter(piece_images.values()))
    piece_height, piece_width = sample_img.shape[:2]

    # Créer une image vide
    canvas_width = max_pieces_per_row * piece_width
    canvas_height = (rows_needed + 3) * piece_height  # +3 pour le titre et les légendes
    canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255

    # Ajouter un titre
    title = f"Classification des Pieces: {corner_count} coins, {edge_count} bords, {regular_count} regulieres"
    cv2.putText(canvas, title, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    # Dessiner des légendes colorées
    legend_y = 60
    cv2.rectangle(canvas, (20, legend_y), (40, legend_y+20), (0, 0, 255), -1)  # Rouge pour les coins
    cv2.putText(canvas, "Coins", (50, legend_y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    cv2.rectangle(canvas, (120, legend_y), (140, legend_y+20), (0, 255, 0), -1)  # Vert pour les bords
    cv2.putText(canvas, "Bords", (150, legend_y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    cv2.rectangle(canvas, (220, legend_y), (240, legend_y+20), (255, 0, 0), -1)  # Bleu pour régulières
    cv2.putText(canvas, "Regulieres", (250, legend_y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    # Disposer les pièces
    start_y = 100  # Décalage pour le titre et les légendes
    for piece_idx, category in piece_categories.items():
        if piece_idx not in piece_images:
            continue

        img = piece_images[piece_idx].copy()

        # Ajouter une bordure colorée selon la catégorie
        border_color = (0, 0, 0)  # Noir par défaut
        if category == "corner":
            border_color = (0, 0, 255)  # Rouge
        elif category == "edge":
            border_color = (0, 255, 0)  # Vert
        else:
            border_color = (255, 0, 0)  # Bleu

        # Ajouter une bordure épaisse
        img = cv2.copyMakeBorder(img, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=border_color)

        # Calculer la position
        row = piece_idx // max_pieces_per_row
        col = piece_idx % max_pieces_per_row

        x = col * piece_width
        y = start_y + row * piece_height

        # Placer l'image
        h, w = img.shape[:2]
        try:
            canvas[y:y+h, x:x+w] = img

            # Ajouter un numéro
            cv2.putText(canvas, str(piece_idx+1), (x+10, y+20),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        except:
            # En cas d'erreur de taille, continuons
            pass

    # Sauvegarder l'image
    cv2.imwrite(output_path, canvas)

# ========= EDGE MATCHING AND PUZZLE ASSEMBLY =========

def normalize_edge_points(points, target_points=50, flip_for_matching=True):
    """
    Normalize edge points for consistent comparison:
    1. Resample to fixed number of points
    2. Flip one edge if needed (for matching intrusion to extrusion)
    3. Scale to unit size
    
    Args:
        points: Array of (x, y) coordinates representing edge points
        target_points: Number of points to resample to
        flip_for_matching: Whether to flip points for tab-slot matching
        
    Returns:
        Normalized array of edge points
    """
    if len(points) < 2:
        # Not enough points to normalize
        return np.array(points)
        
    # Convert to numpy array if it's not already
    points = np.array(points)
    
    # Resample to target number of points using interpolation
    cumulative_distances = [0]
    for i in range(1, len(points)):
        d = np.linalg.norm(points[i] - points[i-1])
        cumulative_distances.append(cumulative_distances[-1] + d)
    
    total_length = cumulative_distances[-1]
    if total_length == 0:
        # All points are the same, can't normalize
        return points
    
    # Generate evenly spaced points
    resampled_points = []
    for i in range(target_points):
        target_dist = (i / (target_points-1)) * total_length
        idx = np.searchsorted(cumulative_distances, target_dist)
        if idx >= len(points):
            idx = len(points) - 1
        
        if idx == 0:
            resampled_points.append(points[0])
        else:
            # Linear interpolation
            d1 = cumulative_distances[idx-1]
            d2 = cumulative_distances[idx]
            frac = (target_dist - d1) / (d2 - d1) if d2 > d1 else 0
            p1 = points[idx-1]
            p2 = points[idx]
            interp_point = p1 + frac * (p2 - p1)
            resampled_points.append(interp_point)
    
    resampled_points = np.array(resampled_points)
    
    # Flip points for tab-slot matching if needed
    if flip_for_matching and len(resampled_points) > 2:
        # Find edge midpoint
        mid_idx = len(resampled_points) // 2
        midpoint = resampled_points[mid_idx]
        
        # Calculate vector from first to last point
        edge_vector = resampled_points[-1] - resampled_points[0]
        
        # Get normal vector (perpendicular to edge_vector)
        normal = np.array([-edge_vector[1], edge_vector[0]])
        normal = normal / np.linalg.norm(normal)
        
        # Flip points along this normal vector
        flipped_points = resampled_points - 2 * np.outer(
            np.dot(resampled_points - midpoint, normal), normal)
        
        return flipped_points
    
    return resampled_points

def calculate_procrustes_similarity(points1, points2):
    """
    Calculate similarity using Procrustes analysis.
    Returns a score between 0 and 1 (1 = perfect match).
    
    Args:
        points1: Array of (x, y) coordinates for first edge
        points2: Array of (x, y) coordinates for second edge
        
    Returns:
        Similarity score between 0 and 1 (higher is better)
    """
    # Input validation
    if len(points1) < 3 or len(points2) < 3:
        return 0.5  # Not enough points for meaningful analysis
        
    points1 = np.array(points1)
    points2 = np.array(points2)
    
    # Step 1: Center both point sets
    centroid1 = np.mean(points1, axis=0)
    centroid2 = np.mean(points2, axis=0)
    
    centered1 = points1 - centroid1
    centered2 = points2 - centroid2
    
    # Step 2: Calculate scale factors
    scale1 = np.sqrt(np.sum(centered1**2) / len(centered1))
    scale2 = np.sqrt(np.sum(centered2**2) / len(centered2))
    
    if scale1 == 0 or scale2 == 0:
        return 0.5  # Can't normalize
    
    # Normalize by scale
    normalized1 = centered1 / scale1
    normalized2 = centered2 / scale2
    
    # Step 3: Find optimal rotation using SVD
    correlation_matrix = normalized1.T @ normalized2
    
    try:
        # Use scipy for more robust SVD
        from scipy import linalg
        U, s, Vt = linalg.svd(correlation_matrix)
        
        # Ensure proper rotation matrix (no reflection)
        rotation = U @ Vt
        if np.linalg.det(rotation) < 0:
            Vt[-1, :] = -Vt[-1, :]
            rotation = U @ Vt
    except:
        # Fallback if SVD fails
        return 0.5
    
    # Step 4: Calculate residual (error after alignment)
    aligned = normalized1 @ rotation
    residual = np.sum((aligned - normalized2)**2)
    
    # Convert to similarity score (lower residual = higher similarity)
    max_possible_residual = len(points1) * 2  # Theoretical maximum
    similarity_score = 1.0 - min(1.0, residual / max_possible_residual)
    
    return similarity_score

def calculate_hausdorff_distance(points1, points2):
    """
    Calculate Hausdorff distance between two point sets.
    Lower distance means better match.
    
    Args:
        points1: Array of (x, y) coordinates for first edge
        points2: Array of (x, y) coordinates for second edge
        
    Returns:
        Hausdorff distance (lower is better)
    """
    if len(points1) == 0 or len(points2) == 0:
        return float('inf')  # Can't calculate
        
    points1 = np.array(points1)
    points2 = np.array(points2)
    
    # Calculate all pairwise distances efficiently
    # distance_matrix[i,j] = distance between points1[i] and points2[j]
    d1 = np.sum(points1**2, axis=1, keepdims=True)
    d2 = np.sum(points2**2, axis=1)
    
    # Use np.maximum to prevent negative values due to numerical instability
    distance_squared = np.maximum(0, d1 + d2[:, np.newaxis] - 2 * np.dot(points2, points1.T))
    distances = np.sqrt(distance_squared)
    
    # Forward Hausdorff: min distance from each point in points1 to any point in points2
    d1_to_2 = np.max(np.min(distances.T, axis=1))
    
    # Backward Hausdorff: min distance from each point in points2 to any point in points1
    d2_to_1 = np.max(np.min(distances, axis=1))
    
    # Hausdorff distance is the maximum of the two
    hausdorff_dist = max(d1_to_2, d2_to_1)
    
    return hausdorff_dist

def calculate_shape_compatibility(edge1_type, edge1_deviation, edge2_type, edge2_deviation):
    """
    Calculate shape compatibility between two edges based on their types.
    
    Compatibility rules:
    - Intrusion matches with extrusion (high score)
    - Two straight edges never match (zero score)
    - Other combinations have a low score
    
    Args:
        edge1_type: Type of first edge ('straight', 'intrusion', 'extrusion')
        edge1_deviation: Deviation value of first edge
        edge2_type: Type of second edge
        edge2_deviation: Deviation value of second edge
        
    Returns:
        Compatibility score between 0 and 1
    """
    # Check complementary types (intrusion should match with extrusion)
    if edge1_type == "intrusion" and edge2_type == "extrusion":
        # Calculate how well the shapes might fit (deviations should be roughly complementary)
        deviation_match = 1.0 - min(1.0, abs(abs(edge1_deviation) - abs(edge2_deviation)) / max(abs(edge1_deviation), abs(edge2_deviation), 1))
        return 0.9 * deviation_match  # High score for complementary types
    
    if edge1_type == "extrusion" and edge2_type == "intrusion":
        deviation_match = 1.0 - min(1.0, abs(abs(edge1_deviation) - abs(edge2_deviation)) / max(abs(edge1_deviation), abs(edge2_deviation), 1))
        return 0.9 * deviation_match  # High score for complementary types
    
    # Prevent matching between two straight edges
    if edge1_type == "straight" and edge2_type == "straight":
        return 0.0  # No match allowed between two straight edges
    
    # Default low compatibility
    return 0.1

def extract_edge_points_from_image(piece_idx, edge_idx, debug_dir='debug'):
    """
    Extract edge points from saved edge images in debug directory.
    
    Args:
        piece_idx: Piece index (0-based)
        edge_idx: Edge index (0-based)
        debug_dir: Debug directory containing edge images
        
    Returns:
        Array of (x, y) points representing the edge
    """
    # Construct path to edge image
    edge_path = os.path.join(debug_dir, 'edges', f"piece_{piece_idx+1}_edge_{edge_idx+1}.png")
    
    if not os.path.exists(edge_path):
        return []
    
    # Read edge image
    edge_img = cv2.imread(edge_path)
    if edge_img is None:
        return []
    
    # Extract edge points (green pixels)
    edge_points = []
    for y in range(edge_img.shape[0]):
        for x in range(edge_img.shape[1]):
            pixel = edge_img[y, x]
            # Check for green pixel (BGR = [0, 255, 0])
            if pixel[1] > 200 and pixel[0] < 50 and pixel[2] < 50:
                edge_points.append((x, y))
    
    return edge_points

def enhanced_edge_compatibility(edge1_points, edge2_points, edge1_type, edge1_deviation, 
                               edge2_type, edge2_deviation, edge1_colors, edge2_colors):
    """
    Enhanced edge compatibility calculation that combines:
    1. Basic shape compatibility from edge types
    2. Procrustes analysis for optimal shape alignment
    3. Hausdorff distance for detailed shape comparison
    4. Color compatibility
    
    Args:
        edge1_points: Array of (x, y) coordinates for first edge
        edge2_points: Array of (x, y) coordinates for second edge
        edge1_type: Type of first edge ('straight', 'intrusion', 'extrusion')
        edge1_deviation: Deviation value of first edge
        edge2_type: Type of second edge
        edge2_deviation: Deviation value of second edge
        edge1_colors: Color features of first edge
        edge2_colors: Color features of second edge
        
    Returns:
        Enhanced compatibility score between 0 and 1
    """
    # Stage 1: Current type-based compatibility (fast initial filter)
    basic_shape_score = calculate_shape_compatibility(edge1_type, edge1_deviation, edge2_type, edge2_deviation)
    
    # Calculate color compatibility separately
    color_score = calculate_color_compatibility(edge1_colors, edge2_colors)
    
    # Skip advanced shape analysis if basic compatibility is very low or if both edges are straight
    if basic_shape_score < 0.2 or (edge1_type == "straight" and edge2_type == "straight"):
        return basic_shape_score * 0.7 + color_score * 0.3
    
    # Stage 2: Advanced shape analysis for promising matches
    procrustes_score = 0.5  # Default value
    hausdorff_score = 0.5  # Default value
    
    # Only perform advanced shape analysis if we have enough points
    if len(edge1_points) > 5 and len(edge2_points) > 5:
        try:
            # Normalize edge points
            edge1_normalized = normalize_edge_points(edge1_points)
            
            # If matching intrusion to extrusion, flip one edge
            flip_for_matching = (edge1_type == "intrusion" and edge2_type == "extrusion") or \
                               (edge1_type == "extrusion" and edge2_type == "intrusion")
            
            edge2_normalized = normalize_edge_points(edge2_points, flip_for_matching=flip_for_matching)
            
            # Apply Procrustes analysis
            procrustes_score = calculate_procrustes_similarity(edge1_normalized, edge2_normalized)
            
            # Calculate Hausdorff distance and convert to a similarity score
            hausdorff_dist = calculate_hausdorff_distance(edge1_normalized, edge2_normalized)
            max_distance = 100.0  # Normalization factor - adjust based on your puzzle's scale
            hausdorff_score = 1.0 - min(1.0, hausdorff_dist / max_distance)
        except Exception as e:
            # Fallback to default values if there's an error
            procrustes_score = 0.5
            hausdorff_score = 0.5
    
    # 5. Analyze color distinctiveness for adaptive weighting
    color_distinctiveness = 0.0
    try:
        if edge1_colors and 'h_hist' in edge1_colors:
            # Measure the "peakiness" of the histograms - more peaks = more distinctive
            h_hist = np.array(edge1_colors['h_hist'], dtype=np.float32)
            peak_threshold = np.mean(h_hist) + np.std(h_hist)
            peak_count = np.sum(h_hist > peak_threshold)
            
            # Normalize - more peaks = more distinctive
            histogram_bins = len(h_hist)
            distinctiveness_from_peaks = min(1.0, peak_count / (histogram_bins / 4))
            
            # Also consider the standard deviation of the hue - higher deviation = more distinctive
            std_h = edge1_colors['std_hsv'][0] / 90.0  # Normalize by half the hue range
            std_distinctiveness = min(1.0, std_h)
            
            # Combine these metrics
            color_distinctiveness = 0.6 * distinctiveness_from_peaks + 0.4 * std_distinctiveness
        else:
            color_distinctiveness = 0.5  # Default value
    except Exception as e:
        color_distinctiveness = 0.5  # Default value
    
    # Adjust color weight based on distinctiveness
    if color_distinctiveness > 0.8:  # Very distinctive colors
        color_weight = 0.4  # Increase color weight
    elif color_distinctiveness < 0.3:  # Similar colors
        color_weight = 0.2  # Decrease color weight
    else:
        color_weight = 0.3  # Default weight
    
    # Combine shape scores (these are all shape-related)
    combined_shape_score = (0.4 * basic_shape_score + 
                           0.35 * procrustes_score + 
                           0.25 * hausdorff_score)
    
    # Final score with adaptive weighting between shape and color
    final_score = (1 - color_weight) * combined_shape_score + color_weight * color_score
    
    return final_score

def calculate_color_compatibility(color_feature1, color_feature2):
    """
    Calculate color compatibility between two edges based on their DTW color features.
    
    Args:
        color_feature1: Color features of first edge
        color_feature2: Color features of second edge
        
    Returns:
        Compatibility score between 0 and 1
    """
    if color_feature1 is None or color_feature2 is None:
        return 0.5  # Default mid-range score if no color data is available
    
    # Extract color sequences and confidence values
    lab_sequence1 = np.array(color_feature1['lab_sequence']) if 'lab_sequence' in color_feature1 else []
    lab_sequence2 = np.array(color_feature2['lab_sequence']) if 'lab_sequence' in color_feature2 else []
    
    confidence1 = color_feature1.get('confidence_sequence', None)
    confidence2 = color_feature2.get('confidence_sequence', None)
    
    # If either sequence is empty, return default score
    if len(lab_sequence1) == 0 or len(lab_sequence2) == 0:
        return 0.5
    
    # Run DTW matching in both directions (normal and reversed)
    # Normal direction
    normal_similarity = dtw_color_matching(lab_sequence1, lab_sequence2, confidence1, confidence2)
    
    # Reversed direction (since edges may match in opposite directions)
    reversed_sequence2 = np.flip(lab_sequence2, axis=0)
    reversed_confidence2 = np.flip(confidence2) if confidence2 is not None else None
    reversed_similarity = dtw_color_matching(lab_sequence1, reversed_sequence2, confidence1, reversed_confidence2)
    
    # Take the better matching direction
    similarity = max(normal_similarity, reversed_similarity)
    
    return similarity

def create_edge_match_visualization(match, piece_results, piece_images, output_path):
    """
    Create a visualization of a matched edge pair.
    
    Args:
        match: Dictionary containing match information
        piece_results: List of processed piece data
        piece_images: Dictionary of piece images
        output_path: Path to save the visualization
    """
    piece1_idx = match['piece1_idx']
    piece2_idx = match['piece2_idx']
    edge1_idx = match['edge1_idx']
    edge2_idx = match['edge2_idx']
    
    # Get piece images
    piece1_img = piece_images.get(piece1_idx)
    piece2_img = piece_images.get(piece2_idx)
    
    if piece1_img is None or piece2_img is None:
        return  # Skip if images are not available
    
    # Get edge types and scores
    edge1_type = piece_results[piece1_idx]['edge_types'][edge1_idx]
    edge2_type = piece_results[piece2_idx]['edge_types'][edge2_idx]
    
    # Since edge points aren't in the results, we'll use a simplified approach
    # We'll highlight the edges on the actual piece images
    
    # Create a combined visualization image
    # Make space for two pieces side by side with padding and info area
    max_height = max(piece1_img.shape[0], piece2_img.shape[0])
    total_width = piece1_img.shape[1] + piece2_img.shape[1] + 100  # Add padding between pieces
    
    # Create canvas with some extra height for text information
    vis_img = np.ones((max_height + 150, total_width, 3), dtype=np.uint8) * 255
    
    # Create copies of the images so we can draw on them
    piece1_draw = piece1_img.copy()
    piece2_draw = piece2_img.copy()
    
    # Calculate the center of each piece for finding corners
    h1, w1 = piece1_img.shape[:2]
    h2, w2 = piece2_img.shape[:2]
    center1 = (w1 // 2, h1 // 2)
    center2 = (w2 // 2, h2 // 2)
    
    # Draw the edge on piece 1 - we'll use a mask approach
    edge_mask1 = np.zeros_like(piece1_img)
    
    # Calculate approximate corners by dividing the piece into quadrants
    # These are just approximated positions, and will highlight roughly where the edges are
    corners1 = [
        (w1 // 4, h1 // 4),           # Top left
        (w1 * 3 // 4, h1 // 4),       # Top right
        (w1 * 3 // 4, h1 * 3 // 4),   # Bottom right
        (w1 // 4, h1 * 3 // 4)        # Bottom left
    ]
    
    # Draw the edge between approximate corners
    start_corner1 = corners1[edge1_idx]
    end_corner1 = corners1[(edge1_idx + 1) % 4]
    
    # Create a thick line along the edge
    cv2.line(edge_mask1, start_corner1, end_corner1, (0, 0, 255), 5)
    
    # Apply a dilation to make the mask cover more of the edge area
    kernel = np.ones((15, 15), np.uint8)
    edge_mask1 = cv2.dilate(edge_mask1, kernel, iterations=1)
    
    # Draw the edge on piece 2
    edge_mask2 = np.zeros_like(piece2_img)
    
    corners2 = [
        (w2 // 4, h2 // 4),           # Top left
        (w2 * 3 // 4, h2 // 4),       # Top right
        (w2 * 3 // 4, h2 * 3 // 4),   # Bottom right
        (w2 // 4, h2 * 3 // 4)        # Bottom left
    ]
    
    start_corner2 = corners2[edge2_idx]
    end_corner2 = corners2[(edge2_idx + 1) % 4]
    
    cv2.line(edge_mask2, start_corner2, end_corner2, (255, 0, 0), 5)
    edge_mask2 = cv2.dilate(edge_mask2, kernel, iterations=1)
    
    # Apply the masks to highlight the edges
    # We'll blend the masks with the original images
    alpha = 0.6  # Transparency factor
    
    # Apply the masks
    piece1_draw = cv2.addWeighted(piece1_draw, 1, edge_mask1, alpha, 0)
    piece2_draw = cv2.addWeighted(piece2_draw, 1, edge_mask2, alpha, 0)
    
    # Add circles at the corners
    cv2.circle(piece1_draw, start_corner1, 8, (0, 255, 0), -1)   # Green start corner
    cv2.circle(piece1_draw, end_corner1, 8, (255, 255, 0), -1)   # Yellow end corner
    
    cv2.circle(piece2_draw, start_corner2, 8, (0, 255, 0), -1)
    cv2.circle(piece2_draw, end_corner2, 8, (255, 255, 0), -1)
    
    # Draw the pieces on the visualization image
    y_offset1 = (max_height - piece1_img.shape[0]) // 2
    vis_img[y_offset1:y_offset1+piece1_img.shape[0], 0:piece1_img.shape[1]] = piece1_draw
    
    x_offset2 = piece1_img.shape[1] + 100  # Add 100px spacing
    y_offset2 = (max_height - piece2_img.shape[0]) // 2
    vis_img[y_offset2:y_offset2+piece2_img.shape[0], x_offset2:x_offset2+piece2_img.shape[1]] = piece2_draw
    
    # Add score and edge type information at the bottom
    info_y = max_height + 20
    cv2.putText(vis_img, f"Piece {piece1_idx+1} Edge {edge1_idx+1} ({edge1_type}) ↔ Piece {piece2_idx+1} Edge {edge2_idx+1} ({edge2_type})",
                (20, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    cv2.putText(vis_img, f"Total Score: {match['total_score']:.3f}   Shape Score: {match['shape_score']:.3f}   Color Score: {match['color_score']:.3f}",
                (20, info_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # Add arrow indicating the match
    mid1_x = (start_corner1[0] + end_corner1[0]) // 2
    mid1_y = (start_corner1[1] + end_corner1[1]) // 2 + y_offset1
    
    mid2_x = (start_corner2[0] + end_corner2[0]) // 2 + x_offset2
    mid2_y = (start_corner2[1] + end_corner2[1]) // 2 + y_offset2
    
    cv2.arrowedLine(vis_img, (int(mid1_x), int(mid1_y)), (int(mid2_x), int(mid2_y)), (0, 165, 255), 2, tipLength=0.05)
    
    # Add legend
    legend_y = info_y + 60
    # First edge
    cv2.circle(vis_img, (20, legend_y), 5, (0, 0, 255), -1)
    cv2.putText(vis_img, f"Piece {piece1_idx+1} Edge {edge1_idx+1}", (35, legend_y + 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    # Second edge
    cv2.circle(vis_img, (250, legend_y), 5, (255, 0, 0), -1)
    cv2.putText(vis_img, f"Piece {piece2_idx+1} Edge {edge2_idx+1}", (265, legend_y + 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    # Corners
    cv2.circle(vis_img, (450, legend_y), 5, (0, 255, 0), -1)
    cv2.putText(vis_img, "Start Corner", (465, legend_y + 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    cv2.circle(vis_img, (600, legend_y), 5, (255, 255, 0), -1)
    cv2.putText(vis_img, "End Corner", (615, legend_y + 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    # Add matching type explanation
    match_y = legend_y + 30
    cv2.putText(vis_img, f"Match Type: {edge1_type} ↔ {edge2_type}", (20, match_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # Add rank information
    cv2.putText(vis_img, f"Match Rank: #{match['rank'] if 'rank' in match else '?'}", (450, match_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # Save the visualization
    cv2.imwrite(output_path, vis_img)


def _process_piece_pair(args):
    """
    Helper function to process a single pair of pieces for edge matching.
    This function is designed to be used with multiprocessing.
    
    This is used by the parallel edge matching algorithm to distribute
    the workload across multiple CPU cores, significantly accelerating
    the matching process, especially for puzzles with many pieces.
    
    Args:
        args: Tuple containing (piece1, piece2, edge_points_cache)
        
    Returns:
        List of potential matches between the two pieces
    """
    try:
        piece1, piece2, edge_points_cache_dict = args
        
        # Local matches for this pair of pieces
        local_matches = []
        
        # Create a copy of the cache dict to avoid modifying the shared dict
        local_cache = dict(edge_points_cache_dict)
        
        # For each combination of edges
        for edge1_idx in range(4):
            for edge2_idx in range(4):
                try:
                    # Get edge data
                    edge1_type = piece1['edge_types'][edge1_idx]
                    edge1_deviation = piece1['edge_deviations'][edge1_idx]
                    edge1_colors = piece1['edge_colors'][edge1_idx]
                    
                    edge2_type = piece2['edge_types'][edge2_idx]
                    edge2_deviation = piece2['edge_deviations'][edge2_idx]
                    edge2_colors = piece2['edge_colors'][edge2_idx]
                    
                    # Skip unknown edges
                    if edge1_type == "unknown" or edge2_type == "unknown":
                        continue
                    
                    # Get edge points from cache or extract them
                    edge1_key = (piece1['piece_idx'], edge1_idx)
                    if edge1_key not in local_cache:
                        try:
                            local_cache[edge1_key] = extract_edge_points_from_image(
                                piece1['piece_idx'], edge1_idx)
                        except Exception:
                            # If extraction fails, use empty points
                            local_cache[edge1_key] = []
                            
                    edge2_key = (piece2['piece_idx'], edge2_idx)
                    if edge2_key not in local_cache:
                        try:
                            local_cache[edge2_key] = extract_edge_points_from_image(
                                piece2['piece_idx'], edge2_idx)
                        except Exception:
                            # If extraction fails, use empty points
                            local_cache[edge2_key] = []
                    
                    edge1_points = local_cache[edge1_key]
                    edge2_points = local_cache[edge2_key]
                    
                    # Use enhanced compatibility calculation
                    try:
                        # Try enhanced compatibility if we have edge points
                        if len(edge1_points) > 5 and len(edge2_points) > 5:
                            total_score = enhanced_edge_compatibility(
                                edge1_points, edge2_points,
                                edge1_type, edge1_deviation,
                                edge2_type, edge2_deviation,
                                edge1_colors, edge2_colors
                            )
                            
                            # Calculate component scores for reference
                            shape_score = calculate_shape_compatibility(
                                edge1_type, edge1_deviation, edge2_type, edge2_deviation
                            )
                            color_score = calculate_color_compatibility(edge1_colors, edge2_colors)
                            
                            # Approximate advanced shape scores for record-keeping
                            advanced_shape_score = (total_score - 0.3 * color_score) / 0.7
                        else:
                            # Fallback to standard scoring
                            shape_score = calculate_shape_compatibility(
                                edge1_type, edge1_deviation, edge2_type, edge2_deviation
                            )
                            color_score = calculate_color_compatibility(edge1_colors, edge2_colors)
                            total_score = 0.7 * shape_score + 0.3 * color_score
                            advanced_shape_score = shape_score
                    except Exception:
                        # Fallback to standard scoring on error
                        shape_score = calculate_shape_compatibility(
                            edge1_type, edge1_deviation, edge2_type, edge2_deviation
                        )
                        color_score = calculate_color_compatibility(edge1_colors, edge2_colors)
                        total_score = 0.7 * shape_score + 0.3 * color_score
                        advanced_shape_score = shape_score
                    
                    # Store the match if score is above threshold
                    if total_score > 0.4:  # Lower threshold to allow more matches
                        local_matches.append({
                            'piece1_idx': piece1['piece_idx'],
                            'piece2_idx': piece2['piece_idx'],
                            'edge1_idx': edge1_idx,
                            'edge2_idx': edge2_idx,
                            'total_score': total_score,
                            'shape_score': shape_score,
                            'color_score': color_score,
                            'advanced_shape_score': advanced_shape_score,
                            'has_advanced_analysis': len(edge1_points) > 5 and len(edge2_points) > 5
                        })
                except Exception as edge_err:
                    # Skip this edge pair if there's an error
                    continue
        
        # Return the matches for this pair of pieces
        return local_matches
    except Exception as e:
        # If the entire piece pair processing fails, return an empty list
        # to ensure the overall process continues
        return []


def match_edges(piece_results, num_processes=None):
    """
    Match edges between all puzzle pieces using enhanced compatibility scoring.
    Uses multiprocessing to parallelize the matching process.
    
    Performance improvement: This parallelized implementation significantly
    reduces processing time by distributing the edge matching workload across
    multiple CPU cores. The speedup is approximately linear with the number
    of cores, with some overhead for process management.
    
    For example:
    - Single-core: 100% processing time (baseline)
    - 4 cores: ~30% processing time (3.3x faster)
    - 8 cores: ~15% processing time (6.7x faster)
    
    Memory usage will increase with the number of processes, so adjust
    the num_processes parameter based on your system capabilities.
    
    Args:
        piece_results: List of processed piece data with edge information
        num_processes: Number of processes to use (None = auto-detect)
        
    Returns:
        Dictionary with edge matches and compatibility scores
    """
    num_pieces = len(piece_results)
    
    # Auto-detect number of processes if not specified
    if num_processes is None:
        num_processes = max(1, multiprocessing.cpu_count() - 1)
    
    # Create a progress counter
    total_comparisons = num_pieces * (num_pieces - 1) * 16 // 2  # For each pair of pieces, 16 possible edge combinations
    
    print(f"Starting parallel edge matching with {num_processes} processes (comparing {total_comparisons} potential matches)...")
    
    # Create a global cache for edge points to avoid repeatedly extracting them
    edge_points_cache = {}
    
    # Prepare piece pairs for parallel processing
    piece_pairs = []
    for i in range(num_pieces):
        for j in range(i + 1, num_pieces):  # Only compare each pair once
            piece_pairs.append((piece_results[i], piece_results[j], edge_points_cache))
    
    # Process piece pairs in parallel
    start_time = time.time()
    all_matches = []
    
    # Use a progress tracking approach
    processed_pairs = 0
    total_pairs = len(piece_pairs)
    
    # Create batches for better progress reporting
    batch_size = max(1, total_pairs // 20)  # Report progress roughly every 5%
    batches = [piece_pairs[i:i+batch_size] for i in range(0, total_pairs, batch_size)]
    
    print(f"Divided work into {len(batches)} batches for better progress tracking")
    
    # Process each batch
    for batch_idx, batch in enumerate(batches):
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
            batch_results = list(executor.map(_process_piece_pair, batch))
        
        # Flatten results and add to all matches
        for result in batch_results:
            all_matches.extend(result)
        
        # Update progress
        processed_pairs += len(batch)
        elapsed = time.time() - start_time
        estimated_total = elapsed * total_pairs / processed_pairs if processed_pairs > 0 else 0
        remaining = estimated_total - elapsed
        
        print(f"Matching progress: {processed_pairs}/{total_pairs} pairs processed ({processed_pairs*100//total_pairs}%)")
        print(f"Time elapsed: {elapsed:.1f}s, estimated remaining: {remaining:.1f}s")
    
    # Sort matches by descending score
    all_matches.sort(key=lambda x: x['total_score'], reverse=True)
    
    # Count matches that used advanced analysis
    advanced_matches = sum(1 for m in all_matches if m.get('has_advanced_analysis', False))
    print(f"Found {len(all_matches)} potential edge matches ({advanced_matches} with advanced shape analysis).")
    print(f"Total edge matching time: {time.time() - start_time:.1f} seconds")
    
    return all_matches

class PuzzleAssembler:
    """Class to handle puzzle assembly from edge matches."""

    def __init__(self, piece_results, edge_matches):
        """
        Initialize the puzzle assembler.

        Args:
            piece_results: List of processed piece data
            edge_matches: List of edge matches between pieces
        """
        self.piece_results = piece_results
        self.edge_matches = edge_matches
        self.num_pieces = len(piece_results)

        # Grid for piece placement - key is (row, col) tuple, value is piece_idx
        self.grid = {}

        # Positions of placed pieces - key is piece_idx, value is (row, col)
        self.placed_positions = {}

        # Edges already used in connections - set of (piece_idx, edge_idx) tuples
        self.used_edges = set()

        # Set of piece indices already placed
        self.placed_pieces = set()

        # Set of piece indices that can potentially be placed next (frontier)
        self.frontier = set()

        # Grid size tracking
        self.min_row = 0
        self.max_row = 0
        self.min_col = 0
        self.max_col = 0

        # Backtracking support
        self.history = []  # Stack to keep track of placement history for backtracking
        self.dead_ends = set()  # Set of piece/position combinations that led to dead ends
        self.backtrack_count = 0  # Counter to track how many times we've backtracked
        self.max_backtrack_depth = 5  # Maximum depth to backtrack before trying a different strategy

        # Dynamic threshold support
        self.initial_match_threshold = 0.4  # Starting match threshold
        self.min_match_threshold = 0.2  # Minimum threshold we'll accept
        self.current_match_threshold = self.initial_match_threshold  # Current threshold value

        # Advanced piece classification with corner types and confidence scores
        self.piece_categories_refined = classify_puzzle_pieces_refined(piece_results)

        # Backward compatibility with old classification
        self.piece_categories = {idx: info["category"] for idx, info in self.piece_categories_refined.items()}

        # Calculate puzzle dimensions
        width, height = calculate_puzzle_dimensions(self.piece_categories, self.num_pieces)
        self.calculated_width = width
        self.calculated_height = height

        # For corner-type based assembly
        self.corner_positions = {
            "top_left": (0, 0),
            "top_right": (0, width-1),
            "bottom_left": (height-1, 0),
            "bottom_right": (height-1, width-1)
        }

        # For fast lookup of corners and edges
        self.corner_pieces = [idx for idx, info in self.piece_categories_refined.items()
                            if info["category"] == "corner"]
        self.edge_pieces = [idx for idx, info in self.piece_categories_refined.items()
                          if info["category"] == "edge"]

        # Sort corners and edges by confidence
        self.corner_pieces.sort(key=lambda idx: self.piece_categories_refined[idx]["confidence"], reverse=True)
        self.edge_pieces.sort(key=lambda idx: self.piece_categories_refined[idx]["confidence"], reverse=True)

        print(f"Puzzle dimensions calculées: {width}x{height}")

        # Count pieces by type with refined classification
        corner_count = sum(1 for info in self.piece_categories_refined.values() if info["category"] == "corner")
        edge_count = sum(1 for info in self.piece_categories_refined.values() if info["category"] == "edge")
        regular_count = sum(1 for info in self.piece_categories_refined.values() if info["category"] == "regular")

        print(f"Classification des pièces: {corner_count} coins, {edge_count} bords, {regular_count} régulières")

        # Print corner types
        corner_types = {idx: info["type"] for idx, info in self.piece_categories_refined.items()
                      if info["category"] == "corner" and info["type"] != "unknown"}
        if corner_types:
            print(f"Types de coins détectés: {corner_types}")
    
    def start_assembly(self, seed_piece_idx=None):
        """
        Start the assembly by placing the first piece.

        Args:
            seed_piece_idx: Optional specific piece to use as the seed (None = auto-select)

        Returns:
            Boolean indicating success
        """
        if not self.edge_matches or not self.piece_results:
            print("No pieces or matches to assemble.")
            return False

        # If no specific seed piece provided, look for a corner piece first
        if seed_piece_idx is None:
            # Chercher d'abord parmi les pièces de coin
            corner_pieces = [idx for idx, cat in self.piece_categories.items()
                           if cat == "corner"]

            if corner_pieces:
                # Si on a des coins, commençons par un coin
                corner_match_scores = {}

                # Calculer un score pour chaque coin
                for piece_idx in corner_pieces:
                    # Compter les matches pour cette pièce
                    match_count = 0
                    match_score_sum = 0

                    for match in self.edge_matches[:min(len(self.edge_matches), 300)]:
                        if match['piece1_idx'] == piece_idx or match['piece2_idx'] == piece_idx:
                            match_count += 1
                            match_score_sum += match['total_score']

                    if match_count > 0:
                        corner_match_scores[piece_idx] = match_count * match_score_sum

                # Choisir le coin avec le meilleur score
                if corner_match_scores:
                    seed_piece_idx = max(corner_match_scores.items(), key=lambda x: x[1])[0]
                    print(f"Selected corner piece: {seed_piece_idx+1} as seed")
                else:
                    # Si aucun coin n'a de correspondances, prendre simplement le premier coin
                    seed_piece_idx = corner_pieces[0]
                    print(f"Selected first corner piece: {seed_piece_idx+1} as seed (no matches found)")
            else:
                # Pas de coins identifiés, revenir à la méthode originale basée uniquement sur les matches
                # Choose seed piece - take the one with the most high-quality matches
                piece_match_counts = {}
                piece_match_scores = {}

                # Consider a larger number of matches
                for match in self.edge_matches[:min(len(self.edge_matches), 300)]:
                    piece1_idx = match['piece1_idx']
                    piece2_idx = match['piece2_idx']
                    score = match['total_score']

                    # Count occurrences
                    if piece1_idx not in piece_match_counts:
                        piece_match_counts[piece1_idx] = 0
                        piece_match_scores[piece1_idx] = 0
                    if piece2_idx not in piece_match_counts:
                        piece_match_counts[piece2_idx] = 0
                        piece_match_scores[piece2_idx] = 0

                    piece_match_counts[piece1_idx] += 1
                    piece_match_counts[piece2_idx] += 1

                    # Also sum up scores
                    piece_match_scores[piece1_idx] += score
                    piece_match_scores[piece2_idx] += score

                # Choose piece with best combination of count and score
                seed_candidates = {}
                for piece_idx in piece_match_counts:
                    seed_candidates[piece_idx] = piece_match_counts[piece_idx] * piece_match_scores[piece_idx]

                # Choose piece with highest combined score
                seed_piece_idx = max(seed_candidates.items(), key=lambda x: x[1])[0] if seed_candidates else 0
                print(f"Selected seed piece: {seed_piece_idx+1} with {piece_match_counts.get(seed_piece_idx, 0)} matches")
        else:
            print(f"Using specified seed piece: {seed_piece_idx+1}")

        # Check if this piece is valid
        if seed_piece_idx < 0 or seed_piece_idx >= self.num_pieces:
            print(f"Invalid seed piece index: {seed_piece_idx}")
            return False

        # Place seed piece at (0, 0)
        self.place_piece(seed_piece_idx, 0, 0)

        # Add neighboring pieces to frontier
        self.update_frontier()

        return True
    
    def place_piece(self, piece_idx, row, col, edge_match=None, track_history=True):
        """
        Place a puzzle piece at the specified grid position.
        
        Args:
            piece_idx: Index of the piece to place
            row, col: Grid coordinates for placement
            edge_match: Optional details about the edge match that led to this placement
            track_history: Whether to add this placement to history for backtracking
            
        Returns:
            Boolean indicating success
        """
        if piece_idx in self.placed_pieces:
            return False
        
        if (row, col) in self.grid:
            return False
        
        # Check if this is a known dead end
        placement_signature = (piece_idx, row, col)
        if placement_signature in self.dead_ends:
            return False
        
        self.grid[(row, col)] = piece_idx
        self.placed_positions[piece_idx] = (row, col)
        self.placed_pieces.add(piece_idx)
        
        # Update grid bounds
        self.min_row = min(self.min_row, row)
        self.max_row = max(self.max_row, row)
        self.min_col = min(self.min_col, col)
        self.max_col = max(self.max_col, col)
        
        # Record placement in history for potential backtracking
        if track_history:
            self.history.append({
                'piece_idx': piece_idx,
                'position': (row, col),
                'edge_match': edge_match,
                'used_edges': set(self.used_edges),  # Make a copy of current used edges
                'match_threshold': self.current_match_threshold
            })
        
        return True
        
    def remove_piece(self, piece_idx):
        """
        Remove a piece from the puzzle assembly for backtracking.
        
        Args:
            piece_idx: Index of the piece to remove
            
        Returns:
            Boolean indicating success
        """
        if piece_idx not in self.placed_pieces:
            return False
        
        # Get piece position
        position = self.placed_positions[piece_idx]
        
        # Remove from tracking structures
        del self.grid[position]
        del self.placed_positions[piece_idx]
        self.placed_pieces.remove(piece_idx)
        
        # Remove edges associated with this piece from used_edges
        self.used_edges = {edge for edge in self.used_edges if edge[0] != piece_idx}
        
        # We don't update grid bounds (min_row, etc.) as that would be complex
        # and unnecessary for the backtracking algorithm
        
        return True
        
    def backtrack(self):
        """
        Backtrack to a previous state in the assembly process.
        
        Returns:
            Boolean indicating whether backtracking was successful
        """
        if not self.history:
            return False
        
        # Get the last placement
        last_placement = self.history.pop()
        piece_idx = last_placement['piece_idx']
        
        # Mark this placement as a dead end to avoid trying it again
        placement_signature = (piece_idx, *last_placement['position'])
        self.dead_ends.add(placement_signature)
        
        # Remove the piece
        self.remove_piece(piece_idx)
        
        # Restore the used edges set to the state before this placement
        self.used_edges = last_placement['used_edges']
        
        # Update frontier after removing the piece
        self.update_frontier()
        
        # Track backtracking activity
        self.backtrack_count += 1
        
        print(f"Backtracked: removed piece {piece_idx+1} from position {last_placement['position']}")
        return True
    
    def update_frontier(self):
        """Update the frontier of pieces that can be placed next."""
        # Clear current frontier
        self.frontier = set()
        
        # For each placed piece, find its unplaced neighbors
        for placed_idx in self.placed_pieces:
            for match in self.edge_matches:
                # Check if this match involves the placed piece
                if match['piece1_idx'] == placed_idx and match['piece2_idx'] not in self.placed_pieces:
                    self.frontier.add(match['piece2_idx'])
                elif match['piece2_idx'] == placed_idx and match['piece1_idx'] not in self.placed_pieces:
                    self.frontier.add(match['piece1_idx'])
    
    def determine_piece_position(self, piece_idx):
        """
        Determine the best position for placing the next piece.
        Now supports rotation by considering different edge orientations.
        
        Args:
            piece_idx: Index of the piece to place
            
        Returns:
            Tuple of (row, col) for the piece position, and edge match details for orientation
        """
        best_score = -1
        best_position = None
        best_edge_match = None
        
        # Look at all potential connections to placed pieces
        for match in self.edge_matches:
            # Case 1: piece_idx is piece1 and piece2 is already placed
            if match['piece1_idx'] == piece_idx and match['piece2_idx'] in self.placed_pieces:
                placed_idx = match['piece2_idx']
                placed_edge = match['edge2_idx']
                new_edge = match['edge1_idx']
                score = match['total_score']
            # Case 2: piece_idx is piece2 and piece1 is already placed
            elif match['piece2_idx'] == piece_idx and match['piece1_idx'] in self.placed_pieces:
                placed_idx = match['piece1_idx']
                placed_edge = match['edge1_idx']
                new_edge = match['edge2_idx']
                score = match['total_score']
            else:
                continue
                
            # Skip if the edge of the placed piece is already used
            if (placed_idx, placed_edge) in self.used_edges:
                continue
                
            # Only consider matches above threshold
            if score < self.current_match_threshold:
                continue
                
            # Determine the position based on the edge orientation
            placed_row, placed_col = self.placed_positions[placed_idx]
            
            # Calculate new position based on edge indices
            # Edge indices: 0=top, 1=right, 2=bottom, 3=left (clockwise from top)
            if placed_edge == 0:  # Top edge of placed piece
                new_row, new_col = placed_row - 1, placed_col
                required_edge = 2  # New piece's bottom should connect
            elif placed_edge == 1:  # Right edge of placed piece
                new_row, new_col = placed_row, placed_col + 1
                required_edge = 3  # New piece's left should connect
            elif placed_edge == 2:  # Bottom edge of placed piece
                new_row, new_col = placed_row + 1, placed_col
                required_edge = 0  # New piece's top should connect
            elif placed_edge == 3:  # Left edge of placed piece
                new_row, new_col = placed_row, placed_col - 1
                required_edge = 1  # New piece's right should connect
            
            # Check if the position is already occupied
            if (new_row, new_col) in self.grid:
                continue
                
            # Calculate rotation needed for proper orientation
            # If new_edge != required_edge, we need rotation
            rotation_needed = (required_edge - new_edge) % 4  # Clockwise rotation steps needed
            
            # Apply a rotation penalty to the score (slightly prefer non-rotated pieces)
            adjusted_score = score
            if rotation_needed > 0:
                # Small penalty for rotation (5% per step of rotation)
                rotation_penalty = 0.05 * rotation_needed
                adjusted_score = score * (1 - rotation_penalty)
                
            # Apply global constraints to ensure consistent assembly
            # Check neighbor consistency - pieces should have matching neighbors on all sides
            neighbor_count = 0
            conflict_count = 0
            
            # Check all 4 adjacent positions
            for neighbor_dir in range(4):
                # Calculate neighbor position
                if neighbor_dir == 0:  # Top
                    neighbor_row, neighbor_col = new_row - 1, new_col
                    piece_edge = (new_edge + 4 - rotation_needed) % 4  # Edge that would connect to top
                    neighbor_edge = 2  # Bottom edge of neighbor
                elif neighbor_dir == 1:  # Right
                    neighbor_row, neighbor_col = new_row, new_col + 1
                    piece_edge = (new_edge + 5 - rotation_needed) % 4  # Edge that would connect to right
                    neighbor_edge = 3  # Left edge of neighbor
                elif neighbor_dir == 2:  # Bottom
                    neighbor_row, neighbor_col = new_row + 1, new_col
                    piece_edge = (new_edge + 6 - rotation_needed) % 4  # Edge that would connect to bottom
                    neighbor_edge = 0  # Top edge of neighbor
                elif neighbor_dir == 3:  # Left
                    neighbor_row, neighbor_col = new_row, new_col - 1
                    piece_edge = (new_edge + 7 - rotation_needed) % 4  # Edge that would connect to left
                    neighbor_edge = 1  # Right edge of neighbor
                
                # Skip if this is the direction we're connecting from
                if (neighbor_row, neighbor_col) == (placed_row, placed_col):
                    continue
                    
                # Check if there's a neighbor in this direction
                neighbor_idx = self.grid.get((neighbor_row, neighbor_col))
                if neighbor_idx is not None:
                    neighbor_count += 1
                    
                    # Check if there's a good match between this piece and the neighbor
                    has_good_match = False
                    for match in self.edge_matches:
                        # Check if this match involves our piece and the neighbor
                        if (match['piece1_idx'] == piece_idx and match['piece2_idx'] == neighbor_idx and
                            match['edge1_idx'] == piece_edge and match['edge2_idx'] == neighbor_edge):
                            # Found a match, check if it's good
                            if match['total_score'] >= self.current_match_threshold:
                                has_good_match = True
                            break
                        elif (match['piece2_idx'] == piece_idx and match['piece1_idx'] == neighbor_idx and
                              match['edge2_idx'] == piece_edge and match['edge1_idx'] == neighbor_edge):
                            # Found a match (reversed), check if it's good
                            if match['total_score'] >= self.current_match_threshold:
                                has_good_match = True
                            break
                    
                    # If there's a neighbor but no good match, it's a conflict
                    if not has_good_match:
                        conflict_count += 1
            
            # Adjust score based on global constraints
            constraint_adjusted_score = adjusted_score
            
            # Penalty for conflicts with existing neighbors
            if conflict_count > 0:
                # Severe penalty for each conflict (50% per conflict)
                constraint_adjusted_score *= (1 - 0.5 * conflict_count)
            
            # Bonus for having multiple consistent neighbors (5% per consistent neighbor)
            consistent_neighbors = neighbor_count - conflict_count
            if consistent_neighbors > 0:
                constraint_adjusted_score *= (1 + 0.05 * consistent_neighbors)
                
            # Skip completely if the conflicts are too severe
            if conflict_count > 1:  # More than one conflict is too problematic
                continue
                
            # Store rotation information in the edge match data
            edge_match_with_rotation = (placed_idx, placed_edge, new_edge, rotation_needed)
                
            # Update best match if this is better
            if constraint_adjusted_score > best_score:
                best_score = constraint_adjusted_score
                best_position = (new_row, new_col)
                best_edge_match = edge_match_with_rotation
        
        return best_position, best_edge_match
    
    def assemble_next_piece(self):
        """
        Place the next piece with the highest score.
        Uses dynamic thresholds and backtracking when necessary.
        
        Returns:
            True if a piece was placed, False otherwise
        """
        # If frontier is empty, try backtracking first
        if not self.frontier:
            # If we can backtrack, we'll try again with a new configuration
            if self.backtrack():
                # Consider lowering the threshold after backtracking
                self.adjust_threshold(lower=True)
                return True  # We made progress by backtracking
            else:
                # No more backtracking possible
                return False
            
        best_piece = None
        best_position = None
        best_score = -1
        best_edge_match = None
        best_match_data = None
        
        # Evaluate each piece in the frontier
        for piece_idx in self.frontier:
            position, edge_match = self.determine_piece_position(piece_idx)
            if position:
                # Find the corresponding match data
                for match in self.edge_matches:
                    if ((match['piece1_idx'] == piece_idx and match['piece2_idx'] == edge_match[0]) or
                        (match['piece2_idx'] == piece_idx and match['piece1_idx'] == edge_match[0])) and \
                       ((match['edge1_idx'] == edge_match[2] and match['edge2_idx'] == edge_match[1]) or
                        (match['edge2_idx'] == edge_match[2] and match['edge1_idx'] == edge_match[1])):
                        score = match['total_score']
                        
                        # Only consider scores above the current threshold
                        if score >= self.current_match_threshold and score > best_score:
                            best_score = score
                            best_piece = piece_idx
                            best_position = position
                            best_edge_match = edge_match
                            best_match_data = match
                        break
        
        # If found a piece to place
        if best_piece and best_position:
            row, col = best_position
            # Record the match data for history
            self.place_piece(best_piece, row, col, edge_match=best_match_data)
            
            # Mark edges as used and track rotation info
            placed_idx, placed_edge, new_edge, rotation = best_edge_match
            self.used_edges.add((placed_idx, placed_edge))
            self.used_edges.add((best_piece, new_edge))
            
            # Store rotation information with the piece
            match_with_rotation = best_match_data.copy() if best_match_data else {}
            match_with_rotation['rotation'] = rotation
            if rotation > 0:
                print(f"  Piece {best_piece+1} rotated {rotation*90}° clockwise")
            
            # Update frontier
            self.frontier.remove(best_piece)
            self.update_frontier()
            
            # Since we placed a piece successfully, reset backtracking metrics
            self.backtrack_count = 0
            
            # If score was very good, consider raising the threshold again
            if best_score > self.current_match_threshold + 0.1:
                self.adjust_threshold(lower=False)
                
            return True
        
        # If we didn't find a piece to place, try backtracking
        if self.backtrack_count < self.max_backtrack_depth:
            if self.backtrack():
                return True  # We made progress by backtracking
        
        # If backtracking limit reached or no backtracking possible, try lowering the threshold
        if self.adjust_threshold(lower=True):
            print(f"Lowered match threshold to {self.current_match_threshold:.2f}")
            return self.assemble_next_piece()  # Try again with lower threshold
        
        return False
        
    def adjust_threshold(self, lower=True):
        """
        Adjust the matching threshold dynamically.
        
        Args:
            lower: If True, lower the threshold; otherwise raise it
            
        Returns:
            Boolean indicating whether the threshold was adjusted
        """
        if lower:
            # Don't go below minimum threshold
            if self.current_match_threshold > self.min_match_threshold:
                # Reduce by 0.05 at a time
                self.current_match_threshold = max(
                    self.current_match_threshold - 0.05, 
                    self.min_match_threshold
                )
                return True
        else:
            # Don't go above initial threshold
            if self.current_match_threshold < self.initial_match_threshold:
                # Increase by 0.05 at a time
                self.current_match_threshold = min(
                    self.current_match_threshold + 0.05,
                    self.initial_match_threshold
                )
                return True
                
        return False
    
    def find_seed_candidates(self, num_candidates=5, candidate_set=None):
        """
        Find the best seed piece candidates based on match quality.

        Args:
            num_candidates: Number of candidates to return
            candidate_set: Optional set of piece indices to consider (e.g., only corners)

        Returns:
            List of piece indices to try as seeds, sorted by potential
        """
        piece_match_counts = {}
        piece_match_scores = {}

        # Consider a larger number of matches
        for match in self.edge_matches[:min(len(self.edge_matches), 300)]:
            piece1_idx = match['piece1_idx']
            piece2_idx = match['piece2_idx']
            score = match['total_score']

            # Ne considérer que les pièces dans candidate_set si spécifié
            if candidate_set is not None:
                if piece1_idx not in candidate_set and piece2_idx not in candidate_set:
                    continue

                # Si un seul des deux est dans candidate_set, ne compter que celui-là
                if piece1_idx not in candidate_set:
                    piece1_idx = None
                if piece2_idx not in candidate_set:
                    piece2_idx = None

            # Count occurrences pour piece1
            if piece1_idx is not None:
                if piece1_idx not in piece_match_counts:
                    piece_match_counts[piece1_idx] = 0
                    piece_match_scores[piece1_idx] = 0
                piece_match_counts[piece1_idx] += 1
                piece_match_scores[piece1_idx] += score

            # Count occurrences pour piece2
            if piece2_idx is not None:
                if piece2_idx not in piece_match_counts:
                    piece_match_counts[piece2_idx] = 0
                    piece_match_scores[piece2_idx] = 0
                piece_match_counts[piece2_idx] += 1
                piece_match_scores[piece2_idx] += score

        # Choose pieces with best combination of count and score
        seed_candidates = {}

        # Si un ensemble de candidats est fourni mais qu'aucun n'a de correspondance
        if candidate_set is not None and not piece_match_counts:
            # Retourner simplement les n premiers candidats fournis
            return list(candidate_set)[:num_candidates]

        for piece_idx in piece_match_counts:
            # Accorder un bonus aux pièces selon leur catégorie
            category_bonus = 1.0
            if piece_idx in self.piece_categories:
                if self.piece_categories[piece_idx] == "corner":
                    category_bonus = 1.3  # Bonus pour les coins
                elif self.piece_categories[piece_idx] == "edge":
                    category_bonus = 1.1  # Petit bonus pour les bords

            seed_candidates[piece_idx] = (piece_match_counts[piece_idx] *
                                         piece_match_scores[piece_idx] *
                                         category_bonus)

        # Sort by score and return top candidates
        sorted_candidates = sorted(seed_candidates.items(), key=lambda x: x[1], reverse=True)
        return [c[0] for c in sorted_candidates[:num_candidates]]
    
    def reset_assembly(self):
        """Reset the assembly state to start fresh."""
        self.grid = {}
        self.placed_positions = {}
        self.used_edges = set()
        self.placed_pieces = set()
        self.frontier = set()
        self.history = []
        self.dead_ends = set()
        self.backtrack_count = 0
        self.current_match_threshold = self.initial_match_threshold
        
        # Reset grid bounds
        self.min_row = 0
        self.max_row = 0
        self.min_col = 0
        self.max_col = 0
        
        return True
    
    def assemble_puzzle_with_corner_types(self):
        """
        Assemble the puzzle using corner type information for better placement.

        Returns:
            Dictionary with assembly results
        """
        print("Starting puzzle assembly with corner types...")

        # First, try to establish a framework with known corner types
        corner_framework = {}
        for corner_type, position in self.corner_positions.items():
            matching_corners = [
                idx for idx in self.corner_pieces
                if self.piece_categories_refined[idx]["type"] == corner_type
            ]

            if matching_corners:
                # Use the highest confidence corner of this type
                matching_corners.sort(
                    key=lambda idx: self.piece_categories_refined[idx]["confidence"],
                    reverse=True
                )
                corner_framework[corner_type] = matching_corners[0]

        # If we have a good framework with at least 2 corners, use them as starting points
        if len(corner_framework) >= 2:
            print(f"Found framework with {len(corner_framework)} typed corners: {corner_framework}")

            # Start with framework assembly
            self.reset_assembly()

            # Place the corners in their correct positions
            for corner_type, piece_idx in corner_framework.items():
                row, col = self.corner_positions[corner_type]
                placed = self.place_piece(piece_idx, row, col)
                if placed:
                    print(f"Placed {corner_type} corner ({piece_idx+1}) at position ({row}, {col})")
                else:
                    print(f"Failed to place {corner_type} corner ({piece_idx+1})")

            # Update frontier with potential next pieces
            self.update_frontier()

            # Now assemble the rest of the puzzle
            pieces_placed = len(corner_framework)
            iterations = 0
            max_iterations = self.num_pieces * 4

            while pieces_placed < self.num_pieces and iterations < max_iterations:
                success = self.assemble_next_piece()
                if success:
                    current_placed = len(self.placed_pieces)
                    if current_placed > pieces_placed:
                        print(f"Placed piece {current_placed}/{self.num_pieces}")
                    pieces_placed = current_placed
                else:
                    print(f"Could not place more pieces after {pieces_placed}/{self.num_pieces}")
                    break
                iterations += 1

            # Calculate grid dimensions
            grid_height = self.max_row - self.min_row + 1
            grid_width = self.max_col - self.min_col + 1

            # Validate against calculated dimensions
            dimension_match = self.is_dimension_match(grid_width, grid_height)

            print(f"Framework-based assembly complete. Placed {pieces_placed}/{self.num_pieces} pieces.")
            print(f"Puzzle dimensions: {grid_width}x{grid_height}, " +
                  ("matches" if dimension_match else "differs from") +
                  f" calculated {self.calculated_width}x{self.calculated_height}")

            # Return assembly results
            return {
                "success": pieces_placed > 0,
                "pieces_placed": pieces_placed,
                "total_pieces": self.num_pieces,
                "grid": dict(self.grid),
                "placed_positions": dict(self.placed_positions),
                "dimensions": (grid_width, grid_height),
                "bounds": (self.min_row, self.min_col, self.max_row, self.max_col),
                "dimension_match": dimension_match,
                "calculated_dimensions": (self.calculated_width, self.calculated_height),
                "assembly_method": "framework"
            }

        # If framework approach didn't fully succeed or we don't have enough typed corners,
        # fall back to regular assembly
        print("Not enough typed corners for framework assembly, falling back to normal assembly...")
        return self.assemble_puzzle(True, 3)

    def is_dimension_match(self, grid_width, grid_height):
        """Check if the current grid dimensions match the calculated dimensions."""
        if not (self.calculated_width and self.calculated_height):
            return False

        return ((grid_width == self.calculated_width and grid_height == self.calculated_height) or
                (grid_width == self.calculated_height and grid_height == self.calculated_width))

    def assemble_puzzle(self, try_multiple_starts=True, max_start_attempts=3):
        """
        Assemble the complete puzzle.

        Args:
            try_multiple_starts: Whether to try multiple starting pieces
            max_start_attempts: Maximum number of different starting pieces to try

        Returns:
            Dictionary with assembly results
        """
        print("Starting puzzle assembly...")

        best_assembly = None
        best_pieces_placed = 0

        # Get seed candidates if using multiple starts
        seed_candidates = [None]  # Default will use automatic selection
        if try_multiple_starts:
            # Utiliser d'abord les coins comme candidats
            corner_pieces = [idx for idx, cat in self.piece_categories.items()
                           if cat == "corner"]

            if corner_pieces and len(corner_pieces) <= max_start_attempts:
                # Si on a assez de coins, essayons-les tous
                seed_candidates = corner_pieces
                print(f"Will try all {len(seed_candidates)} corner pieces as seeds")
            elif corner_pieces:
                # Sinon, sélectionner un sous-ensemble de coins
                best_corner_candidates = self.find_seed_candidates(
                    num_candidates=max_start_attempts,
                    candidate_set=corner_pieces
                )
                seed_candidates = best_corner_candidates
                print(f"Will try {len(seed_candidates)} best corner pieces as seeds")
            else:
                # Pas de coins, utiliser la méthode originale
                seed_candidates = self.find_seed_candidates(num_candidates=max_start_attempts)
                print(f"Will try {len(seed_candidates)} different starting pieces (no corners found)")

        # Try each seed candidate
        for attempt, seed_piece in enumerate(seed_candidates):
            if attempt > 0:
                # Reset for a new attempt
                print(f"\nTrying alternate starting piece {seed_piece+1}...")
                self.reset_assembly()

            # Place the first piece (automatic or specified)
            success = self.start_assembly(seed_piece)
            if not success:
                print(f"Failed to start assembly with seed piece {seed_piece+1 if seed_piece is not None else 'auto'}.")
                continue

            # Keep track of pieces placed
            pieces_placed = 1
            iterations = 0
            max_iterations = self.num_pieces * 3  # Increased for backtracking

            # Assemble pieces until no more can be placed or all are placed
            while pieces_placed < self.num_pieces and iterations < max_iterations:
                success = self.assemble_next_piece()
                if success:
                    # If we backtracked, the count may not increase
                    current_placed = len(self.placed_pieces)
                    if current_placed > pieces_placed:
                        print(f"Placed piece {current_placed}/{self.num_pieces}")
                    pieces_placed = current_placed
                else:
                    print(f"Could not place more pieces after {pieces_placed}/{self.num_pieces}")
                    break
                iterations += 1

            # Calculate grid dimensions
            grid_height = self.max_row - self.min_row + 1
            grid_width = self.max_col - self.min_col + 1

            print(f"Assembly attempt {attempt+1} complete. Placed {pieces_placed}/{self.num_pieces} pieces.")
            print(f"Puzzle dimensions: {grid_width}x{grid_height}")

            # Vérifier si les dimensions correspondent aux dimensions calculées
            dimension_match = False
            if self.calculated_width and self.calculated_height:
                # Vérifier si les dimensions correspondent (dans un sens ou dans l'autre)
                if (grid_width == self.calculated_width and grid_height == self.calculated_height) or \
                   (grid_width == self.calculated_height and grid_height == self.calculated_width):
                    dimension_match = True
                    print(f"✓ Assembled dimensions match calculated dimensions!")
                else:
                    print(f"⚠ Assembled dimensions {grid_width}x{grid_height} differ from calculated {self.calculated_width}x{self.calculated_height}")

            # Calculer un score pour cette assemblage (basé sur les pièces placées et la correspondance des dimensions)
            assembly_score = pieces_placed
            if dimension_match:
                assembly_score *= 1.2  # Bonus pour correspondance des dimensions

            # Save this assembly if it's the best so far
            if assembly_score > best_pieces_placed:
                best_pieces_placed = assembly_score
                best_assembly = {
                    "success": pieces_placed > 0,
                    "pieces_placed": pieces_placed,
                    "total_pieces": self.num_pieces,
                    "grid": dict(self.grid),  # Make copies to avoid reference issues
                    "placed_positions": dict(self.placed_positions),
                    "dimensions": (grid_width, grid_height),
                    "bounds": (self.min_row, self.min_col, self.max_row, self.max_col),
                    "seed_piece": seed_piece,
                    "dimension_match": dimension_match,
                    "calculated_dimensions": (self.calculated_width, self.calculated_height)
                }

                # If we've placed all pieces, no need to try more seeds
                if pieces_placed == self.num_pieces:
                    print("Found a complete solution!")
                    break

        # Restore the best assembly if we tried multiple
        if try_multiple_starts and best_assembly and best_assembly["seed_piece"] != seed_candidates[0]:
            print(f"\nRestoring best assembly (placed {best_assembly['pieces_placed']}/{self.num_pieces} pieces)...")
            self.reset_assembly()
            self.grid = best_assembly["grid"]
            self.placed_positions = best_assembly["placed_positions"]
            self.placed_pieces = set(best_assembly["placed_positions"].keys())
            # We don't restore used_edges or frontier as they're not critical
            
            # Restore bounds
            self.min_row, self.min_col, self.max_row, self.max_col = best_assembly["bounds"]
        
        return best_assembly if best_assembly else {
            "success": False,
            "pieces_placed": 0,
            "total_pieces": self.num_pieces
        }
    
    def visualize_assembly(self, output_path, piece_images=None, piece_masks=None, debug_dir='debug'):
        """
        Create an enhanced visualization of the assembled puzzle that preserves piece shapes
        and shows how pieces actually connect.
        
        Args:
            output_path: Path to save the visualization
            piece_images: Dictionary of piece images (optional)
            piece_masks: Dictionary of piece binary masks (optional)
            debug_dir: Debug directory containing piece data
            
        Returns:
            Assembly visualization image
        """
        if not self.placed_pieces:
            print("No pieces placed to visualize.")
            return None
        
        # Calculate grid dimensions
        grid_height = self.max_row - self.min_row + 1
        grid_width = self.max_col - self.min_col + 1
        
        # Try to load piece masks if not provided
        if piece_masks is None and piece_images is not None:
            piece_masks = {}
            for piece_idx, piece_img in piece_images.items():
                # Create a binary mask from non-black pixels in the image
                gray = cv2.cvtColor(piece_img, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
                piece_masks[piece_idx] = mask
        
        # Determine piece size and spacing
        if piece_images and len(piece_images) > 0:
            # Find average piece size
            avg_height = sum(img.shape[0] for img in piece_images.values()) / len(piece_images)
            avg_width = sum(img.shape[1] for img in piece_images.values()) / len(piece_images)
            piece_size = max(int(avg_height), int(avg_width))
        else:
            piece_size = 150  # Default size if no images
        
        # Create three visualizations:
        # 1. Grid-based simple view (for reference)
        # 2. Real-shape assembly view (more realistic)
        # 3. Interactive exploded view (showing pieces with connections)
        
        # --- 1. Grid-based Simple View ---
        grid_cell_size = piece_size
        grid_canvas_height = grid_height * grid_cell_size + 50  # Extra space for labels
        grid_canvas_width = grid_width * grid_cell_size + 50    # Extra space for labels
        grid_canvas = np.ones((grid_canvas_height, grid_canvas_width, 3), dtype=np.uint8) * 255
        
        # Draw grid lines
        for i in range(grid_height + 1):
            y = i * grid_cell_size + 25  # Offset for labels
            cv2.line(grid_canvas, (25, y), (grid_canvas_width-25, y), (200, 200, 200), 1)
        for i in range(grid_width + 1):
            x = i * grid_cell_size + 25  # Offset for labels
            cv2.line(grid_canvas, (x, 25), (x, grid_canvas_height-25), (200, 200, 200), 1)
        
        # Place pieces on grid canvas
        for (row, col), piece_idx in self.grid.items():
            # Convert grid position to canvas coordinates
            canvas_row = row - self.min_row
            canvas_col = col - self.min_col
            
            # Calculate cell position with offset
            y1 = canvas_row * grid_cell_size + 25
            y2 = (canvas_row + 1) * grid_cell_size + 25
            x1 = canvas_col * grid_cell_size + 25
            x2 = (canvas_col + 1) * grid_cell_size + 25
            
            # Draw piece on canvas
            if piece_images and piece_idx in piece_images:
                # Resize piece image to fit cell
                piece_img = piece_images[piece_idx]
                resized_img = cv2.resize(piece_img, (grid_cell_size, grid_cell_size))
                grid_canvas[y1:y2, x1:x2] = resized_img
            else:
                # Draw colored rectangle with piece index
                color = ((piece_idx * 40) % 256, (piece_idx * 70) % 256, (piece_idx * 110) % 256)
                cv2.rectangle(grid_canvas, (x1, y1), (x2, y2), color, -1)
            
            # Always add piece number (even on images)
            cv2.putText(grid_canvas, f"{piece_idx+1}", (x1 + 10, y1 + 20), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Add labels for rows and columns
        for i in range(grid_height):
            row_label = str(i + self.min_row)
            cv2.putText(grid_canvas, row_label, (5, i * grid_cell_size + grid_cell_size//2 + 25), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        for i in range(grid_width):
            col_label = str(i + self.min_col)
            cv2.putText(grid_canvas, col_label, (i * grid_cell_size + grid_cell_size//2 + 25, 15), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        # --- 2. Real-shape Assembly View ---
        if piece_images and piece_masks:
            # Step 1: Prepare full-size canvas for realistic assembly
            # Use a large canvas and calculate where to place each piece
            padding = 100  # Extra padding around the assembled puzzle
            
            # Find max height and width needed
            if piece_images and len(piece_images) > 0:
                real_width = grid_width * piece_size + padding * 2
                real_height = grid_height * piece_size + padding * 2
            else:
                real_width = grid_width * 150 + padding * 2
                real_height = grid_height * 150 + padding * 2
            
            real_canvas = np.ones((real_height, real_width, 3), dtype=np.uint8) * 255
            assembled_mask = np.zeros((real_height, real_width), dtype=np.uint8)
            
            # Track each piece's position for edge visualization
            piece_positions = {}
            
            # Step 2: Process each piece and position it correctly on the canvas
            for (row, col), piece_idx in self.grid.items():
                if piece_idx not in piece_images:
                    continue
                    
                # Get piece image and mask
                piece_img = piece_images[piece_idx]
                piece_mask = piece_masks[piece_idx]
                
                # Calculate piece position in the real canvas
                center_y = padding + (row - self.min_row) * piece_size + piece_size // 2
                center_x = padding + (col - self.min_col) * piece_size + piece_size // 2
                
                # Store piece center position for edge visualization
                piece_positions[piece_idx] = (center_x, center_y)
                
                # Get piece dimensions
                h, w = piece_img.shape[:2]
                
                # Calculate top-left corner for placement
                y1 = center_y - h // 2
                x1 = center_x - w // 2
                y2 = y1 + h
                x2 = x1 + w
                
                # Ensure within bounds
                if y1 < 0 or x1 < 0 or y2 >= real_height or x2 >= real_width:
                    continue
                
                # Create a bigger mask for the overlapping area
                overlap_mask = assembled_mask[y1:y2, x1:x2].copy()
                
                # Calculate overlap with already placed pieces
                overlap = cv2.bitwise_and(overlap_mask, piece_mask)
                
                # Place the piece using the mask
                roi = real_canvas[y1:y2, x1:x2]
                
                # Copy piece to canvas where the mask is set
                np.copyto(roi, piece_img, where=cv2.cvtColor(piece_mask, cv2.COLOR_GRAY2BGR) > 0)
                
                # Update assembled mask (add this piece's mask to it)
                assembled_mask[y1:y2, x1:x2] = cv2.bitwise_or(assembled_mask[y1:y2, x1:x2], piece_mask)
                
                # Add a small text label with piece number
                label_x = x1 + w // 2 - 10
                label_y = y1 + h // 2
                cv2.putText(real_canvas, f"{piece_idx+1}", (label_x, label_y), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            # If we have pieces and edge matches, let's draw connection lines between matched pieces
            if len(self.used_edges) > 0:
                # Create a copy of the canvas for marking edges
                edge_canvas = real_canvas.copy()
                
                # Draw lines between pieces that are connected
                # Go through used_edges to draw connections
                for piece_idx, edge_idx in self.used_edges:
                    # Find the matching piece for this edge
                    matching_piece = None
                    matching_edge = None
                    
                    for match in self.edge_matches:
                        if match['piece1_idx'] == piece_idx and match['edge1_idx'] == edge_idx:
                            if match['piece2_idx'] in self.placed_pieces:
                                matching_piece = match['piece2_idx']
                                matching_edge = match['edge2_idx']
                                break
                        elif match['piece2_idx'] == piece_idx and match['edge2_idx'] == edge_idx:
                            if match['piece1_idx'] in self.placed_pieces:
                                matching_piece = match['piece1_idx']
                                matching_edge = match['edge1_idx']
                                break
                    
                    # If we found a match and both pieces are positioned
                    if matching_piece is not None and piece_idx in piece_positions and matching_piece in piece_positions:
                        # Get piece centers
                        p1_x, p1_y = piece_positions[piece_idx]
                        p2_x, p2_y = piece_positions[matching_piece]
                        
                        # Draw a connection line between the pieces
                        cv2.line(edge_canvas, (int(p1_x), int(p1_y)), (int(p2_x), int(p2_y)), 
                               (0, 255, 0), 2, cv2.LINE_AA)
                
                # Save the edge-highlighted canvas
                edge_output_path = output_path.replace('.png', '_edges.png')
                cv2.imwrite(edge_output_path, edge_canvas)
                print(f"Edge-highlighted assembly saved to {edge_output_path}")
            
            # Step 3: Crop to fit the actual assembled puzzle
            # Find the bounding box of the assembled puzzle
            non_zero_points = cv2.findNonZero(assembled_mask)
            if non_zero_points is not None:
                x, y, w, h = cv2.boundingRect(non_zero_points)
                
                # Add some margin
                margin = 20
                x = max(0, x - margin)
                y = max(0, y - margin)
                w = min(real_width - x, w + 2 * margin)
                h = min(real_height - y, h + 2 * margin)
                
                # Crop the canvas
                real_canvas = real_canvas[y:y+h, x:x+w]
            
            # Save the realistic assembly visualization
            real_output_path = output_path.replace('.png', '_realistic.png')
            cv2.imwrite(real_output_path, real_canvas)
            print(f"Realistic assembly visualization saved to {real_output_path}")
            
            # --- 3. Create an exploded view visualization ---
            # This view shows pieces slightly separated to better see their shapes and connections
            exploded_padding = 150  # Space between pieces
            exploded_width = grid_width * (piece_size + exploded_padding) + padding * 2
            exploded_height = grid_height * (piece_size + exploded_padding) + padding * 2
            exploded_canvas = np.ones((exploded_height, exploded_width, 3), dtype=np.uint8) * 255
            
            # Draw pieces with spacing between them
            piece_centers = {}  # To track piece centers for connection lines
            
            for (row, col), piece_idx in self.grid.items():
                if piece_idx not in piece_images:
                    continue
                    
                # Get piece image and mask
                piece_img = piece_images[piece_idx]
                piece_mask = piece_masks[piece_idx]
                
                # Calculate piece position in the exploded canvas (with spacing)
                center_y = padding + (row - self.min_row) * (piece_size + exploded_padding) + piece_size // 2
                center_x = padding + (col - self.min_col) * (piece_size + exploded_padding) + piece_size // 2
                
                # Store center for connection lines
                piece_centers[piece_idx] = (center_x, center_y)
                
                # Get piece dimensions
                h, w = piece_img.shape[:2]
                
                # Calculate top-left corner for placement
                y1 = center_y - h // 2
                x1 = center_x - w // 2
                y2 = y1 + h
                x2 = x1 + w
                
                # Skip if out of bounds
                if y1 < 0 or x1 < 0 or y2 >= exploded_height or x2 >= exploded_width:
                    continue
                
                # Place the piece using the mask
                roi = exploded_canvas[y1:y2, x1:x2]
                
                # Copy piece to canvas where the mask is set
                np.copyto(roi, piece_img, where=cv2.cvtColor(piece_mask, cv2.COLOR_GRAY2BGR) > 0)
                
                # Add piece number
                cv2.putText(exploded_canvas, f"{piece_idx+1}", (x1 + 10, y1 + 20), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Draw connection lines between matching edges
            for piece_idx, edge_idx in self.used_edges:
                # Find the matching piece
                for match in self.edge_matches:
                    if (match['piece1_idx'] == piece_idx and match['edge1_idx'] == edge_idx and 
                        match['piece2_idx'] in self.placed_pieces):
                        matching_piece = match['piece2_idx']
                        
                        # If both pieces are positioned in our visualization
                        if piece_idx in piece_centers and matching_piece in piece_centers:
                            # Get piece centers
                            p1_x, p1_y = piece_centers[piece_idx]
                            p2_x, p2_y = piece_centers[matching_piece]
                            
                            # Draw a dashed connection line
                            # Calculate a vector from p1 to p2
                            dx = p2_x - p1_x
                            dy = p2_y - p1_y
                            
                            # Get unit vector
                            length = np.sqrt(dx**2 + dy**2)
                            if length > 0:
                                dx /= length
                                dy /= length
                            
                            # Draw dashed line with score indicator
                            num_dashes = 10
                            dash_length = length / (num_dashes * 2)
                            
                            for i in range(num_dashes):
                                start_x = int(p1_x + dx * i * dash_length * 2)
                                start_y = int(p1_y + dy * i * dash_length * 2)
                                end_x = int(start_x + dx * dash_length)
                                end_y = int(start_y + dy * dash_length)
                                
                                # Color based on edge score (red to green)
                                score = match['total_score']
                                # Color from red (bad) to green (good) based on score
                                if score < 0.3:
                                    color = (0, 0, 255)  # Red for poor match
                                elif score < 0.6:
                                    color = (0, 165, 255)  # Orange for medium match
                                else:
                                    color = (0, 255, 0)  # Green for good match
                                    
                                cv2.line(exploded_canvas, (start_x, start_y), (end_x, end_y), 
                                       color, 2, cv2.LINE_AA)
                        
                        break
                    
                    elif (match['piece2_idx'] == piece_idx and match['edge2_idx'] == edge_idx and 
                          match['piece1_idx'] in self.placed_pieces):
                        matching_piece = match['piece1_idx']
                        
                        # If both pieces are positioned in our visualization
                        if piece_idx in piece_centers and matching_piece in piece_centers:
                            # Get piece centers
                            p1_x, p1_y = piece_centers[piece_idx]
                            p2_x, p2_y = piece_centers[matching_piece]
                            
                            # Draw a dashed connection line
                            # Calculate a vector from p1 to p2
                            dx = p2_x - p1_x
                            dy = p2_y - p1_y
                            
                            # Get unit vector
                            length = np.sqrt(dx**2 + dy**2)
                            if length > 0:
                                dx /= length
                                dy /= length
                            
                            # Draw dashed line with score indicator
                            num_dashes = 10
                            dash_length = length / (num_dashes * 2)
                            
                            for i in range(num_dashes):
                                start_x = int(p1_x + dx * i * dash_length * 2)
                                start_y = int(p1_y + dy * i * dash_length * 2)
                                end_x = int(start_x + dx * dash_length)
                                end_y = int(start_y + dy * dash_length)
                                
                                # Color based on edge score (red to green)
                                score = match['total_score']
                                # Color from red (bad) to green (good) based on score
                                if score < 0.3:
                                    color = (0, 0, 255)  # Red for poor match
                                elif score < 0.6:
                                    color = (0, 165, 255)  # Orange for medium match
                                else:
                                    color = (0, 255, 0)  # Green for good match
                                    
                                cv2.line(exploded_canvas, (start_x, start_y), (end_x, end_y), 
                                       color, 2, cv2.LINE_AA)
                        
                        break
            
            # Draw legend for the connection colors
            legend_y = 30
            legend_x = 30
            cv2.putText(exploded_canvas, "Connection Strength:", (legend_x, legend_y), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            
            # Poor match
            cv2.line(exploded_canvas, (legend_x, legend_y + 30), (legend_x + 40, legend_y + 30), 
                   (0, 0, 255), 2)
            cv2.putText(exploded_canvas, "Poor Match", (legend_x + 50, legend_y + 35), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            # Medium match
            cv2.line(exploded_canvas, (legend_x, legend_y + 60), (legend_x + 40, legend_y + 60), 
                   (0, 165, 255), 2)
            cv2.putText(exploded_canvas, "Medium Match", (legend_x + 50, legend_y + 65), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            # Good match
            cv2.line(exploded_canvas, (legend_x, legend_y + 90), (legend_x + 40, legend_y + 90), 
                   (0, 255, 0), 2)
            cv2.putText(exploded_canvas, "Good Match", (legend_x + 50, legend_y + 95), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            # Save exploded view
            exploded_output_path = output_path.replace('.png', '_exploded.png')
            cv2.imwrite(exploded_output_path, exploded_canvas)
            print(f"Exploded view visualization saved to {exploded_output_path}")
            
            # Create a combined view with all visualizations
            # Create a 2x2 grid with all visualizations
            # First resize all images to similar heights for consistent layout
            target_height = 600
            
            # Resize grid canvas
            grid_h, grid_w = grid_canvas.shape[:2]
            grid_scale = target_height / grid_h
            resized_grid = cv2.resize(grid_canvas, (int(grid_w * grid_scale), target_height))
            
            # Resize realistic canvas
            real_h, real_w = real_canvas.shape[:2]
            real_scale = target_height / real_h
            resized_real = cv2.resize(real_canvas, (int(real_w * real_scale), target_height))
            
            # Resize exploded canvas
            exploded_h, exploded_w = exploded_canvas.shape[:2]
            exploded_scale = target_height / exploded_h
            resized_exploded = cv2.resize(exploded_canvas, (int(exploded_w * exploded_scale), target_height))
            
            # Create a combined canvas with all visualizations
            # Calculate the total width needed (all images + padding)
            total_width = resized_grid.shape[1] + resized_real.shape[1] + resized_exploded.shape[1] + 40  # 20px padding between images
            
            # Create the combined canvas
            combined_canvas = np.ones((target_height + 50, total_width, 3), dtype=np.uint8) * 255  # Extra 50px for titles
            
            # Add titles at the top
            title_y = 30
            grid_title_x = resized_grid.shape[1] // 2 - 40
            real_title_x = resized_grid.shape[1] + 20 + resized_real.shape[1] // 2 - 70
            exploded_title_x = resized_grid.shape[1] + resized_real.shape[1] + 40 + resized_exploded.shape[1] // 2 - 60
            
            cv2.putText(combined_canvas, "Grid View", (grid_title_x, title_y), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            cv2.putText(combined_canvas, "Realistic View", (real_title_x, title_y), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            cv2.putText(combined_canvas, "Exploded View", (exploded_title_x, title_y), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            
            # Place the images on the canvas
            combined_canvas[50:50+target_height, :resized_grid.shape[1]] = resized_grid
            combined_canvas[50:50+target_height, resized_grid.shape[1]+20:resized_grid.shape[1]+20+resized_real.shape[1]] = resized_real
            combined_canvas[50:50+target_height, resized_grid.shape[1]+resized_real.shape[1]+40:] = resized_exploded
            
            # Save combined view
            combined_output_path = output_path.replace('.png', '_all_views.png')
            cv2.imwrite(combined_output_path, combined_canvas)
            print(f"Combined visualization with all views saved to {combined_output_path}")
            
            # Also save the original grid view
            cv2.imwrite(output_path, grid_canvas)
            
            return combined_canvas
        else:
            # If no piece images available, just save the grid view
            cv2.imwrite(output_path, grid_canvas)
            print(f"Grid assembly visualization saved to {output_path}")
            return grid_canvas

def main():
    """Fonction principale simplifiée."""
    # Régler la priorité du processus
    try:
        if sys.platform == 'win32':
            psutil.Process().nice(psutil.HIGH_PRIORITY_CLASS)
        else:
            os.nice(-10)
        print("Process priority set to high")
    except Exception as e:
        print(f"Could not set process priority: {e}")
    
    # Préparer les répertoires
    dirs = setup_output_directories()
    output_dirs = (dirs['edges'], dirs['edge_types'], dirs['corners'], dirs['contours'])
    
    # Détecter les pièces de puzzle (utilise le cache si disponible)
    with Timer("Puzzle detection"):
        puzzle_data = detect_puzzle_pieces(INPUT_PATH, THRESHOLD_VALUE, MIN_CONTOUR_AREA)
        
        # Lire directement l'image pour les tâches restantes
        img = cv2.imread(INPUT_PATH)
        if img is None:
            raise ValueError(f"Could not read image from {INPUT_PATH}")
        
        # Recréer le masque pour simplifier
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary_mask = cv2.threshold(gray_img, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)
        
        # Morphologie
        closing_kernel = np.ones((9, 9), np.uint8)
        dilation_kernel = np.ones((3, 3), np.uint8)
        
        closed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, closing_kernel)
        processed_mask = cv2.dilate(closed_mask, dilation_kernel, iterations=1)
        
        # Trouver et filtrer les contours
        contours, _ = cv2.findContours(processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = [c for c in contours if cv2.contourArea(c) > MIN_CONTOUR_AREA]
        
        # Créer le masque final
        filled_mask = np.zeros_like(processed_mask)
        cv2.drawContours(filled_mask, valid_contours, -1, 255, -1)
    
    # Extraire les données des pièces pour le traitement parallèle
    piece_count = puzzle_data['count']
    pieces = puzzle_data['pieces']
    
    print(f"Detected {piece_count} puzzle pieces")
    
    # Sauvegarder les masques et pièces
    with Timer("Saving initial files"):
        save_masks(img, filled_mask, valid_contours, dirs)
        save_pieces(pieces, img, filled_mask, dirs)
    
    # Traiter les pièces en parallèle
    results = parallel_process_pieces(pieces, output_dirs, MAX_WORKERS)
    
    # Résumer les résultats
    edge_type_counts = {"straight": 0, "intrusion": 0, "extrusion": 0, "unknown": 0}
    for result in results:
        for edge_type in result['edge_types']:
            edge_type_counts[edge_type] += 1
    
    print("\nEdge Classification Summary:")
    for edge_type, count in edge_type_counts.items():
        print(f"  - {edge_type}: {count}")
    

if __name__ == "__main__":
    with Timer("Total execution"):
        main()