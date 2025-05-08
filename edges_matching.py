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
    debug_dir = "debug"
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
        
def extract_edge_color_features(piece_img, edge_points, corner1, corner2, edge_index):
    """
    Extract enhanced color features for an edge and return visualization data.
    Includes spatial awareness, gradient features, and higher resolution histograms.
    """
    if len(edge_points) == 0:
        return None, None
    
    # Create a mask for sampling colors along the edge with a small margin
    mask = np.zeros(piece_img.shape[:2], dtype=np.uint8)
    
    # Draw the edge points with a small buffer to capture edge colors
    for x, y in edge_points:
        if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
            cv2.circle(mask, (int(x), int(y)), 3, 255, -1)  # 3-pixel radius
    
    # Get the color samples from the masked region
    # Important: Keep the BGR structure for calcHist
    samples_bgr = piece_img[mask == 255]
    
    if len(samples_bgr) == 0:
        return None, None
    
    # Convert to HSV after extracting the samples to preserve the 3-channel structure
    samples_hsv = cv2.cvtColor(samples_bgr.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    
    # Calculate mean and standard deviation
    mean_hsv = np.mean(samples_hsv, axis=0).tolist()
    std_hsv = np.std(samples_hsv, axis=0).tolist()
    
    # IMPROVEMENT 1: Increased histogram resolution (32 bins instead of 16)
    bins = 32
    h_hist = np.histogram(samples_hsv[:, 0], bins=bins, range=(0, 180))[0]
    s_hist = np.histogram(samples_hsv[:, 1], bins=bins, range=(0, 256))[0]
    v_hist = np.histogram(samples_hsv[:, 2], bins=bins, range=(0, 256))[0]
    
    # Normalize the histograms
    h_hist = h_hist.astype(np.float32) / np.sum(h_hist) if np.sum(h_hist) > 0 else h_hist
    s_hist = s_hist.astype(np.float32) / np.sum(s_hist) if np.sum(s_hist) > 0 else s_hist
    v_hist = v_hist.astype(np.float32) / np.sum(v_hist) if np.sum(v_hist) > 0 else v_hist
    
    # IMPROVEMENT 2: Spatial awareness - Divide edge into segments
    # Sort edge points to follow the edge path consistently
    if len(edge_points) > 10:
        # Sort points by their projection onto the line from corner1 to corner2
        line_vec = (corner2[0] - corner1[0], corner2[1] - corner1[1])
        line_len = np.sqrt(line_vec[0]**2 + line_vec[1]**2)
        
        if line_len > 0:
            # Project each point onto the line
            projections = []
            for x, y in edge_points:
                point_vec = (x - corner1[0], y - corner1[1])
                proj = (point_vec[0]*line_vec[0] + point_vec[1]*line_vec[1]) / line_len
                projections.append((proj, (x, y)))
            
            # Sort by projection
            sorted_points = [p[1] for p in sorted(projections)]
            
            # Divide edge into 5 segments
            num_segments = 5
            segment_length = max(1, len(sorted_points) // num_segments)
            
            # Calculate histogram for each segment
            spatial_h_hists = []
            spatial_s_hists = []
            spatial_v_hists = []
            
            for i in range(num_segments):
                start_idx = i * segment_length
                end_idx = min((i + 1) * segment_length, len(sorted_points))
                segment_points = sorted_points[start_idx:end_idx]
                
                if len(segment_points) > 0:
                    segment_mask = np.zeros(piece_img.shape[:2], dtype=np.uint8)
                    for x, y in segment_points:
                        if 0 <= y < segment_mask.shape[0] and 0 <= x < segment_mask.shape[1]:
                            cv2.circle(segment_mask, (int(x), int(y)), 3, 255, -1)
                    
                    segment_samples = piece_img[segment_mask == 255]
                    if len(segment_samples) > 0:
                        segment_hsv = cv2.cvtColor(segment_samples.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
                        
                        segment_h_hist = np.histogram(segment_hsv[:, 0], bins=bins, range=(0, 180))[0]
                        segment_s_hist = np.histogram(segment_hsv[:, 1], bins=bins, range=(0, 256))[0]
                        segment_v_hist = np.histogram(segment_hsv[:, 2], bins=bins, range=(0, 256))[0]
                        
                        # Normalize segment histograms
                        if np.sum(segment_h_hist) > 0:
                            segment_h_hist = segment_h_hist.astype(np.float32) / np.sum(segment_h_hist)
                        if np.sum(segment_s_hist) > 0:
                            segment_s_hist = segment_s_hist.astype(np.float32) / np.sum(segment_s_hist)
                        if np.sum(segment_v_hist) > 0:
                            segment_v_hist = segment_v_hist.astype(np.float32) / np.sum(segment_v_hist)
                        
                        spatial_h_hists.append(segment_h_hist.tolist())
                        spatial_s_hists.append(segment_s_hist.tolist())
                        spatial_v_hists.append(segment_v_hist.tolist())
                    else:
                        # Empty segment
                        spatial_h_hists.append(np.zeros(bins).tolist())
                        spatial_s_hists.append(np.zeros(bins).tolist())
                        spatial_v_hists.append(np.zeros(bins).tolist())
                else:
                    # Empty segment
                    spatial_h_hists.append(np.zeros(bins).tolist())
                    spatial_s_hists.append(np.zeros(bins).tolist())
                    spatial_v_hists.append(np.zeros(bins).tolist())
        else:
            # Fallback if line length is too small
            spatial_h_hists = [h_hist.tolist()] * 5
            spatial_s_hists = [s_hist.tolist()] * 5
            spatial_v_hists = [v_hist.tolist()] * 5
    else:
        # Too few points for meaningful segmentation
        spatial_h_hists = [h_hist.tolist()] * 5
        spatial_s_hists = [s_hist.tolist()] * 5
        spatial_v_hists = [v_hist.tolist()] * 5
    
    # IMPROVEMENT 3: Gradient/Transition features
    transition_features = {}
    if len(edge_points) > 10:
        # Use sorted points if available, otherwise sort again
        if 'sorted_points' not in locals():
            # Sort points by their projection onto the line from corner1 to corner2
            line_vec = (corner2[0] - corner1[0], corner2[1] - corner1[1])
            line_len = np.sqrt(line_vec[0]**2 + line_vec[1]**2)
            
            if line_len > 0:
                # Project each point onto the line
                projections = []
                for x, y in edge_points:
                    point_vec = (x - corner1[0], y - corner1[1])
                    proj = (point_vec[0]*line_vec[0] + point_vec[1]*line_vec[1]) / line_len
                    projections.append((proj, (x, y)))
                
                # Sort by projection
                sorted_points = [p[1] for p in sorted(projections)]
            else:
                sorted_points = edge_points
        
        # Sample colors along the edge path
        edge_colors = []
        for x, y in sorted_points:
            if 0 <= y < piece_img.shape[0] and 0 <= x < piece_img.shape[1]:
                color = piece_img[int(y), int(x)]
                edge_colors.append(color)
        
        if len(edge_colors) > 1:
            edge_colors = np.array(edge_colors)
            
            # Convert to HSV
            edge_hsv = cv2.cvtColor(edge_colors.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
            
            # Calculate gradients (differences between adjacent points)
            h_diffs = np.abs(np.diff(edge_hsv[:, 0]))
            s_diffs = np.abs(np.diff(edge_hsv[:, 1]))
            v_diffs = np.abs(np.diff(edge_hsv[:, 2]))
            
            # Handle hue circular difference (0 and 180 are close)
            h_diffs = np.minimum(h_diffs, 180 - h_diffs)
            
            # Gradient statistics
            transition_features = {
                'mean_h_gradient': float(np.mean(h_diffs)),
                'max_h_gradient': float(np.max(h_diffs)),
                'mean_s_gradient': float(np.mean(s_diffs)),
                'max_s_gradient': float(np.max(s_diffs)),
                'mean_v_gradient': float(np.mean(v_diffs)),
                'max_v_gradient': float(np.max(v_diffs)),
                'gradient_count': int(np.sum(h_diffs > 20) + np.sum(s_diffs > 30) + np.sum(v_diffs > 30))
            }
    
    if not transition_features:
        # Default values if transition features couldn't be calculated
        transition_features = {
            'mean_h_gradient': 0.0,
            'max_h_gradient': 0.0,
            'mean_s_gradient': 0.0,
            'max_s_gradient': 0.0,
            'mean_v_gradient': 0.0,
            'max_v_gradient': 0.0,
            'gradient_count': 0
        }
    
    # Create enhanced feature vector
    color_feature = {
        'h_hist': h_hist.tolist(),
        's_hist': s_hist.tolist(),
        'v_hist': v_hist.tolist(),
        'mean_hsv': mean_hsv,
        'std_hsv': std_hsv,
        'spatial_h_hists': spatial_h_hists,
        'spatial_s_hists': spatial_s_hists,
        'spatial_v_hists': spatial_v_hists,
        'transition_features': transition_features
    }
    
    # Create visualization data to be used later
    edge_img = cv2.bitwise_and(piece_img, piece_img, mask=mask)
    vis_data = {
        'edge_img': edge_img,
        'h_hist': h_hist,
        's_hist': s_hist,
        'v_hist': v_hist,
        'edge_index': edge_index,
        'spatial_h_hists': spatial_h_hists,
        'spatial_s_hists': spatial_s_hists,
        'spatial_v_hists': spatial_v_hists,
        'transition_features': transition_features
    }
    
    return color_feature, vis_data

def create_color_feature_visualization(piece_img, vis_data_list, piece_index, output_path):
    """
    Create an enhanced visualization of color features for all edges of a piece,
    including spatial segments and gradient data.
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
    
    plt.suptitle(f"Piece {piece_index+1} - Enhanced Edge Color Features", fontsize=20)
    
    # Define color names and colors for histograms
    hist_names = ['Hue', 'Saturation', 'Value']
    hist_colors = ['green', 'blue', 'red']
    
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
        
        # 2. Global Histograms - Full edge histograms (higher resolution with 32 bins)
        hist_list = [vis_data['h_hist'], vis_data['s_hist'], vis_data['v_hist']]
        
        for j in range(3):  # Show all three channels
            ax_hist = plt.subplot(gs[row_start, j+1])
            hist = hist_list[j].flatten()
            ax_hist.bar(range(len(hist)), hist, color=hist_colors[j], alpha=0.7)
            ax_hist.set_title(f"{hist_names[j]} Histogram", fontsize=12)
            ax_hist.set_xlim([0, len(hist)-1])
            ax_hist.grid(alpha=0.3)
            
            # Add small ticks at 25% intervals
            for tick in np.linspace(0, len(hist)-1, 5):
                ax_hist.axvline(tick, color='gray', linestyle='--', alpha=0.3)
        
        # 3. Spatial Segments - If available
        if ('spatial_h_hists' in vis_data and len(vis_data['spatial_h_hists']) > 0):
            # Create a subplot for spatial segments visualization
            ax_spatial = plt.subplot(gs[row_start+1, 0:2])
            
            # Determine how many segments we have
            num_segments = len(vis_data['spatial_h_hists'])
            
            # Create a visualization showing color distribution along the edge
            segment_width = 100
            segment_height = 50
            spatial_img = np.ones((segment_height, segment_width * num_segments, 3), dtype=np.uint8) * 255
            
            for seg_idx, seg_h_hist in enumerate(vis_data['spatial_h_hists']):
                # Extract segment histograms
                seg_h = np.array(vis_data['spatial_h_hists'][seg_idx])
                seg_s = np.array(vis_data['spatial_s_hists'][seg_idx])
                seg_v = np.array(vis_data['spatial_v_hists'][seg_idx])
                
                # Calculate dominant colors for this segment
                if np.sum(seg_h) > 0:
                    # Find dominant hue
                    dominant_h_idx = np.argmax(seg_h)
                    dominant_h = int(dominant_h_idx * 180 / len(seg_h))
                    
                    # Find dominant saturation and value
                    dominant_s_idx = np.argmax(seg_s)
                    dominant_s = int(dominant_s_idx * 255 / len(seg_s))
                    
                    dominant_v_idx = np.argmax(seg_v)
                    dominant_v = int(dominant_v_idx * 255 / len(seg_v))
                    
                    # Ensure saturation and value are high enough to show visible colors
                    # This solves the issue of black rectangles
                    dominant_s = max(dominant_s, 150)  # Ensure sufficient saturation
                    dominant_v = max(dominant_v, 200)  # Ensure sufficient brightness
                    
                    # Fix the predominantly red hue issue by ensuring we have better color distribution
                    # Adjust hue based on its index in the histogram to get better color variation
                    # Convert to actual angle in HSV color wheel (0-180 range for OpenCV)
                    bins_count = len(seg_h)  # Number of bins in the histogram
                    dominant_h = int((dominant_h_idx * 180.0) / bins_count)
                    
                    # Create a colored rectangle for this segment
                    color_patch = np.ones((segment_height, segment_width, 3), dtype=np.uint8)
                    color_patch[:, :, 0] = dominant_h
                    color_patch[:, :, 1] = dominant_s
                    color_patch[:, :, 2] = dominant_v
                    
                    # Convert from HSV to BGR for display
                    color_patch = cv2.cvtColor(color_patch, cv2.COLOR_HSV2BGR)
                    
                    # Place into the spatial visualization
                    x_start = seg_idx * segment_width
                    x_end = (seg_idx + 1) * segment_width
                    spatial_img[:, x_start:x_end, :] = color_patch
                    
                    # Add segment number
                    cv2.putText(spatial_img, str(seg_idx+1), 
                               (x_start + 5, segment_height - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            # Display the spatial visualization
            spatial_img_with_borders = spatial_img.copy()
            
            # Add black borders between segments for better visibility
            for seg_idx in range(1, num_segments):
                x_border = seg_idx * segment_width
                cv2.line(spatial_img_with_borders, 
                         (x_border, 0), 
                         (x_border, segment_height), 
                         (0, 0, 0), 2)
            
            # Add a border around the entire image
            cv2.rectangle(spatial_img_with_borders, (0, 0), 
                         (segment_width * num_segments - 1, segment_height - 1), 
                         (0, 0, 0), 2)
                         
            ax_spatial.imshow(cv2.cvtColor(spatial_img_with_borders, cv2.COLOR_BGR2RGB))
            ax_spatial.set_title(f"Edge {edge_index+1} Spatial Color Distribution", fontsize=12)
            ax_spatial.axis('off')
        
        # 4. Gradient/Transition Features - If available
        if 'transition_features' in vis_data:
            tf = vis_data['transition_features']
            ax_gradient = plt.subplot(gs[row_start+1, 2:4])
            
            # Create a text-based summary of gradient features
            gradient_text = f"Color Transition Features:\n\n"
            gradient_text += f"Gradient Count: {tf['gradient_count']}\n"
            gradient_text += f"Mean Gradients (H/S/V): {tf['mean_h_gradient']:.1f} / {tf['mean_s_gradient']:.1f} / {tf['mean_v_gradient']:.1f}\n"
            gradient_text += f"Max Gradients (H/S/V): {tf['max_h_gradient']:.1f} / {tf['max_s_gradient']:.1f} / {tf['max_v_gradient']:.1f}\n"
            
            # Determine transition level
            if tf['gradient_count'] > 5:
                transition_level = "High color variation"
            elif tf['gradient_count'] > 2:
                transition_level = "Moderate color variation"
            else:
                transition_level = "Mostly uniform color"
                
            gradient_text += f"\nCharacterization: {transition_level}"
            
            # Display the gradient information
            ax_gradient.text(0.05, 0.5, gradient_text, 
                           fontsize=12, va='center', transform=ax_gradient.transAxes)
            ax_gradient.axis('off')
    
    if valid_edges == 0:
        plt.figtext(0.5, 0.5, "No color data available", 
                  ha='center', va='center', fontsize=16)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95, hspace=0.4)
    
    # Save figure directly to file instead of converting to OpenCV image
    plt.savefig(output_path, dpi=120)
    plt.close()

# ========= EDGE MATCHING AND PUZZLE ASSEMBLY =========

def calculate_shape_compatibility(edge1_type, edge1_deviation, edge2_type, edge2_deviation):
    """
    Calculate shape compatibility between two edges based on their types.
    
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
    
    if edge1_type == "straight" and edge2_type == "straight":
        return 0.7  # Moderate score for straight-straight
    
    # Default low compatibility
    return 0.1

def calculate_color_compatibility(color_feature1, color_feature2):
    """
    Calculate enhanced color compatibility between two edges based on their color features.
    Includes spatial awareness, gradient features, and higher resolution histograms.
    
    Args:
        color_feature1: Enhanced color features of first edge
        color_feature2: Enhanced color features of second edge
        
    Returns:
        Compatibility score between 0 and 1
    """
    if color_feature1 is None or color_feature2 is None:
        return 0.5  # Default mid-range score if no color data is available
    
    # 1. Global histogram comparison (using full histograms)
    h_corr = cv2.compareHist(np.array(color_feature1['h_hist'], dtype=np.float32), 
                           np.array(color_feature2['h_hist'], dtype=np.float32), 
                           cv2.HISTCMP_CORREL)
    s_corr = cv2.compareHist(np.array(color_feature1['s_hist'], dtype=np.float32), 
                           np.array(color_feature2['s_hist'], dtype=np.float32), 
                           cv2.HISTCMP_CORREL)
    v_corr = cv2.compareHist(np.array(color_feature1['v_hist'], dtype=np.float32), 
                           np.array(color_feature2['v_hist'], dtype=np.float32), 
                           cv2.HISTCMP_CORREL)
    
    # Normalize correlation (-1 to 1) to score (0 to 1)
    h_score = (h_corr + 1) / 2  
    s_score = (s_corr + 1) / 2
    v_score = (v_corr + 1) / 2
    
    # Weighted combination of channel correlations
    # Hue is more important for color matching, then saturation, then value
    global_color_score = 0.5 * h_score + 0.3 * s_score + 0.2 * v_score
    
    # 2. Spatial histogram comparison (comparing corresponding segments)
    spatial_scores = []
    
    # Check if spatial histograms are available in both features
    if ('spatial_h_hists' in color_feature1 and 'spatial_h_hists' in color_feature2 and
        len(color_feature1['spatial_h_hists']) > 0 and len(color_feature2['spatial_h_hists']) > 0):
        
        num_segments = min(len(color_feature1['spatial_h_hists']), len(color_feature2['spatial_h_hists']))
        
        for i in range(num_segments):
            # Compare corresponding segments
            if i < len(color_feature1['spatial_h_hists']) and i < len(color_feature2['spatial_h_hists']):
                segment_h_corr = cv2.compareHist(
                    np.array(color_feature1['spatial_h_hists'][i], dtype=np.float32),
                    np.array(color_feature2['spatial_h_hists'][i], dtype=np.float32),
                    cv2.HISTCMP_CORREL
                )
                
                segment_s_corr = cv2.compareHist(
                    np.array(color_feature1['spatial_s_hists'][i], dtype=np.float32),
                    np.array(color_feature2['spatial_s_hists'][i], dtype=np.float32),
                    cv2.HISTCMP_CORREL
                )
                
                segment_v_corr = cv2.compareHist(
                    np.array(color_feature1['spatial_v_hists'][i], dtype=np.float32),
                    np.array(color_feature2['spatial_v_hists'][i], dtype=np.float32),
                    cv2.HISTCMP_CORREL
                )
                
                # Normalize and combine
                segment_h_score = (segment_h_corr + 1) / 2
                segment_s_score = (segment_s_corr + 1) / 2
                segment_v_score = (segment_v_corr + 1) / 2
                
                segment_score = 0.5 * segment_h_score + 0.3 * segment_s_score + 0.2 * segment_v_score
                spatial_scores.append(segment_score)
        
        # Also compare in reverse order (for opposite edges that should match)
        spatial_scores_reversed = []
        for i in range(num_segments):
            rev_idx = num_segments - 1 - i
            if i < len(color_feature1['spatial_h_hists']) and rev_idx < len(color_feature2['spatial_h_hists']):
                segment_h_corr = cv2.compareHist(
                    np.array(color_feature1['spatial_h_hists'][i], dtype=np.float32),
                    np.array(color_feature2['spatial_h_hists'][rev_idx], dtype=np.float32),
                    cv2.HISTCMP_CORREL
                )
                
                segment_s_corr = cv2.compareHist(
                    np.array(color_feature1['spatial_s_hists'][i], dtype=np.float32),
                    np.array(color_feature2['spatial_s_hists'][rev_idx], dtype=np.float32),
                    cv2.HISTCMP_CORREL
                )
                
                segment_v_corr = cv2.compareHist(
                    np.array(color_feature1['spatial_v_hists'][i], dtype=np.float32),
                    np.array(color_feature2['spatial_v_hists'][rev_idx], dtype=np.float32),
                    cv2.HISTCMP_CORREL
                )
                
                # Normalize and combine
                segment_h_score = (segment_h_corr + 1) / 2
                segment_s_score = (segment_s_corr + 1) / 2
                segment_v_score = (segment_v_corr + 1) / 2
                
                segment_score = 0.5 * segment_h_score + 0.3 * segment_s_score + 0.2 * segment_v_score
                spatial_scores_reversed.append(segment_score)
        
        # Use the better matching direction (normal or reversed)
        spatial_score = max(
            sum(spatial_scores) / max(1, len(spatial_scores)),
            sum(spatial_scores_reversed) / max(1, len(spatial_scores_reversed))
        )
    else:
        # Fall back to global score if spatial data is not available
        spatial_score = global_color_score
    
    # 3. Compare color transition/gradient features
    gradient_score = 0.5  # Default score
    
    if ('transition_features' in color_feature1 and 'transition_features' in color_feature2):
        tf1 = color_feature1['transition_features']
        tf2 = color_feature2['transition_features']
        
        # Compare gradient counts - edges with similar number of color transitions should match
        gradient_count_diff = abs(tf1['gradient_count'] - tf2['gradient_count'])
        gradient_count_score = max(0, 1.0 - gradient_count_diff / 10.0)  # Normalize difference
        
        # Compare mean gradients - edges with similar color transition intensity should match
        h_gradient_diff = abs(tf1['mean_h_gradient'] - tf2['mean_h_gradient']) / 90.0  # Normalize
        s_gradient_diff = abs(tf1['mean_s_gradient'] - tf2['mean_s_gradient']) / 128.0
        v_gradient_diff = abs(tf1['mean_v_gradient'] - tf2['mean_v_gradient']) / 128.0
        
        mean_gradient_score = 1.0 - (0.5 * h_gradient_diff + 0.3 * s_gradient_diff + 0.2 * v_gradient_diff)
        
        # Max gradients comparison - maximum color transitions should be similar
        max_h_gradient_diff = abs(tf1['max_h_gradient'] - tf2['max_h_gradient']) / 90.0
        max_s_gradient_diff = abs(tf1['max_s_gradient'] - tf2['max_s_gradient']) / 128.0
        max_v_gradient_diff = abs(tf1['max_v_gradient'] - tf2['max_v_gradient']) / 128.0
        
        max_gradient_score = 1.0 - (0.5 * max_h_gradient_diff + 0.3 * max_s_gradient_diff + 0.2 * max_v_gradient_diff)
        
        # Combine gradient scores
        gradient_score = 0.4 * gradient_count_score + 0.3 * mean_gradient_score + 0.3 * max_gradient_score
    
    # 4. Compare means of HSV (basic feature)
    mean_diff_h = abs(color_feature1['mean_hsv'][0] - color_feature2['mean_hsv'][0]) / 180.0
    mean_diff_s = abs(color_feature1['mean_hsv'][1] - color_feature2['mean_hsv'][1]) / 255.0
    mean_diff_v = abs(color_feature1['mean_hsv'][2] - color_feature2['mean_hsv'][2]) / 255.0
    
    mean_score = 1.0 - (0.5 * mean_diff_h + 0.3 * mean_diff_s + 0.2 * mean_diff_v)
    
    # 5. Combine all scores with appropriate weights
    # - Global histogram: Overall color distribution
    # - Spatial histogram: Color patterns along the edge
    # - Gradient features: Color transitions 
    # - Mean values: Basic color similarity
    final_score = (
        0.3 * global_color_score +  # Base color distribution
        0.3 * spatial_score +       # Spatial color patterns
        0.25 * gradient_score +     # Color transitions
        0.15 * mean_score           # Basic color similarity
    )
    
    return final_score

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


def match_edges(piece_results):
    """
    Match edges between all puzzle pieces based on shape and color compatibility.
    
    Args:
        piece_results: List of processed piece data with edge information
        
    Returns:
        Dictionary with edge matches and compatibility scores
    """
    num_pieces = len(piece_results)
    matches = []
    
    # Create a progress counter
    total_comparisons = num_pieces * (num_pieces - 1) * 16 // 2  # For each pair of pieces, 16 possible edge combinations
    progress_counter = 0
    progress_interval = max(1, total_comparisons // 20)  # Show progress at 5% intervals
    
    print(f"Starting edge matching (comparing {total_comparisons} potential matches)...")
    
    # For each pair of pieces
    for i in range(num_pieces):
        for j in range(i + 1, num_pieces):  # Only compare each pair once
            piece1 = piece_results[i]
            piece2 = piece_results[j]
            
            # For each combination of edges
            for edge1_idx in range(4):
                for edge2_idx in range(4):
                    # Get edge data
                    edge1_type = piece1['edge_types'][edge1_idx]
                    edge1_deviation = piece1['edge_deviations'][edge1_idx]
                    edge1_colors = piece1['edge_colors'][edge1_idx]
                    
                    edge2_type = piece2['edge_types'][edge2_idx]
                    edge2_deviation = piece2['edge_deviations'][edge2_idx]
                    edge2_colors = piece2['edge_colors'][edge2_idx]
                    
                    # Skip unknown edges
                    if edge1_type == "unknown" or edge2_type == "unknown":
                        progress_counter += 1
                        continue
                    
                    # Calculate compatibility scores
                    shape_score = calculate_shape_compatibility(
                        edge1_type, edge1_deviation, edge2_type, edge2_deviation
                    )
                    
                    color_score = calculate_color_compatibility(edge1_colors, edge2_colors)
                    
                    # Combine scores (shape is more important than color)
                    total_score = 0.7 * shape_score + 0.3 * color_score
                    
                    # Store the match if score is above threshold
                    if total_score > 0.4:  # Lower threshold to allow more matches
                        matches.append({
                            'piece1_idx': piece1['piece_idx'],
                            'piece2_idx': piece2['piece_idx'],
                            'edge1_idx': edge1_idx,
                            'edge2_idx': edge2_idx,
                            'total_score': total_score,
                            'shape_score': shape_score,
                            'color_score': color_score
                        })
                    
                    # Update progress
                    progress_counter += 1
                    if progress_counter % progress_interval == 0:
                        print(f"Matching progress: {progress_counter}/{total_comparisons} ({progress_counter*100//total_comparisons}%)")
    
    # Sort matches by descending score
    matches.sort(key=lambda x: x['total_score'], reverse=True)
    
    print(f"Found {len(matches)} potential edge matches.")
    return matches

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
    
    def start_assembly(self):
        """Start the assembly by placing the first piece."""
        if not self.edge_matches or not self.piece_results:
            print("No pieces or matches to assemble.")
            return False
        
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
        
        # Place seed piece at (0, 0)
        self.place_piece(seed_piece_idx, 0, 0)
        
        # Add neighboring pieces to frontier
        self.update_frontier()
        
        return True
    
    def place_piece(self, piece_idx, row, col):
        """Place a puzzle piece at the specified grid position."""
        if piece_idx in self.placed_pieces:
            return False
        
        if (row, col) in self.grid:
            return False
        
        self.grid[(row, col)] = piece_idx
        self.placed_positions[piece_idx] = (row, col)
        self.placed_pieces.add(piece_idx)
        
        # Update grid bounds
        self.min_row = min(self.min_row, row)
        self.max_row = max(self.max_row, row)
        self.min_col = min(self.min_col, col)
        self.max_col = max(self.max_col, col)
        
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
        
        Args:
            piece_idx: Index of the piece to place
            
        Returns:
            Tuple of (row, col, rotation) for the piece, or None if no valid placement
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
                
            # Check if edge orientation matches (this would require rotation if not)
            # For now, we'll consider compatible edges even if they need rotation
            # This is to expand our matching possibilities
            compatible_edges = True
            # Commented out to allow more matches:
            # if new_edge != required_edge:
            #     continue
                
            # Update best match if this is better
            if score > best_score:
                best_score = score
                best_position = (new_row, new_col)
                best_edge_match = (placed_idx, placed_edge, new_edge)
        
        return best_position, best_edge_match
    
    def assemble_next_piece(self):
        """
        Place the next piece with the highest score.
        
        Returns:
            True if a piece was placed, False otherwise
        """
        if not self.frontier:
            return False
            
        best_piece = None
        best_position = None
        best_score = -1
        best_edge_match = None
        
        # Evaluate each piece in the frontier
        for piece_idx in self.frontier:
            position, edge_match = self.determine_piece_position(piece_idx)
            if position:
                # Find the corresponding match
                for match in self.edge_matches:
                    if ((match['piece1_idx'] == piece_idx and match['piece2_idx'] == edge_match[0]) or
                        (match['piece2_idx'] == piece_idx and match['piece1_idx'] == edge_match[0])) and \
                       ((match['edge1_idx'] == edge_match[2] and match['edge2_idx'] == edge_match[1]) or
                        (match['edge2_idx'] == edge_match[2] and match['edge1_idx'] == edge_match[1])):
                        score = match['total_score']
                        if score > best_score:
                            best_score = score
                            best_piece = piece_idx
                            best_position = position
                            best_edge_match = edge_match
                        break
        
        # If found a piece to place
        if best_piece and best_position:
            row, col = best_position
            self.place_piece(best_piece, row, col)
            
            # Mark edges as used
            placed_idx, placed_edge, new_edge = best_edge_match
            self.used_edges.add((placed_idx, placed_edge))
            self.used_edges.add((best_piece, new_edge))
            
            # Update frontier
            self.frontier.remove(best_piece)
            self.update_frontier()
            return True
        
        return False
    
    def assemble_puzzle(self):
        """
        Assemble the complete puzzle.
        
        Returns:
            Dictionary with assembly results
        """
        print("Starting puzzle assembly...")
        
        # Place the first piece
        if not self.start_assembly():
            print("Failed to start assembly.")
            return {"success": False}
        
        # Keep track of pieces placed
        pieces_placed = 1
        iterations = 0
        max_iterations = self.num_pieces * 2  # Avoid infinite loops
        
        # Assemble pieces until no more can be placed or all are placed
        while pieces_placed < self.num_pieces and iterations < max_iterations:
            success = self.assemble_next_piece()
            if success:
                pieces_placed += 1
                print(f"Placed piece {pieces_placed}/{self.num_pieces}")
            else:
                print(f"Could not place more pieces after {pieces_placed}/{self.num_pieces}")
                break
            iterations += 1
        
        # Calculate grid dimensions
        grid_height = self.max_row - self.min_row + 1
        grid_width = self.max_col - self.min_col + 1
        
        print(f"Assembly complete. Placed {pieces_placed}/{self.num_pieces} pieces.")
        print(f"Puzzle dimensions: {grid_width}x{grid_height}")
        
        return {
            "success": pieces_placed > 0,
            "pieces_placed": pieces_placed,
            "total_pieces": self.num_pieces,
            "grid": self.grid,
            "placed_positions": self.placed_positions,
            "dimensions": (grid_width, grid_height),
            "bounds": (self.min_row, self.min_col, self.max_row, self.max_col)
        }
    
    def visualize_assembly(self, output_path, piece_images=None):
        """
        Create a visualization of the assembled puzzle.
        
        Args:
            output_path: Path to save the visualization
            piece_images: Dictionary of piece images (optional)
            
        Returns:
            Assembly visualization image
        """
        if not self.placed_pieces:
            print("No pieces placed to visualize.")
            return None
        
        # Calculate grid dimensions and cell size
        grid_height = self.max_row - self.min_row + 1
        grid_width = self.max_col - self.min_col + 1
        
        # Default cell size if no images provided
        cell_size = 100
        
        # Use piece images if provided
        if piece_images and len(piece_images) > 0:
            # Find average piece size
            avg_height = sum(img.shape[0] for img in piece_images.values()) / len(piece_images)
            avg_width = sum(img.shape[1] for img in piece_images.values()) / len(piece_images)
            cell_size = max(int(avg_height), int(avg_width), 100)
        
        # Create canvas for the assembly visualization
        canvas_height = grid_height * cell_size
        canvas_width = grid_width * cell_size
        canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255
        
        # Draw grid lines
        for i in range(grid_height + 1):
            y = i * cell_size
            cv2.line(canvas, (0, y), (canvas_width, y), (200, 200, 200), 1)
        for i in range(grid_width + 1):
            x = i * cell_size
            cv2.line(canvas, (x, 0), (x, canvas_height), (200, 200, 200), 1)
        
        # Place pieces on canvas
        for (row, col), piece_idx in self.grid.items():
            # Convert grid position to canvas coordinates
            canvas_row = row - self.min_row
            canvas_col = col - self.min_col
            
            # Draw piece representation
            y1 = canvas_row * cell_size
            y2 = (canvas_row + 1) * cell_size
            x1 = canvas_col * cell_size
            x2 = (canvas_col + 1) * cell_size
            
            # Draw piece on canvas
            if piece_images and piece_idx in piece_images:
                # Resize piece image to fit cell
                piece_img = piece_images[piece_idx]
                resized_img = cv2.resize(piece_img, (cell_size, cell_size))
                canvas[y1:y2, x1:x2] = resized_img
            else:
                # Draw colored rectangle with piece index
                color = ((piece_idx * 40) % 256, (piece_idx * 70) % 256, (piece_idx * 110) % 256)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), color, -1)
                cv2.putText(canvas, f"{piece_idx}", (x1 + cell_size//4, y1 + cell_size//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Add labels for rows and columns
        for i in range(grid_height):
            row_label = str(i + self.min_row)
            cv2.putText(canvas, row_label, (5, i * cell_size + cell_size//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        for i in range(grid_width):
            col_label = str(i + self.min_col)
            cv2.putText(canvas, col_label, (i * cell_size + cell_size//2, 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        # Save visualization
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, canvas)
        
        print(f"Assembly visualization saved to {output_path}")
        return canvas

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
    
    # Match edges between pieces
    with Timer("Edge matching"):
        edge_matches = match_edges(results)
        
        # Save top matches information
        os.makedirs(dirs['edges'], exist_ok=True)
        top_matches_path = os.path.join(dirs['edges'], "top_matches.txt")
        with open(top_matches_path, 'w') as f:
            f.write("Top Edge Matches:\n")
            for i, match in enumerate(edge_matches[:20]):  # Save top 20 matches
                f.write(f"{i+1}. Piece {match['piece1_idx']+1} Edge {match['edge1_idx']+1} ⟷ "
                        f"Piece {match['piece2_idx']+1} Edge {match['edge2_idx']+1} "
                        f"(Score: {match['total_score']:.3f}, Shape: {match['shape_score']:.3f}, "
                        f"Color: {match['color_score']:.3f})\n")
        
        # Create matching visualization directory
        matching_dir = os.path.join("debug", "matching")
        os.makedirs(matching_dir, exist_ok=True)
        
        # Load piece images for visualization
        piece_images = {}
        for result in results:
            piece_idx = result['piece_idx']
            piece_path = os.path.join(dirs['pieces'], f"piece_{piece_idx+1}.png")
            if os.path.exists(piece_path):
                piece_img = cv2.imread(piece_path)
                if piece_img is not None:
                    piece_images[piece_idx] = piece_img
        
        # Create visualizations for top matches
        print("Creating match visualizations...")
        for i, match in enumerate(edge_matches[:30]):  # Visualize top 30 matches
            # Add rank to the match info for visualization
            match_with_rank = match.copy()
            match_with_rank['rank'] = i + 1
            
            match_vis_path = os.path.join(matching_dir, f"match_{i+1:02d}_p{match['piece1_idx']+1}e{match['edge1_idx']+1}_p{match['piece2_idx']+1}e{match['edge2_idx']+1}.png")
            create_edge_match_visualization(match_with_rank, results, piece_images, match_vis_path)
            
        print(f"Created visualization for {min(30, len(edge_matches))} top matches in {matching_dir}/")
    
    # Assemble the puzzle
    with Timer("Puzzle assembly"):
        assembler = PuzzleAssembler(results, edge_matches)
        assembly_result = assembler.assemble_puzzle()
        
        # Create assembly visualization directory
        assembly_dir = os.path.join(dirs['edge_types'], "assembly")
        os.makedirs(assembly_dir, exist_ok=True)
        
        # Load piece images for visualization (if available)
        piece_images = {}
        for result in results:
            piece_idx = result['piece_idx']
            piece_path = os.path.join(dirs['pieces'], f"piece_{piece_idx+1}.png")
            if os.path.exists(piece_path):
                piece_img = cv2.imread(piece_path)
                if piece_img is not None:
                    piece_images[piece_idx] = piece_img
        
        # Visualize the assembly
        viz_path = os.path.join(assembly_dir, "puzzle_assembly.png")
        assembler.visualize_assembly(viz_path, piece_images)
        
        # Save assembly data
        assembly_data_path = os.path.join(assembly_dir, "assembly_data.txt")
        with open(assembly_data_path, 'w') as f:
            f.write(f"Puzzle Assembly Results:\n")
            f.write(f"Pieces placed: {assembly_result['pieces_placed']}/{assembly_result['total_pieces']}\n")
            f.write(f"Puzzle dimensions: {assembly_result['dimensions'][0]}x{assembly_result['dimensions'][1]}\n\n")
            
            f.write("Piece Placements:\n")
            for piece_idx, (row, col) in assembler.placed_positions.items():
                f.write(f"Piece {piece_idx+1}: Position ({row}, {col})\n")
    
    print("Processing completed!")
    print(f"All output saved to subdirectories in debug/")

if __name__ == "__main__":
    with Timer("Total execution"):
        main()