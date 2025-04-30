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
    
    for j in range(4):
        next_j = (j + 1) % 4
        edge_points = extract_edge_between_corners(corner_points, j, next_j, edge_coordinates, centroid)
        
        if len(edge_points) > 0:
            edge_type, deviation = classify_edge(edge_points, corner_points[j], corner_points[next_j], centroid)
        else:
            edge_type, deviation = "unknown", 0
        
        edges.append(edge_points)
        edge_types.append(edge_type)
        edge_deviations.append(deviation)
        
        # Créer image pour chaque bord
        if not DISABLE_DEBUG_FILES:
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
        'edge_deviations': edge_deviations
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
    
    print("Processing completed!")
    print(f"All output saved to subdirectories in debug/")

if __name__ == "__main__":
    with Timer("Total execution"):
        main()