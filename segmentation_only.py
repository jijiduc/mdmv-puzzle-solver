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
from functools import partial
import tempfile
import pickle
import time
import gc
import sys
import psutil  # Vous devrez peut-être installer ce module: pip install psutil

# Paramètres pour préserver la mémoire et optimiser le parallélisme
BATCH_SIZE = 2  # Taille réduite pour plus de tâches parallèles
CHUNK_SIZE = 10  # Nombre de points à traiter par tâche d'analyse d'arête
USE_RAMDISK = False
DEBUG_MODE = True  # Pour afficher les informations de performance
SET_HIGH_PRIORITY = True  # Augmente la priorité du processus

input_path = "picture/puzzle_24-1/b-2.jpg"
threshold_value = 135  # Threshold value for binary mask
min_contour_area = 150  # Minimum contour area to filter out noise

# Configuration VSCode: Désactiver l'intégration de débogage qui peut ralentir le multiprocessing
os.environ["PYTHONUNBUFFERED"] = "1"  # Désactive la mise en mémoire tampon qui peut ralentir VSCode

# Créer un dossier temporaire pour accélérer les opérations
if USE_RAMDISK:
    temp_dir = "/mnt/ramdisk"  # Ajustez selon votre système
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir, exist_ok=True)
else:
    temp_dir = tempfile.mkdtemp()

# Augmenter la priorité du processus principal
def set_process_priority(high_priority=True):
    try:
        process = psutil.Process(os.getpid())
        if high_priority:
            if sys.platform == 'win32':
                process.nice(psutil.HIGH_PRIORITY_CLASS)
            else:
                process.nice(-10)  # Valeur négative = priorité plus élevée sur Unix
        print(f"Process priority set to {'high' if high_priority else 'normal'}")
    except Exception as e:
        print(f"Could not set process priority: {e}")

# Moniteur de performances
class PerformanceMonitor:
    def __init__(self, name):
        self.name = name
        self.start_time = time.time()
        self.checkpoints = []
        
    def checkpoint(self, label):
        now = time.time()
        self.checkpoints.append((label, now - self.start_time))
        if DEBUG_MODE:
            print(f"{self.name} - {label}: {now - self.start_time:.3f}s")
            sys.stdout.flush()  # Force l'affichage immédiat
        
    def get_total_time(self):
        return time.time() - self.start_time

# Fonction plus efficace pour extraire les bords
def extract_edge_between_corners(corners, corner_idx1, corner_idx2, edge_coords, centroid):
    """Version optimisée d'extraction d'arêtes utilisant un algorithme plus direct et rapide."""
    corner1 = corners[corner_idx1]
    corner2 = corners[corner_idx2]
    
    if len(edge_coords) == 0:
        return []
    
    # Utilisation de numpy pour les calculs vectorisés (plus rapide)
    edge_array = np.array(edge_coords)
    
    # Calculer les angles par rapport au centroïde
    angles = np.arctan2(edge_array[:, 1] - centroid[1], edge_array[:, 0] - centroid[0])
    
    # Calculer les angles des deux coins
    angle1 = math.atan2(corner1[1] - centroid[1], corner1[0] - centroid[0])
    angle2 = math.atan2(corner2[1] - centroid[1], corner2[0] - centroid[0])
    
    # Assurer que angle2 > angle1 pour la sélection d'arc
    if angle2 < angle1:
        angle2 += 2 * np.pi
    
    # Normaliser les angles pour qu'ils soient dans la même plage
    normalized_angles = angles.copy()
    normalized_angles[normalized_angles < angle1] += 2 * np.pi
    
    # Sélectionner les points dans la plage angulaire
    mask = (normalized_angles >= angle1) & (normalized_angles <= angle2)
    filtered_points = edge_array[mask]
    
    # Trier les points par angle
    sorted_indices = np.argsort(normalized_angles[mask])
    sorted_points = filtered_points[sorted_indices]
    
    return sorted_points.tolist()

def classify_edge_chunk(edge_points_chunk, corner1, corner2, centroid):
    """Classifie une partie des points d'une arête - pour parallélisation fine."""
    if not edge_points_chunk or len(edge_points_chunk) < 2:
        return [], []
    
    x1, y1 = corner1
    x2, y2 = corner2
    
    # Calcul vectorisé pour la ligne et la normale
    line_vec = (x2-x1, y2-y1)
    line_length = math.sqrt(line_vec[0]**2 + line_vec[1]**2)
    if line_length < 1:
        return [], []
    
    normal_vec = (-line_vec[1]/line_length, line_vec[0]/line_length)
    
    # Vecteur du centroïde au milieu de la ligne
    mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
    centroid_to_mid = (mid_x - centroid[0], mid_y - centroid[1])
    
    # Direction de la normale
    normal_direction = centroid_to_mid[0]*normal_vec[0] + centroid_to_mid[1]*normal_vec[1]
    outward_normal = normal_vec if normal_direction > 0 else (-normal_vec[0], -normal_vec[1])
    
    # Calcul des déviations pour ce morceau
    deviations = []
    deviation_points = []
    
    # Calcul vectorisé des points d'arête
    for x, y in edge_points_chunk:
        point_vec = (x-x1, y-y1)
        
        if line_length > 0:
            line_dot = (point_vec[0]*line_vec[0] + point_vec[1]*line_vec[1]) / line_length
            proj_x = x1 + line_dot * line_vec[0] / line_length
            proj_y = y1 + line_dot * line_vec[1] / line_length
            
            dev_vec = (x-proj_x, y-proj_y)
            deviation = math.sqrt(dev_vec[0]**2 + dev_vec[1]**2)
            
            sign = 1 if (dev_vec[0]*outward_normal[0] + dev_vec[1]*outward_normal[1]) > 0 else -1
            signed_deviation = sign * deviation
            
            deviations.append(signed_deviation)
            deviation_points.append((proj_x, proj_y, x, y))
    
    return deviations, deviation_points

def classify_edge(edge_points, corner1, corner2, centroid):
    """Classifie une arête complète en utilisant une approche parallélisée."""
    if not edge_points or len(edge_points) < 3:
        return "unknown", 0, []
    
    # Utiliser ThreadPoolExecutor pour cette opération car c'est CPU-bound et petit
    deviations = []
    deviation_points = []
    
    # Diviser les points en morceaux pour traitement parallèle
    chunks = [edge_points[i:i+CHUNK_SIZE] for i in range(0, len(edge_points), CHUNK_SIZE)]
    
    # Traitement en parallèle des morceaux
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for chunk in chunks:
            future = executor.submit(classify_edge_chunk, chunk, corner1, corner2, centroid)
            futures.append(future)
        
        # Collecter les résultats
        for future in concurrent.futures.as_completed(futures):
            chunk_deviations, chunk_points = future.result()
            deviations.extend(chunk_deviations)
            deviation_points.extend(chunk_points)
    
    # Calculer le seuil adaptatif
    x1, y1 = corner1
    x2, y2 = corner2
    line_length = math.sqrt((x2-x1)**2 + (y2-y1)**2)
    straight_threshold = max(5, line_length * 0.05)
    
    # Déterminer le type d'arête
    if deviations:
        mean_deviation = sum(deviations) / len(deviations)
        abs_deviations = [abs(d) for d in deviations]
        max_abs_deviation = max(abs_deviations)
        
        significant_positive = sum(1 for d in deviations if d > straight_threshold)
        significant_negative = sum(1 for d in deviations if d < -straight_threshold)
        portion_significant = (significant_positive + significant_negative) / len(deviations)
        
        if max_abs_deviation < straight_threshold or portion_significant < 0.2:
            edge_type = "straight"
            max_deviation = mean_deviation
        elif abs(mean_deviation) < straight_threshold * 0.5:
            if significant_positive > significant_negative * 2:
                edge_type = "extrusion"
                max_deviation = max([d for d in deviations if d > 0], default=0)
            elif significant_negative > significant_positive * 2:
                edge_type = "intrusion"
                max_deviation = min([d for d in deviations if d < 0], default=0)
            else:
                edge_type = "straight"
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
    
    return edge_type, max_deviation, deviation_points

class CompressedPieceData:
    """Classe optimisée pour stocker les données d'une pièce de façon plus compacte."""
    def __init__(self, corners, centroid, edge_coords, piece_mask, piece_img):
        self.corners = corners
        self.centroid = centroid
        self.edge_coords = edge_coords
        
        # Compresser le masque (binaire)
        _, self.mask_compressed = cv2.imencode('.png', piece_mask)
        self.mask_shape = piece_mask.shape
        
        # Compresser l'image (qualité réduite mais suffisante)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
        _, self.img_compressed = cv2.imencode('.jpg', piece_img, encode_param)
        self.img_shape = piece_img.shape
    
    def get_mask(self):
        mask_data = cv2.imdecode(self.mask_compressed, cv2.IMREAD_GRAYSCALE)
        return mask_data

    def get_img(self):
        img_data = cv2.imdecode(self.img_compressed, cv2.IMREAD_COLOR)
        return img_data

def process_piece_parallel(piece_data_pkl, piece_index, output_dirs_pkl):
    """Traite une pièce avec une meilleure gestion de la mémoire et de la performance."""
    # Support du suivi de performance
    monitor = PerformanceMonitor(f"Piece {piece_index+1}")
    
    # Désérialiser les données
    try:
        piece_data = pickle.loads(piece_data_pkl)
        output_dirs = pickle.loads(output_dirs_pkl)
        edges_dir, edge_types_dir = output_dirs
        
        corners = piece_data.corners
        centroid = piece_data.centroid
        edge_coords = piece_data.edge_coords
        piece_mask = piece_data.get_mask()
        piece_img = piece_data.get_img()
        i = piece_index
        
        monitor.checkpoint("Data unpacked")
        
        # Assurer que les coins sont arrangés dans le sens des aiguilles d'une montre
        cx = sum(p[0] for p in corners) / len(corners)
        cy = sum(p[1] for p in corners) / len(corners)
        corners = sorted(corners, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))
        
        # Extraire et classifier les quatre bords
        edges = []
        edge_types = []
        edge_deviations = []
        deviation_points = []
        
        for j in range(4):
            next_j = (j + 1) % 4
            edge_points = extract_edge_between_corners(corners, j, next_j, edge_coords, centroid)
            edge_type, deviation, dev_points = classify_edge(edge_points, corners[j], corners[next_j], centroid)
            
            edges.append(edge_points)
            edge_types.append(edge_type)
            edge_deviations.append(deviation)
            deviation_points.append(dev_points)
        
        monitor.checkpoint("Edges classified")
        
        # Fichiers temporaires pour réduire la consommation mémoire
        edge_images_temp = []
        
        for j in range(4):
            next_j = (j + 1) % 4
            edge_points = edges[j]
            
            # Calculer la boîte englobante pour les points du bord avec un padding
            if edge_points:
                min_x = max(0, int(min(x for x, y in edge_points)) - 20)
                min_y = max(0, int(min(y for x, y in edge_points)) - 20)
                max_x = min(piece_mask.shape[1], int(max(x for x, y in edge_points)) + 20)
                max_y = min(piece_mask.shape[0], int(max(y for x, y in edge_points)) + 20)
                
                # Inclure les coins dans la boîte
                min_x = min(min_x, int(corners[j][0]) - 10, int(corners[next_j][0]) - 10)
                min_y = min(min_y, int(corners[j][1]) - 10, int(corners[next_j][1]) - 10)
                max_x = max(max_x, int(corners[j][0]) + 10, int(corners[next_j][0]) + 10)
                max_y = max(max_y, int(corners[j][1]) + 10, int(corners[next_j][1]) + 10)
                
                # Dimensions finales
                width = max_x - min_x
                height = max_y - min_y
                
                # Assurer dimensions minimales
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
                for k, corner in enumerate(corners):
                    if k == j or k == next_j:
                        new_x = int(corner[0] - min_x)
                        new_y = int(corner[1] - min_y)
                        if 0 <= new_y < edge_img.shape[0] and 0 <= new_x < edge_img.shape[1]:
                            cv2.circle(edge_img, (new_x, new_y), 5, [255, 0, 0], -1)
                            cv2.putText(edge_img, str(k), (new_x+7, new_y+7), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, [255, 0, 0], 2)
                
                # Sauvegarder d'abord dans un fichier temporaire
                temp_edge_path = os.path.join(temp_dir, f"temp_edge_{i+1}_{j+1}.jpg")
                cv2.imwrite(temp_edge_path, edge_img)
                final_edge_path = os.path.join(edges_dir, f"piece_{i+1}_edge_{j+1}.png")
                edge_images_temp.append((temp_edge_path, final_edge_path))
            else:
                # Image vide
                edge_img = np.zeros((100, 100, 3), dtype=np.uint8)
                temp_edge_path = os.path.join(temp_dir, f"temp_edge_{i+1}_{j+1}.jpg")
                cv2.imwrite(temp_edge_path, edge_img)
                final_edge_path = os.path.join(edges_dir, f"piece_{i+1}_edge_{j+1}.png")
                edge_images_temp.append((temp_edge_path, final_edge_path))
        
        monitor.checkpoint("Edge images created")
        
        # Créer la visualisation dans un fichier temporaire
        fig, axes = plt.subplots(2, 3, figsize=(12, 8), dpi=100)
        fig.suptitle(f"Piece {i+1} Edge Classification", fontsize=16)
        
        for j in range(4):
            row = j // 2
            col = j % 2
            
            # Créer la visualisation
            edge_vis = (piece_img.copy() * 0.5).astype(np.uint8)
            
            # Dessiner les points
            for x, y in edges[j]:
                if 0 <= y < edge_vis.shape[0] and 0 <= x < edge_vis.shape[1]:
                    edge_vis[int(y), int(x)] = [0, 255, 0]
            
            # Dessiner les coins
            for k, corner in enumerate(corners):
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
            
            # Dessiner les lignes de déviation
            for proj_x, proj_y, point_x, point_y in deviation_points[j]:
                axes[row, col].plot([proj_x, point_x], [proj_y, point_y], 'r-', alpha=0.5)
        
        # Montrer la pièce originale
        axes[0, 2].imshow(cv2.cvtColor(piece_img, cv2.COLOR_BGR2RGB))
        axes[0, 2].set_title("Original Piece")
        axes[0, 2].axis('off')
        
        # Ajouter un résumé
        edge_summary = "\n".join([f"Edge {j+1}: {edge_types[j]} ({edge_deviations[j]:.1f}px)" for j in range(4)])
        axes[1, 2].axis('off')
        axes[1, 2].text(0.1, 0.5, edge_summary, fontsize=12, va='center')
        
        plt.tight_layout()
        
        # Sauvegarder dans un temporaire
        temp_vis_path = os.path.join(temp_dir, f"temp_vis_{i+1}.png")
        plt.savefig(temp_vis_path)
        plt.close(fig)
        
        final_vis_path = os.path.join(edge_types_dir, f"piece_{i+1}_edge_types.png")
        
        monitor.checkpoint("Visualization saved")
        
        # Libérer la mémoire
        del piece_img, piece_mask, edges, deviation_points
        gc.collect()
        
        # Retourner les résultats et chemins de fichiers
        return {
            'piece_idx': i,
            'edge_types': edge_types,
            'edge_deviations': edge_deviations,
            'edge_images_temp': edge_images_temp,
            'vis_paths': (temp_vis_path, final_vis_path),
            'processing_time': monitor.get_total_time()
        }
    
    except Exception as e:
        import traceback
        error_msg = f"Error processing piece {piece_index+1}: {e}\n{traceback.format_exc()}"
        print(error_msg)
        return {'piece_idx': piece_index, 'error': error_msg}

def init_worker():
    """Initialisation des processus de travail pour optimiser les performances."""
    # Désactiver la collecte de déchets automatique pour réduire les pauses
    gc.disable()
    
    # Définir une priorité élevée pour ce processus travailleur
    if SET_HIGH_PRIORITY:
        try:
            process = psutil.Process(os.getpid())
            if sys.platform == 'win32':
                process.nice(psutil.ABOVE_NORMAL_PRIORITY_CLASS)
            else:
                process.nice(-5)  # Priorité moins haute que le processus principal
        except:
            pass  # Ignorer les erreurs de priorité

def main():
    """Fonction principale optimisée pour VSCode."""
    # Augmenter la priorité du processus principal
    if SET_HIGH_PRIORITY:
        set_process_priority(True)
    
    main_monitor = PerformanceMonitor("Main process")
    
    # Lire l'image d'entrée
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"Could not read image from {input_path}")
    
    main_monitor.checkpoint("Image loaded")
    
    # Créer les répertoires nécessaires
    debug_dir = "debug"
    os.makedirs(debug_dir, exist_ok=True)
    
    masks_dir = os.path.join(debug_dir, "masks")
    pieces_dir = os.path.join(debug_dir, "pieces")
    transforms_dir = os.path.join(debug_dir, "transforms")
    contours_dir = os.path.join(debug_dir, "contours")
    corners_dir = os.path.join(debug_dir, "corners")
    edges_dir = os.path.join(debug_dir, "edges")
    edge_types_dir = os.path.join(debug_dir, "edge_types")
    
    for directory in [masks_dir, pieces_dir, transforms_dir, contours_dir, corners_dir, edges_dir, edge_types_dir]:
        os.makedirs(directory, exist_ok=True)
    
    # Traitement d'image initial
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary_mask = cv2.threshold(gray_img, threshold_value, 255, cv2.THRESH_BINARY)
    binary_mask = np.uint8(binary_mask)
    
    # Opérations morphologiques
    closing_kernel = np.ones((9, 9), np.uint8)
    dilation_kernel = np.ones((3, 3), np.uint8)
    
    closed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, closing_kernel)
    processed_mask = cv2.dilate(closed_mask, dilation_kernel, iterations=1)
    
    # Trouver les contours
    contours, _ = cv2.findContours(processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = [c for c in contours if cv2.contourArea(c) > min_contour_area]
    
    filled_mask = np.zeros_like(processed_mask)
    cv2.drawContours(filled_mask, valid_contours, -1, 255, -1)
    
    masked_img = cv2.bitwise_and(img, img, mask=filled_mask)
    
    # Dessiner les boîtes englobantes
    contour_img = masked_img.copy()
    for contour in valid_contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(contour_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Sauvegarder les masques
    cv2.imwrite(os.path.join(masks_dir, "binary_mask.png"), filled_mask)
    cv2.imwrite(os.path.join(masks_dir, "masked_img.png"), masked_img)
    cv2.imwrite(os.path.join(masks_dir, "contour_img.png"), contour_img)
    
    main_monitor.checkpoint("Masks processed")
    
    # Extraire chaque pièce
    piece_images = []
    piece_masks = []
    distance_transforms = []
    padding = 5
    
    for i, contour in enumerate(valid_contours):
        x, y, w, h = cv2.boundingRect(contour)
        
        x1, y1 = max(0, x-padding), max(0, y-padding)
        x2, y2 = min(img.shape[1], x+w+padding), min(img.shape[0], y+h+padding)
        
        piece_img = img[y1:y2, x1:x2].copy()
        piece_mask = filled_mask[y1:y2, x1:x2].copy()
        
        masked_piece = cv2.bitwise_and(piece_img, piece_img, mask=piece_mask)
        
        piece_path = os.path.join(pieces_dir, f"piece_{i+1}.png")
        cv2.imwrite(piece_path, masked_piece)
        
        dist_transform = cv2.distanceTransform(piece_mask, cv2.DIST_L2, 5)
        cv2.normalize(dist_transform, dist_transform, 0, 1.0, cv2.NORM_MINMAX)
        
        dt_path = os.path.join(transforms_dir, f"distance_transform_{i+1}.png")
        plt.imsave(dt_path, dist_transform, cmap='gray')
        
        piece_images.append(masked_piece)
        piece_masks.append(piece_mask)
        distance_transforms.append(dist_transform)
    
    print(f"Processed {len(piece_images)} puzzle pieces")
    main_monitor.checkpoint("Pieces extracted")
    
    # Détection des coins
    piece_corners = []
    for i, (piece_img, piece_mask) in enumerate(zip(piece_images, piece_masks)):
        if len(piece_img.shape) == 3:
            piece_gray = cv2.cvtColor(piece_img, cv2.COLOR_BGR2GRAY)
        else:
            piece_gray = piece_img.copy()
        
        edges = cv2.Canny(piece_mask, 50, 150)
        
        edge_overlay = piece_img.copy()
        edge_overlay[edges > 0] = [0, 255, 0]
        
        contour_path = os.path.join(contours_dir, f"contour_piece_{i+1}.png")
        cv2.imwrite(contour_path, edge_overlay)
        
        # Trouver les points de bord
        edge_points = np.where(edges > 0)
        y_edge, x_edge = edge_points[0], edge_points[1]
        edge_coordinates = np.column_stack((x_edge, y_edge))
        
        # Centroïde
        moments = cv2.moments(piece_mask)
        if moments["m00"] != 0:
            centroid_x = int(moments["m10"] / moments["m00"])
            centroid_y = int(moments["m01"] / moments["m00"])
        else:
            centroid_x = piece_mask.shape[1] // 2
            centroid_y = piece_mask.shape[0] // 2
        
        # Calculer les vecteurs
        distances = []
        angles = []
        coords = []
        
        for x, y in edge_coordinates:
            dist = euclidean((centroid_x, centroid_y), (x, y))
            angle = math.atan2(y - centroid_y, x - centroid_x)
            distances.append(dist)
            angles.append(angle)
            coords.append((x, y))
        
        # Trier par angle
        sorted_triples = sorted(zip(angles, distances, coords))
        sorted_angles, sorted_distances, sorted_coords = zip(*sorted_triples)
        
        sorted_angles = np.array(sorted_angles)
        sorted_distances = np.array(sorted_distances)
        sorted_coords = np.array(sorted_coords)
        
        # Appliquer un filtre
        window_length = min(51, len(sorted_distances) // 5 * 2 + 1)
        if window_length >= 3:
            sorted_distances_smooth = savgol_filter(sorted_distances, window_length, 3)
        else:
            sorted_distances_smooth = sorted_distances
        
        # Trouver les pics
        peaks, properties = find_peaks(
            sorted_distances_smooth, 
            prominence=5,
            distance=len(sorted_distances_smooth)/15
        )
        
        if len(peaks) < 6:
            peaks, properties = find_peaks(
                sorted_distances_smooth, 
                prominence=3,
                distance=len(sorted_distances_smooth)/20
            )
        
        if len(peaks) < 6:
            highest_dist_indices = np.argsort(sorted_distances_smooth)[-10:]
            peaks = np.unique(np.concatenate([peaks, highest_dist_indices]))
        
        if len(peaks) >= 6:
            peak_coords = [sorted_coords[p] for p in peaks]
            
            def quad_area(points):
                x = [p[0] for p in points]
                y = [p[1] for p in points]
                return 0.5 * abs(sum(x[i] * y[i+1] - x[i+1] * y[i] 
                               for i in range(len(points)-1)) + 
                           x[-1] * y[0] - x[0] * y[-1])
            
            def rectangle_score(points):
                area = quad_area(points)
                
                angles_deg = []
                for j in range(4):
                    p1 = points[j]
                    p2 = points[(j+1) % 4]
                    p3 = points[(j+2) % 4]
                    
                    v1 = (p2[0] - p1[0], p2[1] - p1[1])
                    v2 = (p3[0] - p2[0], p3[1] - p2[1])
                    
                    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
                    mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
                    mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
                    
                    if mag1 * mag2 == 0:
                        angle_deg = 0
                    else:
                        cos_angle = dot_product / (mag1 * mag2)
                        cos_angle = max(-1, min(1, cos_angle))
                        angle_deg = math.degrees(math.acos(cos_angle))
                    
                    angles_deg.append(angle_deg)
                
                angle_penalty = sum(abs(angle - 90) for angle in angles_deg)
                
                perimeter = sum(math.sqrt((points[j][0] - points[(j+1)%4][0])**2 + 
                                        (points[j][1] - points[(j+1)%4][1])**2)
                               for j in range(4))
                
                compactness = area / (perimeter**2)
                
                return area * compactness * (1000 / (angle_penalty + 1))
            
            best_score = -1
            best_corners = None
            
            for corner_comb in combinations(range(len(peak_coords)), 4):
                points = [peak_coords[idx] for idx in corner_comb]
                
                cx = sum(p[0] for p in points) / 4
                cy = sum(p[1] for p in points) / 4
                points.sort(key=lambda p: math.atan2(p[1] - cy, p[0] - cx))
                
                score = rectangle_score(points)
                if score > best_score:
                    best_score = score
                    best_corners = points
            
            if best_corners:
                corner_points = best_corners
            else:
                corner_points = [sorted_coords[p] for p in peaks[:4]]
        else:
            corner_points = [sorted_coords[p] for p in peaks[:min(4, len(peaks))]]
        
        # Visualiser les coins
        corner_img = piece_img.copy()
        
        for j, (x, y) in enumerate(corner_points):
            cv2.circle(corner_img, (int(x), int(y)), 5, (255, 0, 0), -1)
            cv2.putText(corner_img, str(j), (int(x)+5, int(y)+5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.circle(corner_img, (centroid_x, centroid_y), 3, (0, 0, 255), -1)
        
        corner_path = os.path.join(corners_dir, f"corner_piece_{i+1}.png")
        cv2.imwrite(corner_path, corner_img)
        
        # Stocker de façon compressée
        compressed_data = CompressedPieceData(
            corners=corner_points,
            centroid=(centroid_x, centroid_y),
            edge_coords=edge_coordinates,
            piece_mask=piece_mask,
            piece_img=piece_img
        )
        
        piece_corners.append(compressed_data)
    
    main_monitor.checkpoint("Corners detected")
    
    # EXTRACTION ET CARACTÉRISATION DES BORDS
    print("Starting edge extraction and characterization...")
    
    # Compter les cœurs disponibles
    num_cores = multiprocessing.cpu_count()
    print(f"Detected {num_cores} CPU cores")
    
    # Utiliser (max - 1) cœurs comme demandé
    num_cores_to_use = max(1, num_cores - 1)
    print(f"Using {num_cores_to_use} cores for parallel processing")
    
    # Sérialiser les arguments communs
    output_dirs_pkl = pickle.dumps((edges_dir, edge_types_dir))
    
    # Diviser en lots plus petits pour plus de parallélisme
    piece_batches = [piece_corners[i:i+BATCH_SIZE] for i in range(0, len(piece_corners), BATCH_SIZE)]
    
    # Structure de coordonnateur multiprocess adaptée à VSCode
    all_results = []
    
    # Activer le mode multiprocessing avec protection
    if __name__ == "__main__" or True:  # La deuxième condition permet de fonctionner aussi sans protection
        for batch_idx, batch in enumerate(piece_batches):
            print(f"Processing batch {batch_idx+1}/{len(piece_batches)} ({len(batch)} pieces)...")
            
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=num_cores_to_use,
                initializer=init_worker
            ) as executor:
                # Traiter les pièces en parallèle
                futures = []
                for i, piece_data in enumerate(batch):
                    global_idx = batch_idx * BATCH_SIZE + i
                    piece_data_pkl = pickle.dumps(piece_data)
                    
                    # Soumettre le travail
                    future = executor.submit(
                        process_piece_parallel,
                        piece_data_pkl, 
                        global_idx, 
                        output_dirs_pkl
                    )
                    futures.append(future)
                
                # Collecter les résultats et écrire les fichiers
                batch_results = []
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result()
                        if 'error' in result:
                            print(f"Error with piece {result['piece_idx']+1}: {result['error']}")
                            continue
                            
                        batch_results.append(result)
                        print(f"Completed piece {result['piece_idx']+1} in {result['processing_time']:.2f}s")
                        
                        # Copier les fichiers
                        for temp_path, final_path in result['edge_images_temp']:
                            if os.path.exists(temp_path):
                                os.replace(temp_path, final_path)
                        
                        # Copier la visualisation
                        temp_vis, final_vis = result['vis_paths']
                        if os.path.exists(temp_vis):
                            os.replace(temp_vis, final_vis)
                    
                    except Exception as e:
                        print(f"Error processing result: {e}")
                
                all_results.extend(batch_results)
            
            # Forcer une collecte des déchets entre les lots
            gc.collect()
    
    main_monitor.checkpoint("All pieces processed")
    
    # Résumer les résultats
    edge_type_counts = {"straight": 0, "intrusion": 0, "extrusion": 0, "unknown": 0}
    for result in all_results:
        for edge_type in result['edge_types']:
            edge_type_counts[edge_type] += 1
    
    print("\nEdge Classification Summary:")
    for edge_type, count in edge_type_counts.items():
        print(f"  - {edge_type}: {count}")
    
    print("Edge extraction and characterization completed.")
    print(f"Total processing time: {main_monitor.get_total_time():.2f}s")
    print(f"All debug images saved to subdirectories in {debug_dir}/")
    
    # Nettoyer le répertoire temporaire
    if not USE_RAMDISK and os.path.exists(temp_dir):
        import shutil
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"Warning: Could not clean up temporary directory {temp_dir}: {e}")

# Protection pour le multiprocessing sous Windows
if __name__ == "__main__":
    main()