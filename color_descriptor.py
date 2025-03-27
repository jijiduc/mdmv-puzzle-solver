from image_processing import save_image, find_contour
import scipy 
import numpy as np
from scipy.spatial.distance import cdist
import cv2
import matplotlib.pyplot as plt

def extract_contour_color_descriptor(image, contour, sample_count=100):
    """
    Extrait un descripteur de couleur le long du contour d'une pièce de puzzle.
    
    Args:
        image: Image source (BGR)
        contour: Contour de la pièce
        sample_count: Nombre de points d'échantillonnage le long du contour
    
    Returns:
        Un tableau numpy contenant les valeurs BGR pour chaque point échantillonné
    """
    
    
    # Vérifier que le contour est au format attendu
    if not isinstance(contour, np.ndarray):
        return None
    
    # Vérifier la forme et convertir si nécessaire
    if len(contour.shape) == 2 and contour.shape[1] == 2:
        # Format (n, 2) - convertir à (n, 1, 2)
        contour = contour.reshape((-1, 1, 2)).astype(np.int32)
    elif len(contour.shape) != 3 or contour.shape[1] != 1 or contour.shape[2] != 2:
        print(f"Format de contour invalide: {contour.shape}")
        return None
        
    # S'assurer que le contour est de type int32
    contour = contour.astype(np.int32)
    
    # Vérifier que le contour a suffisamment de points
    if len(contour) < 5:
        return None
        
    # Convertir en HSV pour une meilleure représentation des couleurs
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    try:
        # Rééchantillonnage du contour pour avoir un nombre fixe de points
        # Cela permet de comparer des contours de différentes tailles
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx_contour = cv2.approxPolyDP(contour, epsilon, True)
    except cv2.error as e:
        print(f"Erreur OpenCV: {e}")
        print(f"Shape du contour: {contour.shape}, type: {contour.dtype}")
        print(f"Premiers points: {contour[:5]}")
        return None
    
    # S'assurer d'avoir un nombre minimum de points
    if len(approx_contour) < 5:
        return None
    
    # Créer un contour avec un nombre fixe de points équidistants
    perimeter = cv2.arcLength(approx_contour, True)
    step = perimeter / sample_count
    
    resampled_points = []
    dist_traveled = 0
    
    for i in range(len(approx_contour)):
        p1 = approx_contour[i][0]
        p2 = approx_contour[(i + 1) % len(approx_contour)][0]
        
        segment_length = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        segment_dir = np.array([p2[0] - p1[0], p2[1] - p1[1]]) / (segment_length + 1e-10)
        
        while dist_traveled < perimeter and len(resampled_points) < sample_count:
            segment_pos = dist_traveled % perimeter
            if segment_pos <= segment_length:
                # Point à l'intérieur du segment actuel
                point = p1 + segment_dir * segment_pos
                resampled_points.append(point.astype(int))
                dist_traveled += step
            else:
                # Passer au segment suivant
                break
                
    # S'assurer d'avoir exactement sample_count points
    while len(resampled_points) < sample_count:
        resampled_points.append(resampled_points[-1])
    
    # Extraire les couleurs à chaque point échantillonné
    colors = []
    
    # Taille du voisinage pour chaque point (moyenne locale)
    neighborhood = 3
    
    for point in resampled_points:
        x, y = point
        
        # Vérifier que le point est dans l'image
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
            # Extraire une petite région autour du point
            roi_x_start = max(0, x - neighborhood)
            roi_x_end = min(image.shape[1], x + neighborhood + 1)
            roi_y_start = max(0, y - neighborhood)
            roi_y_end = min(image.shape[0], y + neighborhood + 1)
            
            roi = hsv_image[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
            
            # Calculer la couleur moyenne dans la région
            if roi.size > 0:
                color = np.mean(roi, axis=(0, 1)).astype(np.uint8)
                colors.append(color)
            else:
                # Utiliser la couleur du pixel si la région est vide
                colors.append(hsv_image[y, x])
        else:
            # Point hors de l'image, utiliser une valeur par défaut
            colors.append(np.array([0, 0, 0], dtype=np.uint8))
    
    return np.array(colors)

def compare_color_descriptors(descriptor1, descriptor2):
    """
    Compare deux descripteurs de couleur et retourne une mesure de similarité.
    
    Args:
        descriptor1: Premier descripteur de couleur
        descriptor2: Deuxième descripteur de couleur
    
    Returns:
        Score de similarité entre 0 et 1 (1 = identique)
    """
    
    if descriptor1 is None or descriptor2 is None:
        return 0.0
    
    # Normaliser pour se concentrer sur la teinte et la saturation (HSV)
    # H est cyclique (0-180 en OpenCV), S et V sont 0-255
    descriptor1_norm = descriptor1.copy().astype(float)
    descriptor2_norm = descriptor2.copy().astype(float)
    
    # Normaliser H, S, V
    descriptor1_norm[:, 0] /= 180.0  # H
    descriptor1_norm[:, 1] /= 255.0  # S
    descriptor1_norm[:, 2] /= 255.0  # V
    
    descriptor2_norm[:, 0] /= 180.0  # H
    descriptor2_norm[:, 1] /= 255.0  # S
    descriptor2_norm[:, 2] /= 255.0  # V
    
    # Comparer en tenant compte de la cyclicité de H
    # et en donnant des poids différents à H, S et V
    weights = np.array([2.0, 1.5, 1.0])  # H est plus important, puis S, puis V
    
    # Distance entre descripteurs (en tenant compte des poids)
    distances = []
    
    # Calculer distance pour chaque orientation possible (décalage circulaire)
    min_dist = float('inf')
    best_shift = 0
    
    for shift in range(len(descriptor1)):
        # Décalage circulaire du descripteur2
        shifted = np.roll(descriptor2_norm, shift, axis=0)
        
        # Calculer distance en tenant compte de la cyclicité de H
        h_dist = np.minimum(
            np.abs(descriptor1_norm[:, 0] - shifted[:, 0]),
            1.0 - np.abs(descriptor1_norm[:, 0] - shifted[:, 0])
        )
        
        # Distances pour S et V (pas cycliques)
        s_dist = np.abs(descriptor1_norm[:, 1] - shifted[:, 1])
        v_dist = np.abs(descriptor1_norm[:, 2] - shifted[:, 2])
        
        # Distance pondérée
        weighted_dist = np.mean(
            weights[0] * h_dist + weights[1] * s_dist + weights[2] * v_dist
        )
        
        if weighted_dist < min_dist:
            min_dist = weighted_dist
            best_shift = shift
    
    # Convertir la distance en similarité
    similarity = 1.0 - min_dist / (weights.sum() / len(weights))
    
    return similarity, best_shift

def visualize_color_descriptor(descriptor, size=(500, 100)):
    """
    Visualise un descripteur de couleur sous forme d'image.
    
    Args:
        descriptor: Descripteur de couleur (format HSV)
        size: Taille de l'image de sortie (largeur, hauteur)
    
    Returns:
        Image représentant le descripteur de couleur
    """
    if descriptor is None:
        return np.zeros((size[1], size[0], 3), dtype=np.uint8)
    
    # Créer une image vide
    width, height = size
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Calculer la largeur de chaque segment
    segment_width = width / len(descriptor)
    
    # Dessiner chaque couleur
    for i, color in enumerate(descriptor):
        x_start = int(i * segment_width)
        x_end = int((i + 1) * segment_width)
        
        # Dessiner un rectangle de la couleur correspondante
        cv2.rectangle(
            image,
            (x_start, 0),
            (x_end, height),
            color.tolist(),
            -1
        )
    
    # Convertir de HSV à BGR pour l'affichage
    image_bgr = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    
    return image_bgr

def find_matching_pieces(image, pieces_or_contours, threshold=0.7):
    """
    Trouve les pièces correspondantes en fonction de la similarité de couleur.
    
    Args:
        image: Image source
        pieces_or_contours: Liste des pièces ou des contours des pièces
        threshold: Seuil de similarité pour considérer une correspondance
    
    Returns:
        Liste des paires de pièces correspondantes avec leur score
    """
    import numpy as np
    import cv2
    
    # Détecter si on a reçu des pièces ou des contours
    is_contour_list = True
    if len(pieces_or_contours) > 0:
        first_item = pieces_or_contours[0]
        # Si c'est une image (pièce) et non un contour
        if isinstance(first_item, np.ndarray) and len(first_item.shape) == 3 and first_item.shape[2] == 3:
            is_contour_list = False
    
    # Extraire les descripteurs pour toutes les pièces
    descriptors = []
    for i, item in enumerate(pieces_or_contours):
        if is_contour_list:
            # Si c'est déjà un contour
            contour = item
        else:
            # Si c'est une pièce (image), on trouve son contour
            piece_contours = find_contour(item)
            if not piece_contours or len(piece_contours) == 0:
                continue
            # Prendre le plus grand contour (la pièce)
            contour = max(piece_contours, key=cv2.contourArea)
        
        descriptor = extract_contour_color_descriptor(image, contour)
        descriptors.append((i, descriptor))
    
    # Filtrer les descripteurs valides
    valid_descriptors = [(i, desc) for i, desc in descriptors if desc is not None]
    
    # Comparer toutes les paires possibles
    matches = []
    for i in range(len(valid_descriptors)):
        idx1, desc1 = valid_descriptors[i]
        
        for j in range(i + 1, len(valid_descriptors)):
            idx2, desc2 = valid_descriptors[j]
            
            similarity, shift = compare_color_descriptors(desc1, desc2)
            
            if similarity > threshold:
                matches.append((idx1, idx2, similarity, shift))
    
    # Trier par similarité décroissante
    matches.sort(key=lambda x: x[2], reverse=True)
    
    return matches

# Fonction principale pour tester
def test_color_descriptor(image_path):
    """
    Teste le descripteur de couleur sur une image contenant des pièces de puzzle.
    
    Args:
        image_path: Chemin vers l'image
    """
    # Lire l'image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Erreur: Impossible de lire l'image {image_path}")
        return
    
    # Prétraitement et recherche des contours
    # (Utiliser les fonctions existantes du projet)
    from image_processing import pre_process_image, find_contour, show_contour
    
    # Trouver les contours
    contours = find_contour(image)
    print(f"Nombre de contours trouvés: {len(contours)}")
    
    # Filtrer les contours pour ne garder que les pièces de puzzle
    valid_contours = []
    for contour in contours:
        if len(contour) < 5:
            continue
            
        area = cv2.contourArea(contour)
        if area < 1000:
            continue
            
        perimeter = cv2.arcLength(contour, True)
        if perimeter < 100:
            continue
            
        compactness = 4 * np.pi * area / (perimeter * perimeter)
        if compactness < 0.02:
            continue
        
        valid_contours.append(contour)
    
    print(f"Nombre de contours valides: {len(valid_contours)}")
    
    # Extraire et visualiser les descripteurs pour les 5 premières pièces
    plt.figure(figsize=(15, 10))
    
    for i, contour in enumerate(valid_contours[:5]):
        # Extraire le descripteur
        descriptor = extract_contour_color_descriptor(image, contour)
        
        if descriptor is None:
            continue
        
        # Visualiser le contour
        piece_img = np.zeros_like(image)
        cv2.drawContours(piece_img, [contour], 0, (0, 255, 0), 2)
        
        # Convertir BGR à RGB pour matplotlib
        piece_img_rgb = cv2.cvtColor(piece_img, cv2.COLOR_BGR2RGB)
        
        # Visualiser le descripteur
        desc_img = visualize_color_descriptor(descriptor)
        desc_img_rgb = cv2.cvtColor(desc_img, cv2.COLOR_BGR2RGB)
        
        # Afficher
        plt.subplot(5, 2, i*2 + 1)
        plt.imshow(piece_img_rgb)
        plt.title(f"Pièce {i+1}")
        plt.axis('off')
        
        plt.subplot(5, 2, i*2 + 2)
        plt.imshow(desc_img_rgb)
        plt.title(f"Descripteur de couleur")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("debug/color_descriptors.jpg")
    plt.close()
    
    # Trouver des correspondances
    matches = find_matching_pieces(image, valid_contours, threshold=0.6)
    
    # Visualiser les 3 meilleures correspondances
    plt.figure(figsize=(15, 10))
    
    for i, (idx1, idx2, similarity, shift) in enumerate(matches[:3]):
        # Extraire les descripteurs
        desc1 = extract_contour_color_descriptor(image, valid_contours[idx1])
        desc2 = extract_contour_color_descriptor(image, valid_contours[idx2])
        
        # Décaler le second descripteur pour l'alignement optimal
        desc2_shifted = np.roll(desc2, shift, axis=0)
        
        # Visualiser les contours
        img1 = np.zeros_like(image)
        img2 = np.zeros_like(image)
        
        cv2.drawContours(img1, [valid_contours[idx1]], 0, (0, 255, 0), 2)
        cv2.drawContours(img2, [valid_contours[idx2]], 0, (0, 255, 0), 2)
        
        # Convertir BGR à RGB pour matplotlib
        img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        
        # Visualiser les descripteurs
        desc1_img = visualize_color_descriptor(desc1)
        desc2_img = visualize_color_descriptor(desc2_shifted)
        
        desc1_img_rgb = cv2.cvtColor(desc1_img, cv2.COLOR_BGR2RGB)
        desc2_img_rgb = cv2.cvtColor(desc2_img, cv2.COLOR_BGR2RGB)
        
        # Afficher
        plt.subplot(3, 4, i*4 + 1)
        plt.imshow(img1_rgb)
        plt.title(f"Pièce {idx1+1}")
        plt.axis('off')
        
        plt.subplot(3, 4, i*4 + 2)
        plt.imshow(desc1_img_rgb)
        plt.title("Descripteur")
        plt.axis('off')
        
        plt.subplot(3, 4, i*4 + 3)
        plt.imshow(img2_rgb)
        plt.title(f"Pièce {idx2+1}")
        plt.axis('off')
        
        plt.subplot(3, 4, i*4 + 4)
        plt.imshow(desc2_img_rgb)
        plt.title(f"Sim: {similarity:.2f}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("debug/color_matches.jpg")
    
    return matches