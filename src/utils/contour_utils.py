"""
Utilitaires optimisés pour le traitement des contours de pièces de puzzle
avec focus sur la performance et la qualité de segmentation.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import logging
from scipy import stats

# Configuration du logger
try:
    from src.utils.logging_utils import log_manager
    logger = log_manager.get_logger(__name__)
except ImportError:
    # Fallback si log_manager n'est pas disponible
    logger = logging.getLogger(__name__)


def find_contours(binary_image: np.ndarray,
                 mode: int = cv2.RETR_EXTERNAL,
                 method: int = cv2.CHAIN_APPROX_SIMPLE) -> List[np.ndarray]:
    """
    Trouve les contours dans une image binaire avec méthode optimisée pour les performances.
    Version avec optimisations de mémoire.
    
    Args:
        binary_image: Image binaire d'entrée
        mode: Mode de récupération des contours
        method: Méthode d'approximation des contours
    
    Returns:
        Liste des contours
    """
    # Optimisation: éviter de copier l'image si possible
    # Vérifier si l'image est continue en mémoire, sinon faire une copie
    if not binary_image.flags['C_CONTIGUOUS']:
        binary_copy = np.ascontiguousarray(binary_image)
    else:
        binary_copy = binary_image
    
    # Optimisation: utiliser directement RETR_EXTERNAL pour les performances
    contours, _ = cv2.findContours(binary_copy, mode, method)
    
    # Pour les images avec des pièces très complexes ou partiellement fusionnées,
    # essayer également RETR_LIST comme alternative, mais seulement si nécessaire
    if len(contours) < 5 and cv2.countNonZero(binary_image) > binary_image.size * 0.05:
        # Utiliser RETR_LIST qui peut trouver des contours internes
        contours_alt, _ = cv2.findContours(binary_copy, cv2.RETR_LIST, method)
        
        # Si la méthode alternative trouve plus de contours, l'utiliser
        if len(contours_alt) > len(contours):
            # Optimisation : retourner directement les contours sans conversion
            return contours_alt
    
    # Retourner les contours sans conversion supplémentaire
    return contours


def detect_contours(self, binary_image: np.ndarray, original_image: np.ndarray,
                  expected_pieces: Optional[int] = None) -> List[np.ndarray]:
    """
    Détecte les contours des pièces de puzzle dans une image binaire.
    Version optimisée pour la performance.
    
    Args:
        binary_image: Image binaire d'entrée
        original_image: Image originale (pour filtrage basé sur la taille)
        expected_pieces: Nombre attendu de pièces
    
    Returns:
        Liste des contours détectés
    """
    start_time = time.time()
    self.logger.info("Détection des contours avec approche optimisée...")
    
    # Optimisation: utiliser directement detect_puzzle_pieces si disponible
    # Évite plusieurs étapes intermédiaires
    if hasattr(self, 'quick_detect') and self.quick_detect:
        # Utilisation de la fonction optimisée intégrée
        min_size = self.config.contour.MIN_AREA if hasattr(self.config, 'contour') else 1000
        cleaned_binary, contours = detect_puzzle_pieces(
            original_image, 
            expected_min_size=min_size
        )
        self.save_debug_image(cleaned_binary, "05_optimized_binary.jpg")
        
        # Suivi des statistiques de performance
        elapsed = time.time() - start_time
        self.detection_stats['timing']['contour_detection'] = elapsed
        self.logger.info(f"Détection rapide terminée en {elapsed:.3f}s, {len(contours)} contours trouvés")
        
        if contours:
            # Dessiner les contours pour visualisation
            contour_vis = original_image.copy()
            cv2.drawContours(contour_vis, contours, -1, (0, 255, 0), 2)
            self.save_debug_image(contour_vis, "06_contours.jpg")
        
        return contours
    
    # Approche standard: optimisation des paramètres
    params = self.optimize_detection_parameters(binary_image, original_image, expected_pieces)
    
    # Détection initiale des contours - utiliser le mode et la méthode les plus rapides
    contours = find_contours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    self.logger.info(f"Trouvé {len(contours)} contours initiaux")
    
    # Filtrage des contours - passage unique à travers les contours
    filtered_contours = filter_contours(contours, **params)
    
    # Si le nombre de contours trouvés est insuffisant, essayer une récupération
    if expected_pieces and len(filtered_contours) < expected_pieces * 0.7:
        self.logger.info(f"Récupération: trouvé {len(filtered_contours)}/{expected_pieces} pièces attendues")
        
        # Paramètres plus permissifs pour la récupération
        recovery_params = params.copy()
        recovery_params['min_area'] *= 0.7
        recovery_params['solidity_range'] = (0.5, 0.99)
        
        # Filtrage avec les paramètres de récupération
        recovery_contours = filter_contours(contours, **recovery_params)
        
        if len(recovery_contours) > len(filtered_contours):
            filtered_contours = recovery_contours
            self.logger.info(f"Récupération réussie: {len(filtered_contours)} contours")
    
    # Optimisation finale des contours - élimination des doublons et simplification
    optimized_contours = optimize_contours(filtered_contours, min_area=params['min_area'])
    
    # Création de la visualisation des contours
    if optimized_contours:
        contour_vis = original_image.copy()
        cv2.drawContours(contour_vis, optimized_contours, -1, (0, 255, 0), 2)
        self.save_debug_image(contour_vis, "06_contours.jpg")
    
    elapsed = time.time() - start_time
    self.detection_stats['timing']['contour_detection'] = elapsed
    self.logger.info(f"Détection des contours terminée en {elapsed:.3f}s, {len(optimized_contours)} contours filtrés")
    
    return optimized_contours

def filter_contours(contours: List[np.ndarray],
                   min_area: float = 500,
                   max_area: Optional[float] = None,
                   min_perimeter: float = 50,
                   solidity_range: Tuple[float, float] = (0.6, 0.99),
                   aspect_ratio_range: Tuple[float, float] = (0.2, 5.0),
                   use_statistical_filtering: bool = True,
                   expected_piece_count: Optional[int] = None) -> List[np.ndarray]:
    """
    Filtre les contours pour identifier les pièces de puzzle valides avec optimisation de performance.
    
    Args:
        contours: Liste des contours d'entrée
        min_area: Aire minimale du contour
        max_area: Aire maximale du contour (si None, pas de limite supérieure)
        min_perimeter: Périmètre minimal du contour
        solidity_range: Plage (min, max) pour la solidité (area/convex_hull_area)
        aspect_ratio_range: Plage (min, max) pour le ratio d'aspect
        use_statistical_filtering: Utiliser ou non le filtrage statistique
        expected_piece_count: Nombre attendu de pièces de puzzle (optionnel)
    
    Returns:
        Liste filtrée des contours
    """
    # Optimisation pour le cas de liste vide
    if not contours:
        return []
        
    # Filtrage initial avec critères de base
    initial_filtered = []
    
    # Précalcul des areas et périmètres pour éviter les recalculs
    areas = [cv2.contourArea(contour) for contour in contours]
    perimeters = [cv2.arcLength(contour, True) for contour in contours]
    
    for i, (contour, area, perimeter) in enumerate(zip(contours, areas, perimeters)):
        # Filtre rapide sur l'aire et le périmètre
        if area < min_area or perimeter < min_perimeter:
            continue
            
        # Filtre sur l'aire maximum si spécifiée
        if max_area is not None and area > max_area:
            continue
        
        # Calcul des propriétés de forme supplémentaires
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h > 0 else 0
        
        if aspect_ratio < aspect_ratio_range[0] or aspect_ratio > aspect_ratio_range[1]:
            continue
        
        # Calcul de la solidity (opération coûteuse)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        
        if hull_area == 0:  # Éviter la division par zéro
            continue
            
        solidity = area / hull_area
        
        if solidity < solidity_range[0] or solidity > solidity_range[1]:
            continue
        
        # Calcul de la compacité (indicateur de forme)
        # Compacité = périmètre² / (4 * π * aire)
        compactness = perimeter**2 / (4 * np.pi * area) if area > 0 else float('inf')
        
        # Les pièces de puzzle ont typiquement une compacité entre 1,5 et 10
        if compactness < 1.2 or compactness > 15.0:
            continue
            
        initial_filtered.append((contour, {
            'area': area,
            'perimeter': perimeter,
            'solidity': solidity,
            'aspect_ratio': aspect_ratio,
            'compactness': compactness
        }))
    
    # Si pas de filtrage statistique ou trop peu de contours, retourner les contours initialement filtrés
    if not use_statistical_filtering or len(initial_filtered) < 3:
        return [c[0] for c in initial_filtered]
    
    # Extraction des aires pour analyse statistique
    areas = np.array([metrics['area'] for _, metrics in initial_filtered])
    
    # Utiliser des mesures robustes: médiane et MAD (déviation absolue médiane)
    median_area = np.median(areas)
    mad_area = stats.median_abs_deviation(areas)
    
    # Si nous avons un nombre attendu de pièces, affiner les statistiques
    if expected_piece_count is not None and expected_piece_count > 0 and len(initial_filtered) > expected_piece_count * 0.5:
        # Estimation de l'aire attendue par pièce
        total_filtered_area = sum(areas)
        estimated_area_per_piece = total_filtered_area / (len(initial_filtered) * 0.75)
        
        # Pondération entre les statistiques observées et les attentes
        confidence = min(len(initial_filtered) / (expected_piece_count * 2), 0.8)
        median_area = median_area * (1 - confidence) + estimated_area_per_piece * confidence
    
    # Définition des seuils en fonction de la distribution
    cv_value = mad_area / (median_area + 1e-6)  # Coefficient de variation avec MAD
    
    # Adapter le seuil selon la variabilité
    if cv_value < 0.2:
        deviation_factor = 2.0  # Distribution cohérente
    else:
        deviation_factor = 3.0 + cv_value * 5.0  # Plus permissif pour grande variabilité
    
    # Limites d'aire acceptables
    min_acceptable_area = max(min_area, median_area - deviation_factor * mad_area)
    max_acceptable_area = median_area + deviation_factor * mad_area
    
    # Filtrage final
    filtered_contours = []
    for contour, metrics in initial_filtered:
        if min_acceptable_area <= metrics['area'] <= max_acceptable_area:
            filtered_contours.append(contour)
    
    # Récupération des pièces si nécessaire
    if expected_piece_count is not None and len(filtered_contours) < expected_piece_count * 0.7:
        # Seuils plus permissifs pour la récupération
        lenient_min = median_area - deviation_factor * 1.5 * mad_area
        lenient_max = median_area + deviation_factor * 1.5 * mad_area
        
        for contour, metrics in initial_filtered:
            # Check if this contour is already in filtered_contours
            is_duplicate = False
            for existing_contour in filtered_contours:
                # Simple check based on basic properties
                if abs(cv2.contourArea(contour) - cv2.contourArea(existing_contour)) < 1.0:
                    # For contours with very similar areas, do a more detailed check
                    similarity = cv2.matchShapes(contour, existing_contour, cv2.CONTOURS_MATCH_I2, 0.0)
                    if similarity < 0.1:  # Adjust threshold as needed
                        is_duplicate = True
                        break
                        
            if is_duplicate:
                continue
                
            if lenient_min <= metrics['area'] <= lenient_max:
                if validate_shape_as_puzzle_piece(contour):
                    filtered_contours.append(contour)
    
    # Journalisation des statistiques
    logger.info(f"Statistiques de filtrage des contours:")
    logger.info(f"  Contours initiaux: {len(contours)}")
    logger.info(f"  Après filtrage de base: {len(initial_filtered)}")
    logger.info(f"  Après filtrage statistique: {len(filtered_contours)}")
    logger.info(f"  Aire médiane: {median_area:.2f}, MAD: {mad_area:.2f}")
    logger.info(f"  Plage d'acceptation d'aire: {min_acceptable_area:.2f} à {max_acceptable_area:.2f}")
    
    # Fallback si trop peu de contours après filtrage statistique
    if expected_piece_count is not None and len(filtered_contours) < expected_piece_count * 0.5:
        logger.warning(f"Taux de détection faible après filtrage statistique. Utilisation du filtrage de base.")
        return [c[0] for c in initial_filtered]
    
    return filtered_contours

def validate_shape_as_puzzle_piece(contour: np.ndarray) -> bool:
    """
    Valide si la forme d'un contour correspond à une pièce de puzzle.
    Version optimisée pour la performance.
    
    Args:
        contour: Contour d'entrée
    
    Returns:
        True si le contour est probablement une pièce de puzzle
    """
    # Vérification rapide si le contour a assez de points pour une pièce de puzzle
    if len(contour) < 20:
        return False
        
    # Calcul des métriques de forme essentielles
    area = cv2.contourArea(contour)
    if area <= 0:
        return False
        
    perimeter = cv2.arcLength(contour, True)
    
    # Calcul de la compacité (compactness) - indicateur clé de la complexité de forme
    # Les pièces de puzzle ont généralement des valeurs plus élevées que les formes simples
    compactness = perimeter**2 / (4 * np.pi * area)
    
    # Les pièces de puzzle ont généralement une compacité entre 2 et 10
    if not (2.0 <= compactness <= 12.0):
        return False
    
    # Calcul de la solidité (proportion de l'aire par rapport à l'enveloppe convexe)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    
    if hull_area <= 0:
        return False
        
    solidity = area / hull_area
    
    # Les pièces de puzzle ont généralement une solidité assez élevée (mais pas trop)
    if not (0.7 <= solidity <= 0.98):
        return False
    
    # Vérification du rectangle englobant et ratio d'aspect
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else float('inf')
    
    # Les pièces de puzzle ne sont généralement pas trop allongées
    if aspect_ratio > 4.0:
        return False
    
    # Vérification de la présence de "bosses" (défauts de convexité)
    # Cette vérification est plus coûteuse, ne l'appliquer qu'aux candidats prometteurs
    try:
        hull_indices = cv2.convexHull(contour, returnPoints=False)
        defects = cv2.convexityDefects(contour, hull_indices)
        
        if defects is None:
            return False
            
        # Compter les défauts significatifs
        significant_defects = 0
        for i in range(defects.shape[0]):
            _, _, _, depth = defects[i, 0]
            if depth > 300:  # Seuil arbitraire basé sur les dimensions typiques des pièces de puzzle
                significant_defects += 1
        
        # Les pièces de puzzle ont généralement quelques défauts de convexité significatifs
        if significant_defects < 1:
            return False
    except:
        # Si le calcul des défauts de convexité échoue, être conservateur
        return False
    
    return True


def calculate_contour_features(contour: np.ndarray) -> Dict[str, Any]:
    """
    Calcule les caractéristiques essentielles d'un contour, optimisé pour la performance.
    
    Args:
        contour: Contour d'entrée
    
    Returns:
        Dictionnaire des caractéristiques du contour
    """
    # Mesures de base - optimisées pour la performance
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    
    # Formes englobantes
    x, y, w, h = cv2.boundingRect(contour)
    rect_area = w * h
    
    # Rectangle d'aire minimale
    min_area_rect = cv2.minAreaRect(contour)
    min_area_rect_area = min_area_rect[1][0] * min_area_rect[1][1]
    
    # Enveloppe convexe
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    
    # Descripteurs de forme
    extent = area / rect_area if rect_area > 0 else 0
    solidity = area / hull_area if hull_area > 0 else 0
    equivalent_diameter = np.sqrt(4 * area / np.pi)
    
    # Moments et caractéristiques dérivées
    moments = cv2.moments(contour)
    
    centroid_x = moments['m10'] / moments['m00'] if moments['m00'] != 0 else 0
    centroid_y = moments['m01'] / moments['m00'] if moments['m00'] != 0 else 0
    
    # Compacité (perimeter^2 / area)
    compactness = perimeter**2 / (4 * np.pi * area) if area > 0 else 0
    
    # Ellipticité (rapport des axes majeur et mineur)
    if min_area_rect[1][0] > 0 and min_area_rect[1][1] > 0:
        major_axis = max(min_area_rect[1][0], min_area_rect[1][1])
        minor_axis = min(min_area_rect[1][0], min_area_rect[1][1])
        ellipticity = major_axis / minor_axis
    else:
        ellipticity = 1.0
    
    # Retour du dictionnaire des caractéristiques essentielles
    # Version optimisée pour se concentrer sur les métriques les plus utiles
    return {
        'area': area,
        'perimeter': perimeter,
        'bbox': (x, y, w, h),
        'bbox_area': rect_area,
        'min_area_rect': min_area_rect,
        'min_area_rect_area': min_area_rect_area,
        'hull_area': hull_area,
        'extent': extent,
        'solidity': solidity,
        'equivalent_diameter': equivalent_diameter,
        'centroid': (centroid_x, centroid_y),
        'compactness': compactness,
        'ellipticity': ellipticity
    }


def cluster_contours(contours: List[np.ndarray],
                    features: Optional[List[Dict[str, Any]]] = None) -> Dict[str, List[int]]:
    """
    Regroupe les contours par caractéristiques similaires en utilisant scikit-learn.
    
    Args:
        contours: Liste des contours
        features: Liste des caractéristiques précalculées (optionnel)
    
    Returns:
        Dictionnaire de cluster_id -> liste des indices de contours
    """
    # Importation conditionnelle de scikit-learn
    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        logger.warning("scikit-learn non disponible. Utilisation du clustering simplifié.")
        return _simple_cluster_contours(contours, features)
    
    # Optimisation pour les cas triviaux
    if len(contours) <= 1:
        return {'0': list(range(len(contours)))}
    
    # Calcul des caractéristiques si non fournies
    if features is None:
        features = [calculate_contour_features(c) for c in contours]
    
    # Extraction des métriques clés pour le clustering
    feature_matrix = np.array([
        [f['area'], f['compactness'], f['solidity']] 
        for f in features
    ])
    
    # Normalisation des caractéristiques
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(feature_matrix)
    
    # Déterminer le nombre de clusters
    n_samples = len(contours)
    n_clusters = min(3, max(1, n_samples // 10 + 1))
    
    # Clustering avec K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(normalized_features)
    
    # Organisation des résultats
    cluster_dict = {}
    for i, cluster_id in enumerate(clusters):
        cluster_str = str(cluster_id)
        if cluster_str not in cluster_dict:
            cluster_dict[cluster_str] = []
        cluster_dict[cluster_str].append(i)
    
    # Journalisation des résultats
    logger.info(f"Résultats du clustering des contours:")
    for cluster_id, indices in cluster_dict.items():
        mean_area = np.mean([features[i]['area'] for i in indices])
        logger.info(f"  Cluster {cluster_id}: {len(indices)} contours, aire moyenne: {mean_area:.2f}")
    
    return cluster_dict


def _simple_cluster_contours(contours: List[np.ndarray], 
                           features: Optional[List[Dict[str, Any]]] = None) -> Dict[str, List[int]]:
    """
    Implémentation simplifiée de clustering pour les cas où scikit-learn n'est pas disponible.
    
    Args:
        contours: Liste des contours
        features: Liste des caractéristiques précalculées (optionnel)
        
    Returns:
        Dictionnaire de cluster_id -> liste des indices de contours
    """
    # Cas trivial: un seul cluster pour tous
    if len(contours) <= 3:
        return {'0': list(range(len(contours)))}
    
    # Calcul des caractéristiques si non fournies
    if features is None:
        features = [calculate_contour_features(c) for c in contours]
    
    # Extraction des aires pour clustering basique - vectorisé avec NumPy
    areas = np.array([f['area'] for f in features])
    
    # Classification simple basée sur l'aire
    # Utiliser des opérations vectorisées pour le clustering
    median_area = np.median(areas)
    mad = np.median(np.abs(areas - median_area))
    
    # Définir les seuils
    small_threshold = median_area - mad
    large_threshold = median_area + mad
    
    # Créer des masques booléens pour chaque cluster - vectorisé
    small_mask = areas < small_threshold
    large_mask = areas > large_threshold
    medium_mask = ~(small_mask | large_mask)
    
    # Obtenir les indices pour chaque cluster
    small_indices = np.where(small_mask)[0].tolist()
    medium_indices = np.where(medium_mask)[0].tolist()
    large_indices = np.where(large_mask)[0].tolist()
    
    # Créer le dictionnaire de résultats
    clusters = {
        '0': small_indices,
        '1': medium_indices,
        '2': large_indices
    }
    
    return clusters


def contour_match_score(contour1: np.ndarray, contour2: np.ndarray, threshold: float = 0.7) -> float:
    """
    Calcule un score de correspondance entre deux contours.
    Optimisé pour la performance.
    
    Args:
        contour1: Premier contour
        contour2: Second contour
        threshold: Seuil de correspondance
        
    Returns:
        Score de correspondance (0-1)
    """
    # Calcul rapide des aires
    area1 = cv2.contourArea(contour1)
    area2 = cv2.contourArea(contour2)
    
    # Si les aires sont trop différentes, les contours ne correspondent pas
    area_ratio = min(area1, area2) / max(area1, area2)
    if area_ratio < 0.5:
        return 0.0
    
    # Calcul des rectangles englobants
    x1, y1, w1, h1 = cv2.boundingRect(contour1)
    x2, y2, w2, h2 = cv2.boundingRect(contour2)
    
    # Vérification du chevauchement des rectangles englobants
    intersection_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    intersection_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    
    if intersection_x <= 0 or intersection_y <= 0:
        return 0.0
    
    # Calcul du score de forme - méthode plus rapide que de comparer des masques complets
    # Utilisation des moments de Hu pour comparer les formes
    moments1 = cv2.moments(contour1)
    moments2 = cv2.moments(contour2)
    
    hu1 = cv2.HuMoments(moments1).flatten()
    hu2 = cv2.HuMoments(moments2).flatten()
    
    # Utilisation des 3 premiers moments uniquement pour la vitesse
    shape_diff = np.mean(np.abs(hu1[:3] - hu2[:3]))
    shape_score = np.exp(-shape_diff)
    
    # Score combiné
    score = 0.5 * area_ratio + 0.5 * shape_score
    
    return score if score >= threshold else 0.0


def find_duplicate_contours(contours: List[np.ndarray], threshold: float = 0.85) -> List[int]:
    """
    Identifie les contours dupliqués à éliminer.
    Optimisé pour la performance.
    
    Args:
        contours: Liste des contours
        threshold: Seuil de similarité pour considérer des doublons
        
    Returns:
        Liste des indices des contours à supprimer
    """
    if len(contours) <= 1:
        return []
    
    # Précalcul des aires pour une comparaison rapide
    areas = [cv2.contourArea(c) for c in contours]
    
    duplicates = []
    n = len(contours)
    
    # Pour chaque contour
    for i in range(n):
        if i in duplicates:
            continue
            
        area_i = areas[i]
        
        # Comparer uniquement avec les contours d'aire similaire (optimisation)
        for j in range(i+1, n):
            if j in duplicates:
                continue
                
            area_j = areas[j]
            
            # Vérification rapide des aires
            if min(area_i, area_j) / max(area_i, area_j) < 0.7:
                continue
            
            # Calcul plus approfondi de la similarité
            score = contour_match_score(contours[i], contours[j], threshold)
            
            if score >= threshold:
                # Garder le contour avec la plus grande aire
                if area_i >= area_j:
                    duplicates.append(j)
                else:
                    duplicates.append(i)
                    break  # Si i est un doublon, arrêter de le comparer à d'autres
    
    return duplicates


def optimize_contours(contours: List[np.ndarray], min_area: float = 1000) -> List[np.ndarray]:
    """
    Optimise une liste de contours en éliminant les doublons et les formes non valides.
    
    Args:
        contours: Liste des contours à optimiser
        min_area: Aire minimale pour un contour valide
        
    Returns:
        Liste optimisée des contours
    """
    # Filtrage rapide par aire
    filtered = [c for c in contours if cv2.contourArea(c) >= min_area]
    
    # Identification et suppression des doublons
    duplicates = find_duplicate_contours(filtered)
    unique_contours = [c for i, c in enumerate(filtered) if i not in duplicates]
    
    # Simplification des contours pour améliorer les performances
    simplified_contours = []
    for contour in unique_contours:
        epsilon = 0.005 * cv2.arcLength(contour, True)
        simplified = cv2.approxPolyDP(contour, epsilon, True)
        simplified_contours.append(simplified)
    
    return simplified_contours