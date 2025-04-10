"""
Utilitaires de vérification pour la détection des pièces de puzzle.
Ce module fournit des fonctions optimisées pour vérifier et filtrer les pièces détectées
en se concentrant sur la qualité de la segmentation.
"""

import numpy as np
import cv2
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from scipy import stats

# Obtention du logger pour ce module
try:
    from src.utils.logging_utils import log_manager
    logger = log_manager.get_logger(__name__)
except ImportError:
    # Fallback si log_manager n'est pas disponible
    logger = logging.getLogger(__name__)


def validate_puzzle_piece(contour: np.ndarray, features: Optional[Dict] = None, 
                        validation_level: str = 'standard') -> Tuple[bool, float, str]:
    """
    Fonction unifiée pour valider si un contour représente une pièce de puzzle valide.
    
    Args:
        contour: Contour à valider
        features: Caractéristiques précalculées (optionnel)
        validation_level: Niveau de rigueur ('permissive', 'standard', 'strict')
        
    Returns:
        Tuple de (est_valide, score_qualité, raison)
    """
    # Calculer les caractéristiques si non fournies
    if not features:
        features = _calculate_basic_features(contour)
    
    # Vérifications de base (communes à tous les niveaux)
    basic_valid, basic_score, basic_reason = _validate_basic_requirements(features)
    if not basic_valid:
        return False, basic_score, basic_reason
    
    # Ajuster les seuils en fonction du niveau de validation
    thresholds = _get_validation_thresholds(validation_level)
    
    # Vérifications avancées
    shape_valid, shape_score, shape_reason = _validate_shape_properties(features, contour, thresholds)
    if not shape_valid:
        return False, shape_score, shape_reason
    
    # Calcul du score global
    quality_score = _calculate_quality_score(features, basic_score, shape_score, thresholds)
    
    # Validation finale
    if quality_score < thresholds['min_quality_score']:
        return False, quality_score, "low_quality_score"
    
    return True, quality_score, "valid"


def _calculate_basic_features(contour: np.ndarray) -> Dict[str, Any]:
    """
    Calcule les caractéristiques de base d'un contour nécessaires pour la validation.
    
    Args:
        contour: Contour à analyser
        
    Returns:
        Dictionnaire des caractéristiques essentielles
    """
    # Mesures de base
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    
    # Rectangle englobant
    x, y, w, h = cv2.boundingRect(contour)
    
    # Calcul du ratio d'aspect
    aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else float('inf')
    
    # Enveloppe convexe et solidité
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0
    
    # Compacité (perimeter^2 / (4 * π * area))
    compactness = (perimeter**2) / (4 * np.pi * area) if area > 0 else float('inf')
    
    # Moments et centroïde
    M = cv2.moments(contour)
    cx = M['m10'] / M['m00'] if M['m00'] != 0 else 0
    cy = M['m01'] / M['m00'] if M['m00'] != 0 else 0
    
    return {
        'area': area,
        'perimeter': perimeter,
        'bbox': (x, y, w, h),
        'aspect_ratio': aspect_ratio,
        'solidity': solidity,
        'compactness': compactness,
        'centroid': (cx, cy)
    }


def _validate_basic_requirements(features: Dict) -> Tuple[bool, float, str]:
    """
    Vérifie les exigences de base pour une pièce de puzzle valide.
    
    Args:
        features: Caractéristiques du contour
        
    Returns:
        Tuple de (est_valide, score_basique, raison)
    """
    # Vérification de l'aire minimale
    if features['area'] < 500:  # Valeur minimale absolue
        return False, 0.0, "area_too_small"
    
    # Vérification du périmètre minimum
    if features['perimeter'] < 50:
        return False, 0.1, "perimeter_too_small"
    
    # Score de base proportionnel à l'aire
    # Les pièces trop petites obtiennent un score plus faible
    area_score = min(1.0, features['area'] / 2000)
    
    return True, area_score, "basic_requirements_met"


def _validate_shape_properties(features: Dict, contour: np.ndarray, 
                            thresholds: Dict) -> Tuple[bool, float, str]:
    """
    Vérifie les propriétés de forme avancées pour une pièce de puzzle.
    
    Args:
        features: Caractéristiques du contour
        contour: Contour à analyser
        thresholds: Seuils de validation
        
    Returns:
        Tuple de (est_valide, score_forme, raison)
    """
    # 1. Vérification de la compacité
    compactness = features['compactness']
    min_compactness, max_compactness = thresholds['compactness_range']
    
    if compactness < min_compactness:
        return False, 0.3, "too_compact"
    if compactness > max_compactness:
        return False, 0.2, "too_irregular"
    
    # Score de compacité normalisé
    # Optimal autour de 4-5 (typique pour les pièces de puzzle)
    compactness_score = 1.0 - min(abs(compactness - 4.5) / 5.0, 1.0)
    
    # 2. Vérification du ratio d'aspect
    aspect_ratio = features['aspect_ratio']
    if aspect_ratio > thresholds['aspect_ratio_max']:
        return False, 0.2, "bad_aspect_ratio"
    
    # Score de ratio d'aspect
    aspect_score = 1.0 - min((aspect_ratio - 1.0) / thresholds['aspect_ratio_max'], 1.0)
    
    # 3. Vérification de la solidité
    solidity = features['solidity']
    min_solidity, max_solidity = thresholds['solidity_range']
    
    if solidity < min_solidity:
        return False, 0.3, "low_solidity"
    if solidity > max_solidity:
        return False, 0.4, "too_solid"
    
    # Score de solidité (optimal autour de 0.85)
    solidity_score = 1.0 - min(abs(solidity - 0.85) / 0.15, 1.0)
    
    # 4. Vérification des défauts de convexité (caractéristique des pièces de puzzle)
    # Cette vérification est plus coûteuse, ne l'appliquer qu'aux candidats prometteurs
    try:
        hull_indices = cv2.convexHull(contour, returnPoints=False)
        defects = cv2.convexityDefects(contour, hull_indices)
        
        if defects is None or len(defects) == 0:
            defect_score = 0.6
            defect_valid = False
        else:
            # Compter les défauts significatifs
            significant_defects = 0
            for i in range(defects.shape[0]):
                _, _, _, depth = defects[i, 0]
                if depth > 300:  # Seuil adapté aux dimensions typiques
                    significant_defects += 1
            
            # Les pièces de puzzle ont généralement quelques défauts significatifs
            defect_score = min(significant_defects / 4.0, 1.0)
            defect_valid = significant_defects >= 1
    except:
        # En cas d'erreur, utiliser une valeur par défaut
        defect_score = 0.7
        defect_valid = True
    
    # Si aucun défaut significatif et validation stricte, rejeter
    if not defect_valid and validation_level == 'strict':
        return False, defect_score, "no_significant_defects"
    
    # Score global de forme
    shape_score = (
        0.3 * compactness_score +
        0.2 * aspect_score +
        0.3 * solidity_score +
        0.2 * defect_score
    )
    
    return True, shape_score, "valid_shape"


def _calculate_quality_score(features: Dict, basic_score: float, 
                           shape_score: float, thresholds: Dict) -> float:
    """
    Calcule le score de qualité global pour une pièce de puzzle.
    
    Args:
        features: Caractéristiques du contour
        basic_score: Score des critères de base
        shape_score: Score des propriétés de forme
        thresholds: Seuils de validation
        
    Returns:
        Score de qualité global
    """
    # Poids des différents scores
    # Le score de forme est plus important pour déterminer si c'est une pièce de puzzle
    weights = {
        'basic': 0.3,
        'shape': 0.7
    }
    
    # Score pondéré
    quality_score = weights['basic'] * basic_score + weights['shape'] * shape_score
    
    return quality_score


def _get_validation_thresholds(validation_level: str) -> Dict:
    """
    Retourne les seuils appropriés selon le niveau de validation.
    
    Args:
        validation_level: Niveau de validation ('permissive', 'standard', 'strict')
        
    Returns:
        Dictionnaire des seuils de validation
    """
    if validation_level == 'permissive':
        return {
            'compactness_range': (1.3, 15.0),
            'solidity_range': (0.5, 0.99),
            'aspect_ratio_max': 5.0,
            'min_quality_score': 0.4
        }
    elif validation_level == 'strict':
        return {
            'compactness_range': (1.8, 10.0),
            'solidity_range': (0.7, 0.95),
            'aspect_ratio_max': 3.0,
            'min_quality_score': 0.6
        }
    else:  # standard
        return {
            'compactness_range': (1.5, 12.0),
            'solidity_range': (0.6, 0.98),
            'aspect_ratio_max': 4.0,
            'min_quality_score': 0.5
        }


def _recover_pieces_by_area(verified_pieces, rejected_pieces, expected_pieces, central_tendency):
    """
    Récupère des pièces rejetées pour atteindre le nombre attendu.
    
    Args:
        verified_pieces: Liste des pièces déjà vérifiées
        rejected_pieces: Liste des pièces rejetées
        expected_pieces: Nombre attendu de pièces
        central_tendency: Tendance centrale (moyenne ou médiane)
        
    Returns:
        Liste des pièces récupérées
    """
    # Nombre de pièces à récupérer
    to_recover = expected_pieces - len(verified_pieces)
    
    if to_recover <= 0 or not rejected_pieces:
        return []
    
    logger.info(f"Tentative de récupération de {to_recover} pièces pour atteindre {expected_pieces}")
    
    # Tri des pièces rejetées par proximité à la tendance centrale
    rejected_by_distance = [
        (abs(p.features['area'] - central_tendency), p) for p in rejected_pieces
    ]
    rejected_by_distance.sort(key=lambda x: x[0])  # Tri par distance
    
    # Sélectionner les pièces les plus proches du centre
    pieces_to_recover = [p for _, p in rejected_by_distance[:to_recover]]
    
    # Restaurer leur statut
    for piece in pieces_to_recover:
        piece.is_valid = True
        piece.validation_status = "recovered_by_area"
    
    logger.info(f"Récupéré {len(pieces_to_recover)} pièces par proximité d'aire")
    
    return pieces_to_recover


def fast_shape_verification(pieces, validation_level: str = 'standard', expected_pieces: Optional[int] = None) -> List:
    """
    Vérification rapide basée uniquement sur les propriétés de forme des contours.
    
    Args:
        pieces: Liste des objets PuzzlePiece
        validation_level: Niveau de validation ('permissive', 'standard', 'strict')
        expected_pieces: Nombre attendu de pièces
        
    Returns:
        Liste des pièces vérifiées
    """
    start_time = time.time()
    logger.info(f"Exécution de la vérification rapide par forme (niveau: {validation_level})")
    
    verified_pieces = []
    rejected_pieces = []
    
    # Obtenir les seuils de validation
    thresholds = _get_validation_thresholds(validation_level)
    
    for piece in pieces:
        try:
            # Vérification de la forme en utilisant la fonction unifiée
            is_valid, quality_score, reason = validate_puzzle_piece(
                piece.contour, piece.features, validation_level
            )
            
            # Mise à jour du score de validation de la pièce
            piece.validation_score = quality_score
            
            if is_valid:
                piece.is_valid = True
                piece.validation_status = f"valid_shape:{quality_score:.2f}"
                verified_pieces.append(piece)
            else:
                piece.is_valid = False
                piece.validation_status = f"invalid_shape:{reason}:{quality_score:.2f}"
                rejected_pieces.append(piece)
        except Exception as e:
            logger.error(f"Erreur lors de la validation de la pièce: {str(e)}")
            piece.is_valid = False
            piece.validation_status = f"validation_error:{str(e)}"
            rejected_pieces.append(piece)
    
    elapsed_time = time.time() - start_time
    logger.info(f"Vérification par forme ({validation_level}): conservé {len(verified_pieces)}/{len(pieces)} " +
              f"pièces en {elapsed_time:.3f}s")
    
    # Récupération si nécessaire et si nombre attendu fourni
    if expected_pieces and len(verified_pieces) < expected_pieces * 0.8:
        logger.info(f"Trop peu de pièces ({len(verified_pieces)}/{expected_pieces}). Tentative de récupération.")
        
        # Si nous avons moins de 80% des pièces attendues, essayer une validation plus permissive
        if validation_level != 'permissive':
            recovery_result = fast_shape_verification(
                rejected_pieces, 'permissive', expected_pieces - len(verified_pieces)
            )
            
            if recovery_result:
                # Mise à jour du statut pour indiquer la récupération
                for piece in recovery_result:
                    piece.validation_status = f"recovered_shape:{piece.validation_score:.2f}"
                
                verified_pieces.extend(recovery_result)
                logger.info(f"Récupéré {len(recovery_result)} pièces supplémentaires avec validation permissive")
        
        # Si toujours insuffisant, essayer de récupérer par score de qualité
        if len(verified_pieces) < expected_pieces:
            more_pieces = _recover_pieces_by_score(
                verified_pieces, rejected_pieces, expected_pieces, 'shape'
            )
            
            if more_pieces:
                verified_pieces.extend(more_pieces)
    
    return verified_pieces


def _recover_pieces_by_score(verified_pieces, rejected_pieces, expected_pieces, recovery_type='shape'):
    """
    Récupère des pièces en se basant sur leur score de qualité.
    
    Args:
        verified_pieces: Liste des pièces déjà vérifiées
        rejected_pieces: Liste des pièces rejetées
        expected_pieces: Nombre attendu de pièces
        recovery_type: Type de récupération ('shape', 'combined')
        
    Returns:
        Liste des pièces récupérées
    """
    # Nombre de pièces à récupérer
    to_recover = expected_pieces - len(verified_pieces)
    
    if to_recover <= 0 or not rejected_pieces:
        return []
    
    logger.info(f"Récupération de {to_recover} pièces par score ({recovery_type})")
    
    # Tri des pièces rejetées par score de validation
    rejected_by_score = sorted(
        rejected_pieces, 
        key=lambda p: p.validation_score if hasattr(p, 'validation_score') else 0.0,
        reverse=True
    )
    
    # Sélectionner les pièces avec les meilleurs scores
    pieces_to_recover = rejected_by_score[:to_recover]
    
    # Restaurer leur statut
    for piece in pieces_to_recover:
        piece.is_valid = True
        piece.validation_status = f"recovered_by_{recovery_type}_score"
    
    logger.info(f"Récupéré {len(pieces_to_recover)} pièces par score")
    
    return pieces_to_recover


def final_area_verification(pieces, area_threshold: float = 2.0, expected_pieces: Optional[int] = None):
    """
    Effectue une vérification finale basée sur l'aire des pièces avec optimisation de performance.
    
    Args:
        pieces: Liste des objets PuzzlePiece
        area_threshold: Seuil d'écart-type pour le filtrage (défaut: 2.0)
        expected_pieces: Nombre attendu de pièces (pour la récupération)
    
    Returns:
        Liste des pièces vérifiées
    """
    # Optimisation pour les cas triviaux
    if not pieces or len(pieces) < 2:
        return pieces
    
    start_time = time.time()
    logger.info(f"Vérification finale par aire avec seuil {area_threshold}")
    
    # Extraction des aires - vectorisé dans un tableau NumPy
    areas = np.array([piece.features['area'] for piece in pieces])
    
    # Utiliser un estimateur robuste (médiane/MAD) pour les distributions non-normales
    is_normal = _is_normal_distribution(areas)
    
    if is_normal:
        # Statistiques paramétriques pour distributions normales
        mean_area = np.mean(areas)
        std_area = np.std(areas)
        logger.info(f"Distribution normale: moyenne={mean_area:.2f}, écart-type={std_area:.2f}")
        
        min_acceptable = mean_area - area_threshold * std_area
        max_acceptable = mean_area + area_threshold * std_area
    else:
        # Statistiques robustes pour distributions non-normales
        median_area = np.median(areas)
        # Calcul de la déviation absolue médiane (MAD)
        mad = np.median(np.abs(areas - median_area))
        # Facteur de correction pour équivalence avec l'écart-type dans une distribution normale
        mad_scaled = 1.4826 * mad
        
        logger.info(f"Distribution non-normale: médiane={median_area:.2f}, MAD={mad_scaled:.2f}")
        
        min_acceptable = median_area - area_threshold * mad_scaled
        max_acceptable = median_area + area_threshold * mad_scaled
    
    logger.info(f"Plage d'aire acceptable: {min_acceptable:.2f} à {max_acceptable:.2f}")
    
    # Vectorisation du filtrage avec NumPy
    valid_mask = (areas >= min_acceptable) & (areas <= max_acceptable)
    invalid_mask = ~valid_mask
    
    # Utiliser np.where pour obtenir les indices - plus rapide que des boucles
    valid_indices = np.where(valid_mask)[0]
    invalid_indices = np.where(invalid_mask)[0]
    
    # Création des listes filtrées en une seule fois
    verified_pieces = [pieces[i] for i in valid_indices]
    rejected_pieces = [pieces[i] for i in invalid_indices]
    
    # Mise à jour du statut des pièces rejetées - une seule boucle au lieu de vérifications répétées
    for piece in rejected_pieces:
        piece.validation_status = f"area_outlier:{piece.features['area']:.2f}"
        piece.is_valid = False
    
    # Journalisation des résultats
    elapsed_time = time.time() - start_time
    logger.info(f"Vérification par aire: conservé {len(verified_pieces)}/{len(pieces)} pièces, " +
              f"rejeté {len(rejected_pieces)} valeurs aberrantes ({elapsed_time:.3f}s)")
    
    # Étape de récupération si nécessaire
    if expected_pieces and len(verified_pieces) < expected_pieces:
        # Calculer le nombre de pièces à récupérer
        to_recover = expected_pieces - len(verified_pieces)
        if to_recover > 0 and rejected_pieces:
            # Calcul vectorisé des distances à la valeur centrale
            central_value = mean_area if is_normal else median_area
            distances = np.abs(np.array([p.features['area'] for p in rejected_pieces]) - central_value)
            
            # Trier les indices par distance croissante
            sorted_indices = np.argsort(distances)
            
            # Sélectionner les pièces les plus proches de la valeur centrale
            recovered_pieces = [rejected_pieces[i] for i in sorted_indices[:to_recover]]
            
            # Restauration en une seule boucle au lieu de rechercher des correspondances
            for piece in recovered_pieces:
                piece.is_valid = True
                piece.validation_status = "recovered_by_area"
            
            logger.info(f"Récupération de {len(recovered_pieces)} pièces pour atteindre {expected_pieces}")
            verified_pieces.extend(recovered_pieces)
    
    return verified_pieces


def _is_normal_distribution(data, significance=0.05):
    """
    Vérifie si les données suivent une distribution normale.
    
    Args:
        data: Données à tester (array-like)
        significance: Seuil de signification (alpha)
        
    Returns:
        True si les données suivent une distribution normale
    """
    try:
        from scipy import stats
        
        # Le test de Shapiro-Wilk a des limites sur la taille de l'échantillon
        # Typiquement entre 3 et 5000 échantillons
        if len(data) >= 3 and len(data) <= 5000:
            _, p_value = stats.shapiro(data)
            return p_value > significance
        elif len(data) > 5000:
            # Pour des échantillons plus grands, utiliser D'Agostino-Pearson
            k2, p_value = stats.normaltest(data)
            return p_value > significance
        else:
            # Trop peu d'échantillons pour un test fiable
            return True
            
    except ImportError:
        # Si scipy n'est pas disponible, utiliser une heuristique simple
        # basée sur le coefficient d'asymétrie (skewness) et d'aplatissement (kurtosis)
        n = len(data)
        if n < 8:  # Trop peu de données pour une estimation fiable
            return True
            
        # Calcul manuel du skewness
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return True
            
        # Skewness - mesure l'asymétrie de la distribution
        skewness = np.sum(((data - mean) / std) ** 3) / n
        
        # Kurtosis - mesure l'aplatissement de la distribution
        kurtosis = np.sum(((data - mean) / std) ** 4) / n - 3
        
        # Des valeurs proches de 0 indiquent une distribution normale
        return abs(skewness) < 1.0 and abs(kurtosis) < 2.0


def create_verification_visualization(image, verified_pieces, rejected_pieces):
    """
    Crée une visualisation montrant les pièces vérifiées et rejetées.
    Version optimisée pour la performance.
    
    Args:
        image: Image originale
        verified_pieces: Liste des objets PuzzlePiece vérifiés
        rejected_pieces: Liste des objets PuzzlePiece rejetés
    
    Returns:
        Image de visualisation
    """
    # Créer une copie de l'image pour la visualisation
    vis = image.copy()
    
    # 1. Préparation des informations textuelles pour éviter les calculs répétés
    piece_labels = {}
    
    # Calculer les centroïdes et les textes en une seule boucle pour les pièces vérifiées
    for piece in verified_pieces:
        M = cv2.moments(piece.contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            text = f"#{piece.id}"
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            
            piece_labels[piece] = {
                'position': (cx, cy),
                'text': text,
                'size': text_size,
                'color': (0, 255, 0)
            }
    
    # Faire de même pour les pièces rejetées
    for piece in rejected_pieces:
        M = cv2.moments(piece.contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # Extraire la raison du rejet du statut de validation
            status_parts = piece.validation_status.split(":") if piece.validation_status else ["rejected"]
            reason = status_parts[0]
            
            text_size, _ = cv2.getTextSize(reason, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            
            piece_labels[piece] = {
                'position': (cx, cy),
                'text': reason,
                'size': text_size,
                'color': (0, 0, 255)
            }
    
    # 2. Dessiner tous les contours en une seule opération pour chaque groupe
    if verified_pieces:
        cv2.drawContours(vis, [p.contour for p in verified_pieces], -1, (0, 255, 0), 2)
    
    if rejected_pieces:
        cv2.drawContours(vis, [p.contour for p in rejected_pieces], -1, (0, 0, 255), 2)
    
    # 3. Dessiner les étiquettes en une boucle unique
    for piece, label_info in piece_labels.items():
        cx, cy = label_info['position']
        text = label_info['text']
        text_size = label_info['size']
        color = label_info['color']
        
        # Ajouter un arrière-plan pour améliorer la lisibilité
        cv2.rectangle(vis, 
                    (cx - 5, cy - text_size[1] - 5), 
                    (cx + text_size[0] + 5, cy + 5), 
                    (0, 0, 0), -1)
        
        cv2.putText(vis, text, (cx, cy),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # Ajouter une légende
    cv2.putText(vis, f"Vert: {len(verified_pieces)} Pièces Valides", (20, 30),
              cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.putText(vis, f"Rouge: {len(rejected_pieces)} Pièces Rejetées", (20, 70),
              cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    
    return vis


def quick_contour_verification(contours: List[np.ndarray], 
                              validation_level: str = 'standard',
                              min_area: float = 1000,
                              max_area: Optional[float] = None) -> Tuple[List[np.ndarray], List[np.ndarray], List[float]]:
    """
    Vérification rapide des contours sans instancier des objets PuzzlePiece.
    Idéal pour le filtrage initial ou la prévisualisation.
    
    Args:
        contours: Liste des contours à vérifier
        validation_level: Niveau de validation ('permissive', 'standard', 'strict')
        min_area: Aire minimale acceptable
        max_area: Aire maximale acceptable (si None, pas de limite supérieure)
        
    Returns:
        Tuple de (contours valides, contours rejetés, scores de qualité)
    """
    valid_contours = []
    rejected_contours = []
    quality_scores = []
    
    # Obtenir les seuils de validation
    thresholds = _get_validation_thresholds(validation_level)
    
    for contour in contours:
        # Vérification rapide de l'aire
        area = cv2.contourArea(contour)
        if area < min_area or (max_area is not None and area > max_area):
            rejected_contours.append(contour)
            quality_scores.append(0.0)
            continue
        
        # Vérification avec la fonction unifiée
        is_valid, quality_score, _ = validate_puzzle_piece(contour, None, validation_level)
        
        if is_valid:
            # Utiliser le contour original sans créer de copie
            valid_contours.append(contour)
        else:
            # Utiliser le contour original sans créer de copie
            rejected_contours.append(contour)
        
        quality_scores.append(quality_score)
    
    return valid_contours, rejected_contours, quality_scores


def combine_verification_methods(pieces, expected_pieces: Optional[int] = None, 
                               validation_level: str = 'standard') -> List:
    """
    Combine plusieurs méthodes de vérification pour obtenir les meilleurs résultats.
    Sélectionne automatiquement la méthode appropriée en fonction du contexte.
    
    Args:
        pieces: Liste des objets PuzzlePiece
        expected_pieces: Nombre attendu de pièces
        validation_level: Niveau de validation par défaut
        
    Returns:
        Liste des pièces vérifiées
    """
    logger.info("Application de la vérification combinée adaptative")
    
    # Cas 1: Très peu de pièces
    if len(pieces) <= 5:
        logger.info("Peu de pièces détectées, utilisation de la vérification de forme uniquement")
        return fast_shape_verification(pieces, validation_level, expected_pieces)
    
    # Cas 2: Nombre attendu fourni
    if expected_pieces:
        # Si le nombre de pièces est proche du nombre attendu
        if 0.9 * expected_pieces <= len(pieces) <= 1.1 * expected_pieces:
            logger.info(f"Nombre de pièces ({len(pieces)}) proche du nombre attendu ({expected_pieces})")
            logger.info("Utilisation du filtrage par aire uniquement")
            return final_area_verification(pieces, 2.5, expected_pieces)
        
        # Si beaucoup plus de pièces que prévu (probablement des faux positifs)
        if len(pieces) > 1.5 * expected_pieces:
            logger.info(f"Beaucoup de pièces détectées ({len(pieces)} vs {expected_pieces} attendues)")
            logger.info("Utilisation de la vérification stricte")
            return final_validation_check(
                pieces, expected_pieces, 1.8, 3.0, 'strict', True
            )
    
    # Cas 3: Par défaut, utiliser la vérification complète standard
    logger.info("Utilisation de la vérification complète standard")
    return final_validation_check(
        pieces, expected_pieces, 2.0, 4.0, validation_level, True
    )


def get_verification_statistics(pieces, original_count: int) -> Dict[str, Any]:
    """
    Collecte des statistiques sur le processus de vérification.
    
    Args:
        pieces: Liste des pièces après vérification
        original_count: Nombre initial de pièces avant vérification
        
    Returns:
        Dictionnaire des statistiques
    """
    stats = {}
    
    # Statistiques de base
    stats['original_count'] = original_count
    stats['verified_count'] = len(pieces)
    stats['removed_count'] = original_count - len(pieces)
    stats['removal_rate'] = (original_count - len(pieces)) / original_count if original_count > 0 else 0
    
    # Statistiques des pièces valides
    if pieces:
        areas = [p.features['area'] for p in pieces]
        scores = [p.validation_score for p in pieces if hasattr(p, 'validation_score')]
        
        stats['mean_area'] = np.mean(areas)
        stats['std_area'] = np.std(areas)
        stats['min_area'] = min(areas)
        stats['max_area'] = max(areas)
        
        if scores:
            stats['mean_score'] = np.mean(scores)
            stats['min_score'] = min(scores)
            stats['max_score'] = max(scores)
    
    # Collecter les raisons de rejet
    rejection_reasons = {}
    for piece in pieces:
        if not piece.is_valid and piece.validation_status:
            reason = piece.validation_status.split(':')[0]
            rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
    
    stats['rejection_reasons'] = rejection_reasons
    
    return stats

def final_validation_check(pieces, expected_pieces: Optional[int] = None,
                          area_threshold: float = 2.0,
                          aspect_threshold: float = 4.0,
                          validation_level: str = 'standard',
                          use_recovery: bool = True) -> List:
    """
    Effectue une vérification de validation finale pour filtrer les pièces
    invalides en utilisant plusieurs critères.
    
    Args:
        pieces: Liste des pièces de puzzle
        expected_pieces: Nombre attendu de pièces
        area_threshold: Seuil pour la vérification d'aire
        aspect_threshold: Seuil pour le ratio d'aspect
        validation_level: Niveau de validation ('permissive', 'standard', 'strict')
        use_recovery: Tenter de récupérer des pièces si trop sont filtrées
        
    Returns:
        Liste des pièces vérifiées
    """
    logger.info(f"Exécution de la vérification finale avec niveau {validation_level}")
    
    if not pieces:
        return []
    
    # Étape 1: Vérification de forme rapide
    shape_verified = fast_shape_verification(pieces, validation_level, expected_pieces)
    
    # Étape 2: Vérification par aire
    area_verified = final_area_verification(shape_verified, area_threshold, expected_pieces)
    
    # Si trop peu de pièces sont conservées et que la récupération est activée
    if expected_pieces and len(area_verified) < expected_pieces * 0.7 and use_recovery:
        logger.info(f"Trop peu de pièces ({len(area_verified)}/{expected_pieces}). Tentative de récupération.")
        
        # Essayer une vérification plus permissive
        return fast_shape_verification(pieces, 'permissive', expected_pieces)
    
    return area_verified