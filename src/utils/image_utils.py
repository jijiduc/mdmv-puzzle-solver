"""
Utilitaires de traitement d'image optimisés pour la segmentation des pièces de puzzle
avec un accent sur la performance et l'efficacité.
"""

import cv2
import numpy as np
import os
from typing import Tuple, Optional, Union, List, Dict, Any


def read_image(path: str) -> np.ndarray:
    """
    Lit une image à partir d'un fichier de manière efficace.
    
    Args:
        path: Chemin vers le fichier image
    
    Returns:
        Image sous forme de tableau NumPy
    
    Raises:
        FileNotFoundError: Si le fichier n'existe pas
        ValueError: Si l'image ne peut pas être lue
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fichier image non trouvé: {path}")
    
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Impossible de lire l'image {path}")
    
    return image


def save_image(image: np.ndarray, path: str) -> None:
    """
    Sauvegarde une image dans un fichier.
    
    Args:
        image: Image sous forme de tableau NumPy
        path: Chemin de destination
    """
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    
    cv2.imwrite(path, image)


def resize_image(image: np.ndarray, width: Optional[int] = None, height: Optional[int] = None, 
                scale: Optional[float] = None, max_dimension: Optional[int] = None) -> np.ndarray:
    """
    Redimensionne une image efficacement avec plusieurs options.
    
    Args:
        image: Image d'entrée
        width: Largeur cible
        height: Hauteur cible
        scale: Facteur d'échelle
        max_dimension: Dimension maximale (largeur ou hauteur) pour limitation de la taille
        
    Returns:
        Image redimensionnée
    """
    # Méthode rapide si aucun redimensionnement n'est nécessaire
    if (width is None and height is None and scale is None and max_dimension is None):
        return image
        
    h, w = image.shape[:2]
    
    # Option de dimension maximale pour limiter la taille des grandes images
    if max_dimension is not None:
        if max(h, w) > max_dimension:
            scale = max_dimension / max(h, w)
            return cv2.resize(image, None, fx=scale, fy=scale, 
                            interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        return image
        
    if scale is not None:
        # Redimensionnement par facteur d'échelle
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        return cv2.resize(image, None, fx=scale, fy=scale, interpolation=interp)
    
    # Calcul des dimensions cibles
    if width is None:
        aspect_ratio = w / h
        width = int(height * aspect_ratio)
    elif height is None:
        aspect_ratio = h / w
        height = int(width * aspect_ratio)
    
    # Choix de l'interpolation en fonction du redimensionnement
    is_downscale = (width * height) < (w * h)
    interp = cv2.INTER_AREA if is_downscale else cv2.INTER_LINEAR
    
    return cv2.resize(image, (width, height), interpolation=interp)


def fast_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convertit rapidement une image en niveaux de gris.
    
    Args:
        image: Image d'entrée
        
    Returns:
        Image en niveaux de gris
    """
    if len(image.shape) == 2:
        return image
    elif len(image.shape) == 3:
        # Utilise le canal vert si l'image est BGR/RGB (meilleure perception pour l'oeil humain)
        if image.shape[2] == 3:
            return image[:, :, 1]
        # Conversion standard pour les autres types
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def analyze_image(image: np.ndarray) -> Dict[str, Any]:
    """
    Analyse complète des caractéristiques d'une image pour optimiser le traitement.
    
    Args:
        image: Image d'entrée
        
    Returns:
        Dictionnaire avec les caractéristiques détaillées de l'image
    """
    # Utiliser d'abord l'analyse rapide
    quick_analysis = quick_analyze_image(image)
    
    # Conversion en niveaux de gris si nécessaire
    gray = fast_grayscale(image) if len(image.shape) == 3 else image.copy()
    
    # Analyses supplémentaires plus détaillées
    # 1. Mesures de netteté
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # 2. Détection des bords pour évaluer la densité
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.count_nonzero(edges) / edges.size
    
    # 3. Analyse de l'histogramme plus détaillée
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist.flatten() / hist.sum()
    
    # Identifier les pics et vallées significatifs
    peaks = []
    valleys = []
    for i in range(1, 255):
        if hist[i] > hist[i-1] and hist[i] > hist[i+1] and hist[i] > 0.01:
            peaks.append((i, hist[i]))
        if hist[i] < hist[i-1] and hist[i] < hist[i+1] and hist[i] < 0.01:
            valleys.append((i, hist[i]))
            
    # Estimation du bruit
    # Méthode simple : écart-type du filtre laplacien sur une région lisse
    # Trouver une région probablement lisse (faible variance locale)
    noise_level = 0
    try:
        block_size = 16
        h, w = gray.shape
        best_std = float('inf')
        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                block = gray[y:y+block_size, x:x+block_size]
                std = np.std(block)
                if std < best_std:
                    best_std = std
        
        noise_level = best_std
    except:
        # En cas d'erreur, utiliser une estimation de base
        noise_level = quick_analysis['std'] * 0.1
    
    # Combiner les analyses
    result = {
        **quick_analysis,  # Inclure l'analyse rapide
        'sharpness': laplacian_var,
        'edge_density': edge_density,
        'noise_level': noise_level,
        'histogram_peaks': len(peaks),
        'histogram_valleys': len(valleys),
        'dominant_peaks': sorted(peaks, key=lambda x: x[1], reverse=True)[:3] if peaks else []
    }
    
    # Détecter s'il s'agit d'une image de pièces de puzzle typique
    # (fond sombre, objets clairs bien séparés)
    result['is_typical_puzzle_image'] = (
        result['is_dark_background'] and 
        edge_density > 0.01 and edge_density < 0.1 and 
        len(peaks) >= 2
    )
    
    return result


def quick_analyze_image(image: np.ndarray) -> Dict[str, Any]:
    """
    Analyse rapide des caractéristiques de l'image pour la segmentation.
    
    Args:
        image: Image d'entrée
        
    Returns:
        Dictionnaire avec les caractéristiques de base
    """
    # Convertir en niveaux de gris pour l'analyse rapide
    gray = fast_grayscale(image)
    
    # Calcul des métriques rapides
    mean_val = np.mean(gray)
    std_val = np.std(gray)
    
    # Détection rapide du type d'arrière-plan
    dark_ratio = np.sum(gray < 50) / gray.size
    is_dark_background = dark_ratio > 0.3
    
    # Calcul simplifié du contraste
    p5 = np.percentile(gray, 5)
    p95 = np.percentile(gray, 95)
    contrast = (p95 - p5) / 255.0
    
    # Création d'une version sous-échantillonnée pour l'histogramme (plus rapide)
    small_gray = gray[::4, ::4]
    hist = cv2.calcHist([small_gray], [0], None, [64], [0, 256])
    hist = hist.flatten() / hist.sum()
    
    # Détermination rapide si l'histogramme est bimodal
    hist_peaks = []
    for i in range(1, 63):
        if hist[i] > hist[i-1] and hist[i] > hist[i+1] and hist[i] > 0.05:
            hist_peaks.append((i * 4, hist[i]))
    
    is_bimodal = len(hist_peaks) >= 2
    
    return {
        'mean': mean_val,
        'std': std_val,
        'contrast': contrast,
        'is_dark_background': is_dark_background,
        'is_bimodal': is_bimodal,
        'background_value': np.mean(gray[gray < 50]) if is_dark_background else np.mean(gray[gray > 200])
    }


def fast_enhance_contrast(image: np.ndarray) -> np.ndarray:
    """
    Amélioration rapide du contraste pour la segmentation.
    
    Args:
        image: Image en niveaux de gris
        
    Returns:
        Image avec contraste amélioré
    """
    # Analyse rapide
    analysis = quick_analyze_image(image)
    
    # Utilisation de CLAHE avec des paramètres adaptés
    clip_limit = 3.0 if analysis['contrast'] < 0.3 else 2.0
    
    # Application de CLAHE
    gray = fast_grayscale(image)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    return clahe.apply(gray)


def preprocess_image(image: np.ndarray, strategy: str = 'auto', expected_pieces: Optional[int] = None, 
                   params: Optional[Dict] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prétraite une image pour la détection des pièces de puzzle en utilisant différentes stratégies.
    
    Args:
        image: Image d'entrée
        strategy: Stratégie de prétraitement ('fast', 'adaptive', 'standard', ou 'auto')
        expected_pieces: Nombre attendu de pièces (pour optimisation)
        params: Paramètres spécifiques à la stratégie
        
    Returns:
        Tuple de (image prétraitée, image binaire, image des bords)
    """
    # Valeurs par défaut pour les paramètres
    if params is None:
        params = {}
        
    # Déterminer automatiquement la meilleure stratégie si 'auto'
    if strategy == 'auto':
        strategy = _select_preprocessing_strategy(image)
    
    # Appeler la fonction de prétraitement appropriée
    if strategy == 'fast':
        return _preprocess_fast(image, params)
    elif strategy == 'adaptive':
        return _preprocess_adaptive(image, expected_pieces, params)
    else:  # standard
        return _preprocess_standard(image, params)


def _select_preprocessing_strategy(image: np.ndarray) -> str:
    """
    Sélectionne automatiquement la meilleure stratégie de prétraitement.
    
    Args:
        image: Image d'entrée
        
    Returns:
        Stratégie de prétraitement ('fast', 'adaptive', ou 'standard')
    """
    # Analyse rapide
    analysis = quick_analyze_image(image)
    
    # Sélection basée sur les caractéristiques de l'image
    if analysis['contrast'] > 0.6 and analysis['is_dark_background']:
        return 'fast'  # Bon contraste et fond sombre -> stratégie rapide
    elif analysis['contrast'] < 0.4 or analysis['is_bimodal']:
        return 'adaptive'  # Faible contraste ou histogramme complexe -> adaptatif
    else:
        return 'standard'  # Cas par défaut -> standard


def _preprocess_fast(image: np.ndarray, params: Optional[Dict] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prétraitement rapide d'une image pour la détection des pièces.
    
    Args:
        image: Image couleur d'entrée
        params: Paramètres optionnels
    
    Returns:
        Tuple de (image prétraitée, image binaire, image des bords)
    """
    if params is None:
        params = {}
    
    # Conversion en niveaux de gris
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
    
    # Flou gaussien pour réduire le bruit
    blur_size = params.get('blur_size', 5)
    blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
    
    # Analyse rapide pour déterminer le type d'image
    analysis = quick_analyze_image(image)
    
    # Choisir la méthode de seuillage en fonction des caractéristiques de l'image
    if analysis['is_dark_background']:
        # Pour fond sombre - méthode directe rapide
        threshold_value = min(100, analysis['background_value'] + 30)
        _, binary = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY)
    else:
        # Pour d'autres types d'images, utiliser Otsu
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Opérations morphologiques pour nettoyer l'image binaire
    morph_kernel_size = params.get('morph_kernel_size', 5)
    morph_iterations = params.get('morph_iterations', 1)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=morph_iterations)
    
    # Détection des bords - optionnelle, pour compatibilité
    edges = cv2.Canny(cleaned, 50, 150)
    
    return blurred, cleaned, edges


def _preprocess_adaptive(image: np.ndarray, expected_pieces: Optional[int] = None, 
                       params: Optional[Dict] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prétraitement adaptatif avancé avec analyse multi-canal.
    
    Args:
        image: Image couleur d'entrée
        expected_pieces: Nombre attendu de pièces (pour optimisation)
        params: Paramètres optionnels
    
    Returns:
        Tuple de (image prétraitée, image binaire, image des bords)
    """
    if params is None:
        params = {}
    
    # Analyse de l'image
    analysis = analyze_image(image)
    
    # Utilisation du prétraitement multi-canal pour créer la meilleure représentation en niveaux de gris
    best_preprocessed, all_channels = multi_channel_preprocess(image)
    
    # Trouver les paramètres de seuillage optimaux
    threshold_params = find_optimal_threshold_parameters(best_preprocessed)
    
    # Appliquer le seuillage optimal
    if threshold_params['method'] == 'otsu':
        _, binary = cv2.threshold(best_preprocessed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif threshold_params['method'] == 'adaptive':
        binary = cv2.adaptiveThreshold(
            best_preprocessed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, threshold_params['block_size'], threshold_params['c']
        )
    elif threshold_params['method'] == 'hybrid':
        # Approche hybride
        _, otsu_binary = cv2.threshold(best_preprocessed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        adaptive_binary = cv2.adaptiveThreshold(
            best_preprocessed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 35, 10
        )
        binary = cv2.bitwise_or(otsu_binary, adaptive_binary)
    
    # Opérations morphologiques pour nettoyer l'image binaire
    kernel_size = params.get('morph_kernel_size', 5)
    iterations = params.get('morph_iterations', 2)
    
    cleaned = apply_morphology(
        binary, 
        operation=cv2.MORPH_CLOSE, 
        kernel_size=kernel_size,
        iterations=iterations
    )
    
    # Détection des bords - principalement pour la visualisation
    edges = cv2.Canny(best_preprocessed, 50, 150)
    
    return best_preprocessed, cleaned, edges


def _preprocess_standard(image: np.ndarray, params: Optional[Dict] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prétraitement standard pour la détection des pièces de puzzle.
    
    Args:
        image: Image couleur d'entrée
        params: Paramètres optionnels
    
    Returns:
        Tuple de (image prétraitée, image binaire, image des bords)
    """
    if params is None:
        params = {}
    
    # Conversion en niveaux de gris
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
    
    # Réduction du bruit
    blur_size = params.get('blur_size', 5)
    blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
    
    # Amélioration du contraste
    enhanced = fast_enhance_contrast(blurred)
    
    # Seuillage automatique
    use_auto_threshold = params.get('use_auto_threshold', True)
    
    if use_auto_threshold:
        binary, method, _ = compare_threshold_methods(
            enhanced,
            params.get('adaptive_block_size', 35),
            params.get('adaptive_c', 10)
        )
    else:
        # Par défaut: utiliser Otsu
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Opérations morphologiques pour nettoyer l'image binaire
    kernel_size = params.get('morph_kernel_size', 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Détection des bords
    edges = cv2.Canny(cleaned, 30, 200)
    
    return enhanced, cleaned, edges


def multi_channel_preprocess(image: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Prétraitement multi-canal pour extraire la meilleure représentation de l'image.
    Version corrigée pour éviter l'erreur 'dictionary changed size during iteration'.
    
    Args:
        image: Image couleur d'entrée
        
    Returns:
        Tuple de (meilleure image prétraitée, dictionnaire des canaux)
    """
    # Vérifier si l'image est déjà en niveaux de gris
    if len(image.shape) == 2:
        enhanced = fast_enhance_contrast(image)
        return enhanced, {'gray': enhanced}
    
    # Extraire différents canaux et espaces de couleur
    channels = {}
    
    # Canaux BGR originaux
    b, g, r = cv2.split(image)
    channels['blue'] = b
    channels['green'] = g
    channels['red'] = r
    
    # Espace HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    channels['hue'] = h
    channels['saturation'] = s
    channels['value'] = v
    
    # Espace LAB
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    channels['lightness'] = l
    channels['a'] = a
    channels['b'] = b
    
    # Grayscale standard
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    channels['gray'] = gray
    
    # Créer un nouveau dictionnaire pour les versions améliorées
    enhanced_channels = {}
    scores = {}
    
    # Traiter chaque canal original sans modifier le dictionnaire pendant l'itération
    for name, channel in channels.items():
        # Rehausser le contraste
        enhanced = fast_enhance_contrast(channel)
        enhanced_key = name + '_enhanced'
        enhanced_channels[enhanced_key] = enhanced
        
        # Calculer un score basé sur le contraste et la clarté des bords
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        edges = cv2.Canny(enhanced, 50, 150)
        
        # Score: mesure la qualité de la segmentation potentielle
        edge_count = np.count_nonzero(edges)
        edge_score = edge_count / edges.size
        contrast_score = quick_analyze_image(enhanced)['contrast']
        
        scores[enhanced_key] = edge_score * 0.6 + contrast_score * 0.4
    
    # Fusionner les dictionnaires de canaux originaux et améliorés
    all_channels = {**channels, **enhanced_channels}
    
    # Sélectionner le meilleur canal
    best_channel_name = max(scores, key=scores.get)
    best_channel = enhanced_channels[best_channel_name]
    
    return best_channel, all_channels


def find_optimal_threshold_parameters(gray_image: np.ndarray) -> Dict[str, Any]:
    """
    Trouve les paramètres de seuillage optimaux pour une image.
    
    Args:
        gray_image: Image en niveaux de gris
        
    Returns:
        Dictionnaire des paramètres optimaux
    """
    # Tester différentes méthodes de seuillage
    # 1. Otsu
    _, otsu_binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_score = evaluate_segmentation(otsu_binary)
    
    # 2. Seuillage adaptatif avec différentes tailles de bloc
    best_adaptive_score = 0
    best_block_size = 35
    best_c = 10
    
    # Tester quelques paramètres clés rapidement plutôt que toutes les combinaisons
    for block_size in [25, 35, 51, 75]:
        for c in [5, 10, 15]:
            adaptive_binary = cv2.adaptiveThreshold(
                gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, block_size, c
            )
            score = evaluate_segmentation(adaptive_binary)
            
            if score > best_adaptive_score:
                best_adaptive_score = score
                best_block_size = block_size
                best_c = c
    
    # 3. Approche hybride
    adaptive_binary = cv2.adaptiveThreshold(
        gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, best_block_size, best_c
    )
    hybrid_binary = cv2.bitwise_or(otsu_binary, adaptive_binary)
    hybrid_score = evaluate_segmentation(hybrid_binary)
    
    # Sélectionner la meilleure méthode
    scores = {
        'otsu': otsu_score,
        'adaptive': best_adaptive_score,
        'hybrid': hybrid_score
    }
    
    best_method = max(scores, key=scores.get)
    
    # Retourner les paramètres optimaux
    if best_method == 'otsu':
        return {'method': 'otsu'}
    elif best_method == 'adaptive':
        return {
            'method': 'adaptive',
            'block_size': best_block_size,
            'c': best_c
        }
    else:  # hybrid
        return {
            'method': 'hybrid',
            'block_size': best_block_size,
            'c': best_c
        }


def compare_threshold_methods(gray_image: np.ndarray, adaptive_block_size: int = 35, 
                            adaptive_c: int = 10) -> Tuple[np.ndarray, str, float]:
    """
    Compare différentes méthodes de seuillage et retourne la meilleure.
    
    Args:
        gray_image: Image en niveaux de gris
        adaptive_block_size: Taille de bloc pour le seuillage adaptatif
        adaptive_c: Constante pour le seuillage adaptatif
    
    Returns:
        Tuple de (image binaire, nom de la méthode, score)
    """
    # 1. Otsu
    _, otsu_binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_score = evaluate_segmentation(otsu_binary)
    
    # 2. Seuillage adaptatif
    adaptive_binary = cv2.adaptiveThreshold(
        gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, adaptive_block_size, adaptive_c
    )
    adaptive_score = evaluate_segmentation(adaptive_binary)
    
    # 3. Approche hybride
    hybrid_binary = cv2.bitwise_or(otsu_binary, adaptive_binary)
    hybrid_score = evaluate_segmentation(hybrid_binary)
    
    # Sélectionner la meilleure méthode
    methods = {
        'otsu': (otsu_binary, otsu_score),
        'adaptive': (adaptive_binary, adaptive_score),
        'hybrid': (hybrid_binary, hybrid_score)
    }
    
    best_method = max(methods, key=lambda k: methods[k][1])
    best_binary, best_score = methods[best_method]
    
    return best_binary, best_method, best_score


def apply_morphology(binary: np.ndarray, operation: int = cv2.MORPH_CLOSE, 
                   kernel_size: int = 5, iterations: int = 1) -> np.ndarray:
    """
    Applique une opération morphologique à une image binaire.
    
    Args:
        binary: Image binaire d'entrée
        operation: Opération morphologique (cv2.MORPH_*)
        kernel_size: Taille du noyau
        iterations: Nombre d'itérations
    
    Returns:
        Image binaire traitée
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.morphologyEx(binary, operation, kernel, iterations=iterations)


def detect_edges(image: np.ndarray, threshold1: int = 50, threshold2: int = 150) -> np.ndarray:
    """
    Détecte les bords dans une image.
    
    Args:
        image: Image en niveaux de gris
        threshold1: Premier seuil pour l'algorithme de Canny
        threshold2: Second seuil pour l'algorithme de Canny
    
    Returns:
        Image des bords détectés
    """
    # Assurer que l'image est en niveaux de gris
    gray = fast_grayscale(image)
    return cv2.Canny(gray, threshold1, threshold2)


def optimize_for_segmentation(image: np.ndarray, downscale_factor: float = 1.0) -> np.ndarray:
    """
    Prétraitement optimisé pour la segmentation des pièces de puzzle.
    
    Args:
        image: Image d'entrée
        downscale_factor: Facteur de réduction d'échelle pour accélérer le traitement
        
    Returns:
        Image prétraitée pour segmentation
    """
    # Redimensionnement pour accélérer le traitement si nécessaire
    if downscale_factor != 1.0 and downscale_factor > 0:
        processed = resize_image(image, scale=downscale_factor)
    else:
        processed = image.copy()
    
    # Conversion en niveaux de gris rapide
    gray = fast_grayscale(processed)
    
    # Analyse rapide
    analysis = quick_analyze_image(gray)
    
    # Appliquer une réduction de bruit
    if analysis['std'] > 15:  # Image bruitée
        blur_size = 5
    else:
        blur_size = 3
        
    blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
    
    # Amélioration du contraste seulement si nécessaire
    if analysis['contrast'] < 0.5:
        # CLAHE est coûteux, ne l'utiliser que si nécessaire
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)
    else:
        enhanced = blurred
        
    # Optimisation pour les images à fond sombre (cas typique des puzzles)
    if analysis['is_dark_background']:
        # Prétraitement optimisé pour fond sombre
        # Suppression rapide du fond
        threshold_value = min(120, analysis['background_value'] + 30)
        _, mask = cv2.threshold(enhanced, threshold_value, 255, cv2.THRESH_BINARY)
        return mask
    else:
        # Pour les autres types d'images, utiliser Otsu (rapide et efficace)
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary


def adaptive_preprocess(image: np.ndarray) -> np.ndarray:
    """
    Prétraitement adaptatif qui sélectionne la meilleure méthode pour chaque image.
    
    Args:
        image: Image d'entrée
        
    Returns:
        Image binaire prétraitée
    """
    # Analyse de l'image pour déterminer ses caractéristiques
    analysis = analyze_image(image)
    
    # Sélection de la méthode appropriée en fonction des caractéristiques
    if analysis['is_dark_background'] and analysis['contrast'] > 0.4:
        # Fond sombre avec bon contraste - méthode simple et rapide
        gray = fast_grayscale(image)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        threshold_value = min(100, analysis['background_value'] + 30)
        _, binary = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY)
    elif analysis['contrast'] < 0.3:
        # Faible contraste - amélioration et seuillage adaptatif
        gray = fast_grayscale(image)
        enhanced = fast_enhance_contrast(gray)
        binary = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 35, 10
        )
    else:
        # Cas général - Otsu
        gray = fast_grayscale(image)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Nettoyage morphologique
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    return cleaned


def fast_adaptive_threshold(image: np.ndarray) -> np.ndarray:
    """
    Seuillage adaptatif optimisé pour la performance.
    
    Args:
        image: Image en niveaux de gris
        
    Returns:
        Image binaire
    """
    # Analyse rapide
    analysis = quick_analyze_image(image)
    
    # Pour les images à faible contraste, un seuillage adaptatif est meilleur
    if analysis['contrast'] < 0.4:
        # Taillez les paramètres pour la vitesse - bloc plus grand est plus rapide
        block_size = 35  # Une valeur plus élevée accélère le traitement
        c = 10
        return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, block_size, c)
    else:
        # Otsu est beaucoup plus rapide et fonctionne bien avec un bon contraste
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

def adaptive_threshold(image: np.ndarray, block_size: int = 35, c: int = 10) -> np.ndarray:
    """
    Applique un seuillage adaptatif à une image en niveaux de gris.
    
    Args:
        image: Image en niveaux de gris
        block_size: Taille du bloc pour le calcul du seuil adaptatif
        c: Constante soustraite de la moyenne
        
    Returns:
        Image binaire seuillée
    """
    # S'assurer que block_size est impair
    if block_size % 2 == 0:
        block_size += 1
        
    return cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, block_size, c
    )

def clean_binary_image(binary: np.ndarray, min_size: int = 500) -> np.ndarray:
    """
    Nettoie une image binaire en éliminant les petits artefacts.
    Optimisé pour la performance.
    
    Args:
        binary: Image binaire
        min_size: Taille minimale des composantes à conserver
        
    Returns:
        Image binaire nettoyée
    """
    # Application d'une opération morphologique pour fermer les petits trous
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Suppression des petits artefacts - optimisé avec l'analyse de composantes connectées
    # Cette approche est beaucoup plus rapide que de traiter chaque contour individuellement
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    
    # Créer un masque pour les composantes valides
    mask = np.zeros_like(cleaned)
    
    # Optimisation: utiliser un accès direct aux statistiques et une opération vectorisée
    # pour créer le masque des composantes valides (sauf l'arrière-plan à l'indice 0)
    valid_components = np.where(stats[1:, cv2.CC_STAT_AREA] >= min_size)[0] + 1
    
    # Créer le masque avec une seule boucle sur les composantes valides
    for label in valid_components:
        mask[labels == label] = 255
    
    return mask


def detect_puzzle_pieces(image: np.ndarray, expected_min_size: int = 1000) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Pipeline optimisé en performance pour détecter les pièces de puzzle.
    
    Args:
        image: Image d'entrée
        expected_min_size: Taille minimale attendue pour une pièce de puzzle
        
    Returns:
        Tuple contenant (image binaire des pièces, liste des contours)
    """
    # 1. Prétraitement optimisé
    # Utiliser une échelle réduite pour l'analyse rapide, si l'image est grande
    h, w = image.shape[:2]
    if max(h, w) > 2000:
        analysis_scale = 0.5
    else:
        analysis_scale = 1.0
        
    # Analyse rapide
    small_image = resize_image(image, scale=analysis_scale) if analysis_scale < 1.0 else image
    analysis = quick_analyze_image(small_image)
    
    # 2. Pipeline de traitement selon le type d'image
    if analysis['is_dark_background']:
        # Optimisation pour fond sombre (cas typique des puzzles)
        gray = fast_grayscale(image)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Seuillage simple et rapide pour fond sombre
        threshold_value = min(100, analysis['background_value'] + 25)
        _, binary = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY)
    else:
        # Pour d'autres cas, utiliser le pipeline général mais optimisé
        preprocessed = optimize_for_segmentation(image)
        binary = fast_adaptive_threshold(preprocessed)
    
    # 3. Nettoyage et extraction des contours
    cleaned = clean_binary_image(binary, min_size=expected_min_size)
    
    # Détection des contours - utilisation de CHAIN_APPROX_SIMPLE pour la performance
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filtrage direct par aire
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= expected_min_size]
    
    return cleaned, filtered_contours


def create_piece_mask(image: np.ndarray, contour: np.ndarray) -> np.ndarray:
    """
    Crée un masque pour une pièce de puzzle à partir de son contour.
    Optimisé pour la performance.
    
    Args:
        image: Image originale
        contour: Contour de la pièce
        
    Returns:
        Masque binaire de la pièce
    """
    # Création d'un masque rapide
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], 0, 255, -1)
    return mask


def extract_piece_image(image: np.ndarray, contour: np.ndarray, padding: int = 5) -> np.ndarray:
    """
    Extrait l'image d'une pièce de puzzle avec un fond propre.
    Optimisé pour la performance.
    
    Args:
        image: Image originale
        contour: Contour de la pièce
        padding: Marge autour de la pièce
        
    Returns:
        Image de la pièce sur fond blanc
    """
    # Obtenir le rectangle englobant avec marge
    x, y, w, h = cv2.boundingRect(contour)
    x_min = max(0, x - padding)
    y_min = max(0, y - padding)
    x_max = min(image.shape[1], x + w + padding)
    y_max = min(image.shape[0], y + h + padding)
    
    # Créer un masque
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], 0, 255, -1)
    mask_roi = mask[y_min:y_max, x_min:x_max]
    
    # Extraire la région d'intérêt
    roi = image[y_min:y_max, x_min:x_max]
    
    # Création optimisée de l'image avec fond blanc
    result = np.ones_like(roi) * 255
    mask_3ch = cv2.merge([mask_roi, mask_roi, mask_roi]) if len(image.shape) == 3 else mask_roi
    
    # Application du masque (opération vectorisée pour la vitesse)
    result = np.where(mask_3ch > 0, roi, result)
    
    return result


def evaluate_segmentation(binary: np.ndarray, min_area: int = 1000) -> float:
    """
    Évalue rapidement la qualité d'une segmentation pour les pièces de puzzle.
    
    Args:
        binary: Image binaire segmentée
        min_area: Aire minimale pour les pièces valides
        
    Returns:
        Score de qualité (0-1)
    """
    # Détection rapide des contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Pas de contours est mauvais
    if not contours:
        return 0.0
    
    # Calcul vectorisé des aires de tous les contours
    areas = np.array([cv2.contourArea(cnt) for cnt in contours])
    
    # Filtrage vectorisé par aire
    valid_mask = areas >= min_area
    valid_areas = areas[valid_mask]
    
    # Pas de contours valides est mauvais
    if len(valid_areas) == 0:
        return 0.0
    
    # Métriques vectorisées pour la vitesse
    count_score = min(len(valid_areas) / 30.0, 1.0)  # Idéalement pas trop de pièces
    
    # Cohérence des aires (les pièces de puzzle ont généralement des tailles similaires)
    mean_area = np.mean(valid_areas)
    std_area = np.std(valid_areas)
    area_consistency = max(0, 1.0 - (std_area / (mean_area + 1e-5)) / 0.5)
    
    # Score combiné simplifié
    score = 0.5 * count_score + 0.5 * area_consistency
    
    return score



def find_best_threshold(preprocessed: np.ndarray) -> np.ndarray:
    """
    Trouve rapidement le meilleur seuillage pour la segmentation.
    
    Args:
        preprocessed: Image prétraitée
        
    Returns:
        Image binaire avec le meilleur seuillage
    """
    # Test rapide de deux méthodes principales
    
    # 1. Otsu (rapide)
    _, otsu_binary = cv2.threshold(preprocessed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_score = evaluate_segmentation(otsu_binary)
    
    # 2. Seuillage adaptatif (un peu plus lent)
    # Seulement si Otsu n'est pas satisfaisant
    if otsu_score < 0.7:
        adaptive_binary = cv2.adaptiveThreshold(preprocessed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                            cv2.THRESH_BINARY, 35, 10)
        adaptive_score = evaluate_segmentation(adaptive_binary)
        
        if adaptive_score > otsu_score:
            return adaptive_binary
    
    return otsu_binary