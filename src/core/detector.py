"""
Détecteur optimisé de pièces de puzzle avec focus sur la segmentation
et la performance.
"""

from src.core.piece import PuzzlePiece
from src.config.settings import Config
from src.utils.contour_utils import (
    find_contours, filter_contours, calculate_contour_features,
    validate_shape_as_puzzle_piece, cluster_contours, optimize_contours
)
from src.utils.image_utils import (
    preprocess_image, adaptive_threshold, apply_morphology, detect_edges,
    multi_channel_preprocess, analyze_image, adaptive_preprocess,
    find_optimal_threshold_parameters, compare_threshold_methods,
    optimize_for_segmentation, fast_adaptive_threshold, clean_binary_image,
    detect_puzzle_pieces
)
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import os
import sys
import time
import logging
from multiprocessing import Pool, cpu_count
import math
from src.utils.cache_utils import PipelineCache

# Ajout du répertoire parent au chemin pour permettre les imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class PuzzleDetector:
    """
    Détecteur optimisé pour la segmentation des pièces de puzzle.
    Se concentre uniquement sur la détection des pièces sans analyser leurs caractéristiques internes.
    """

    def __init__(self, config: Config = None, pipeline_cache: Optional[PipelineCache] = None):
        """
        Initialise le détecteur avec la configuration fournie.

        Args:
            config: Paramètres de configuration
            pipeline_cache: Cache du pipeline (optionnel)
        """
        self.config = config or Config()
        self.logger = self._setup_logger()
        self.debug_images = {}
        self.pipeline_cache = pipeline_cache

        # Suivi des performances de détection
        self.detection_stats = {
            'params': {},
            'results': {},
            'timing': {}
        }

    def _setup_logger(self) -> logging.Logger:
        """Configure un logger pour le détecteur"""
        logger = logging.getLogger(__name__)
        return logger

    def save_debug_image(self, image: np.ndarray, filename: str) -> None:
        """
        Sauvegarde une image pour le débogage et la conserve en mémoire.

        Args:
            image: Image à sauvegarder
            filename: Nom du fichier
        """
        if not self.config.DEBUG:
            return

        # Garder une copie en mémoire pour analyse
        self.debug_images[filename.split('.')[0]] = image.copy()

        # Sauvegarder sur disque si nécessaire
        os.makedirs(self.config.DEBUG_DIR, exist_ok=True)
        path = os.path.join(self.config.DEBUG_DIR, filename)
        cv2.imwrite(path, image)
        self.logger.debug(f"Image de débogage sauvegardée: {path}")

    def preprocess_fast(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prétraitement rapide d'une image pour la détection des pièces.
        Version optimisée.
        
        Args:
            image: Image couleur d'entrée
        
        Returns:
            Tuple de (image prétraitée, image binaire, image des bords)
        """
        start_time = time.time()
        self.logger.info("Prétraitement rapide de l'image...")
        
        # Conversion en niveaux de gris avec masque de bits au lieu de np.copy
        # C'est plus rapide que cv2.cvtColor pour les conversions simples
        gray = image[:,:,1] if len(image.shape) == 3 else image.copy()
        self.save_debug_image(gray, "01_gray.jpg")
        
        # Flou gaussien pour réduire le bruit
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        self.save_debug_image(blurred, "02_blurred.jpg")
        
        # Analyse rapide pour déterminer le type d'image
        analysis = analyze_image(image)
        
        # Choisir la méthode de seuillage en fonction des caractéristiques de l'image
        if analysis['is_dark_background']:
            # Pour fond sombre - méthode directe rapide
            threshold_value = min(100, analysis['background_value'] + 30)
            _, binary = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY)
        else:
            # Pour d'autres types d'images, utiliser Otsu
            _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        self.save_debug_image(binary, "03_binary.jpg")
        
        # Optimisation: utiliser des versions précompilées des noyaux morphologiques
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        self.save_debug_image(cleaned, "04_cleaned.jpg")
        
        # Détection des bords - optionnelle, pour compatibilité
        # Utiliser une seule passe de Canny au lieu de plusieurs seuils
        edges = cv2.Canny(cleaned, 50, 150)
        self.save_debug_image(edges, "05_edges.jpg")
        
        self.detection_stats['timing']['preprocessing'] = time.time() - start_time
        
        return blurred, cleaned, edges

    def preprocess_adaptive(self, image: np.ndarray, expected_pieces: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prétraitement adaptatif avancé avec analyse multi-canal.

        Args:
            image: Image couleur d'entrée
            expected_pieces: Nombre attendu de pièces (pour optimisation)

        Returns:
            Tuple de (image prétraitée, image binaire, image des bords)
        """
        start_time = time.time()
        self.logger.info("Utilisation du prétraitement adaptatif avancé...")

        # Analyse de l'image
        analysis = analyze_image(image)
        self.logger.info(
            f"Analyse d'image: contraste={analysis['contrast']:.2f}, fond={analysis['background_value']:.2f}")

        # Utilisation du prétraitement multi-canal pour créer la meilleure représentation en niveaux de gris
        best_preprocessed, all_channels = multi_channel_preprocess(image)
        self.save_debug_image(best_preprocessed, "01_best_preprocessed.jpg")

        # Sauvegarder des images de diagnostic des différents canaux
        for name, channel in all_channels.items():
            if len(channel.shape) == 2:  # Uniquement les canaux en niveaux de gris
                self.save_debug_image(channel, f"01_channel_{name}.jpg")

        # Trouver les paramètres de seuillage optimaux
        threshold_params = find_optimal_threshold_parameters(best_preprocessed)
        self.logger.info(
            f"Méthode de seuillage sélectionnée: {threshold_params['method']}")

        # Appliquer le seuillage optimal
        if threshold_params['method'] == 'otsu':
            _, binary = cv2.threshold(
                best_preprocessed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif threshold_params['method'] == 'adaptive':
            binary = cv2.adaptiveThreshold(
                best_preprocessed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, threshold_params['block_size'], threshold_params['c']
            )
        elif threshold_params['method'] == 'hybrid':
            # Approche hybride
            _, otsu_binary = cv2.threshold(
                best_preprocessed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            adaptive_binary = cv2.adaptiveThreshold(
                best_preprocessed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 35, 10
            )
            binary = cv2.bitwise_or(otsu_binary, adaptive_binary)

        self.save_debug_image(binary, "02_binary.jpg")

        # Opérations morphologiques pour nettoyer l'image binaire
        kernel_size = max(
            3, min(7, int(min(image.shape[0], image.shape[1]) / 500)))
        cleaned = apply_morphology(
            binary,
            operation=cv2.MORPH_CLOSE,
            kernel_size=kernel_size,
            iterations=2
        )
        self.save_debug_image(cleaned, "03_cleaned.jpg")

        # Détection des bords - principalement pour la visualisation
        edges = cv2.Canny(best_preprocessed, 50, 150)
        self.save_debug_image(edges, "04_edges.jpg")

        self.detection_stats['timing']['preprocessing'] = time.time(
        ) - start_time

        return best_preprocessed, cleaned, edges

    def preprocess(self, image: np.ndarray, expected_pieces: Optional[int] = None, 
             fast_mode: bool = False, image_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prétraite une image pour la détection des pièces de puzzle.
        Sélectionne automatiquement la meilleure méthode selon la configuration.

        Args:
            image: Image couleur d'entrée
            expected_pieces: Nombre attendu de pièces (optionnel)
            fast_mode: Utiliser le mode rapide de prétraitement
            image_path: Chemin de l'image pour le cache (optionnel)

        Returns:
            Tuple de (image prétraitée, image binaire, image des bords)
        """
        # Vérifier le cache si le chemin de l'image est fourni
        if self.pipeline_cache and image_path:
            from src.utils.cache_utils import cache_preprocessing
            
            # Essayer de récupérer les résultats depuis le cache
            adaptive = hasattr(self.config.preprocessing, 'USE_ADAPTIVE') and self.config.preprocessing.USE_ADAPTIVE
            cached_result = cache_preprocessing(
                self.pipeline_cache, 
                image_path, 
                fast_mode, 
                adaptive,
                expected_pieces
            )
            
            if cached_result is not None:
                self.logger.info("Résultats de prétraitement récupérés depuis le cache")
                # Sauvegarder les images de débogage
                if hasattr(self, 'save_debug_image'):
                    self.save_debug_image(cached_result[0], "01_preprocessed.jpg")
                    self.save_debug_image(cached_result[1], "02_binary.jpg")
                    self.save_debug_image(cached_result[2], "03_edges.jpg")
                return cached_result

        # Optimisation pour les cas triviaux
        if fast_mode:
            result = self.preprocess_fast(image)
            
        # Choix du pipeline à utiliser en fonction de la configuration
        elif hasattr(self.config.preprocessing, 'USE_ADAPTIVE') and self.config.preprocessing.USE_ADAPTIVE:
            result = self.preprocess_adaptive(image, expected_pieces)
            
        else:
            # Pipeline original simplifié
            self.logger.info("Prétraitement de l'image avec pipeline standard...")

            h, w = image.shape[:2]
            self.logger.info(f"Dimensions originales de l'image: {w}x{h}")

            # Prétraitement simplifié
            preprocessed, binary_preproc, edges_preproc = preprocess_image(image)
            self.save_debug_image(preprocessed, "01_preprocessed.jpg")

            # Seuillage automatique
            if self.config.preprocessing.USE_AUTO_THRESHOLD:
                binary, method, _ = compare_threshold_methods(
                    preprocessed,
                    self.config.preprocessing.ADAPTIVE_BLOCK_SIZE,
                    self.config.preprocessing.ADAPTIVE_C
                )
                self.logger.info(f"Méthode de seuillage sélectionnée: {method}")
                self.save_debug_image(binary, f"02_{method}_binary.jpg")
            else:
                # Par défaut: utiliser Otsu
                _, binary = cv2.threshold(
                    preprocessed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                self.save_debug_image(binary, "02_binary.jpg")

            # Opérations morphologiques pour nettoyer l'image binaire
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            self.save_debug_image(cleaned, "03_cleaned.jpg")

            # Détection des bords
            edges = cv2.Canny(cleaned, 30, 200)
            self.save_debug_image(edges, "04_edges.jpg")

            result = (preprocessed, cleaned, edges)

        # Mettre en cache les résultats si le cache est activé
        if self.pipeline_cache and image_path:
            from src.utils.cache_utils import save_preprocessing_to_cache
            
            adaptive = hasattr(self.config.preprocessing, 'USE_ADAPTIVE') and self.config.preprocessing.USE_ADAPTIVE
            save_preprocessing_to_cache(
                self.pipeline_cache,
                image_path,
                fast_mode,
                adaptive,
                expected_pieces,
                result
            )

        return result

    def optimize_detection_parameters(self, binary_image: np.ndarray, original_image: np.ndarray,
                                      expected_pieces: Optional[int] = None) -> Dict[str, Any]:
        """
        Optimise les paramètres de détection en fonction de l'image.

        Args:
            binary_image: Image binaire d'entrée
            original_image: Image originale
            expected_pieces: Nombre attendu de pièces

        Returns:
            Dictionnaire des paramètres optimaux
        """
        # Analyse de l'image
        analysis = analyze_image(original_image)

        # Calcul de l'aire maximale basée sur la taille de l'image
        img_area = original_image.shape[0] * original_image.shape[1]
        max_area = self.config.contour.MAX_AREA_RATIO * img_area

        # Estimation de l'aire minimale en fonction de l'image
        if expected_pieces:
            # Estimation basée sur le nombre attendu de pièces
            estimated_min_area = img_area / \
                (expected_pieces * 2.5)  # Facteur de correction
            min_area = max(self.config.contour.MIN_AREA,
                           estimated_min_area * 0.5)
        else:
            # Estimation basée sur la taille de l'image
            min_area = max(self.config.contour.MIN_AREA, img_area / 1000)

        # Optimisation du seuil de solidité en fonction du contraste
        if analysis['contrast'] < 0.4:
            # Pour les images à faible contraste, être plus permissif
            solidity_min = 0.6
        else:
            solidity_min = 0.7

        # Optimisation des paramètres de filtrage
        params = {
            'min_area': min_area,
            'max_area': max_area,
            'min_perimeter': self.config.contour.MIN_PERIMETER,
            'solidity_range': (solidity_min, 0.99),
            'aspect_ratio_range': self.config.contour.ASPECT_RATIO_RANGE,
            'use_statistical_filtering': True,
            'expected_piece_count': expected_pieces
        }

        self.logger.info(
            f"Paramètres optimisés: min_area={min_area:.0f}, solidity_min={solidity_min}")

        return params

    def detect_contours(self, binary_image: np.ndarray, original_image: np.ndarray,
                  expected_pieces: Optional[int] = None, image_path: Optional[str] = None) -> List[np.ndarray]:
        """
        Détecte les contours des pièces de puzzle dans une image binaire.

        Args:
            binary_image: Image binaire d'entrée
            original_image: Image originale (pour filtrage basé sur la taille)
            expected_pieces: Nombre attendu de pièces
            image_path: Chemin de l'image pour le cache (optionnel)

        Returns:
            Liste des contours détectés
        """
        start_time = time.time()
        self.logger.info("Détection des contours avec approche optimisée...")
        
        # Vérifier le cache si activé
        if self.pipeline_cache and image_path:
            from src.utils.cache_utils import cache_contours
            
            # Paramètres pour la clé de cache
            cache_params = {
                'expected_pieces': expected_pieces,
                'min_area': self.config.contour.MIN_AREA if hasattr(self.config.contour, 'MIN_AREA') else 1000,
                'max_area_ratio': self.config.contour.MAX_AREA_RATIO if hasattr(self.config.contour, 'MAX_AREA_RATIO') else 0.3,
                'image_size': original_image.shape[:2]
            }
            
            # Essayer de récupérer depuis le cache
            cached_contours = cache_contours(
                self.pipeline_cache,
                image_path,
                binary_image,
                cache_params['min_area'],
                cache_params
            )
            
            if cached_contours is not None:
                self.logger.info(f"Contours récupérés depuis le cache: {len(cached_contours)} contours")
                
                # Mesurer le temps d'accès au cache
                elapsed = time.time() - start_time
                self.detection_stats['timing']['contour_detection'] = elapsed
                self.detection_stats['timing']['cache_hit'] = True
                
                # Créer la visualisation des contours pour la cohérence
                if cached_contours:
                    contour_vis = original_image.copy()
                    cv2.drawContours(contour_vis, cached_contours, -1, (0, 255, 0), 2)
                    self.save_debug_image(contour_vis, "06_contours.jpg")
                
                return cached_contours

        # Si pas de cache ou cache miss, continuer avec la détection normale
        self.detection_stats['timing']['cache_hit'] = False
        
        # Optimisation: utiliser directement detect_puzzle_pieces si disponible
        if hasattr(self, 'quick_detect') and self.quick_detect:
            # Utilisation de la fonction optimisée intégrée
            min_size = self.config.contour.MIN_AREA if hasattr(self.config.contour, 'MIN_AREA') else 1000
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
            
            # Mettre en cache les contours si activé
            if self.pipeline_cache and image_path:
                from src.utils.cache_utils import save_contours_to_cache
                
                cache_params = {
                    'expected_pieces': expected_pieces,
                    'min_area': min_size,
                    'max_area_ratio': self.config.contour.MAX_AREA_RATIO if hasattr(self.config.contour, 'MAX_AREA_RATIO') else 0.3,
                    'image_size': original_image.shape[:2]
                }
                
                save_contours_to_cache(
                    self.pipeline_cache,
                    image_path,
                    binary_image,
                    min_size,
                    cache_params,
                    contours
                )
            
            return contours
        
        # Approche standard: trouver les contours puis les filtrer
        # Optimisation des paramètres
        params = self.optimize_detection_parameters(binary_image, original_image, expected_pieces)
        
        # Détection initiale des contours
        contours = find_contours(binary_image)
        self.logger.info(f"Trouvé {len(contours)} contours initiaux")
        
        # Filtrage des contours
        filtered_contours = filter_contours(contours, **params)
        
        # Si trop peu de contours et nombre attendu fourni, essayer une récupération
        if expected_pieces and len(filtered_contours) < expected_pieces * 0.7:
            self.logger.info(f"Récupération: trouvé {len(filtered_contours)}/{expected_pieces} pièces attendues")
            
            # Essayer avec des paramètres plus permissifs
            recovery_params = params.copy()
            recovery_params['min_area'] *= 0.7
            recovery_params['solidity_range'] = (0.5, 0.99)
            
            recovery_contours = filter_contours(contours, **recovery_params)
            
            if len(recovery_contours) > len(filtered_contours):
                filtered_contours = recovery_contours
                self.logger.info(f"Récupération réussie: {len(filtered_contours)} contours")
        
        # Optimisation finale des contours
        optimized_contours = optimize_contours(filtered_contours, min_area=params['min_area'])
        
        # Création de la visualisation des contours
        if optimized_contours:
            contour_vis = original_image.copy()
            cv2.drawContours(contour_vis, optimized_contours, -1, (0, 255, 0), 2)
            self.save_debug_image(contour_vis, "06_contours.jpg")
        
        elapsed = time.time() - start_time
        self.detection_stats['timing']['contour_detection'] = elapsed
        self.logger.info(f"Détection des contours terminée en {elapsed:.3f}s, {len(optimized_contours)} contours filtrés")
        
        # Mettre en cache les contours si le cache est activé
        if self.pipeline_cache and image_path:
            from src.utils.cache_utils import save_contours_to_cache
            
            # Paramètres pour la clé de cache
            cache_params = {
                'expected_pieces': expected_pieces,
                'min_area': params['min_area'],
                'max_area_ratio': self.config.contour.MAX_AREA_RATIO if hasattr(self.config.contour, 'MAX_AREA_RATIO') else 0.3,
                'image_size': original_image.shape[:2]
            }
            
            # Mettre en cache les contours
            save_contours_to_cache(
                self.pipeline_cache,
                image_path,
                binary_image,
                params['min_area'],
                cache_params,
                optimized_contours
            )
        
        return optimized_contours

    def recover_missed_pieces(self, binary_image, detected_contours, original_image, expected_pieces=None):
        """Tente de récupérer les pièces manquantes de la détection initiale."""
        # Ignorer si le nombre attendu est satisfait
        if not expected_pieces or len(detected_contours) >= expected_pieces:
            return []

        start_time = time.time()
        self.logger.info(
            f"Tentative de récupération: {len(detected_contours)}/{expected_pieces} pièces")

        # Créer un masque d'exclusion des pièces déjà détectées
        exclusion_mask = self._create_exclusion_mask(
            binary_image, detected_contours)

        # Appliquer le masque à l'image binaire
        masked_binary = cv2.bitwise_and(
            binary_image, binary_image, mask=exclusion_mask)
        self.save_debug_image(masked_binary, "07_masked_binary.jpg")

        # Trouver et filtrer les contours dans la zone masquée
        recovery_contours = self._find_recovery_contours(
            masked_binary, original_image)

        # Validation et déduplication
        valid_recovered = self._validate_and_deduplicate(
            recovery_contours, detected_contours)

        # Visualisation des résultats
        if valid_recovered:
            self._visualize_recovery(
                original_image, detected_contours, valid_recovered)

        elapsed = time.time() - start_time
        self.detection_stats['timing']['contour_recovery'] = elapsed
        self.logger.info(
            f"Récupération: {len(valid_recovered)} pièces en {elapsed:.3f}s")

        return valid_recovered

    def _create_exclusion_mask(self, binary_image, contours):
        """Crée un masque qui exclut les régions des contours détectés"""
        mask = np.ones_like(binary_image)

        for contour in contours:
            # Créer un contour légèrement agrandi
            x, y, w, h = cv2.boundingRect(contour)
            padding = 15
            x_min = max(0, x - padding)
            y_min = max(0, y - padding)
            x_max = min(binary_image.shape[1], x + w + padding)
            y_max = min(binary_image.shape[0], y + h + padding)

            # Bloquer cette région dans le masque
            mask[y_min:y_max, x_min:x_max] = 0

        return mask

    def _find_recovery_contours(self, masked_binary, original_image):
        """Trouve les contours dans l'image binaire masquée avec des paramètres plus permissifs"""
        # Paramètres de récupération plus permissifs
        img_area = original_image.shape[0] * original_image.shape[1]

        recovery_params = {
            'min_area': self.config.contour.MIN_AREA * 0.7,
            'max_area': self.config.contour.MAX_AREA_RATIO * img_area,
            'min_perimeter': self.config.contour.MIN_PERIMETER * 0.8,
            'solidity_range': (0.5, 0.99),
            'aspect_ratio_range': (0.2, 5.0)
        }

        # Trouver et filtrer les contours
        mask_contours = find_contours(
            masked_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return filter_contours(mask_contours, **recovery_params)

    def _validate_and_deduplicate(self, recovery_contours, existing_contours):
        """Valide chaque contour et élimine les doublons"""
        valid_recoveries = [
            c for c in recovery_contours if validate_shape_as_puzzle_piece(c)]
        final_recovered = []

        for new_contour in valid_recoveries:
            # Vérifier s'il s'agit d'un doublon
            if not any(self._contours_match(new_contour, existing)
                       for existing in existing_contours + final_recovered):
                final_recovered.append(new_contour)

        return final_recovered

    def _visualize_recovery(self, original_image, original_contours, recovered_contours):
        """Crée une visualisation des contours originaux et récupérés"""
        recovery_vis = original_image.copy()

        # Contours originaux en vert
        cv2.drawContours(recovery_vis, original_contours, -1, (0, 255, 0), 2)

        # Contours récupérés en rouge
        cv2.drawContours(recovery_vis, recovered_contours, -1, (0, 0, 255), 2)

        self.save_debug_image(recovery_vis, "08_recovered_contours.jpg")

    def _process_contour(self, args: Tuple[np.ndarray, np.ndarray, int]) -> Optional[PuzzlePiece]:
        """
        Traite un seul contour pour créer un objet PuzzlePiece.

        Args:
            args: Tuple de (contour, image, indice)

        Returns:
            Objet PuzzlePiece ou None si invalide
        """
        contour, image, idx = args
        try:
            piece = PuzzlePiece(image, contour, self.config)
            piece.id = idx
            return piece
        except Exception as e:
            self.logger.error(
                f"Erreur lors du traitement du contour {idx}: {str(e)}")
            return None

    def process_contours(self, contours: List[np.ndarray], image: np.ndarray, 
                   image_path: Optional[str] = None) -> List[PuzzlePiece]:
        """
        Traite les contours pour créer des objets PuzzlePiece.

        Args:
            contours: Liste des contours
            image: Image originale
            image_path: Chemin de l'image pour le cache (optionnel)

        Returns:
            Liste des pièces de puzzle valides
        """
        if not contours:
            return []

        start_time = time.time()
        self.logger.info("Traitement des contours pour créer les pièces...")
        
        # Vérifier le cache si activé
        if self.pipeline_cache and image_path:
            from src.utils.cache_utils import cache_pieces
            
            # Paramètres pour la clé de cache
            cache_params = {
                'image_size': image.shape[:2],
                'config_hash': hash(tuple(sorted(self.config.to_dict().items())))
            }
            
            # Essayer de récupérer depuis le cache
            cached_pieces = cache_pieces(
                self.pipeline_cache,
                image_path,
                contours,
                cache_params
            )
            
            if cached_pieces is not None:
                self.logger.info(f"Pièces récupérées depuis le cache: {len(cached_pieces)} pièces")
                
                # Mesurer le temps d'accès au cache
                elapsed = time.time() - start_time
                self.detection_stats['timing']['contour_processing'] = elapsed
                self.detection_stats['timing']['cache_hit'] = True
                
                return cached_pieces

        # Si pas de cache ou cache miss, continuer avec le traitement normal
        self.detection_stats['timing']['cache_hit'] = False
        
        pieces = []

        # Traitement des contours en parallèle si activé
        if self.config.performance.USE_MULTIPROCESSING and len(contours) > 1:
            # Préparation des arguments pour le multitraitement
            args = [(contour, image, i) for i, contour in enumerate(contours)]

            # Utilisation d'un pool de processus pour le traitement parallèle
            with Pool(processes=min(self.config.performance.NUM_PROCESSES, cpu_count())) as pool:
                results = pool.map(self._process_contour, args)
                pieces = [p for p in results if p is not None]
        else:
            # Traitement séquentiel
            for i, contour in enumerate(contours):
                try:
                    piece = PuzzlePiece(image, contour, self.config)
                    piece.id = i
                    pieces.append(piece)
                except Exception as e:
                    self.logger.error(
                        f"Erreur lors du traitement du contour {i}: {str(e)}")

        # Filtrer les pièces non valides
        valid_pieces = [p for p in pieces if p.is_valid]

        elapsed = time.time() - start_time
        self.detection_stats['timing']['contour_processing'] = elapsed
        self.logger.info(
            f"Traitement terminé en {elapsed:.3f}s: {len(valid_pieces)}/{len(pieces)} pièces valides")
        
        # À la fin, mettre en cache les pièces si le cache est activé
        if self.pipeline_cache and image_path:
            from src.utils.cache_utils import save_pieces_to_cache
            
            # Paramètres pour la clé de cache
            cache_params = {
                'image_size': image.shape[:2],
                'config_hash': hash(tuple(sorted(self.config.to_dict().items())))
            }
            
            # Mettre en cache les pièces
            save_pieces_to_cache(
                self.pipeline_cache,
                image_path,
                contours,
                cache_params,
                valid_pieces
            )

        return valid_pieces

    def detect(self, image: np.ndarray, expected_pieces: Optional[int] = None,
           fast_mode: bool = False, image_path: Optional[str] = None) -> Tuple[List[PuzzlePiece], Dict[str, np.ndarray]]:
        """
        Détecte les pièces de puzzle dans une image.

        Args:
            image: Image couleur d'entrée
            expected_pieces: Nombre attendu de pièces
            fast_mode: Utiliser le mode rapide de détection
            image_path: Chemin de l'image pour le cache (optionnel)

        Returns:
            Tuple de (liste des pièces de puzzle, dictionnaire des images de débogage)
        """
        total_start_time = time.time()
        self.logger.info("Démarrage de la détection des pièces de puzzle")
        self.debug_images = {}  # Réinitialisation des images de débogage

        # Utiliser le pipeline de détection rapide si demandé
        if fast_mode:
            self.logger.info("Utilisation du mode rapide de détection")
            self.quick_detect = True
            preprocessed, binary, edges = self.preprocess_fast(image)
        else:
            self.quick_detect = False
            preprocessed, binary, edges = self.preprocess(
                image, expected_pieces, fast_mode, image_path)

        # Détection des contours
        contours = self.detect_contours(binary, image, expected_pieces, image_path)

        # Tentative de récupération des pièces manquantes si approprié
        if expected_pieces and len(contours) < expected_pieces:
            recovered_contours = self.recover_missed_pieces(
                binary, contours, image, expected_pieces)
            if recovered_contours:
                self.logger.info(
                    f"Récupéré {len(recovered_contours)} pièces supplémentaires")
                contours.extend(recovered_contours)

        # Traitement des contours pour créer des objets PuzzlePiece
        pieces = self.process_contours(contours, image, image_path)

        # Création d'une visualisation de toutes les pièces valides
        piece_vis = image.copy()
        for piece in pieces:
            piece_vis = piece.draw(piece_vis)

        self.save_debug_image(piece_vis, "09_detected_pieces.jpg")

        # Mesure du temps total
        total_elapsed = time.time() - total_start_time
        self.detection_stats['timing']['total'] = total_elapsed
        self.logger.info(f"Détection terminée en {total_elapsed:.3f} secondes")

        # Enregistrement des résultats de détection
        self.detection_stats['results'] = {
            'pieces_found': len(pieces),
            'expected_pieces': expected_pieces,
            'detection_rate': len(pieces) / expected_pieces if expected_pieces else None,
            'total_elapsed_time': total_elapsed,
            'cache_used': self.pipeline_cache is not None
        }

        # Retour des pièces et des images de débogage
        debug_images = {
            'preprocessed': preprocessed,
            'binary': binary,
            'edges': edges,
            'piece_visualization': piece_vis
        }

        return pieces, debug_images

    def detect_optimal(self, image: np.ndarray, expected_pieces: Optional[int] = None) -> List[PuzzlePiece]:
        """
        Détection optimisée avec sélection automatique des meilleurs paramètres.

        Args:
            image: Image couleur d'entrée
            expected_pieces: Nombre attendu de pièces

        Returns:
            Liste des pièces détectées
        """
        start_time = time.time()
        self.logger.info("Démarrage de la détection optimisée")

        # Analyse de l'image pour déterminer la meilleure stratégie
        analysis = analyze_image(image)

        # Déterminer si on utilise le mode rapide ou complet
        use_fast_mode = analysis['contrast'] > 0.6 and analysis['is_dark_background']

        # Exécuter la détection avec le mode approprié
        pieces, _ = self.detect(image, expected_pieces,
                                fast_mode=use_fast_mode)

        # Si la détection rapide n'a pas bien fonctionné, essayer le mode complet
        if use_fast_mode and (not pieces or (expected_pieces and len(pieces) < expected_pieces * 0.7)):
            self.logger.info(
                "Mode rapide insuffisant, passage au mode complet")
            pieces, _ = self.detect(image, expected_pieces, fast_mode=False)

        elapsed = time.time() - start_time
        self.logger.info(
            f"Détection optimisée terminée en {elapsed:.3f}s: {len(pieces)} pièces trouvées")

        return pieces

    def _contours_match(self, contour1: np.ndarray, contour2: np.ndarray) -> bool:
        """
        Vérifie si deux contours correspondent (représentent la même pièce).

        Args:
            contour1: Premier contour
            contour2: Deuxième contour

        Returns:
            True si les contours représentent probablement la même pièce
        """
        # Obtenir les rectangles englobants
        x1, y1, w1, h1 = cv2.boundingRect(contour1)
        x2, y2, w2, h2 = cv2.boundingRect(contour2)

        # Calculer l'IoU des rectangles englobants
        # Rectangle d'intersection
        x_inter = max(x1, x2)
        y_inter = max(y1, y2)
        w_inter = min(x1 + w1, x2 + w2) - x_inter
        h_inter = min(y1 + h1, y2 + h2) - y_inter

        # S'il n'y a pas de chevauchement, ils ne correspondent pas
        if w_inter <= 0 or h_inter <= 0:
            return False

        # Calculer les aires
        area_inter = w_inter * h_inter
        area1 = w1 * h1
        area2 = w2 * h2
        area_union = area1 + area2 - area_inter

        # Calculer l'IoU
        iou = area_inter / area_union

        # Vérifier la distance des centroïdes
        m1 = cv2.moments(contour1)
        m2 = cv2.moments(contour2)

        if m1["m00"] > 0 and m2["m00"] > 0:
            cx1 = m1["m10"] / m1["m00"]
            cy1 = m1["m01"] / m1["m00"]
            cx2 = m2["m10"] / m2["m00"]
            cy2 = m2["m01"] / m2["m00"]

            # Calculer la distance entre les centroïdes
            distance = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)

            # Si les centroïdes sont très proches, probablement la même pièce
            if distance < 50:
                return True

        # Retourner vrai si l'IoU dépasse le seuil
        return iou > 0.3  # Seuil de 30% de chevauchement

    def get_detection_statistics(self) -> Dict[str, Any]:
        """
        Obtient les statistiques de performance de la dernière détection.

        Returns:
            Dictionnaire des statistiques de détection
        """
        return self.detection_stats

    def get_debug_images(self) -> Dict[str, np.ndarray]:
        """
        Obtient les images de débogage générées pendant la détection.

        Returns:
            Dictionnaire des images de débogage
        """
        return self.debug_images