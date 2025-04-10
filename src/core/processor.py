"""
Module processeur optimisé pour la détection et l'analyse de pièces de puzzle.
Coordonne toutes les étapes du traitement des images de puzzle.
"""

import cv2
import numpy as np
import os
import json
import time
import gc  # Pour la gestion de mémoire optimisée
from typing import List, Dict, Any, Tuple, Optional, Union
from collections import Counter
import logging

# Imports absolus pour éviter les problèmes
from src.utils.image_utils import read_image, save_image
from src.utils.visualization import (
    create_processing_visualization, display_metrics, draw_contours,
    generate_piece_gallery
)
from src.utils.verification import (
    final_area_verification, fast_shape_verification, create_verification_visualization,
    combine_verification_methods, final_validation_check  # Assurez-vous d'inclure final_validation_check ici
)
from src.core.piece import PuzzlePiece
from src.core.detector import PuzzleDetector
from src.config.settings import Config
from src.utils.cache_utils import PipelineCache

class PuzzleProcessor:
    """
    Processeur optimisé pour la détection et l'analyse de pièces de puzzle
    avec capacités adaptatives et validation avancée.

    Cette classe coordonne l'ensemble du processus:
    1. Détection des pièces avec PuzzleDetector
    2. Validation et vérification des pièces détectées
    3. Analyse des métriques et caractéristiques
    4. Génération des visualisations
    5. Enregistrement des résultats
    """

    def __init__(self, config: Optional[Config] = None, enable_cache: bool = True, 
            cache_dir: str = "cache", max_cache_size_mb: int = 1000):
        """
        Initialise le processeur avec une configuration spécifiée.

        Args:
            config: Paramètres de configuration. Si None, utilise la configuration par défaut.
            enable_cache: Activer ou désactiver le cache du pipeline
            cache_dir: Répertoire pour le cache
            max_cache_size_mb: Taille maximale du cache en méga-octets
        """
        self.config = config or Config()
        
        # Initialisation du cache du pipeline
        self.pipeline_cache = PipelineCache(
            cache_dir=cache_dir,
            enable_cache=enable_cache,
            max_cache_size_mb=max_cache_size_mb
        ) if enable_cache else None
        
        # Initialisation du détecteur avec le cache
        self.detector = PuzzleDetector(self.config, self.pipeline_cache)
        self.logger = self._setup_logger()

        # Assurer que les répertoires existent
        if hasattr(self.config, 'DEBUG') and self.config.DEBUG:
            if hasattr(self.config, 'DEBUG_DIR'):
                os.makedirs(self.config.DEBUG_DIR, exist_ok=True)

    def _setup_logger(self) -> logging.Logger:
        """
        Configure le logger pour le processeur.

        Returns:
            Logger configuré
        """
        logger = logging.getLogger(__name__)
        return logger

    def process_image(self,
                  image_path: str,
                  expected_pieces: Optional[int] = None,
                  fast_mode: bool = False,
                  use_multi_pass: bool = False,
                  use_area_verification: bool = False,
                  area_verification_threshold: float = 2.0,
                  use_comprehensive_verification: bool = False) -> Dict[str, Any]:
        """
        Traite une image pour détecter et analyser les pièces de puzzle
        avec optimisation adaptative et vérification finale.

        Args:
            image_path: Chemin vers le fichier image
            expected_pieces: Nombre attendu de pièces (pour métriques et optimisation)
            fast_mode: Utiliser le mode rapide de détection pour les performances
            use_multi_pass: Utiliser la détection à passes multiples pour améliorer les résultats
            use_area_verification: Utiliser la vérification finale par aire
            area_verification_threshold: Seuil pour la vérification finale par aire
            use_comprehensive_verification: Utiliser la vérification complète

        Returns:
            Dictionnaire avec les résultats du traitement
        """
        self.logger.info(f"Traitement de l'image: {image_path}")

        try:
            # Lecture de l'image - sans redimensionnement
            image = read_image(image_path)

            # Journal des dimensions originales
            self.logger.info(
                f"Dimensions d'image originales: {image.shape[1]}x{image.shape[0]}")

            # Heure de début pour mesure de performance
            start_time = time.time()

            # Détecter les pièces avec les méthodes améliorées
            if use_multi_pass:
                # Utiliser la détection à passes multiples pour de meilleurs résultats
                # Vérifier si cette méthode existe, sinon fallback sur detect standard
                if hasattr(self.detector, 'multi_pass_detection'):
                    pieces, debug_images = self.detector.multi_pass_detection(
                        image, expected_pieces, image_path=image_path)
                else:
                    self.logger.warning("Détection multi-passes non disponible, utilisation de la détection standard")
                    pieces, debug_images = self.detector.detect(
                        image, expected_pieces, fast_mode=fast_mode, image_path=image_path)
            else:
                # Mode standard ou rapide
                pieces, debug_images = self.detector.detect(
                    image, expected_pieces, fast_mode=fast_mode, image_path=image_path)

            # Garder trace des pièces originales avant vérification
            original_pieces = pieces.copy()
            original_piece_count = len(pieces)

            # Appliquer la vérification si demandée
            rejected_pieces = []
            verification_vis = None

            if use_comprehensive_verification:
                self.logger.info(
                    "Application de la vérification finale complète...")
                verified_pieces = combine_verification_methods(
                    pieces,
                    expected_pieces=expected_pieces,
                    validation_level='standard'
                )
                rejected_pieces = [
                    p for p in original_pieces if p not in verified_pieces]
                pieces = verified_pieces
            elif use_area_verification:
                self.logger.info(
                    f"Application de la vérification finale par aire avec seuil {area_verification_threshold}...")
                verified_pieces = final_area_verification(
                    pieces, area_verification_threshold, expected_pieces)
                rejected_pieces = [
                    p for p in original_pieces if p not in verified_pieces]
                pieces = verified_pieces

            # Créer une visualisation de vérification si des pièces ont été rejetées
            if rejected_pieces:
                self.logger.info(
                    f"Vérification: suppression de {len(rejected_pieces)} pièces")
                verification_vis = create_verification_visualization(
                    image, pieces, rejected_pieces
                )
                debug_images['verification'] = verification_vis

                if hasattr(self.config, 'DEBUG') and self.config.DEBUG and hasattr(self.config, 'DEBUG_DIR'):
                    save_image(
                        verification_vis,
                        os.path.join(self.config.DEBUG_DIR,
                                    "verification_visualization.jpg")
                    )

            # Calculer le temps de traitement
            processing_time = time.time() - start_time
            self.logger.info(
                f"Détection terminée en {processing_time:.2f} secondes")

            # Calculer les métriques
            metrics = self.calculate_metrics(pieces, image, expected_pieces)
            metrics['processing_time'] = processing_time

            # Ajouter les métriques de vérification
            if use_area_verification or use_comprehensive_verification:
                metrics['original_detected_count'] = original_piece_count
                metrics['pieces_removed_by_verification'] = original_piece_count - \
                    len(pieces)

                if rejected_pieces:
                    # Calculer des statistiques sur les pièces rejetées
                    rejected_areas = [p.features['area']
                        for p in rejected_pieces]
                    metrics['rejected_mean_area'] = np.mean(
                        rejected_areas) if rejected_areas else 0
                    metrics['rejected_min_area'] = np.min(
                        rejected_areas) if rejected_areas else 0
                    metrics['rejected_max_area'] = np.max(
                        rejected_areas) if rejected_areas else 0

            # Ajouter des informations sur l'utilisation du cache
            if self.pipeline_cache:
                metrics['cache_used'] = True
                if hasattr(self.detector, 'detection_stats') and 'timing' in self.detector.detection_stats:
                    timing = self.detector.detection_stats['timing']
                    if 'cache_hit' in timing:
                        metrics['cache_hit'] = timing['cache_hit']

            # Sauvegarder le rapport de métriques si en mode débug
            if hasattr(self.config, 'DEBUG') and self.config.DEBUG and hasattr(self.config, 'DEBUG_DIR'):
                metrics_path = os.path.join(
                    self.config.DEBUG_DIR, "metrics_report.json")
                # Convertir toutes les valeurs numpy en types Python standards pour JSON
                metrics_json = {k: float(v) if isinstance(
                    v, np.number) else v for k, v in metrics.items()}
                with open(metrics_path, 'w') as f:
                    json.dump(metrics_json, f, indent=4)
                self.logger.info(f"Métriques sauvegardées vers {metrics_path}")

            # Créer des visualisations de pièces
            piece_visualizations = []
            for i, piece in enumerate(pieces):
                # Créer deux types de visualisations:
                # 1. Image complète avec contour de pièce pour le résumé
                full_vis = piece.draw(image.copy())

                # 2. Juste l'image de pièce extraite pour visualisation individuelle
                piece_only_vis = piece.get_extracted_image(
                    clean_background=True)

                # Sauvegarder juste l'image de pièce pour débogage
                if hasattr(self.config, 'DEBUG') and self.config.DEBUG and hasattr(self.config, 'DEBUG_DIR'):
                    save_path = os.path.join(
                        self.config.DEBUG_DIR, f"piece_{i}.jpg")
                    save_image(piece_only_vis, save_path)

                piece_visualizations.append(full_vis)

            # Créer la visualisation résumée
            pieces_info = []
            for i, piece in enumerate(pieces):
                pieces_info.append({
                    'id': i,
                    'visualization': piece_visualizations[i],
                    'is_valid': piece.is_valid,
                    'border_types': getattr(piece, 'border_types', None),
                    'validation_score': getattr(piece, 'validation_score', 0.0)
                })

            # Créer et sauvegarder la visualisation résumée
            summary_vis = create_processing_visualization(
                image,
                debug_images.get('preprocessed', np.zeros_like(image)),
                debug_images.get('binary', np.zeros_like(image)),
                debug_images.get('piece_visualization', np.zeros_like(image)),
                pieces_info
            )

            if hasattr(self.config, 'DEBUG') and self.config.DEBUG and hasattr(self.config, 'DEBUG_DIR'):
                save_image(summary_vis, os.path.join(
                    self.config.DEBUG_DIR, "processing_summary.jpg"))

            # Créer la visualisation de métriques
            metrics_vis = display_metrics(metrics)
            if hasattr(self.config, 'DEBUG') and self.config.DEBUG and hasattr(self.config, 'DEBUG_DIR'):
                save_image(metrics_vis, os.path.join(
                    self.config.DEBUG_DIR, "metrics_visualization.jpg"))

            # Sauvegarder la visualisation combinée si elle existe
            if 'combined_visualization' in debug_images and hasattr(self.config, 'DEBUG') and self.config.DEBUG and hasattr(self.config, 'DEBUG_DIR'):
                save_image(debug_images['combined_visualization'],
                        os.path.join(self.config.DEBUG_DIR, "combined_detection.jpg"))
                # Libérer la mémoire
                del debug_images['combined_visualization']

            # Créer une galerie de pièces pour visualisation facile
            gallery = None
            if pieces:
                gallery = generate_piece_gallery(pieces)
                if hasattr(self.config, 'DEBUG') and self.config.DEBUG and hasattr(self.config, 'DEBUG_DIR'):
                    save_image(gallery, os.path.join(
                        self.config.DEBUG_DIR, "piece_gallery.jpg"))

            # Libérer la mémoire des images intermédiaires volumineuses
            for key in list(debug_images.keys()):
                if key in ['preprocessed', 'binary', 'edges']:
                    del debug_images[key]

            # Forcer la collecte de mémoire
            gc.collect()

            # Retourner les résultats
            return {
                'image_path': image_path,
                'pieces': pieces,
                'metrics': metrics,
                'visualizations': {
                    'summary': summary_vis,
                    'metrics': metrics_vis,
                    'pieces': piece_visualizations,
                    'verification': verification_vis,
                    'gallery': gallery
                },
                'processing_time': processing_time,
                'cache_used': self.pipeline_cache is not None
            }

        except Exception as e:
            # Gestion d'erreurs améliorée
            self.logger.error(f"Erreur lors du traitement de l'image {image_path}: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            
            # Créer des visualisations minimales en cas d'erreur
            try:
                # Tenter de créer une image vide pour la visualisation d'erreur
                try:
                    # Essayer de lire l'image originale si possible
                    original_image = read_image(image_path)
                except:
                    # Si ça échoue, créer une image vide
                    original_image = np.ones((500, 700, 3), dtype=np.uint8) * 255
                
                # Créer un message d'erreur sur l'image
                error_text = f"Erreur: {str(e)}"
                cv2.putText(original_image, error_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                cv2.putText(original_image, "Vérifiez le fichier journal pour plus de détails", (20, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Créer une image pour les métriques
                metrics_img = np.ones((300, 500, 3), dtype=np.uint8) * 255
                cv2.putText(metrics_img, "Erreur de traitement", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                cv2.putText(metrics_img, "Aucune métrique disponible", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                
                visualizations = {
                    'summary': original_image,
                    'metrics': metrics_img
                }
            except Exception as vis_error:
                # En cas d'échec de création d'images, utiliser des images vides
                self.logger.error(f"Erreur lors de la création des visualisations: {str(vis_error)}")
                blank_image = np.ones((300, 500, 3), dtype=np.uint8) * 255
                cv2.putText(blank_image, "Erreur de traitement", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                visualizations = {
                    'summary': blank_image.copy(),
                    'metrics': blank_image.copy()
                }
            
            # Retourner un résultat minimal en cas d'erreur, mais avec des visualisations valides
            return {
                'image_path': image_path,
                'error': str(e),
                'pieces': [],
                'metrics': {'error': str(e)},
                'visualizations': visualizations,
                'processing_time': 0,
                'cache_used': self.pipeline_cache is not None
            }

    def calculate_metrics(self,
                      pieces: List[PuzzlePiece],
                      image: np.ndarray,
                      expected_count: Optional[int] = None) -> Dict[str, Any]:
        """
        Calcule des métriques complètes pour les pièces de puzzle détectées.
        Version optimisée avec des opérations vectorisées.

        Args:
            pieces: Liste des pièces détectées
            image: Image originale
            expected_count: Nombre attendu de pièces

        Returns:
            Dictionnaire de métriques
        """
        self.logger.info("Calcul des métriques...")

        metrics = {}
        image_area = image.shape[0] * image.shape[1]

        # Métriques basées sur le comptage
        metrics['detected_count'] = len(pieces)
        metrics['expected_count'] = expected_count
        metrics['valid_pieces'] = sum(1 for p in pieces if p.is_valid)

        if expected_count:
            metrics['detection_rate'] = len(pieces) / expected_count
            metrics['valid_detection_rate'] = metrics['valid_pieces'] / \
                expected_count

        # Si pas de pièces, retourner des métriques par défaut
        if not pieces:
            default_metrics = [
                'mean_area', 'median_area', 'std_area', 'min_area', 'max_area',
                'total_piece_area', 'area_coverage', 'area_cv', 'mean_solidity',
                'std_solidity', 'mean_compactness', 'std_compactness',
                'mean_equivalent_diameter', 'std_equivalent_diameter',
                'mean_validation_score', 'min_validation_score', 'max_validation_score',
                'edge_alignment', 'valid_piece_count', 'valid_piece_ratio'
            ]
            for metric in default_metrics:
                metrics[metric] = 0.0

            metrics['border_types'] = {}
            metrics['tab_pocket_ratio'] = 0.0

            return metrics

        # --- Début de l'optimisation vectorisée ---

        # 1. Extraction vectorisée des caractéristiques avec un seul passage sur les pièces
        # Préallouer les tableaux numpy pour toutes les caractéristiques
        num_pieces = len(pieces)

        # Caractéristiques de base (garanties d'exister)
        areas = np.zeros(num_pieces)

        # Caractéristiques de forme (peuvent ne pas exister dans toutes les pièces)
        solidities = np.zeros(num_pieces)
        compactness = np.zeros(num_pieces)
        eq_diameters = np.zeros(num_pieces)

        # Scores de validation
        has_validation_scores = False
        validation_scores = np.zeros(num_pieces)

        # Tableaux pour les pièces valides et les types de bordure
        valid_mask = np.zeros(num_pieces, dtype=bool)
        all_border_types = []

        # Extraction en une seule boucle au lieu de multiples boucles
        for i, piece in enumerate(pieces):
            # Caractéristiques de base
            areas[i] = piece.features['area']

            # Caractéristiques de forme
            solidities[i] = piece.features.get('solidity', 0)
            compactness[i] = piece.features.get('compactness', 0)
            eq_diameters[i] = piece.features.get('equivalent_diameter', 0)

            # Score de validation
            if hasattr(piece, 'validation_score'):
                validation_scores[i] = piece.validation_score
                has_validation_scores = True

            # Validité de la pièce
            valid_mask[i] = piece.is_valid

            # Types de bordure
            if hasattr(piece, 'border_types') and piece.border_types:
                all_border_types.extend(piece.border_types)

        # 2. Calcul vectorisé des métriques d'aire
        metrics['mean_area'] = float(np.mean(areas))
        metrics['median_area'] = float(np.median(areas))
        metrics['std_area'] = float(np.std(areas))
        metrics['min_area'] = float(np.min(areas))
        metrics['max_area'] = float(np.max(areas))
        metrics['total_piece_area'] = float(np.sum(areas))
        metrics['area_coverage'] = float(
            metrics['total_piece_area'] / image_area)
        metrics['area_cv'] = float(
            metrics['std_area'] / metrics['mean_area'] if metrics['mean_area'] > 0 else 0)

        # 3. Calcul vectorisé des métriques de forme
        metrics['mean_solidity'] = float(np.mean(solidities))
        metrics['std_solidity'] = float(np.std(solidities))
        metrics['mean_compactness'] = float(np.mean(compactness))
        metrics['std_compactness'] = float(np.std(compactness))
        metrics['mean_equivalent_diameter'] = float(np.mean(eq_diameters))
        metrics['std_equivalent_diameter'] = float(np.std(eq_diameters))

        # 4. Scores de validation s'ils sont disponibles
        if has_validation_scores:
            metrics['mean_validation_score'] = float(
                np.mean(validation_scores))
            metrics['min_validation_score'] = float(np.min(validation_scores))
            metrics['max_validation_score'] = float(np.max(validation_scores))
        else:
            metrics['mean_validation_score'] = 0.0
            metrics['min_validation_score'] = 0.0
            metrics['max_validation_score'] = 0.0

        # 5. Distribution des types de bordure
        from collections import Counter
        border_counter = Counter(all_border_types)
        metrics['border_types'] = dict(border_counter)

        # Calculer le rapport languette/poche (devrait être proche de 1 pour les puzzles valides)
        tab_count = border_counter.get('tab', 0)
        pocket_count = border_counter.get('pocket', 0)
        metrics['tab_pocket_ratio'] = float(
            tab_count / pocket_count if pocket_count > 0 else 0)

        # 6. Métriques d'alignement des bords (optimisé)
        edge_map = cv2.Canny(image, 50, 150)

        # Création du masque en une seule opération
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        # Dessiner tous les contours en une seule fois est plus efficace
        contours = [piece.contour for piece in pieces]
        cv2.drawContours(mask, contours, -1, 255, 2)

        # Calculer le chevauchement entre les contours détectés et la carte des bords
        overlap = cv2.bitwise_and(edge_map, mask)
        metrics['edge_alignment'] = float(
            np.sum(overlap > 0) / (np.sum(mask > 0) + 1e-6))

        # 7. Métriques de pièces valides
        # Utilisation du masque booléen préalablement calculé
        valid_count = np.sum(valid_mask)
        metrics['valid_piece_count'] = int(valid_count)
        metrics['valid_piece_ratio'] = float(valid_count / len(pieces))

        # Libérer la mémoire des structures intermédiaires de grande taille
        del edge_map, mask, overlap

        # Forcer la collecte de mémoire
        gc.collect()

        return metrics

    def extract_pieces(self, 
                   pieces: List[PuzzlePiece], 
                   output_dir: str = "extracted_pieces") -> List[str]:
        """
        Extrait les pièces de puzzle individuelles dans des fichiers image séparés.
        Version optimisée pour la performance.
        
        Args:
            pieces: Liste des pièces de puzzle
            output_dir: Répertoire pour sauvegarder les pièces extraites
        
        Returns:
            Liste des chemins vers les images sauvegardées
        """
        self.logger.info(f"Extraction de {len(pieces)} pièces vers {output_dir}...")
        
        # Création du répertoire une seule fois avant la boucle
        os.makedirs(output_dir, exist_ok=True)
        
        # Préallocation de la liste pour un code plus efficace
        saved_paths = [None] * len(pieces)
        
        # Traitement par lots pour réduire les coûts de GC et améliorer la cache locality
        batch_size = 10
        for batch_start in range(0, len(pieces), batch_size):
            batch_end = min(batch_start + batch_size, len(pieces))
            
            for i in range(batch_start, batch_end):
                piece = pieces[i]
                try:
                    # Utiliser l'extraction améliorée avec fond propre
                    piece_image = piece.get_extracted_image(clean_background=True)
                    
                    # Utiliser un format de nom de fichier cohérent
                    path = os.path.join(output_dir, f"piece_{i:03d}.jpg")
                    save_image(piece_image, path)
                    saved_paths[i] = path
                    
                    # Libérer la mémoire explicitement
                    del piece_image
                except Exception as e:
                    self.logger.error(f"Erreur lors de l'extraction de la pièce {i}: {str(e)}")
                    saved_paths[i] = None
            
            # Forcer la collecte de mémoire après chaque lot
            gc.collect()
        
        # Filtrer les chemins None (erreurs) avant de retourner
        valid_paths = [path for path in saved_paths if path is not None]
        self.logger.info(f"Sauvegardé {len(valid_paths)} pièces vers {output_dir}")
        return valid_paths


    def analyze_piece_matches(self, pieces: List[PuzzlePiece]) -> Dict[str, Any]:
        """
        Analyse les correspondances potentielles entre les pièces.

        Args:
            pieces: Liste des pièces de puzzle

        Returns:
            Dictionnaire avec les résultats d'analyse des correspondances
        """
        self.logger.info(
            f"Analyse des correspondances potentielles entre {len(pieces)} pièces...")

        # Uniquement analyser les pièces valides
        valid_pieces = [p for p in pieces if p.is_valid]

        if len(valid_pieces) < 2:
            return {'matches': [], 'match_count': 0, 'analyzed_pieces': len(valid_pieces)}

        # Pré-calcul des IDs pour éviter d'accéder à l'attribut dans la boucle
        piece_ids = [p.id for p in valid_pieces]

        # Tableau pour stocker les matches
        matches = []

        for i, piece1 in enumerate(valid_pieces):
            for j, piece2 in enumerate(valid_pieces):
                if i >= j:  # Ignorer les auto-correspondances et doublons
                    continue

                try:
                    # Calculer le score de correspondance
                    match_score = piece1.calculate_match_score(piece2)

                    # Normaliser le format pour la compatibilité
                    if isinstance(match_score, dict):
                        overall_score = match_score.get('overall', 0.0)
                        match_details = match_score
                    else:
                        overall_score = match_score
                        match_details = {'overall': overall_score}

                    # Ne garder que les correspondances significatives
                    if overall_score > 0.5:
                        matches.append({
                            'piece1_id': piece_ids[i],
                            'piece2_id': piece_ids[j],
                            'match_score': overall_score,
                            'match_details': match_details
                        })
                except Exception as e:
                    self.logger.error(
                        f"Erreur lors du calcul de correspondance: {str(e)}")

        # Trier les correspondances par score (décroissant)
        matches.sort(key=lambda x: x['match_score'], reverse=True)

        # Limiter aux meilleures correspondances pour plus de clarté
        top_matches = matches[:min(20, len(matches))]

        self.logger.info(
            f"Trouvé {len(top_matches)} correspondances significatives")

        return {
            'matches': top_matches,
            'match_count': len(top_matches),
            'analyzed_pieces': len(valid_pieces)}

    def save_results(self,
                     results: Dict[str, Any],
                     output_dir: str="results") -> str:
        """
        Sauvegarde les résultats du traitement dans un répertoire.

        Args:
            results: Résultats du traitement
            output_dir: Répertoire pour sauvegarder les résultats

        Returns:
            Chemin vers les résultats sauvegardés
        """
        os.makedirs(output_dir, exist_ok=True)

        # Créer un répertoire basé sur l'horodatage
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        result_dir = os.path.join(output_dir, f"puzzle_{timestamp}")
        os.makedirs(result_dir, exist_ok=True)

        try:
            # Sauvegarder les visualisations
            vis_dir = os.path.join(result_dir, "visualizations")
            os.makedirs(vis_dir, exist_ok=True)

            if 'visualizations' in results:
                visualizations = results['visualizations']
                # Sauvegarder chaque visualisation si elle existe
                for vis_name, vis_image in visualizations.items():
                    # Vérification que l'image est valide avant de la sauvegarder
                    if vis_image is not None and isinstance(vis_image, np.ndarray):
                        save_image(vis_image, os.path.join(
                            vis_dir, f"{vis_name}.jpg"))
                    else:
                        self.logger.warning(f"Impossible de sauvegarder la visualisation '{vis_name}': image invalide ou None")

                # Sauvegarder les visualisations de pièces individuelles
                if 'pieces' in visualizations and visualizations['pieces']:
                    pieces_dir = os.path.join(vis_dir, "pieces")
                    os.makedirs(pieces_dir, exist_ok=True)

                    for i, vis in enumerate(visualizations['pieces']):
                        # Vérification que l'image de la pièce est valide
                        if vis is not None and isinstance(vis, np.ndarray):
                            save_image(vis, os.path.join(
                                pieces_dir, f"piece_{i:03d}.jpg"))
                        else:
                            self.logger.warning(f"Impossible de sauvegarder la pièce #{i}: image invalide ou None")

            # Sauvegarder les métriques comme JSON
            if 'metrics' in results:
                # Convertir les valeurs numpy en types Python standards
                metrics_json = {}
                for k, v in results['metrics'].items():
                    if isinstance(v, np.number):
                        metrics_json[k] = float(v)
                    elif isinstance(v, np.ndarray):
                        metrics_json[k] = v.tolist()
                    else:
                        metrics_json[k] = v

                with open(os.path.join(result_dir, "metrics.json"), 'w') as f:
                    json.dump(metrics_json, f, indent=4)

            # Sauvegarder les données des pièces
            if 'pieces' in results:
                pieces_data = []
                for piece in results['pieces']:
                    try:
                        # Utiliser la méthode to_dict si disponible
                        if hasattr(piece, 'to_dict') and callable(piece.to_dict):
                            pieces_data.append(piece.to_dict())
                        else:
                            # Créer un dictionnaire simplifié
                            pieces_data.append({
                                'id': piece.id,
                                'is_valid': piece.is_valid,
                                'area': piece.features.get('area', 0),
                                'validation_status': getattr(piece, 'validation_status', None)
                            })
                    except Exception as e:
                        self.logger.error(
                            f"Erreur lors de la sérialisation de la pièce: {str(e)}")

                with open(os.path.join(result_dir, "pieces.json"), 'w') as f:
                    json.dump(pieces_data, f, indent=4)

            self.logger.info(f"Résultats sauvegardés vers {result_dir}")
            return result_dir

        except Exception as e:
            self.logger.error(
                f"Erreur lors de la sauvegarde des résultats: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return result_dir  # Retourner quand même le répertoire même en cas d'erreur partielle
    def optimize_parameters_for_image(self,
                                     image_path: str,
                                     expected_pieces: int,
                                     parameter_grid: Optional[Dict[str, List[Any]]]=None) -> Dict[str, Any]:
        """
        Optimise les paramètres de détection pour une image spécifique.

        Args:
            image_path: Chemin vers le fichier image
            expected_pieces: Nombre attendu de pièces
            parameter_grid: Grille des paramètres à essayer

        Returns:
            Dictionnaire avec les paramètres optimaux et les résultats
        """
        self.logger.info(f"Optimisation des paramètres pour {image_path}...")

        try:
            # Lire l'image
            image = read_image(image_path)

            # Définir la grille de paramètres par défaut si non fournie
            if parameter_grid is None:
                parameter_grid = {
                    'MIN_CONTOUR_AREA': [500, 1000, 2000, 3000],
                    'MEAN_DEVIATION_THRESHOLD': [1.0, 1.5, 2.0, 2.5],
                    'USE_ADAPTIVE_PREPROCESSING': [True, False]
                }

            # Suivre les meilleurs paramètres et résultats
            best_score = -1
            best_params = {}
            best_pieces = []

            # Définir un sous-ensemble réduit de combinaisons à essayer
            # (une recherche complète serait trop lente)
            total_combinations = 1
            for param_values in parameter_grid.values():
                total_combinations *= len(param_values)

            # Limiter les combinaisons à un nombre raisonnable
            max_combinations = 12
            sample_ratio = min(1.0, max_combinations / total_combinations)

            self.logger.info(
                f"Test de {max_combinations} combinaisons de paramètres...")

            # Conserver les valeurs originales des paramètres
            original_params = {}
            for param_name in parameter_grid.keys():
                if hasattr(self.config, param_name):
                    original_params[param_name] = getattr(
                        self.config, param_name)

            # Essayer des combinaisons de paramètres (échantillonnage pour limiter les combinaisons)
            import random
            combinations_to_try = []

            # Générer des combinaisons de paramètres
            for _ in range(min(max_combinations, total_combinations)):
                combination = {}
                for param_name, param_values in parameter_grid.items():
                    combination[param_name] = random.choice(param_values)
                combinations_to_try.append(combination)

            # S'assurer que nous testons à la fois les prétraitements adaptatifs et standard
            if 'USE_ADAPTIVE_PREPROCESSING' not in parameter_grid:
                for i, combo in enumerate(combinations_to_try):
                    combo['USE_ADAPTIVE_PREPROCESSING'] = (i % 2 == 0)

            # Essayer chaque combinaison
            for i, param_set in enumerate(combinations_to_try):
                # Mise à jour de la configuration avec les paramètres actuels
                for param_name, param_value in param_set.items():
                    setattr(self.config, param_name, param_value)

                self.logger.info(
                    f"Test de la combinaison {i+1}/{len(combinations_to_try)}: {param_set}")

                # Exécuter la détection
                pieces, _ = self.detector.detect(image, expected_pieces)

                # Calculer le score (taux de détection et qualité)
                if expected_pieces > 0:
                    detection_rate = len(pieces) / expected_pieces
                else:
                    detection_rate = 0

                valid_pieces = sum(1 for p in pieces if p.is_valid)
                valid_rate = valid_pieces / expected_pieces if expected_pieces > 0 else 0

                # Score de validation moyen
                validation_scores = [
                    piece.validation_score for piece in pieces
                    if hasattr(piece, 'validation_score') and piece.is_valid
                ]
                avg_validation = np.mean(
                    validation_scores) if validation_scores else 0

                # Score combiné
                score = 0.5 * detection_rate + 0.3 * valid_rate + 0.2 * avg_validation

                self.logger.info(f"  Résultats: {len(pieces)}/{expected_pieces} pièces, " +
                               f"score: {score:.3f}")

                # Suivre les meilleurs paramètres
                if score > best_score:
                    best_score = score
                    best_params = param_set.copy()
                    best_pieces = pieces

                # Libérer la mémoire après chaque test
                del pieces
                gc.collect()

            # Restaurer les paramètres originaux
            for param_name, param_value in original_params.items():
                setattr(self.config, param_name, param_value)

            # Journal des meilleurs paramètres
            self.logger.info(f"Meilleurs paramètres: {best_params}")
            self.logger.info(f"Meilleur score: {best_score:.3f}")

            # Retourner les résultats d'optimisation
            return {
                'best_params': best_params,
                'best_score': best_score,
                'best_piece_count': len(best_pieces),
                'expected_pieces': expected_pieces,
                'detection_rate': len(best_pieces) / expected_pieces if expected_pieces > 0 else 0
            }
        except Exception as e:
            self.logger.error(
                f"Erreur lors de l'optimisation des paramètres: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())

            # Retourner un résultat d'échec
            return {
                'error': str(e),
                'best_params': {},
                'best_score': 0,
                'best_piece_count': 0,
                'expected_pieces': expected_pieces,
                'detection_rate': 0
            }

    def analyze_image_characteristics(self, image_path: str) -> Dict[str, Any]:
        """
        Analyse les caractéristiques de l'image pour déterminer les paramètres
        de traitement optimaux.

        Args:
            image_path: Chemin vers le fichier image

        Returns:
            Dictionnaire avec les résultats d'analyse de l'image
        """
        self.logger.info(
            f"Analyse des caractéristiques de l'image: {image_path}")

        try:
            # Lire l'image
            image = read_image(image_path)

            # Convertir en niveaux de gris pour l'analyse
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()

            # Calculer les statistiques de base
            mean = float(np.mean(gray))
            std = float(np.std(gray))
            median = float(np.median(gray))

            # Calculer l'histogramme
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist = hist.flatten() / hist.sum()  # Normaliser

            # Calculer les pics (modes) dans l'histogramme
            peak_indices = []
            for i in range(1, 255):
                if hist[i] > hist[i-1] and hist[i] > hist[i+1] and hist[i] > 0.01:
                    peak_indices.append(i)

            # Obtenir les valeurs des pics
            peaks = [(int(i), float(hist[i])) for i in peak_indices]
            peaks.sort(key=lambda x: x[1], reverse=True)  # Trier par hauteur

            # Vérifier si l'histogramme est bimodal (typique pour les puzzles sur fond contrasté)
            is_bimodal = len(peaks) >= 2

            # Calculer le contraste
            p5 = float(np.percentile(gray, 5))
            p95 = float(np.percentile(gray, 95))
            contrast = float((p95 - p5) / 255.0)

            # Détecter les bords
            edges = cv2.Canny(gray, 50, 150)
            edge_density = float(np.count_nonzero(
                edges) / (gray.shape[0] * gray.shape[1]))

            # Estimer la couleur de fond
            # En supposant que les régions plus sombres sont le fond pour les pièces de puzzle
            dark_ratio = float(np.sum(gray < 50) / gray.size)
            is_dark_background = dark_ratio > 0.3

            # Déterminer les paramètres recommandés basés sur l'analyse
            recommended_params = {}

            # Prétraitement adaptatif si l'image a un faible contraste ou un éclairage inégal
            recommended_params['USE_ADAPTIVE_PREPROCESSING'] = contrast < 0.5

            # Utiliser le pipeline Sobel pour les images avec des caractéristiques de bord subtiles
            recommended_params['USE_SOBEL_PIPELINE'] = edge_density < 0.05

            # Ajuster le seuil d'aire de contour en fonction de la taille de l'image
            img_area = image.shape[0] * image.shape[1]
            # Estimation grossière pour un puzzle de 24 pièces
            estimated_piece_area = img_area / 30
            recommended_params['MIN_CONTOUR_AREA'] = max(
                500, int(estimated_piece_area * 0.1))

            # Ajuster le seuil de filtrage moyen en fonction du contraste
            if contrast < 0.4:
                # Plus permissif pour les images à faible contraste
                recommended_params['MEAN_DEVIATION_THRESHOLD'] = 2.0
            else:
                # Plus strict pour les images à contraste élevé
                recommended_params['MEAN_DEVIATION_THRESHOLD'] = 1.5

            analysis_results = {
                'mean': mean,
                'std': std,
                'median': median,
                'contrast': contrast,
                'edge_density': edge_density,
                'is_bimodal': is_bimodal,
                'is_dark_background': is_dark_background,
                'peaks': peaks[:3] if peaks else [],  # Top 3 pics
                'recommended_params': recommended_params
            }

            self.logger.info(f"Analyse terminée: contraste={contrast:.2f}, " +
                           f"densité de bord={edge_density:.3f}, " +
                           f"fond sombre={is_dark_background}")

            # Libérer la mémoire
            del gray, edges
            gc.collect()

            return analysis_results

        except Exception as e:
            self.logger.error(f"Erreur lors de l'analyse de l'image: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())

            # Retourner un résultat minimal en cas d'erreur
            return {
                'error': str(e),
                'contrast': 0,
                'edge_density': 0,
                'is_dark_background': False,
                'recommended_params': {}
            }

    def final_validation_check(self,
                              pieces: List[PuzzlePiece],
                              expected_pieces: Optional[int]=None,
                              area_threshold: float=2.0,
                              aspect_threshold: float=4.0,
                              validation_level: str='standard',
                              use_recovery: bool=True) -> List[PuzzlePiece]:
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
        self.logger.info(
            f"Exécution de la vérification finale avec niveau {validation_level}")

        if not pieces:
            return []

        # Utiliser la fonction de vérification combinée de verification.py
        verified_pieces = combine_verification_methods(
            pieces,
            expected_pieces=expected_pieces,
            validation_level=validation_level
        )

        # Si trop peu de pièces sont conservées et que la récupération est activée
        if (expected_pieces and
            len(verified_pieces) < expected_pieces * 0.7 and
            use_recovery):

            self.logger.info(
                f"Trop peu de pièces après vérification ({len(verified_pieces)}/{expected_pieces}). Tentative de récupération.")

            # Essayer une vérification plus permissive
            recovered_pieces = combine_verification_methods(
                pieces,
                expected_pieces=expected_pieces,
                validation_level='permissive'
            )

            # Utiliser les résultats de récupération si meilleurs
            if len(recovered_pieces) > len(verified_pieces):
                self.logger.info(
                    f"Récupération réussie: {len(recovered_pieces)} pièces (vs {len(verified_pieces)} précédemment)")
                return recovered_pieces

        return verified_pieces
    
    def manage_cache(self, clear_cache: bool = False) -> Dict[str, Any]:
        """
        Gère le cache du pipeline.
        
        Args:
            clear_cache: Si True, vide le cache
            
        Returns:
            Informations sur le cache
        """
        if not self.pipeline_cache:
            return {
                'status': 'disabled',
                'message': 'Le cache du pipeline est désactivé'
            }
        
        if clear_cache:
            self.pipeline_cache.clear_cache()
            return {
                'status': 'cleared',
                'message': 'Cache vidé avec succès'
            }
        
        # Calculer la taille du cache
        cache_size_mb = 0
        cache_entries = 0
        
        try:
            cache_dir = self.pipeline_cache.cache_dir
            if os.path.exists(cache_dir):
                for root, _, files in os.walk(cache_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        cache_size_mb += os.path.getsize(file_path) / (1024 * 1024)
                        cache_entries += 1
        except Exception as e:
            self.logger.error(f"Erreur lors du calcul de la taille du cache: {str(e)}")
        
        return {
            'status': 'active',
            'entries': cache_entries,
            'size_mb': round(cache_size_mb, 2),
            'max_size_mb': self.pipeline_cache.max_cache_size_mb,
            'directory': self.pipeline_cache.cache_dir
        }
