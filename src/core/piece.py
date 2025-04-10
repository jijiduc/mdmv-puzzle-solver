"""
Représentation optimisée d'une pièce de puzzle avec focus sur la segmentation.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import sys
import os
import logging
import time

# Ajout du répertoire parent au chemin pour permettre les imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.contour_utils import (
    calculate_contour_features, validate_shape_as_puzzle_piece
)
from src.config.settings import Config


class PuzzlePiece:
    """
    Classe optimisée représentant une pièce de puzzle détectée.
    Se concentre sur les caractéristiques de forme et l'extraction d'image.
    """
    
    def __init__(self, 
                image: np.ndarray, 
                contour: np.ndarray, 
                config: Config = None):
        """
        Initialise une pièce de puzzle à partir d'une image et d'un contour.
        
        Args:
            image: Image source
            contour: Contour de la pièce
            config: Paramètres de configuration
        """
        start_time = time.time()
        self.config = config or Config()
        self.image = image
        self.contour = contour
        self.id = None  # Identifiant unique pour la pièce
        self.logger = logging.getLogger(__name__)
        
        # Propriétés à calculer
        self.features = None
        self.extracted_image = None
        self.is_valid = False
        self.validation_status = None
        self.validation_score = 0.0
        
        # Traitement de la pièce
        self._extract_features()
        self._extract_piece_image()
        self._validate_piece()
        
        self.creation_time = time.time() - start_time
    
    def _extract_features(self) -> None:
        """Extrait les caractéristiques essentielles du contour."""
        try:
            self.features = calculate_contour_features(self.contour)
            self.validation_status = "features_extracted"
        except Exception as e:
            self.logger.error(f"Erreur lors de l'extraction des caractéristiques: {str(e)}")
            self.validation_status = f"feature_extraction_error: {str(e)}"
            self.features = {
                'area': 0,
                'perimeter': 0,
                'bbox': (0, 0, 0, 0),
                'centroid': (0, 0)
            }
    
    def _extract_piece_image(self) -> None:
        """
        Extrait l'image de la pièce en utilisant le contour comme masque.
        Version optimisée pour la performance.
        """
        try:
            # Utiliser un rectangle englobant précalculé plutôt que de le recalculer
            x, y, w, h = self.features['bbox']
            
            # Créer un masque à la taille de l'image avec un seul appel cv2.drawContours
            mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [self.contour], 0, 255, -1)
            
            # Calculer les coordonnées avec marge une seule fois
            margin = 5  # Petite marge pour éviter de couper les bords
            x_min = max(0, x - margin)
            y_min = max(0, y - margin)
            x_max = min(self.image.shape[1], x + w + margin)
            y_max = min(self.image.shape[0], y + h + margin)
            
            # Extraire les régions d'intérêt une seule fois
            roi = self.image[y_min:y_max, x_min:x_max]
            mask_roi = mask[y_min:y_max, x_min:x_max]
            
            # Optimisation pour la création du fond blanc
            # Créer un fond blanc et une version 3 canaux du masque ROI une seule fois
            white_bg = np.ones_like(roi) * 255
            
            # Astuce pour traiter les images RGB et grayscale sans if
            channels = 3 if len(self.image.shape) == 3 else 1
            mask_channels = np.repeat(mask_roi[:, :, np.newaxis], channels, axis=2) if channels == 3 else mask_roi
            
            # Appliquer le masque avec une opération vectorisée NumPy
            # Utiliser np.where au lieu d'opérations par pixels
            self.extracted_image = np.where(mask_channels > 0, roi, white_bg)
            
            self.validation_status = "valid_extraction"
        except Exception as e:
            self.logger.error(f"Erreur lors de l'extraction de l'image de la pièce: {str(e)}")
            self.validation_status = f"extraction_error:{str(e)}"
            # Créer une image vide en cas d'erreur
            self.extracted_image = np.ones((100, 100, 3), dtype=np.uint8) * 255

    
    def get_extracted_image(self, clean_background: bool = True) -> np.ndarray:
        """
        Obtient l'image extraite de la pièce avec options pour le fond.
        
        Args:
            clean_background: Si True, retourne la pièce avec un fond blanc propre
                            Si False, retourne la pièce avec le fond original recadré
        
        Returns:
            Image de la pièce uniquement
        """
        if self.extracted_image is None:
            # Si l'extraction a échoué, créer une visualisation simple
            result = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.putText(result, "Pas d'extraction", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                      0.5, (0, 0, 255), 1)
            return result
        
        # L'image extraite a déjà un fond blanc grâce à l'extraction optimisée
        return self.extracted_image.copy()
    
    def _validate_piece(self) -> None:
        """Valide si c'est une pièce de puzzle valide en utilisant uniquement les caractéristiques de forme."""
        if not self.features:
            self.is_valid = False
            self.validation_status = "missing_features"
            return
            
        # Vérification rapide de la taille minimale
        if self.features['area'] < (self.config.MIN_CONTOUR_AREA if hasattr(self.config, 'MIN_CONTOUR_AREA') else 1000):
            self.is_valid = False
            self.validation_status = f"too_small:{self.features['area']:.0f}"
            return
            
        # Initialiser avec la validation de forme de base
        is_valid_shape, quality_score, shape_reason = self._validate_shape()
        
        # Calcul du score de validation
        self.validation_score = quality_score
        
        # La pièce est valide si elle passe un seuil
        threshold = 0.5  # Seuil assez permissif
        self.is_valid = quality_score >= threshold
        
        # Définir le statut de validation
        if self.is_valid:
            self.validation_status = f"valid_piece:{self.validation_score:.2f}"
        else:
            self.validation_status = f"invalid_piece:{shape_reason}:{self.validation_score:.2f}"
    
    def _validate_shape(self) -> Tuple[bool, float, str]:
        """
        Valide la forme de la pièce en se basant sur les caractéristiques géométriques.
        
        Returns:
            Tuple de (est_valide, score_qualité, raison)
        """
        # Extraction des caractéristiques clés
        area = self.features['area']
        perimeter = self.features['perimeter']
        compactness = self.features.get('compactness', perimeter**2 / (4 * np.pi * area) if area > 0 else float('inf'))
        solidity = self.features.get('solidity', 0)
        
        # Vérification de la compacité (indicateur clé de forme pour les pièces de puzzle)
        # Les pièces de puzzle ont généralement une compacité entre 1.5 et 8.0
        compactness_score = 0.0
        if 1.5 <= compactness <= 10.0:
            compactness_score = 1.0 - min(abs(compactness - 4.0) / 6.0, 1.0)
        else:
            if compactness < 1.5:
                return False, 0.3, "too_compact"
            else:
                return False, 0.2, "too_irregular"
        
        # Vérification du ratio d'aspect
        x, y, w, h = self.features['bbox']
        aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else float('inf')
        
        aspect_score = 0.0
        if aspect_ratio <= 4.0:
            aspect_score = 1.0 - min((aspect_ratio - 1.0) / 3.0, 1.0)
        else:
            return False, 0.2, "bad_aspect_ratio"
        
        # Vérification de la solidité (proportion de l'aire par rapport à l'enveloppe convexe)
        solidity_score = 0.0
        if 0.6 <= solidity <= 0.98:
            solidity_score = min(solidity / 0.85, 1.0) if solidity <= 0.9 else 2.0 - solidity / 0.9
        else:
            if solidity < 0.6:
                return False, 0.3, "low_solidity"
            else:
                return False, 0.4, "too_solid"
        
        # Utilisation de validate_shape_as_puzzle_piece comme vérification supplémentaire
        if validate_shape_as_puzzle_piece(self.contour):
            shape_score = 1.0
        else:
            shape_score = 0.6
        
        # Score global combiné
        quality_score = (
            0.3 * compactness_score +
            0.2 * aspect_score +
            0.2 * solidity_score +
            0.3 * shape_score
        )
        
        return quality_score >= 0.5, quality_score, "valid_shape"
    
    def draw(self, image: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Visualisation optimisée de la pièce avec contour.
        
        Args:
            image: Image optionnelle sur laquelle dessiner (si None, utiliser l'image originale)
        
        Returns:
            Image avec visualisation de la pièce
        """
        if image is None:
            vis_img = self.image.copy()
        else:
            vis_img = image.copy()
        
        # Dessiner le contour
        color = (0, 255, 0) if self.is_valid else (0, 0, 255)  # Vert pour valide, rouge pour invalide
        cv2.drawContours(vis_img, [self.contour], -1, color, 2)
        
        # Dessiner l'ID de la pièce si disponible
        if self.id is not None:
            # Calculer le centroïde pour le placement du texte
            M = cv2.moments(self.contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Dessiner l'ID avec un fond contrastant pour la visibilité
                text = f"#{self.id}"
                text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(vis_img, 
                            (cx - 5, cy - text_size[1] - 5), 
                            (cx + text_size[0] + 5, cy + 5), 
                            (0, 0, 0), 
                            -1)
                cv2.putText(vis_img, text, (cx, cy),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Dessiner le score de validation si disponible
        if hasattr(self, 'validation_score') and self.validation_score > 0:
            # Trouver le coin supérieur gauche de la pièce pour le placement du texte
            x, y, _, _ = self.features['bbox']
            score_text = f"{self.validation_score:.2f}"
            cv2.putText(vis_img, score_text, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        return vis_img
    
    def calculate_match_score(self, other_piece: 'PuzzlePiece') -> float:
        """
        Calcule un score de correspondance simple entre cette pièce et une autre pièce.
        Basé uniquement sur des caractéristiques générales sans analyse des coins/bordures.
        
        Args:
            other_piece: Autre pièce de puzzle à comparer
        
        Returns:
            Score de correspondance (0-1)
        """
        if not self.is_valid or not other_piece.is_valid:
            return 0.0
        
        # Vérification de proximité spatiale
        c1 = self.features['centroid']
        c2 = other_piece.features['centroid']
        
        # Distance entre les centroïdes
        distance = np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)
        
        # Tailles des pièces
        size1 = np.sqrt(self.features['area'])
        size2 = np.sqrt(other_piece.features['area'])
        
        # Les pièces adjacentes devraient être proches mais pas trop (éviter les chevauchements)
        # La distance idéale est environ la moitié de la somme des dimensions des pièces
        ideal_distance = (size1 + size2) * 0.5
        distance_score = max(0, 1.0 - abs(distance - ideal_distance) / ideal_distance)
        
        # Similarité de taille (les pièces adjacentes ont généralement des tailles similaires)
        size_ratio = min(size1, size2) / max(size1, size2)
        
        # Score combiné
        match_score = 0.7 * distance_score + 0.3 * size_ratio
        
        return match_score
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convertit la pièce en représentation de dictionnaire.
        
        Returns:
            Dictionnaire avec les données de la pièce
        """
        # Créer une copie sérialisable des caractéristiques
        serializable_features = {}
        for key, value in self.features.items():
            if isinstance(value, np.ndarray):
                serializable_features[key] = value.tolist()
            elif key == 'min_area_rect':
                # Gérer min_area_rect qui a une structure spéciale
                center, size, angle = value
                serializable_features[key] = {
                    'center': (float(center[0]), float(center[1])),
                    'size': (float(size[0]), float(size[1])),
                    'angle': float(angle)
                }
            elif key == 'moments':
                # Convertir les moments (dictionnaire OpenCV avec types numpy)
                serializable_features[key] = {k: float(v) for k, v in value.items()}
            elif key == 'hu_moments' and isinstance(value, np.ndarray):
                # Convertir hu_moments (tableau numpy)
                serializable_features[key] = value.flatten().tolist()
            else:
                serializable_features[key] = value
        
        return {
            'id': self.id,
            'features': serializable_features,
            'is_valid': self.is_valid,
            'validation_status': self.validation_status,
            'validation_score': self.validation_score if hasattr(self, 'validation_score') else 0.0,
            'contour': self.contour.tolist() if self.contour is not None else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], image: np.ndarray, contour: Optional[np.ndarray] = None, config: Config = None) -> 'PuzzlePiece':
        """
        Crée une pièce à partir d'une représentation de dictionnaire.
        
        Args:
            data: Dictionnaire avec les données de la pièce
            image: Image source
            contour: Contour de la pièce (optionnel, peut être extrait des données)
            config: Paramètres de configuration
        
        Returns:
            Objet PuzzlePiece
        """
        # Récupérer le contour soit à partir du paramètre, soit des données
        if contour is None and 'contour' in data:
            # Utiliser asarray au lieu de array pour éviter une copie inutile
            contour = np.asarray(data['contour'])
        
        if contour is None:
            raise ValueError("Le contour doit être fourni soit comme paramètre, soit dans les données")
        
        piece = cls(image, contour, config)
        
        # Remplacer par les données sauvegardées
        piece.id = data.get('id')
        piece.is_valid = data.get('is_valid', False)
        piece.validation_status = data.get('validation_status')
        
        if 'validation_score' in data:
            piece.validation_score = data['validation_score']
        
        return piece