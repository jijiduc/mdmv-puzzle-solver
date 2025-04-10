"""
Utilitaires de mise en cache pour accélérer les réexécutions du pipeline de traitement d'image.
"""

import os
import json
import time
import hashlib
import pickle
import logging
import cv2
import numpy as np
from typing import Dict, Any, Optional, Tuple, List, Union

# Obtenir le logger
logger = logging.getLogger(__name__)

class PipelineCache:
    """
    Gère la mise en cache des résultats intermédiaires du pipeline de détection de pièces de puzzle.
    Optimise les performances en évitant de refaire des calculs coûteux lors des réexécutions.
    """
    
    def __init__(self, cache_dir: str = "cache", enable_cache: bool = True, 
                max_cache_size_mb: int = 1000, max_cache_age_days: int = 7):
        """
        Initialise le gestionnaire de cache.
        
        Args:
            cache_dir: Répertoire pour stocker les fichiers de cache
            enable_cache: Activer ou désactiver le cache
            max_cache_size_mb: Taille maximale du cache en méga-octets
            max_cache_age_days: Âge maximal des fichiers de cache en jours
        """
        self.cache_dir = cache_dir
        self.enable_cache = enable_cache
        self.max_cache_size_mb = max_cache_size_mb
        self.max_cache_age_days = max_cache_age_days
        self.cache_index = {}
        self.index_path = os.path.join(cache_dir, "cache_index.json")
        
        # Créer le répertoire de cache s'il n'existe pas
        if self.enable_cache:
            os.makedirs(cache_dir, exist_ok=True)
            self._load_index()
            self._clean_old_cache()
    
    def _load_index(self):
        """Charge l'index de cache à partir du disque."""
        if os.path.exists(self.index_path):
            try:
                with open(self.index_path, 'r') as f:
                    self.cache_index = json.load(f)
                logger.info(f"Index de cache chargé: {len(self.cache_index)} entrées")
            except Exception as e:
                logger.warning(f"Erreur lors du chargement de l'index de cache: {str(e)}")
                self.cache_index = {}
        else:
            self.cache_index = {}
    
    def _save_index(self):
        """Enregistre l'index de cache sur le disque."""
        if not self.enable_cache:
            return
            
        try:
            with open(self.index_path, 'w') as f:
                json.dump(self.cache_index, f)
        except Exception as e:
            logger.warning(f"Erreur lors de l'enregistrement de l'index de cache: {str(e)}")
    
    def _clean_old_cache(self):
        """Nettoie les fichiers de cache obsolètes et vérifie la taille totale."""
        if not self.enable_cache:
            return
            
        try:
            # Calculer l'âge maximum en secondes
            max_age_seconds = self.max_cache_age_days * 24 * 60 * 60
            current_time = time.time()
            
            # Vérifier chaque entrée de cache
            entries_to_remove = []
            total_size_mb = 0
            
            for key, entry in self.cache_index.items():
                # Vérifier l'âge
                if current_time - entry.get('timestamp', 0) > max_age_seconds:
                    entries_to_remove.append(key)
                    continue
                
                # Vérifier que le fichier existe
                cache_path = entry.get('path')
                if not cache_path or not os.path.exists(cache_path):
                    entries_to_remove.append(key)
                    continue
                
                # Calculer la taille totale
                total_size_mb += os.path.getsize(cache_path) / (1024 * 1024)
            
            # Supprimer les entrées obsolètes
            for key in entries_to_remove:
                self._remove_cache_entry(key)
            
            # Si le cache est trop grand, supprimer les entrées les plus anciennes
            if total_size_mb > self.max_cache_size_mb:
                logger.info(f"Taille du cache ({total_size_mb:.1f} MB) dépasse la limite ({self.max_cache_size_mb} MB)")
                
                # Trier les entrées par date (ancien -> récent)
                entries = [(k, v.get('timestamp', 0)) for k, v in self.cache_index.items()]
                entries.sort(key=lambda x: x[1])
                
                # Supprimer les entrées les plus anciennes jusqu'à ce que la taille soit acceptable
                current_size_mb = total_size_mb
                for key, _ in entries:
                    if current_size_mb <= self.max_cache_size_mb * 0.9:  # Conserver une marge de 10%
                        break
                        
                    entry = self.cache_index.get(key)
                    if entry and 'path' in entry and os.path.exists(entry['path']):
                        file_size_mb = os.path.getsize(entry['path']) / (1024 * 1024)
                        self._remove_cache_entry(key)
                        current_size_mb -= file_size_mb
            
            logger.info(f"Nettoyage du cache: {len(entries_to_remove)} entrées obsolètes supprimées")
            
        except Exception as e:
            logger.warning(f"Erreur lors du nettoyage du cache: {str(e)}")
    
    def _remove_cache_entry(self, key: str):
        """
        Supprime une entrée de cache.
        
        Args:
            key: Clé de l'entrée à supprimer
        """
        if key in self.cache_index:
            try:
                cache_path = self.cache_index[key].get('path')
                if cache_path and os.path.exists(cache_path):
                    os.unlink(cache_path)
                del self.cache_index[key]
            except Exception as e:
                logger.warning(f"Erreur lors de la suppression de l'entrée de cache {key}: {str(e)}")
    
    def clear_cache(self):
        """Vide complètement le cache."""
        if not self.enable_cache:
            return
            
        try:
            logger.info("Vidage du cache...")
            # Supprimer tous les fichiers de cache
            for key, entry in self.cache_index.items():
                cache_path = entry.get('path')
                if cache_path and os.path.exists(cache_path):
                    os.unlink(cache_path)
            
            # Réinitialiser l'index
            self.cache_index = {}
            self._save_index()
            logger.info("Cache vidé avec succès")
            
        except Exception as e:
            logger.warning(f"Erreur lors du vidage du cache: {str(e)}")
    
    def generate_cache_key(self, image_path: str, stage: str, parameters: Dict[str, Any]) -> str:
        """
        Génère une clé de cache unique basée sur le chemin de l'image, l'étape et les paramètres.
        
        Args:
            image_path: Chemin vers l'image d'entrée
            stage: Nom de l'étape du pipeline
            parameters: Paramètres de traitement utilisés
        
        Returns:
            Clé de cache unique
        """
        if not self.enable_cache:
            return ""
            
        try:
            # Obtenir le timestamp du fichier et la taille
            file_info = os.stat(image_path)
            file_timestamp = file_info.st_mtime
            file_size = file_info.st_size
            
            # Créer un dictionnaire avec toutes les informations pertinentes
            key_data = {
                'path': os.path.abspath(image_path),
                'timestamp': file_timestamp,
                'size': file_size,
                'stage': stage,
                'params': parameters
            }
            
            # Convertir en JSON et calculer un hash
            key_str = json.dumps(key_data, sort_keys=True)
            hash_obj = hashlib.md5(key_str.encode())
            
            return hash_obj.hexdigest()
            
        except Exception as e:
            logger.warning(f"Erreur lors de la génération de la clé de cache: {str(e)}")
            return ""
    
    def get_from_cache(self, cache_key: str) -> Optional[Any]:
        """
        Récupère un résultat mis en cache.
        
        Args:
            cache_key: Clé de cache
        
        Returns:
            Données mises en cache ou None si non trouvées
        """
        if not self.enable_cache or not cache_key:
            return None
            
        try:
            # Vérifier si la clé existe
            if cache_key not in self.cache_index:
                return None
                
            entry = self.cache_index[cache_key]
            cache_path = entry.get('path')
            
            # Vérifier que le fichier existe
            if not cache_path or not os.path.exists(cache_path):
                # Nettoyer l'index
                self._remove_cache_entry(cache_key)
                return None
            
            # Charger les données
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
            
            logger.info(f"Résultat chargé depuis le cache: {cache_key}")
            return cached_data
            
        except Exception as e:
            logger.warning(f"Erreur lors de la récupération depuis le cache: {str(e)}")
            return None
    
    def save_to_cache(self, cache_key: str, data: Any, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Enregistre un résultat dans le cache.
        
        Args:
            cache_key: Clé de cache
            data: Données à mettre en cache
            metadata: Métadonnées supplémentaires (optionnel)
        
        Returns:
            True si l'enregistrement a réussi, False sinon
        """
        if not self.enable_cache or not cache_key:
            return False
            
        try:
            # Créer un nom de fichier unique
            cache_filename = f"{cache_key}.pkl"
            cache_path = os.path.join(self.cache_dir, cache_filename)
            
            # Enregistrer les données
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            
            # Mettre à jour l'index
            self.cache_index[cache_key] = {
                'path': cache_path,
                'timestamp': time.time(),
                'metadata': metadata or {}
            }
            
            # Enregistrer l'index
            self._save_index()
            
            logger.info(f"Résultat enregistré dans le cache: {cache_key}")
            return True
            
        except Exception as e:
            logger.warning(f"Erreur lors de l'enregistrement dans le cache: {str(e)}")
            return False

# Fonctions utilitaires pour les types spécifiques de cache

def cache_preprocessing(cache: PipelineCache, image_path: str, fast_mode: bool, adaptive: bool, 
                      expected_pieces: Optional[int] = None) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Récupère ou calcule les résultats de prétraitement d'image.
    
    Args:
        cache: Instance de PipelineCache
        image_path: Chemin vers l'image d'entrée
        fast_mode: Mode rapide
        adaptive: Utiliser le prétraitement adaptatif
        expected_pieces: Nombre attendu de pièces
    
    Returns:
        Tuple de (image prétraitée, image binaire, image des bords) ou None si le cache est désactivé
    """
    # Paramètres pour la clé de cache
    params = {
        'fast_mode': fast_mode,
        'adaptive': adaptive,
        'expected_pieces': expected_pieces
    }
    
    # Générer la clé de cache
    cache_key = cache.generate_cache_key(image_path, 'preprocessing', params)
    
    # Essayer de récupérer depuis le cache
    cached_result = cache.get_from_cache(cache_key)
    if cached_result is not None:
        logger.info("Utilisation des résultats de prétraitement mis en cache")
        return cached_result
    
    # Le résultat n'est pas dans le cache, il sera calculé par le pipeline
    return None

def cache_contours(cache: PipelineCache, image_path: str, binary_image: np.ndarray, 
                 min_area: float, parameters: Dict[str, Any]) -> Optional[List[np.ndarray]]:
    """
    Récupère ou calcule les contours détectés.
    
    Args:
        cache: Instance de PipelineCache
        image_path: Chemin vers l'image d'entrée
        binary_image: Image binaire
        min_area: Aire minimale des contours
        parameters: Paramètres supplémentaires
    
    Returns:
        Liste des contours ou None si le cache est désactivé
    """
    # Paramètres pour la clé de cache
    # Calculer un hash sur l'image binaire pour l'intégrer à la clé
    binary_hash = hashlib.md5(binary_image.tobytes()).hexdigest()
    
    params = {
        'binary_hash': binary_hash,
        'min_area': min_area,
        **parameters
    }
    
    # Générer la clé de cache
    cache_key = cache.generate_cache_key(image_path, 'contours', params)
    
    # Essayer de récupérer depuis le cache
    cached_result = cache.get_from_cache(cache_key)
    if cached_result is not None:
        logger.info("Utilisation des contours mis en cache")
        return cached_result
    
    # Le résultat n'est pas dans le cache, il sera calculé par le pipeline
    return None

def cache_pieces(cache: PipelineCache, image_path: str, contours: List[np.ndarray], 
               parameters: Dict[str, Any]) -> Optional[List[Any]]:
    """
    Récupère ou calcule les objets PuzzlePiece.
    
    Args:
        cache: Instance de PipelineCache
        image_path: Chemin vers l'image d'entrée
        contours: Liste des contours
        parameters: Paramètres supplémentaires
    
    Returns:
        Liste des pièces ou None si le cache est désactivé
    """
    # Paramètres pour la clé de cache
    # Les contours sont difficiles à hasher directement, on utilise leur nombre et quelques propriétés
    contour_summaries = []
    for i, contour in enumerate(contours[:min(20, len(contours))]):  # Limiter pour des raisons de performance
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        x, y, w, h = cv2.boundingRect(contour)
        contour_summaries.append((i, area, perimeter, x, y, w, h))
    
    params = {
        'contour_count': len(contours),
        'contour_summaries': contour_summaries,
        **parameters
    }
    
    # Générer la clé de cache
    cache_key = cache.generate_cache_key(image_path, 'pieces', params)
    
    # Essayer de récupérer depuis le cache
    cached_result = cache.get_from_cache(cache_key)
    if cached_result is not None:
        logger.info("Utilisation des pièces mises en cache")
        return cached_result
    
    # Le résultat n'est pas dans le cache, il sera calculé par le pipeline
    return None

def save_preprocessing_to_cache(cache: PipelineCache, image_path: str, fast_mode: bool, adaptive: bool, 
                              expected_pieces: Optional[int], result: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    """
    Enregistre les résultats de prétraitement dans le cache.
    
    Args:
        cache: Instance de PipelineCache
        image_path: Chemin vers l'image d'entrée
        fast_mode: Mode rapide
        adaptive: Utiliser le prétraitement adaptatif
        expected_pieces: Nombre attendu de pièces
        result: Résultat à mettre en cache
    """
    # Paramètres pour la clé de cache
    params = {
        'fast_mode': fast_mode,
        'adaptive': adaptive,
        'expected_pieces': expected_pieces
    }
    
    # Générer la clé de cache
    cache_key = cache.generate_cache_key(image_path, 'preprocessing', params)
    
    # Métadonnées
    metadata = {
        'fast_mode': fast_mode,
        'adaptive': adaptive,
        'image_path': image_path,
        'timestamp': time.time()
    }
    
    # Enregistrer dans le cache
    cache.save_to_cache(cache_key, result, metadata)

def save_contours_to_cache(cache: PipelineCache, image_path: str, binary_image: np.ndarray, 
                         min_area: float, parameters: Dict[str, Any], contours: List[np.ndarray]) -> None:
    """
    Enregistre les contours dans le cache.
    
    Args:
        cache: Instance de PipelineCache
        image_path: Chemin vers l'image d'entrée
        binary_image: Image binaire
        min_area: Aire minimale des contours
        parameters: Paramètres supplémentaires
        contours: Contours à mettre en cache
    """
    # Calculer un hash sur l'image binaire
    binary_hash = hashlib.md5(binary_image.tobytes()).hexdigest()
    
    params = {
        'binary_hash': binary_hash,
        'min_area': min_area,
        **parameters
    }
    
    # Générer la clé de cache
    cache_key = cache.generate_cache_key(image_path, 'contours', params)
    
    # Métadonnées
    metadata = {
        'contour_count': len(contours),
        'image_path': image_path,
        'timestamp': time.time()
    }
    
    # Enregistrer dans le cache
    cache.save_to_cache(cache_key, contours, metadata)

def save_pieces_to_cache(cache: PipelineCache, image_path: str, contours: List[np.ndarray], 
                       parameters: Dict[str, Any], pieces: List[Any]) -> None:
    """
    Enregistre les objets PuzzlePiece dans le cache.
    
    Args:
        cache: Instance de PipelineCache
        image_path: Chemin vers l'image d'entrée
        contours: Liste des contours
        parameters: Paramètres supplémentaires
        pieces: Pièces à mettre en cache
    """
    # Les contours sont difficiles à hasher directement, on utilise leur nombre et quelques propriétés
    contour_summaries = []
    for i, contour in enumerate(contours[:min(20, len(contours))]):  # Limiter pour des raisons de performance
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        x, y, w, h = cv2.boundingRect(contour)
        contour_summaries.append((i, area, perimeter, x, y, w, h))
    
    params = {
        'contour_count': len(contours),
        'contour_summaries': contour_summaries,
        **parameters
    }
    
    # Générer la clé de cache
    cache_key = cache.generate_cache_key(image_path, 'pieces', params)
    
    # Métadonnées
    metadata = {
        'piece_count': len(pieces),
        'image_path': image_path,
        'timestamp': time.time()
    }
    
    # Enregistrer dans le cache
    cache.save_to_cache(cache_key, pieces, metadata)