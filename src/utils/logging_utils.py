"""
Utilitaires de journalisation optimisés pour le programme de détection de pièces de puzzle.
"""

import logging
import os
import sys
from typing import Optional, Dict
from datetime import datetime


class LogManager:
    """Gestionnaire centralisé de journalisation avec contrôle des niveaux"""
    
    def __init__(self):
        self.loggers = {}
        self.main_logger = None
        self.progress_logger = None
        self.log_file = None
        self.verbosity = logging.INFO
    
    def setup(self, log_dir: str = "logs", verbosity: str = "INFO", log_prefix: str = "puzzle_detection") -> str:
        """
        Configure la journalisation avec différents niveaux de verbosité.
        
        Args:
            log_dir: Répertoire pour les fichiers journaux
            verbosity: Niveau de verbosité ('DEBUG', 'INFO', 'WARNING', 'ERROR')
            log_prefix: Préfixe pour le nom du fichier journal
            
        Returns:
            Chemin vers le fichier journal
        """
        # Convertir le niveau de verbosité
        level_map = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR
        }
        self.verbosity = level_map.get(verbosity.upper(), logging.INFO)
        
        # Créer le répertoire des journaux
        os.makedirs(log_dir, exist_ok=True)
        
        # Créer le nom du fichier journal
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_file = os.path.join(log_dir, f"{log_prefix}_{timestamp}.log")
        
        # Configurer le gestionnaire de fichiers
        file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)  # Toujours journaliser tous les détails dans le fichier
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
        ))
        
        # Configurer le gestionnaire de console
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.verbosity)
        
        # Formateur minimaliste pour l'information
        info_formatter = logging.Formatter('%(message)s')
        # Formateur détaillé pour les avertissements et erreurs
        detail_formatter = logging.Formatter('%(levelname)s: %(message)s')
        
        # Définir le formateur en fonction du niveau
        class LevelDependentFormatter(logging.Formatter):
            def format(self, record):
                if record.levelno <= logging.INFO:
                    return info_formatter.format(record)
                else:
                    return detail_formatter.format(record)
        
        console_handler.setFormatter(LevelDependentFormatter())
        
        # Configurer le logger racine
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)  # Capture tout, filtre au niveau des handlers
        
        # Supprimer les gestionnaires existants pour éviter la duplication
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
            
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        
        # Créer et configurer le logger de progression
        self.progress_logger = logging.getLogger('progress')
        self.progress_logger.setLevel(logging.INFO)
        
        # Garder une référence au logger principal
        self.main_logger = root_logger
        
        self.progress_logger.info(f"Journalisation initialisée. Niveau: {verbosity}")
        self.progress_logger.info(f"Fichier journal: {self.log_file}")
        
        return self.log_file
    
    def get_logger(self, name: str) -> logging.Logger:
        """
        Obtient un logger configuré pour un module spécifique.
        
        Args:
            name: Nom du module/composant
            
        Returns:
            Logger configuré
        """
        if name in self.loggers:
            return self.loggers[name]
        
        logger = logging.getLogger(name)
        self.loggers[name] = logger
        return logger
    
    def set_verbosity(self, verbosity: str) -> None:
        """
        Modifie dynamiquement le niveau de verbosité de la console.
        
        Args:
            verbosity: Niveau de verbosité ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        """
        level_map = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR
        }
        level = level_map.get(verbosity.upper(), logging.INFO)
        
        # Mettre à jour le niveau du gestionnaire de console
        for handler in self.main_logger.handlers:
            if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stdout:
                handler.setLevel(level)
                break
        
        self.verbosity = level
        self.progress_logger.info(f"Niveau de verbosité modifié à: {verbosity}")


# Instance globale pour utilisation dans tous les modules
log_manager = LogManager()