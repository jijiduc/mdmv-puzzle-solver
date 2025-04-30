"""
Ce script traite les images de pièces de puzzle sur un fond sombre et
détecte les pièces individuelles en utilisant des techniques de vision par ordinateur optimisées.
Toutes les sorties du terminal sont également enregistrées dans un fichier journal.
"""
import cv2
import numpy as np
import os
import argparse
import time
import sys
import logging
import shutil
from typing import Optional, List, Dict, Any
from datetime import datetime
from multiprocessing import cpu_count

# Import des modules de détection de pièces de puzzle
from src.config.settings import Config
from src.core.processor import PuzzleProcessor
from src.utils.image_utils import read_image

# Fonction de travailleur pour multiprocessing définie au niveau global
def worker_function(worker_id):
    """Fonction exécutée par chaque processus."""
    import os
    import time
    import random
    process_id = os.getpid()
    print(f"Processus {worker_id} démarré avec PID {process_id}")
    # Simuler un travail avec durée aléatoire
    duration = random.uniform(0.5, 2.0)
    time.sleep(duration)
    print(f"Processus {worker_id} terminé après {duration:.2f}s")
    return process_id

def verify_multiprocessing(num_processes):
    """
    Vérifie que le multiprocessing fonctionne correctement en créant des processus
    et en affichant leur ID.
    
    Args:
        num_processes: Nombre de processus à créer
    """
    import multiprocessing
    import os
    import time
    import random
    
    print(f"\n=== Test de {num_processes} processus parallèles ===")
    print(f"CPU principal: PID {os.getpid()}")
    
    # Mesurer le temps d'exécution
    start_time = time.time()
    
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(worker_function, range(num_processes))
    
    elapsed = time.time() - start_time
    
    # Vérifier que tous les processus ont des IDs différents
    unique_pids = set(results)
    
    print(f"\nTest terminé en {elapsed:.2f} secondes")
    print(f"Nombre de processus créés: {len(results)}")
    print(f"Nombre de PIDs uniques: {len(unique_pids)}")
    print(f"Parallélisation {'réussie' if len(unique_pids) > 1 else 'échouée'}")
    print("=" * 50)

def setup_logging(log_dir="logs"):
    """
    Configure la journalisation vers la console et un fichier avec encodage approprié.
    
    Args:
        log_dir: Répertoire pour sauvegarder les fichiers journaux
    
    Returns:
        Chemin vers le fichier journal
    """
    # Création du répertoire des journaux s'il n'existe pas
    os.makedirs(log_dir, exist_ok=True)
    
    # Création d'un nom de fichier journal avec horodatage
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = os.path.join(log_dir, f"puzzle_detection_{timestamp}.log")
    
    # Configuration du gestionnaire de fichiers avec journalisation détaillée
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s'))
    
    # Configuration du gestionnaire de console avec sortie minimale
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)  # Montrer également les infos dans la console
    console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    
    # Configuration du logger racine
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Création d'un logger de progression séparé pour la sortie console minimale
    progress_logger = logging.getLogger('progress')
    progress_logger.setLevel(logging.INFO)
    
    progress_logger.info(f"Traitement démarré. Journaux détaillés dans : {log_file}")
    
    return log_file


def clear_directory(directory_path):
    """
    Nettoie tous les fichiers d'un répertoire sans supprimer le répertoire lui-même.
    
    Args:
        directory_path: Chemin vers le répertoire à nettoyer
    """
    if os.path.exists(directory_path):
        for file_name in os.listdir(directory_path):
            file_path = os.path.join(directory_path, file_name)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                logging.error(f"Erreur lors de la suppression de {file_path}: {e}")
    else:
        # Création du répertoire s'il n'existe pas
        os.makedirs(directory_path, exist_ok=True)


def parse_arguments():
    """Parse les arguments de ligne de commande avec options améliorées"""
    parser = argparse.ArgumentParser(
        description="Détecteur de Pièces de Puzzle Optimisé pour la Segmentation"
    )
    
    # Paramètres requis
    parser.add_argument("--image", required=True, help="Chemin vers l'image du puzzle")
    
    # Paramètres optionnels standard
    parser.add_argument("--pieces", type=int, help="Nombre attendu de pièces dans le puzzle")
    parser.add_argument("--debug-dir", default="debug", help="Répertoire pour sauvegarder les sorties de débogage")
    parser.add_argument("--log-dir", default="logs", help="Répertoire pour sauvegarder les fichiers journaux")
    parser.add_argument("--extract", action="store_true", help="Extraire les pièces individuelles dans des fichiers séparés")
    parser.add_argument("--extract-dir", default="extracted_pieces", help="Répertoire pour sauvegarder les pièces extraites")
    parser.add_argument("--view", action="store_true", help="Afficher les résultats dans des fenêtres d'image")
    
    # Options de performance
    parser.add_argument("--fast-mode", action="store_true", help="Utiliser le mode rapide pour la détection")
    parser.add_argument("--use-multiprocessing", action="store_true", help="Utiliser le multitraitement pour une détection plus rapide")
    parser.add_argument("--processes", type=int, default=0, help="Nombre de processus à utiliser (0 = auto)")
    
    # Options de segmentation
    parser.add_argument("--min-area", type=float, default=500, help="Aire minimale des pièces en pixels²")
    parser.add_argument("--adaptive-preprocessing", action="store_true", help="Utiliser le prétraitement adaptatif")
    parser.add_argument("--analyze-image", action="store_true", help="Analyser les caractéristiques de l'image")
    
    # Options de vérification
    parser.add_argument("--area-verification", action="store_true", help="Appliquer la vérification finale par aire")
    parser.add_argument("--area-threshold", type=float, default=2.0, help="Seuil d'écart-type pour la vérification par aire")
    parser.add_argument("--comprehensive-verification", action="store_true", 
                   help="Appliquer la vérification complète combinant plusieurs méthodes")
    
    # Options de cache
    parser.add_argument("--use-cache", action="store_true", help="Utiliser le cache du pipeline pour accélérer les réexécutions")
    parser.add_argument("--cache-dir", default="cache", help="Répertoire pour le cache du pipeline")
    parser.add_argument("--max-cache-size", type=int, default=1000, help="Taille maximale du cache en Mo")
    parser.add_argument("--clear-cache", action="store_true", help="Vider le cache avant l'exécution")
    parser.add_argument("--cache-info", action="store_true", help="Afficher des informations sur le cache et quitter")
    
    # Options avancées
    parser.add_argument("--profile", action="store_true", help="Activer le profilage des performances")
    parser.add_argument("--debug", action="store_true", help="Activer le mode débogage avec sauvegarde d'images")
    parser.add_argument("--save-results", action="store_true", help="Sauvegarder les résultats de détection")
    parser.add_argument("--output-dir", default="results", help="Répertoire pour sauvegarder les résultats")
    
    # NOUVELLES OPTIONS pour la méthode simplifiée
    parser.add_argument("--use-simple-approach", action="store_true", 
                       help="Utiliser l'approche simple comme dans test2.py")
    parser.add_argument("--simple-threshold", type=int, default=30, 
                       help="Valeur de seuil pour l'approche simple (default: 30)")
    parser.add_argument("--simple-scale", type=int, default=30, 
                       help="Pourcentage d'échelle pour l'approche simple (default: 30%)")
    parser.add_argument("--contour-smoothing", action="store_true", 
                       help="Appliquer un lissage aux contours pour réduire la sur-segmentation")
    parser.add_argument("--detection-method", choices=["standard", "simple", "watershed", "hybrid"], 
                       default="standard", help="Méthode de détection à utiliser")
    parser.add_argument("--improve-contours", action="store_true", 
                       help="Appliquer des améliorations supplémentaires aux contours")
    
    return parser.parse_args()


def create_config(args):
    """Crée une configuration basée sur les arguments de ligne de commande."""
    config = Config()
    
    # Mise à jour de la configuration générale
    config.DEBUG_DIR = args.debug_dir
    config.DEBUG = args.debug
    
    # Mise à jour de la configuration de performance
    config.performance.USE_MULTIPROCESSING = args.use_multiprocessing
    config.performance.USE_FAST_MODE = args.fast_mode
    
    # Nombre de processus
    if args.processes > 0:
        config.performance.NUM_PROCESSES = args.processes
    else:
        config.performance.NUM_PROCESSES = max(1, cpu_count() - 1)
    
    # Configuration du prétraitement
    config.preprocessing.USE_ADAPTIVE = args.adaptive_preprocessing
    
    # Configuration de la détection des contours
    config.contour.MIN_AREA = args.min_area
    
    # Configuration de la vérification
    if args.area_verification:
        config.verification.VALIDATION_THRESHOLD = args.area_threshold
    
    return config

def process_image(processor, args):
    """
    Traite une image pour détecter les pièces de puzzle.
    
    Args:
        processor: Instance de PuzzleProcessor
        args: Arguments de ligne de commande
    
    Returns:
        Résultats du traitement
    """
    # Configurez les paramètres de traitement
    process_kwargs = {
        'fast_mode': args.fast_mode
    }
    
    # Ajoutez les options de vérification si activées
    if args.area_verification:
        process_kwargs['area_verification_threshold'] = args.area_threshold
        process_kwargs['use_area_verification'] = True
        process_kwargs['use_comprehensive_verification'] = args.comprehensive_verification
    
    # Traitez l'image
    return processor.process_image(args.image, args.pieces, **process_kwargs)


def display_results(results, expected_pieces: Optional[int] = None):
    """
    Affiche les résultats du traitement de manière formatée.
    
    Args:
        results: Dictionnaire avec les résultats du traitement
        expected_pieces: Nombre attendu de pièces
    """
    pieces = results['pieces']
    metrics = results['metrics']
    
    # Journalisation des informations détaillées dans le fichier uniquement
    logging.info("\n=== Résultats d'Analyse du Puzzle ===")
    logging.info(f"Détecté {len(pieces)} pièces de puzzle valides")
    
    if expected_pieces:
        detection_rate = len(pieces) / expected_pieces * 100
        logging.info(f"Taux de détection: {detection_rate:.1f}%")
    
    logging.info(f"Temps de traitement: {results['processing_time']:.2f} secondes")
    
    # Statistiques supplémentaires
    if 'mean_area' in metrics:
        logging.info(f"Aire moyenne des pièces: {metrics['mean_area']:.1f} pixels²")
    if 'std_area' in metrics:
        logging.info(f"Écart-type des aires: {metrics['std_area']:.1f} pixels²")
    
    # Affichage du résumé dans la console
    progress_logger = logging.getLogger('progress')
    progress_logger.info(f"Détecté {len(pieces)} pièces de puzzle")
    if expected_pieces:
        detection_rate = len(pieces) / expected_pieces * 100
        progress_logger.info(f"Taux de détection: {detection_rate:.1f}%")
    progress_logger.info(f"Temps de traitement: {results['processing_time']:.2f} secondes")


def view_images(results):
    """
    Affiche les images dans des fenêtres OpenCV.
    Version améliorée avec gestion des erreurs.
    
    Args:
        results: Dictionnaire avec les résultats du traitement
    """
    # Vérification que les visualisations nécessaires existent
    if 'visualizations' not in results or not results['visualizations']:
        logging.error("Pas de visualisations disponibles")
        
        # Créer une image d'erreur à afficher
        error_img = np.ones((300, 500, 3), dtype=np.uint8) * 255
        cv2.putText(error_img, "Erreur: Visualisations manquantes", (20, 50), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        cv2.imshow("Erreur", error_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return
    
    visualizations = results['visualizations']
    
    # Affichage de la visualisation du résumé si disponible
    if 'summary' in visualizations and visualizations['summary'] is not None and isinstance(visualizations['summary'], np.ndarray):
        cv2.imshow("Résumé d'Analyse du Puzzle", visualizations['summary'])
    
    # Affichage de la visualisation des métriques si disponible
    if 'metrics' in visualizations and visualizations['metrics'] is not None and isinstance(visualizations['metrics'], np.ndarray):
        cv2.imshow("Métriques", visualizations['metrics'])
    
    # Affichage des visualisations de pièces si disponibles
    if 'pieces' in visualizations and visualizations['pieces']:
        # Limiter le nombre de fenêtres à afficher pour ne pas surcharger l'écran
        max_pieces_to_show = min(5, len(visualizations['pieces']))
        for i in range(max_pieces_to_show):
            if i < len(visualizations['pieces']) and visualizations['pieces'][i] is not None and isinstance(visualizations['pieces'][i], np.ndarray):
                cv2.imshow(f"Pièce #{i+1}", visualizations['pieces'][i])
    
    # Affichage de la visualisation de vérification si disponible
    if 'verification' in visualizations and visualizations['verification'] is not None and isinstance(visualizations['verification'], np.ndarray):
        cv2.imshow("Vérification", visualizations['verification'])
    
    # Attente de la fermeture des fenêtres par l'utilisateur
    logging.info("Appuyez sur une touche pour fermer les fenêtres d'image et continuer...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    """Point d'entrée principal pour le détecteur de pièces de puzzle avec options améliorées"""
    # Démarrage du chronométrage
    start_time = time.time()
    
    # Analyse des arguments de ligne de commande
    args = parse_arguments()
    
    # Configuration de la journalisation vers la console et le fichier
    log_file = setup_logging(args.log_dir)
    
    # Création de la configuration
    config = create_config(args)
    
    # Création du processeur avec configuration du cache
    processor = PuzzleProcessor(
        config, 
        enable_cache=args.use_cache,
        cache_dir=args.cache_dir,
        max_cache_size_mb=args.max_cache_size
    )
    
    # Gestion des options du cache si demandées
    if args.cache_info:
        cache_stats = processor.manage_cache()
        progress_logger = logging.getLogger('progress')
        progress_logger.info("=== Informations sur le Cache du Pipeline ===")
        progress_logger.info(f"État: {cache_stats['status']}")
        
        if cache_stats['status'] == 'active':
            progress_logger.info(f"Nombre d'entrées: {cache_stats['entries']}")
            progress_logger.info(f"Taille: {cache_stats['size_mb']:.2f} Mo")
            progress_logger.info(f"Taille maximale: {cache_stats['max_size_mb']} Mo")
            progress_logger.info(f"Répertoire: {cache_stats['directory']}")
        else:
            progress_logger.info(f"Message: {cache_stats['message']}")
        
        progress_logger.info("===================================")
        return
    
    # Vider le cache si demandé
    if args.clear_cache and args.use_cache:
        processor.manage_cache(clear_cache=True)
        progress_logger = logging.getLogger('progress')
        progress_logger.info("Cache du pipeline vidé avec succès")
    
    # Nettoyage des répertoires de débogage et d'extraction de pièces
    logging.info("Nettoyage des répertoires de sortie...")
    clear_directory(args.debug_dir)
    if args.extract:
        clear_directory(args.extract_dir)
    
    # Affichage du message d'initialisation
    logging.info("== Détecteur de Pièces de Puzzle Optimisé ==")
    logging.info(f"Traitement de l'image: {args.image}")
    logging.info(f"Répertoire de sortie de débogage: {args.debug_dir}")
    logging.info(f"Fichier journal: {log_file}")
    
    if args.use_cache:
        logging.info(f"Cache du pipeline activé (répertoire: {args.cache_dir})")
    
    if args.pieces:
        logging.info(f"Pièces attendues: {args.pieces}")
    
    # Journalisation des nouveaux paramètres
    if args.detection_method != "standard":
        logging.info(f"Méthode de détection: {args.detection_method}")
    
    if args.use_simple_approach:
        logging.info(f"Utilisation de l'approche simplifiée avec seuil={args.simple_threshold}, échelle={args.simple_scale}%")
    
    if args.contour_smoothing:
        logging.info("Lissage des contours activé")
    
    if args.improve_contours:
        logging.info("Amélioration des contours activée")
    
    # Analyse des caractéristiques de l'image si demandé
    if args.analyze_image:
        logging.info("Analyse des caractéristiques de l'image...")
        analysis = processor.analyze_image_characteristics(args.image)
        
        logging.info(f"Contraste de l'image: {analysis['contrast']:.2f}")
        logging.info(f"Densité des bords: {analysis['edge_density']:.3f}")
        logging.info(f"Fond sombre: {analysis['is_dark_background']}")
        logging.info(f"Histogramme bimodal: {analysis['is_bimodal']}")
        
        # Mise à jour de la configuration basée sur l'analyse
        config.optimize_for_image_characteristics(analysis)
        logging.info("Configuration optimisée en fonction des caractéristiques de l'image")
    
    # Traitement de l'image avec les nouvelles options
    if args.detection_method != "standard" or args.use_simple_approach:
        # Utiliser la méthode de détection appropriée
        if args.detection_method == "simple" or args.use_simple_approach:
            results = processor.process_image_hybrid(
                args.image,
                expected_pieces=args.pieces,
                use_simple_approach=True,
                simple_threshold=args.simple_threshold,
                simple_scale=args.simple_scale,
                use_area_verification=args.area_verification,
                area_verification_threshold=args.area_threshold,
                use_comprehensive_verification=args.comprehensive_verification,
                improve_contours=args.improve_contours,
                contour_smoothing=args.contour_smoothing
            )
        elif args.detection_method in ["watershed", "hybrid"]:
            results = processor.process_image_with_method(
                args.image,
                method=args.detection_method,
                expected_pieces=args.pieces,
                threshold=args.simple_threshold,
                scale_percent=args.simple_scale,
                use_area_verification=args.area_verification,
                area_verification_threshold=args.area_threshold,
                use_comprehensive_verification=args.comprehensive_verification,
                improve_contours=args.improve_contours
            )
        else:
            # Ne devrait pas se produire à cause des choix restreints dans argparse
            results = processor.process_image(
                args.image,
                expected_pieces=args.pieces,
                fast_mode=args.fast_mode,
                use_area_verification=args.area_verification,
                area_verification_threshold=args.area_threshold,
                use_comprehensive_verification=args.comprehensive_verification
            )
    else:
        # Utiliser la méthode de traitement standard
        results = processor.process_image(
            args.image,
            expected_pieces=args.pieces,
            fast_mode=args.fast_mode,
            use_area_verification=args.area_verification,
            area_verification_threshold=args.area_threshold,
            use_comprehensive_verification=args.comprehensive_verification
        )
    
    # Extraction des pièces individuelles si demandé
    if args.extract:
        extracted_paths = processor.extract_pieces(results['pieces'], args.extract_dir)
        logging.info(f"Extrait {len(extracted_paths)} pièces vers {args.extract_dir}")
    
    # Affichage des résultats
    display_results(results, args.pieces)
    
    # Sauvegarde des résultats si demandé
    if args.save_results:
        result_dir = processor.save_results(results, args.output_dir)
        logging.info(f"Résultats sauvegardés dans {result_dir}")
    
    # Affichage des images si demandé
    if args.view:
        view_images(results)
        
    # Rapport du temps de traitement total
    total_time = time.time() - start_time
    logging.info(f"Temps de traitement total: {total_time:.2f} secondes")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Erreur: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        sys.exit(1)