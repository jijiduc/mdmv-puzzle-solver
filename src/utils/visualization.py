"""
Utilitaires de visualisation optimisés pour l'affichage et le débogage
de la détection de pièces de puzzle, avec focus sur la performance.
"""

import cv2
import numpy as np
import os
from typing import List, Tuple, Dict, Any, Optional
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import io
from PIL import Image
import time


def draw_contours(image: np.ndarray,
                 contours: List[np.ndarray],
                 color: Tuple[int, int, int] = (0, 255, 0),
                 thickness: int = 2,
                 draw_index: bool = False,
                 fill: bool = False) -> np.ndarray:
    """
    Dessine les contours sur une image de façon optimisée.
    
    Args:
        image: Image d'entrée
        contours: Liste des contours à dessiner
        color: Tuple de couleur BGR
        thickness: Épaisseur de la ligne (-1 pour remplir)
        draw_index: Dessiner ou non les indices des contours
        fill: Remplir les contours plutôt que dessiner juste le contour
    
    Returns:
        Image avec les contours dessinés
    """
    result = image.copy()
    
    # Optimisation: traiter séparément le cas de remplissage
    if fill:
        # Créer un masque pour le remplissage
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, contours, -1, 255, -1)
        
        # Créer une image de couleur
        color_img = np.zeros_like(result)
        color_img[:] = (*color, 255) if len(image.shape) == 3 else color[0]
        
        # Appliquer le masque
        mask_3ch = cv2.merge([mask, mask, mask]) if len(image.shape) == 3 else mask
        np.copyto(result, color_img, where=(mask_3ch > 0))
    else:
        # Dessiner les contours normalement
        cv2.drawContours(result, contours, -1, color, thickness)
    
    if draw_index:
        # Dessiner les indices des contours
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        font_color = (255, 255, 255)
        
        for i, contour in enumerate(contours):
            # Calculer le centroïde pour le placement du texte
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Créer un arrière-plan pour le texte pour plus de lisibilité
                text = str(i)
                text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
                cv2.rectangle(result, 
                            (cx - 5, cy - text_size[1] - 5), 
                            (cx + text_size[0] + 5, cy + 5), 
                            (0, 0, 0), -1)
                
                # Dessiner l'indice du contour
                cv2.putText(result, text, (cx, cy), font, font_scale, font_color, font_thickness)
    
    return result


def draw_mask(image: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int] = (0, 255, 0), alpha: float = 0.5) -> np.ndarray:
    """
    Affiche un masque binaire sur une image avec transparence.
    
    Args:
        image: Image d'entrée
        mask: Masque binaire
        color: Couleur à utiliser pour le masque
        alpha: Niveau de transparence (0-1)
    
    Returns:
        Image avec masque superposé
    """
    # Vérifier que le masque est binaire
    if not np.array_equal(np.unique(mask), np.array([0, 255])) and not np.array_equal(np.unique(mask), np.array([0])) and not np.array_equal(np.unique(mask), np.array([255])):
        # Convertir en binaire si ce n'est pas le cas
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # Créer une copie de l'image
    result = image.copy()
    
    # Créer une image de couleur pour le masque
    overlay = np.zeros_like(result)
    
    # Appliquer la couleur au masque
    if len(image.shape) == 3:  # Image couleur
        overlay[mask > 0] = color
    else:  # Image en niveaux de gris
        overlay[mask > 0] = color[0]
    
    # Fusionner les images avec transparence
    cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0, result)
    
    return result


def create_grid_visualization(images: List[Tuple[np.ndarray, str]],
                             cols: int = 3,
                             figsize: Tuple[int, int] = (15, 10),
                             dpi: int = 100,
                             title: str = "") -> np.ndarray:
    """
    Crée une visualisation en grille de plusieurs images.
    
    Args:
        images: Liste de tuples (image, titre)
        cols: Nombre de colonnes dans la grille
        figsize: Taille de la figure en pouces
        dpi: Points par pouce
        title: Titre général pour la figure
    
    Returns:
        Visualisation en grille sous forme de tableau numpy
    """
    # Optimisation: pas besoin de créer une grille pour une seule image
    if len(images) == 1:
        img, img_title = images[0]
        fig = plt.figure(figsize=figsize, dpi=dpi)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 and img.shape[2] == 3 else img, cmap='gray')
        plt.title(img_title)
        plt.axis('off')
        if title:
            plt.suptitle(title, fontsize=16)
        
        # Convertir la figure matplotlib en tableau numpy
        buf = io.BytesIO()
        plt.tight_layout()
        fig.savefig(buf, format='png')
        buf.seek(0)
        
        img_arr = np.array(Image.open(buf))
        plt.close(fig)
        
        # Convertir RGB en BGR pour OpenCV
        if len(img_arr.shape) == 3 and img_arr.shape[2] == 3:
            img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
        
        return img_arr
        
    rows = (len(images) + cols - 1) // cols
    
    fig = plt.figure(figsize=figsize, dpi=dpi)
    
    if title:
        plt.suptitle(title, fontsize=16)
    
    for i, (img, img_title) in enumerate(images):
        ax = fig.add_subplot(rows, cols, i + 1)
        
        # Gestion des images RGB et niveaux de gris
        if len(img.shape) == 3 and img.shape[2] == 3:
            # Convertir BGR en RGB pour matplotlib
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            ax.imshow(img, cmap='gray')
        
        ax.set_title(img_title)
        ax.axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96] if title else [0, 0, 1, 1])
    
    # Convertir la figure matplotlib en tableau numpy
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    
    img_arr = np.array(Image.open(buf))
    plt.close(fig)
    
    # Convertir RGB en BGR pour OpenCV
    if len(img_arr.shape) == 3 and img_arr.shape[2] == 3:
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
    
    return img_arr


def create_processing_visualization(original_image: np.ndarray,
                                  preprocessed: np.ndarray,
                                  binary: np.ndarray,
                                  contours_image: np.ndarray,
                                  detected_pieces: List[Dict[str, Any]],
                                  output_path: str = None) -> np.ndarray:
    """
    Crée une visualisation complète du pipeline de traitement.
    Optimisé pour la performance et la clarté.
    
    Args:
        original_image: Image d'entrée originale
        preprocessed: Image prétraitée en niveaux de gris
        binary: Image binaire après seuillage
        contours_image: Image avec les contours détectés
        detected_pieces: Liste d'informations sur les pièces détectées
        output_path: Chemin optionnel pour sauvegarder la visualisation
    
    Returns:
        Image de visualisation
    """
    start_time = time.time()
    
    # Créer des paires image/titre
    images = [
        (original_image, "Image Originale"),
        (preprocessed, "Image Prétraitée"),
        (binary, "Image Binaire"),
        (contours_image, "Contours Détectés")
    ]
    
    # Ajouter jusqu'à deux pièces détectées si disponibles
    pieces_to_show = min(2, len(detected_pieces))
    
    for i in range(pieces_to_show):
        piece_info = detected_pieces[i]
        piece_img = piece_info.get('visualization', None)
        
        if piece_img is not None:
            images.append((piece_img, f"Pièce #{i+1}"))
    
    # Créer la visualisation en grille
    vis_img = create_grid_visualization(images, cols=2, figsize=(18, 14), dpi=100,
                                      title="Détection de Pièces de Puzzle")
    
    # Ajouter du texte avec le résumé de la détection
    h, w = vis_img.shape[:2]
    
    # S'assurer que l'image de visualisation a 3 canaux (BGR)
    if len(vis_img.shape) == 2:  # Niveaux de gris
        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_GRAY2BGR)
    elif vis_img.shape[2] == 4:  # RGBA
        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGBA2BGR)
    
    # Créer une image de texte avec la même largeur et 3 canaux
    text_img = np.ones((150, w, 3), dtype=np.uint8) * 255
    
    # Ajouter le texte de résumé
    font = cv2.FONT_HERSHEY_SIMPLEX
    elapsed_time = time.time() - start_time
    
    cv2.putText(text_img, f"Pièces Détectées: {len(detected_pieces)}",
                (20, 50), font, 1.2, (0, 0, 0), 2)
    cv2.putText(text_img, f"Dimensions de l'image: {original_image.shape[1]}x{original_image.shape[0]}",
                (20, 90), font, 0.9, (0, 0, 0), 2)
    cv2.putText(text_img, f"Temps de traitement: {elapsed_time:.3f}s",
                (20, 130), font, 0.9, (0, 0, 0), 2)
    
    # Combiner la visualisation et le texte
    result = np.vstack((vis_img, text_img))
    
    # Sauvegarder si un chemin de sortie est fourni
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, result)
    
    return result


def display_metrics(metrics: Dict[str, Any],
                   figsize: Tuple[int, int] = (12, 8)) -> np.ndarray:
    """
    Crée un affichage visuel des métriques de détection.
    Version simplifiée sans visualisation des coins/bordures.
    
    Args:
        metrics: Dictionnaire de métriques
        figsize: Taille de la figure en pouces
    
    Returns:
        Visualisation des métriques sous forme de tableau numpy
    """
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 2, figure=fig)
    
    # Résumé des métriques
    ax1 = fig.add_subplot(gs[0, 0])
    metrics_text = "\n".join([
        f"Pièces Détectées: {metrics.get('detected_count', 'N/A')}",
        f"Pièces Attendues: {metrics.get('expected_count', 'N/A')}",
        f"Pièces Valides: {metrics.get('valid_pieces', 'N/A')}",
        f"Taux de Détection: {metrics.get('detection_rate', 0):.2f}",
        f"Aire Moyenne: {metrics.get('mean_area', 0):.1f}",
        f"Temps de Traitement: {metrics.get('processing_time', 0):.2f}s"
    ])
    ax1.text(0.1, 0.5, metrics_text, fontsize=12, va='center')
    ax1.set_title("Métriques de Détection")
    ax1.axis('off')
    
    # Métriques de vérification si disponibles
    if 'pieces_removed_by_verification' in metrics:
        ax2 = fig.add_subplot(gs[0, 1])
        verification_text = "\n".join([
            f"Détectées Initialement: {metrics.get('original_detected_count', 'N/A')}",
            f"Après Vérification: {metrics.get('detected_count', 'N/A')}",
            f"Pièces Supprimées: {metrics.get('pieces_removed_by_verification', 0)}",
            f"Aire Moyenne Rejetée: {metrics.get('rejected_mean_area', 0):.1f}",
            f"Aire Moyenne Conservée: {metrics.get('mean_area', 0):.1f}",
            f"Score Validation Moyen: {metrics.get('mean_validation_score', 0):.2f}"
        ])
        ax2.text(0.1, 0.5, verification_text, fontsize=12, va='center')
        ax2.set_title("Résultats de Vérification")
        ax2.axis('off')
    
    # Distribution des aires
    ax3 = fig.add_subplot(gs[1, 0])
    if 'pieces' in metrics and metrics['pieces']:
        areas = [p.features['area'] for p in metrics['pieces']]
        ax3.hist(areas, bins=10, color='skyblue', edgecolor='black')
        ax3.set_title("Distribution des Aires")
        ax3.set_xlabel("Aire (pixels)")
        ax3.set_ylabel("Nombre")
    else:
        ax3.text(0.5, 0.5, "Aucune donnée d'aire disponible", fontsize=12, ha='center', va='center')
        ax3.set_title("Distribution des Aires")
        ax3.axis('off')
    
    # Comparaison des nombres avant/après vérification
    ax4 = fig.add_subplot(gs[1, 1])
    if 'pieces_removed_by_verification' in metrics:
        original = metrics.get('original_detected_count', 0)
        after = metrics.get('detected_count', 0)
        expected = metrics.get('expected_count', 0)
        
        bars = ax4.bar(['Original', 'Vérifié', 'Attendu'], 
                      [original, after, expected],
                      color=['skyblue', 'lightgreen', 'salmon'])
        ax4.bar_label(bars, fmt='%d')
        ax4.set_title("Comptage des Pièces")
        ax4.set_ylabel("Nombre de Pièces")
    else:
        # Si aucune vérification n'a été effectuée, afficher la distribution de compacité
        if 'pieces' in metrics and metrics['pieces']:
            compactness = [p.features.get('compactness', 0) for p in metrics['pieces']]
            if any(compactness):
                ax4.hist(compactness, bins=10, color='lightblue', edgecolor='black')
                ax4.set_title("Distribution de Compacité")
                ax4.set_xlabel("Compacité")
                ax4.set_ylabel("Nombre")
            else:
                ax4.text(0.5, 0.5, "Aucune donnée de compacité disponible", fontsize=12, ha='center', va='center')
                ax4.set_title("Compacité des Pièces")
                ax4.axis('off')
        else:
            ax4.text(0.5, 0.5, "Aucune métrique supplémentaire disponible", fontsize=12, ha='center', va='center')
            ax4.set_title("Métriques Supplémentaires")
            ax4.axis('off')
    
    plt.tight_layout()
    
    # Convertir la figure matplotlib en tableau numpy
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    
    img = np.array(Image.open(buf))
    
    # Convertir RGB en BGR pour OpenCV
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    plt.close(fig)
    
    return img


def create_segmentation_visualization(original_image: np.ndarray, 
                                    binary_mask: np.ndarray,
                                    contours: List[np.ndarray]) -> np.ndarray:
    """
    Crée une visualisation de la segmentation montrant l'image originale, le masque binaire et les contours.
    Optimisé pour la performance et la clarté.
    
    Args:
        original_image: Image originale
        binary_mask: Masque binaire de segmentation
        contours: Liste des contours détectés
        
    Returns:
        Image de visualisation
    """
    # Créer des versions des visualisations
    
    # 1. Superposition du masque sur l'image originale avec transparence
    mask_overlay = draw_mask(original_image, binary_mask, color=(0, 255, 0), alpha=0.3)
    
    # 2. Contours dessinés sur l'image originale
    contours_vis = draw_contours(original_image.copy(), contours, color=(0, 0, 255), thickness=2)
    
    # 3. Contours remplis avec couleurs aléatoires pour distinguer les pièces
    filled_contours = original_image.copy()
    
    for i, contour in enumerate(contours):
        # Générer une couleur aléatoire pour chaque contour
        color = (
            np.random.randint(0, 255),
            np.random.randint(0, 255),
            np.random.randint(0, 255)
        )
        
        # Dessiner le contour rempli
        cv2.drawContours(filled_contours, [contour], -1, color, -1)
    
    # Fusionner avec l'original pour voir les détails à travers
    filled_contours = cv2.addWeighted(original_image, 0.7, filled_contours, 0.3, 0)
    
    # Créer la visualisation en grille
    visualizations = [
        (original_image, "Image Originale"),
        (mask_overlay, "Masque de Segmentation"),
        (contours_vis, "Contours Détectés"),
        (filled_contours, "Pièces Segmentées")
    ]
    
    return create_grid_visualization(
        visualizations, 
        cols=2, 
        figsize=(16, 12), 
        title="Visualisation de la Segmentation"
    )


def generate_piece_gallery(pieces, rows: int = 4, cols: int = 5, 
                          max_pieces: int = 20, thumbnail_size: int = 200) -> np.ndarray:
    """
    Génère une galerie d'images des pièces détectées.
    
    Args:
        pieces: Liste des objets PuzzlePiece
        rows: Nombre de lignes dans la galerie
        cols: Nombre de colonnes dans la galerie
        max_pieces: Nombre maximum de pièces à afficher
        thumbnail_size: Taille des vignettes
        
    Returns:
        Image de la galerie
    """
    # Limiter le nombre de pièces
    num_pieces = min(len(pieces), max_pieces)
    pieces = pieces[:num_pieces]
    
    # Ajuster rows/cols si nécessaire
    if num_pieces < rows * cols:
        rows = (num_pieces + cols - 1) // cols
    
    # Créer une image vide pour la galerie
    gallery_height = rows * thumbnail_size
    gallery_width = cols * thumbnail_size
    gallery = np.ones((gallery_height, gallery_width, 3), dtype=np.uint8) * 255
    
    for i, piece in enumerate(pieces):
        if i >= rows * cols:
            break
            
        # Extraire l'image de la pièce
        try:
            piece_img = piece.get_extracted_image(clean_background=True)
            
            # Redimensionner l'image de la pièce pour qu'elle tienne dans la vignette
            h, w = piece_img.shape[:2]
            scale = (thumbnail_size - 20) / max(h, w)
            resized = cv2.resize(piece_img, None, fx=scale, fy=scale, 
                               interpolation=cv2.INTER_AREA)
            
            # Calculer la position de la vignette dans la galerie
            row = i // cols
            col = i % cols
            
            y_offset = row * thumbnail_size + (thumbnail_size - resized.shape[0]) // 2
            x_offset = col * thumbnail_size + (thumbnail_size - resized.shape[1]) // 2
            
            # Insérer la vignette dans la galerie
            gallery[y_offset:y_offset+resized.shape[0], x_offset:x_offset+resized.shape[1]] = resized
            
            # Ajouter un identifiant de pièce
            cv2.putText(gallery, f"#{piece.id}", 
                      (x_offset + 5, y_offset + 15),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            # Dessiner un cadre autour de la vignette
            color = (0, 255, 0) if piece.is_valid else (0, 0, 255)
            cv2.rectangle(gallery, 
                        (x_offset - 1, y_offset - 1), 
                        (x_offset + resized.shape[1], y_offset + resized.shape[0]), 
                        color, 2)
            
        except Exception as e:
            # En cas d'erreur, afficher un rectangle d'erreur
            row = i // cols
            col = i % cols
            
            y_offset = row * thumbnail_size
            x_offset = col * thumbnail_size
            
            # Dessiner un rectangle avec texte d'erreur
            cv2.rectangle(gallery, 
                        (x_offset + 10, y_offset + 10), 
                        (x_offset + thumbnail_size - 10, y_offset + thumbnail_size - 10), 
                        (0, 0, 255), 2)
            cv2.putText(gallery, f"Erreur #{piece.id}", 
                      (x_offset + 20, y_offset + thumbnail_size // 2),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    return gallery


def create_debug_visualization(original_image: np.ndarray, stages: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Crée une visualisation de débogage montrant les différentes étapes du traitement.
    
    Args:
        original_image: Image originale
        stages: Dictionnaire des images des différentes étapes de traitement
        
    Returns:
        Image de visualisation
    """
    # Créer une liste de paires (image, titre)
    visualizations = [("original", original_image)]
    
    # Ajouter chaque étape de traitement
    for name, img in stages.items():
        visualizations.append((name, img))
    
    # Trier par nom pour une visualisation cohérente
    visualizations.sort(key=lambda x: x[0])
    
    # Convertir en format (image, titre)
    vis_pairs = [(img, name.replace('_', ' ').title()) for name, img in visualizations]
    
    # Calculer le nombre de colonnes approprié
    cols = min(3, len(vis_pairs))
    
    # Créer la visualisation
    return create_grid_visualization(
        vis_pairs, 
        cols=cols,
        figsize=(16, 12),
        title="Étapes de Traitement"
    )


def overlay_contours_on_original(image: np.ndarray, contours: List[np.ndarray], 
                               use_random_colors: bool = True) -> np.ndarray:
    """
    Superpose les contours sur l'image originale avec des couleurs pour distinguer les pièces.
    Optimisé pour la clarté visuelle.
    
    Args:
        image: Image originale
        contours: Liste des contours
        use_random_colors: Utiliser des couleurs aléatoires pour chaque contour
        
    Returns:
        Image avec contours superposés
    """
    result = image.copy()
    
    for i, contour in enumerate(contours):
        if use_random_colors:
            # Générer une couleur aléatoire vive
            color = (
                np.random.randint(50, 255),
                np.random.randint(50, 255),
                np.random.randint(50, 255)
            )
        else:
            # Couleur standard
            color = (0, 255, 0)
        
        # Dessiner le contour
        cv2.drawContours(result, [contour], -1, color, 2)
        
        # Ajouter un identifiant de pièce
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # Fond noir pour la lisibilité
            text = f"#{i}"
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(result, 
                        (cx - 5, cy - text_size[1] - 5), 
                        (cx + text_size[0] + 5, cy + 5), 
                        (0, 0, 0), -1)
            
            cv2.putText(result, text, (cx, cy),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return result