"""This file is the main file for the project. It is responsible for running the program and handling the pictures inputs."""

import random
import cv2
import numpy as np
import matplotlib

from image_processing import read_image, show_image, resize_image, save_image, pre_process_image, find_contour, show_contour

# function to show the individual found pieces in image file and saving them in a directory
def save_pieces(pieces):
    """
    Saves the individual found pieces in image files.
    Skips any None values in the pieces list.
    """
    piece_count = 0
    for piece in pieces:
        if piece is not None:
            path = f"pieces/piece_{piece_count}.jpg"
            # Switched parameter order to match your save_image function definition
            save_image(piece, path)
            piece_count += 1
    print(f"Saved {piece_count} valid pieces out of {len(pieces)} total contours.")



# stochastic function to find the missing points in the contour
def find_missing_points(contour, n_missing_points):
    """Finds the missing points in the contour."""
    # get the number of points in the contour
    n_points = len(contour)
    # get the indices of the points in the contour
    indices = list(range(n_points))
    # shuffle the indices
    random.shuffle(indices)
    # get the missing points
    missing_points = [contour[i] for i in indices[:n_missing_points]]
    return missing_points

# function to find the piece in the image using the contour
def find_piece(image, contour, min_area=1000, min_perimeter=100, is_closed_threshold=0.02):
    """
    Extracts a puzzle piece from the image using the contour if it represents a valid piece.
    
    Args:
        image: Source image from which to extract the piece
        contour: Contour of the potential puzzle piece
        min_area: Minimum area of a contour to be considered a piece
        min_perimeter: Minimum perimeter length to be considered a piece
        is_closed_threshold: Threshold for determining if a contour is closed (0-1)
        
    Returns:
        The extracted puzzle piece image or None if not a valid piece
    """
    # Make sure contour has enough points
    if len(contour) < 5:
        return None
    
    # Check area and perimeter
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    
    if area < min_area or perimeter < min_perimeter:
        return None
    
    # Check if the contour is approximately closed using compactness measure
    # For a perfect circle, area/(perimeter^2) = 1/(4*pi)
    # For real puzzles, the ratio will be smaller but still significant
    compactness = 4 * np.pi * area / (perimeter * perimeter)
    
    if compactness < is_closed_threshold:
        # This contour is likely not a closed shape (or very irregular)
        return None
    
    # Get the bounding rectangle of the contour
    x, y, w, h = cv2.boundingRect(contour)
    
    # Create a mask for the piece
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], 0, 255, -1)
    
    # Crop the mask and the image to the bounding rectangle
    mask_roi = mask[y:y+h, x:x+w]
    image_roi = image[y:y+h, x:x+w]
    
    # Apply the mask to extract the piece
    piece = cv2.bitwise_and(image_roi, image_roi, mask=mask_roi)
    
    return piece

def debug_contours(image, contours, rejected_reasons):
    """Visualise les contours avec des codes couleur selon la raison du rejet"""
    debug_img = image.copy()
    
    for i, (contour, reason) in enumerate(zip(contours, rejected_reasons)):
        color = (0, 255, 0)  # Vert pour les contours valides
        if reason == "too_small":
            color = (0, 0, 255)  # Rouge pour les contours trop petits
        elif reason == "not_closed":
            color = (255, 0, 0)  # Bleu pour les contours non fermés
        # etc.
        
        cv2.drawContours(debug_img, [contour], 0, color, 2)
        
    save_image(debug_img, "debug/contours_debug.jpg")


def main():
    """Main function for the program."""
    # Make sure the pieces directory exists
    import os
    if not os.path.exists("pieces"):
        os.makedirs("pieces")
    # clean the pieces directory
    for file in os.listdir("pieces"):
        os.remove(os.path.join("pieces", file))
    
        
    w_1 = read_image("picture/puzzle_24-1/b-2.jpg")
    # w_1 = resize_image(w_1, 1250, 1250)
    
    # find the contour of the pieces on the unified background
    w_1_c = find_contour(w_1)
    
    # Débogage: enregistrez une image avec tous les contours
    debug_img = w_1.copy()
    cv2.drawContours(debug_img, w_1_c, -1, (0, 255, 0), 1)
    save_image(debug_img, "debug/all_contours.jpg")
    
    # Débogage: comptez les rejets par critère
    area_rejects = 0
    perimeter_rejects = 0
    compactness_rejects = 0
    points_rejects = 0
    
    for contour in w_1_c:
        if len(contour) < 5:
            points_rejects += 1
            continue
            
        area = cv2.contourArea(contour)
        if area < 1000:
            area_rejects += 1
            continue
            
        perimeter = cv2.arcLength(contour, True)
        if perimeter < 100:
            perimeter_rejects += 1
            continue
            
        compactness = 4 * np.pi * area / (perimeter * perimeter)
        if compactness < 0.02:
            compactness_rejects += 1
            continue
    
    print(f"Rejets par critère:")
    print(f"  - Points insuffisants: {points_rejects}")
    print(f"  - Aire trop petite: {area_rejects}")
    print(f"  - Périmètre trop petit: {perimeter_rejects}")
    print(f"  - Compacité insuffisante: {compactness_rejects}")
    
    # Find the pieces in the image using the find_piece function and storing them in a list
    w_1_p = [find_piece(w_1, contour) for contour in w_1_c]
    
    # Filter out None values before saving
    valid_pieces = [piece for piece in w_1_p if piece is not None]
    print(f"Found {len(valid_pieces)} valid pieces out of {len(w_1_c)} contours")
    
    # save the pieces in image files
    save_pieces(w_1_p)



if __name__ == '__main__':
    main()