# main-couleur.py
import cv2
import numpy as np
import os
from image_processing import read_image, save_image
from pieces_cutting import PieceProcessor, Visualization, Config

# Define color codes for terminal output
class Colors:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    PURPLE = '\033[95m'
    BOLD = '\033[1m'
    RESET = '\033[0m'

def print_info(message):
    print(f"{Colors.BLUE}{message}{Colors.RESET}")

def print_success(message):
    print(f"{Colors.GREEN}{message}{Colors.RESET}")

def print_warning(message):
    print(f"{Colors.YELLOW}{message}{Colors.RESET}")

def print_error(message):
    print(f"{Colors.RED}{message}{Colors.RESET}")

def print_header(message):
    print(f"{Colors.PURPLE}{Colors.BOLD}{message}{Colors.RESET}")

def main():
    # Initialize configuration
    config = Config()
    
    # Setup directories
    os.makedirs("debug", exist_ok=True)
    [os.remove(f"debug/{f}") for f in os.listdir("debug") if f.endswith(".jpg")]

    # Load image
    input_image = read_image("picture/puzzle_24-1/b-2.jpg")
    if input_image is None:
        print_error("Error: Failed to load input image")
        return []

    # Find and validate contours
    contours = PieceProcessor.find_contour(input_image)
    valid_contours = []
    rejection_reasons = []

    for cnt in contours:
        # Basic contour validation
        if len(cnt) < 5:
            rejection_reasons.append("too_few_points")
            continue
            
        area = cv2.contourArea(cnt)
        if area < config.MIN_AREA:
            rejection_reasons.append("small_area")
            continue
            
        perimeter = cv2.arcLength(cnt, True)
        if perimeter < config.MIN_PERIMETER:
            rejection_reasons.append("short_perimeter")
            continue
            
        # Shape validation
        compactness = 4 * np.pi * area / (perimeter ** 2)
        if compactness < config.CLOSED_THRESHOLD:
            rejection_reasons.append("low_compactness")
            continue
            
        valid_contours.append(cnt)
        rejection_reasons.append("valid")

    print_info(f"Valid contours found: {len(valid_contours)}/{len(contours)}")

    # Visualize contour validation results
    Visualization.debug_contours(input_image, contours, rejection_reasons)

    # Process valid contours
    print_header("\nProcessing puzzle pieces:")
    summary_image = input_image.copy()
    pieces_data = []
    
    for idx, contour in enumerate(valid_contours[:5]):  # Process first 5 pieces
        print_info(f"\nAnalyzing piece #{idx + 1}")
        piece_data = PieceProcessor.analyze_piece(input_image, contour, config)
        
        if not piece_data:
            print_error(f"  Failed to process piece #{idx + 1}")
            continue
            
        print_success(f"  Border types: {piece_data['types']}")
        
        # Save individual visualization
        vis_img = Visualization.draw_borders(input_image, piece_data)
        save_image(vis_img, f"debug/piece_{idx}_borders.jpg")
        
        # Add to summary image
        x, y, w, h = cv2.boundingRect(contour)
        padding = 20
        roi = vis_img[
            max(0, y-padding):min(input_image.shape[0], y+h+padding),
            max(0, x-padding):min(input_image.shape[1], x+w+padding)
        ]
        
        # Resize for grid layout
        cell_size = (summary_image.shape[1]//2, summary_image.shape[0]//3)
        resized_roi = cv2.resize(roi, cell_size)
        
        # Position in 2x3 grid
        row = idx // 2
        col = idx % 2
        summary_image[
            row*cell_size[1]:(row+1)*cell_size[1],
            col*cell_size[0]:(col+1)*cell_size[0]
        ] = resized_roi
        
        pieces_data.append(piece_data)

    # Save summary visualization
    save_image(summary_image, "debug/processing_summary.jpg")

    return valid_contours

if __name__ == '__main__':
    valid = main()
    print_success(f"\nAnalysis complete. Valid pieces: {len(valid)}")