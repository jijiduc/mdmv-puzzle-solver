"""
Visualization utilities for displaying and debugging puzzle piece detection
"""

import cv2
import numpy as np
import os
from typing import List, Tuple, Dict, Any, Optional
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import io
from PIL import Image


def draw_contours(image: np.ndarray,
                 contours: List[np.ndarray],
                 color: Tuple[int, int, int] = (0, 255, 0),
                 thickness: int = 2,
                 draw_index: bool = False) -> np.ndarray:
    """
    Draw contours on an image
    
    Args:
        image: Input image
        contours: List of contours to draw
        color: BGR color tuple
        thickness: Line thickness
        draw_index: Whether to draw contour indices
    
    Returns:
        Image with drawn contours
    """
    result = image.copy()
    
    for i, contour in enumerate(contours):
        cv2.drawContours(result, [contour], -1, color, thickness)
        
        if draw_index:
            # Calculate centroid for text placement
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Draw contour index
                cv2.putText(result, str(i), (cx, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 255, 255), 2)
    
    return result


def draw_corners(image: np.ndarray,
                corners: np.ndarray,
                radius: int = 5,
                color: Tuple[int, int, int] = (0, 0, 255),
                thickness: int = -1,  # -1 for filled circle
                draw_index: bool = True) -> np.ndarray:
    """
    Draw corner points on an image
    
    Args:
        image: Input image
        corners: Array of corner points
        radius: Circle radius
        color: BGR color tuple
        thickness: Circle line thickness
        draw_index: Whether to draw corner indices
    
    Returns:
        Image with drawn corners
    """
    result = image.copy()
    
    for i, corner in enumerate(corners):
        x, y = corner.astype(int)
        cv2.circle(result, (x, y), radius, color, thickness)
        
        if draw_index:
            cv2.putText(result, str(i), (x + 10, y + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1)
    
    return result


def draw_borders(image: np.ndarray,
                borders: List[np.ndarray],
                border_types: List[str],
                colors: Dict[str, Tuple[int, int, int]] = None,
                thickness: int = 2) -> np.ndarray:
    """
    Draw color-coded borders on an image
    
    Args:
        image: Input image
        borders: List of border segments
        border_types: List of border type classifications
        colors: Dictionary mapping border types to BGR colors
        thickness: Line thickness
    
    Returns:
        Image with color-coded borders
    """
    if colors is None:
        colors = {
            "straight": (0, 255, 0),  # Green
            "tab": (0, 0, 255),       # Red
            "pocket": (255, 0, 0)     # Blue
        }
    
    result = image.copy()
    
    for border, border_type in zip(borders, border_types):
        color = colors.get(border_type, (255, 255, 255))
        
        # Draw the border segment
        pts = border.reshape((-1, 1, 2)).astype(np.int32)
        cv2.polylines(result, [pts], False, color, thickness)
    
    return result


def create_grid_visualization(images: List[Tuple[np.ndarray, str]],
                             cols: int = 3,
                             figsize: Tuple[int, int] = (15, 10),
                             dpi: int = 100) -> np.ndarray:
    """
    Create a grid visualization of multiple images
    
    Args:
        images: List of (image, title) tuples
        cols: Number of columns in the grid
        figsize: Figure size in inches
        dpi: Dots per inch
    
    Returns:
        Grid visualization as a numpy array
    """
    rows = (len(images) + cols - 1) // cols
    
    fig = plt.figure(figsize=figsize, dpi=dpi)
    
    for i, (img, title) in enumerate(images):
        ax = fig.add_subplot(rows, cols, i + 1)
        
        # Handle both RGB and grayscale images
        if len(img.shape) == 3 and img.shape[2] == 3:
            # Convert BGR to RGB for matplotlib
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            ax.imshow(img, cmap='gray')
        
        ax.set_title(title)
        ax.axis('off')
    
    plt.tight_layout()
    
    # Convert matplotlib figure to numpy array
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    
    img = np.array(Image.open(buf))
    
    # Convert RGB to BGR for OpenCV
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    plt.close(fig)
    
    return img


def create_processing_visualization(original_image: np.ndarray,
                                  preprocessed: np.ndarray,
                                  binary: np.ndarray,
                                  contours_image: np.ndarray,
                                  detected_pieces: List[Dict[str, Any]],
                                  output_path: str = None) -> np.ndarray:
    """
    Create a comprehensive visualization of the processing pipeline
    Modified to handle full-sized images without resizing
    
    Args:
        original_image: Original input image
        preprocessed: Preprocessed grayscale image
        binary: Binary image after thresholding
        contours_image: Image with detected contours
        detected_pieces: List of detected piece information
        output_path: Optional path to save the visualization
    
    Returns:
        Visualization image
    """
    # Instead of resizing images, we'll use the original dimensions
    # but will adjust the visualization layout
    
    # Create image/title pairs
    images = [
        (original_image, "Original Image"),
        (preprocessed, "Preprocessed Image"),
        (binary, "Binary Image"),
        (contours_image, "Detected Contours")
    ]
    
    # Add up to two detected pieces if available
    pieces_to_show = min(2, len(detected_pieces))
    
    for i in range(pieces_to_show):
        piece_info = detected_pieces[i]
        piece_img = piece_info.get('visualization', None)
        
        if piece_img is not None:
            images.append((piece_img, f"Piece #{i+1}"))
    
    # Create the grid visualization - with increased figure size to 
    # accommodate larger images without resizing them
    vis_img = create_grid_visualization(images, cols=2, figsize=(24, 18), dpi=100)
    
    # Add text with detection summary
    h, w = vis_img.shape[:2]
    
    # Ensure the visualization image has 3 channels (BGR)
    if len(vis_img.shape) == 2:  # Grayscale
        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_GRAY2BGR)
    elif vis_img.shape[2] == 4:  # RGBA
        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGBA2BGR)
    
    # Create text image with matching width and 3 channels
    text_img = np.ones((200, w, 3), dtype=np.uint8) * 255  # Increased height for better readability
    
    # Add summary text
    cv2.putText(text_img, f"Total Detected Pieces: {len(detected_pieces)}",
                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)  # Increased font size
    
    # Log image dimensions
    cv2.putText(text_img, f"Image dimensions: {original_image.shape[1]}x{original_image.shape[0]}",
                (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)
    
    # Combine visualization and text (now both are guaranteed to be 3-channel BGR)
    result = np.vstack((vis_img, text_img))
    
    # Save if output path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, result)
    
    return result


def display_metrics(metrics: Dict[str, Any],
                   figsize: Tuple[int, int] = (10, 6)) -> np.ndarray:
    """
    Create a visual display of detection metrics
    
    Args:
        metrics: Dictionary of metrics
        figsize: Figure size in inches
    
    Returns:
        Metrics visualization as numpy array
    """
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 2, figure=fig)
    
    # Metrics summary
    ax1 = fig.add_subplot(gs[0, 0])
    metrics_text = "\n".join([
        f"Detected Pieces: {metrics.get('detected_count', 'N/A')}",
        f"Expected Pieces: {metrics.get('expected_count', 'N/A')}",
        f"Detection Rate: {metrics.get('detection_rate', 0):.2f}",
        f"Mean Area: {metrics.get('mean_area', 0):.1f}",
        f"Edge Alignment: {metrics.get('edge_alignment', 0):.2f}",
        f"Mean Compactness: {metrics.get('mean_compactness', 0):.2f}"
    ])
    ax1.text(0.1, 0.5, metrics_text, fontsize=12, va='center')
    ax1.set_title("Detection Metrics")
    ax1.axis('off')
    
    # Area distribution
    if 'area_distribution' in metrics:
        ax2 = fig.add_subplot(gs[0, 1])
        areas = metrics['area_distribution']
        ax2.hist(areas, bins=10, color='skyblue', edgecolor='black')
        ax2.set_title("Piece Area Distribution")
        ax2.set_xlabel("Area (pixels)")
        ax2.set_ylabel("Count")
    
    # Border type distribution
    if 'border_types' in metrics:
        ax3 = fig.add_subplot(gs[1, 0])
        border_counts = metrics['border_types']
        ax3.bar(border_counts.keys(), border_counts.values(), color='lightgreen')
        ax3.set_title("Border Type Distribution")
        ax3.set_xlabel("Border Type")
        ax3.set_ylabel("Count")
    
    # Solidity distribution
    if 'solidity_distribution' in metrics:
        ax4 = fig.add_subplot(gs[1, 1])
        solidity = metrics['solidity_distribution']
        ax4.hist(solidity, bins=10, color='salmon', edgecolor='black')
        ax4.set_title("Piece Solidity Distribution")
        ax4.set_xlabel("Solidity")
        ax4.set_ylabel("Count")
    
    plt.tight_layout()
    
    # Convert matplotlib figure to numpy array
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    
    img = np.array(Image.open(buf))
    
    # Convert RGB to BGR for OpenCV
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    plt.close(fig)
    
    return img