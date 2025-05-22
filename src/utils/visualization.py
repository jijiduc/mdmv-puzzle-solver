"""Visualization utilities for puzzle analysis debug output."""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from typing import Dict, List, Tuple, Any, Optional

from ..config.settings import DEBUG_DIRS


def create_input_visualization(img_path: str, img: np.ndarray, output_dir: str):
    """Create visualization of input image and basic info.
    
    Args:
        img_path: Path to original image
        img: Loaded image array
        output_dir: Output directory for visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save original image copy
    cv2.imwrite(os.path.join(output_dir, 'original_image.png'), img)
    
    # Create info file
    info_file = os.path.join(output_dir, 'image_info.txt')
    with open(info_file, 'w') as f:
        f.write(f"Input Image Analysis\n")
        f.write("=" * 30 + "\n")
        f.write(f"Source path: {img_path}\n")
        f.write(f"Image shape: {img.shape}\n")
        f.write(f"Image dtype: {img.dtype}\n")
        f.write(f"Image size: {img.shape[0] * img.shape[1]} pixels\n")
        if len(img.shape) == 3:
            f.write(f"Channels: {img.shape[2]}\n")
        f.write(f"Min pixel value: {img.min()}\n")
        f.write(f"Max pixel value: {img.max()}\n")
        f.write(f"Mean pixel value: {img.mean():.2f}\n")


def create_preprocessing_visualization(img: np.ndarray, binary_mask: np.ndarray, 
                                     processed_mask: np.ndarray, filled_mask: np.ndarray,
                                     threshold_value: int, output_dir: str):
    """Create comprehensive preprocessing visualization.
    
    Args:
        img: Original image
        binary_mask: Initial binary mask
        processed_mask: After morphological operations
        filled_mask: Final mask with filled contours
        threshold_value: Threshold used
        output_dir: Output directory
    """
    masks_dir = os.path.join(output_dir, 'masks')
    os.makedirs(masks_dir, exist_ok=True)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Preprocessing Pipeline (Threshold: {threshold_value})', fontsize=16)
    
    # Original image
    axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    axes[0, 1].imshow(gray, cmap='gray')
    axes[0, 1].set_title('Grayscale')
    axes[0, 1].axis('off')
    
    # Binary threshold
    axes[0, 2].imshow(binary_mask, cmap='gray')
    axes[0, 2].set_title('Binary Threshold')
    axes[0, 2].axis('off')
    
    # Morphological operations
    axes[1, 0].imshow(processed_mask, cmap='gray')
    axes[1, 0].set_title('After Morphology')
    axes[1, 0].axis('off')
    
    # Final mask
    axes[1, 1].imshow(filled_mask, cmap='gray')
    axes[1, 1].set_title('Final Mask')
    axes[1, 1].axis('off')
    
    # Masked result
    masked_img = cv2.bitwise_and(img, img, mask=filled_mask)
    axes[1, 2].imshow(cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB))
    axes[1, 2].set_title('Masked Result')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'preprocessing_pipeline.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save individual masks
    cv2.imwrite(os.path.join(masks_dir, 'binary_mask.png'), binary_mask)
    cv2.imwrite(os.path.join(masks_dir, 'processed_mask.png'), processed_mask)
    cv2.imwrite(os.path.join(masks_dir, 'final_mask.png'), filled_mask)
    cv2.imwrite(os.path.join(masks_dir, 'masked_image.png'), masked_img)


def create_detection_visualization(img: np.ndarray, contours: List, valid_contours: List,
                                 min_area: int, output_dir: str):
    """Create contour detection visualization.
    
    Args:
        img: Original image
        contours: All detected contours
        valid_contours: Filtered valid contours
        min_area: Minimum area threshold used
        output_dir: Output directory
    """
    contours_dir = os.path.join(output_dir, 'contours')
    os.makedirs(contours_dir, exist_ok=True)
    
    # Create comparison visualization
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    
    # All contours
    img_all = img.copy()
    cv2.drawContours(img_all, contours, -1, (0, 0, 255), 2)  # Red for all
    axes[0].imshow(cv2.cvtColor(img_all, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f'All Contours ({len(contours)})')
    axes[0].axis('off')
    
    # Valid contours
    img_valid = img.copy()
    cv2.drawContours(img_valid, valid_contours, -1, (0, 255, 0), 3)  # Green for valid
    axes[1].imshow(cv2.cvtColor(img_valid, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f'Valid Contours (>{min_area}px²) ({len(valid_contours)})')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'contour_detection.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save individual contour images
    cv2.imwrite(os.path.join(contours_dir, 'all_contours.png'), img_all)
    cv2.imwrite(os.path.join(contours_dir, 'valid_contours.png'), img_valid)
    
    # Create contour analysis report
    analysis_file = os.path.join(output_dir, 'detection_analysis.txt')
    with open(analysis_file, 'w') as f:
        f.write("Contour Detection Analysis\n")
        f.write("=" * 30 + "\n")
        f.write(f"Total contours found: {len(contours)}\n")
        f.write(f"Valid contours (>{min_area}px²): {len(valid_contours)}\n")
        f.write(f"Rejection rate: {((len(contours) - len(valid_contours)) / len(contours) * 100):.1f}%\n\n")
        
        f.write("Valid contour areas:\n")
        for i, contour in enumerate(valid_contours):
            area = cv2.contourArea(contour)
            f.write(f"  Piece {i+1}: {area:.0f} px²\n")


def create_piece_gallery(pieces: List[Dict[str, Any]], output_dir: str):
    """Create a gallery view of all extracted pieces.
    
    Args:
        pieces: List of piece data
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate grid size
    n_pieces = len(pieces)
    cols = int(np.ceil(np.sqrt(n_pieces)))
    rows = int(np.ceil(n_pieces / cols))
    
    # Create gallery figure
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    fig.suptitle(f'Extracted Puzzle Pieces ({n_pieces} pieces)', fontsize=16)
    
    # Handle single row/col cases
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    for i, piece_data in enumerate(pieces):
        piece_img = np.array(piece_data['img'], dtype=np.uint8)
        
        if i < len(axes):
            axes[i].imshow(cv2.cvtColor(piece_img, cv2.COLOR_BGR2RGB))
            axes[i].set_title(f'Piece {i+1}')
            axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(len(pieces), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pieces_gallery.png'), dpi=150, bbox_inches='tight')
    plt.close()








def create_edge_type_analysis(piece_results: List[Dict[str, Any]], output_dir: str):
    """Create edge type analysis visualization."""
    # Collect edge type statistics
    edge_type_counts = {}
    total_edges = 0
    
    for result in piece_results:
        edge_types = result.get('edge_types', [])
        for edge_type in edge_types:
            edge_type_counts[edge_type] = edge_type_counts.get(edge_type, 0) + 1
            total_edges += 1
    
    # Create visualization
    if edge_type_counts:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Edge Type Analysis', fontsize=16, fontweight='bold')
        
        # Pie chart
        labels = list(edge_type_counts.keys())
        sizes = list(edge_type_counts.values())
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc']
        
        ax1.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors[:len(labels)], startangle=90)
        ax1.set_title('Edge Type Distribution')
        
        # Bar chart
        bars = ax2.bar(labels, sizes, color=colors[:len(labels)])
        ax2.set_title('Edge Type Counts')
        ax2.set_ylabel('Number of Edges')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, size in zip(bars, sizes):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(size), ha='center', va='bottom', fontweight='bold')
        
        # Edge deviation distribution
        all_deviations = []
        for result in piece_results:
            deviations = result.get('edge_deviations', [])
            all_deviations.extend([abs(d) for d in deviations])
        
        if all_deviations:
            ax3.hist(all_deviations, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax3.set_title('Edge Deviation Distribution')
            ax3.set_xlabel('Absolute Deviation')
            ax3.set_ylabel('Frequency')
            ax3.grid(True, alpha=0.3)
        
        # Edge type per piece
        piece_indices = list(range(1, len(piece_results) + 1))
        edge_type_matrix = []
        
        for result in piece_results:
            edge_types = result.get('edge_types', [])
            type_counts = {t: edge_types.count(t) for t in labels}
            edge_type_matrix.append([type_counts.get(t, 0) for t in labels])
        
        if edge_type_matrix:
            im = ax4.imshow(np.array(edge_type_matrix).T, aspect='auto', cmap='viridis')
            ax4.set_title('Edge Types per Piece')
            ax4.set_xlabel('Piece Index')
            ax4.set_ylabel('Edge Type')
            ax4.set_yticks(range(len(labels)))
            ax4.set_yticklabels(labels)
            ax4.set_xticks(range(len(piece_indices)))
            ax4.set_xticklabels(piece_indices)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax4)
            cbar.set_label('Count')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'edge_type_analysis.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()


def create_edge_classification_visualization(piece_results: List[Dict[str, Any]], output_dir: str):
    """Create edge classification summary visualization (legacy function)."""
    # Call the new comprehensive visualization
    create_edge_type_analysis(piece_results, output_dir)


def create_summary_dashboard(piece_count: int, processing_time: float, 
                           piece_results: List[Dict[str, Any]], output_dir: str):
    """Create overall summary dashboard.
    
    Args:
        piece_count: Number of pieces processed
        processing_time: Total processing time
        piece_results: List of piece results
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create summary report
    summary_file = os.path.join(output_dir, 'analysis_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("Puzzle Analysis Summary\n")
        f.write("=" * 50 + "\n")
        f.write(f"Analysis completed successfully!\n\n")
        
        f.write("Processing Statistics:\n")
        f.write(f"  • Total pieces detected: {piece_count}\n")
        f.write(f"  • Pieces successfully processed: {len(piece_results)}\n")
        f.write(f"  • Total processing time: {processing_time:.2f} seconds\n")
        f.write(f"  • Average time per piece: {(processing_time/piece_count):.3f} seconds\n\n")
        
        if piece_results:
            # Edge type analysis
            edge_type_counts = {}
            for result in piece_results:
                for edge_type in result.get('edge_types', []):
                    edge_type_counts[edge_type] = edge_type_counts.get(edge_type, 0) + 1
            
            f.write("Edge Analysis:\n")
            for edge_type, count in edge_type_counts.items():
                f.write(f"  • {edge_type}: {count} edges\n")
        
        f.write(f"\nOutput Structure:\n")
        f.write(f"  📁 01_input/ - Original image and metadata\n")
        f.write(f"  📁 02_preprocessing/ - Image processing steps\n")
        f.write(f"  📁 03_detection/ - Contour detection results\n")
        f.write(f"  📁 04_pieces/ - Individual piece extractions\n")
        f.write(f"  📁 05_geometry/ - Geometric analysis & corner detection\n")
        f.write(f"  📁 06_features/ - Shape feature analysis\n") 
        f.write(f"    📁 shape/ - Edge classification & individual visualizations\n")
        f.write(f"    📁 color/ - Color feature extraction (future)\n")
        f.write(f"  📁 07_piece_classification/ - Advanced piece classification\n")
        f.write(f"  📁 08_matching/ - Edge matching results\n")
        f.write(f"  📁 09_assembly/ - Final assembly output\n")


def save_piece_with_visualization(piece_data: Dict[str, Any], piece_img: np.ndarray, 
                                piece_idx: int, output_dir: str):
    """Save piece with enhanced visualization.
    
    Args:
        piece_data: Piece data dictionary
        piece_img: Piece image
        piece_idx: Piece index
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save piece image
    piece_filename = os.path.join(output_dir, f"piece_{piece_idx+1}.png")
    cv2.imwrite(piece_filename, piece_img)
    
    # Enhanced piece info
    info_filename = os.path.join(output_dir, f"piece_{piece_idx+1}_info.txt")
    bbox = piece_data.get('bbox', (0, 0, 0, 0))
    
    with open(info_filename, 'w') as f:
        f.write(f"Piece {piece_idx+1} Information\n")
        f.write("=" * 30 + "\n")
        f.write(f"Bounding box: {bbox}\n")
        f.write(f"Width: {bbox[2] - bbox[0]} pixels\n")
        f.write(f"Height: {bbox[3] - bbox[1]} pixels\n")
        f.write(f"Area: {(bbox[2] - bbox[0]) * (bbox[3] - bbox[1])} pixels²\n")
        f.write(f"Image shape: {piece_img.shape}\n")
        f.write(f"Non-zero pixels: {np.count_nonzero(piece_img)}\n")