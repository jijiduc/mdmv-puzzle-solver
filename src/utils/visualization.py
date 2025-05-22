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


def create_geometry_visualization(piece_result: Dict[str, Any], piece_img: np.ndarray, 
                                piece_idx: int, output_dir: str):
    """Create geometric analysis visualization for a piece.
    
    Args:
        piece_result: Processed piece data
        piece_img: Piece image
        piece_idx: Piece index
        output_dir: Output directory
    """
    corners_dir = os.path.join(output_dir, 'corners')
    analysis_dir = os.path.join(output_dir, 'corner_analysis')
    os.makedirs(corners_dir, exist_ok=True)
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Create corner visualization
    if 'corners' in piece_result:
        corner_img = piece_img.copy()
        corners = piece_result['corners']
        centroid = piece_result.get('centroid', (0, 0))
        
        # Draw corners
        for i, corner in enumerate(corners):
            cv2.circle(corner_img, tuple(map(int, corner)), 5, (0, 255, 0), -1)
            cv2.putText(corner_img, f'C{i+1}', (int(corner[0])+10, int(corner[1])-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw centroid
        cv2.circle(corner_img, tuple(map(int, centroid)), 3, (255, 0, 0), -1)
        cv2.putText(corner_img, 'Centroid', (int(centroid[0])+10, int(centroid[1])+10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw lines from centroid to corners
        for corner in corners:
            cv2.line(corner_img, tuple(map(int, centroid)), tuple(map(int, corner)), (0, 0, 255), 1)
        
        cv2.imwrite(os.path.join(corners_dir, f'piece_{piece_idx+1}_corners.png'), corner_img)
        
        # Create corner-distance analysis visualization
        create_corner_distance_analysis(piece_result, piece_img, piece_idx, analysis_dir)


def create_corner_distance_analysis(piece_result: Dict[str, Any], piece_img: np.ndarray, 
                                   piece_idx: int, output_dir: str):
    """Create combined corner and distance analysis visualization.
    
    Args:
        piece_result: Processed piece data
        piece_img: Piece image
        piece_idx: Piece index
        output_dir: Output directory
    """
    if 'corners' not in piece_result:
        return
    
    corners = piece_result['corners']
    centroid = piece_result.get('centroid', (0, 0))
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Piece {piece_idx+1} - Corner & Distance Analysis', fontsize=14, fontweight='bold')
    
    # 1. Original piece with corners
    axes[0, 0].imshow(cv2.cvtColor(piece_img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Corners & Centroid')
    
    # Plot corners and centroid
    if corners:
        corners_array = np.array(corners)
        axes[0, 0].scatter(corners_array[:, 0], corners_array[:, 1], 
                          c='lime', s=100, marker='o', edgecolors='black', linewidth=2, label='Corners')
        
        # Label corners
        for i, corner in enumerate(corners):
            axes[0, 0].annotate(f'C{i+1}', (corner[0], corner[1]), 
                              xytext=(5, 5), textcoords='offset points', 
                              fontsize=10, color='white', fontweight='bold')
    
    # Plot centroid
    axes[0, 0].scatter(centroid[0], centroid[1], c='red', s=80, marker='x', 
                      linewidth=3, label='Centroid')
    
    # Draw lines from centroid to corners
    for corner in corners:
        axes[0, 0].plot([centroid[0], corner[0]], [centroid[1], corner[1]], 
                       'r--', alpha=0.7, linewidth=1)
    
    axes[0, 0].legend()
    axes[0, 0].axis('off')
    
    # 2. Distance from centroid to corners (polar plot)
    if len(corners) >= 3:
        # Calculate distances and angles
        distances = []
        angles = []
        for corner in corners:
            dist = np.sqrt((corner[0] - centroid[0])**2 + (corner[1] - centroid[1])**2)
            angle = np.arctan2(corner[1] - centroid[1], corner[0] - centroid[0])
            distances.append(dist)
            angles.append(angle)
        
        # Close the polygon for plotting
        angles_closed = angles + [angles[0]]
        distances_closed = distances + [distances[0]]
        
        # Create polar subplot
        ax_polar = plt.subplot(2, 2, 2, projection='polar')
        ax_polar.plot(angles_closed, distances_closed, 'o-', linewidth=2, markersize=8)
        ax_polar.fill(angles_closed, distances_closed, alpha=0.3)
        ax_polar.set_title('Distance Profile (Polar)', pad=20)
        ax_polar.grid(True)
        
        # Label points
        for i, (angle, dist) in enumerate(zip(angles, distances)):
            ax_polar.annotate(f'C{i+1}', (angle, dist), xytext=(5, 5), 
                            textcoords='offset points', fontsize=9)
    else:
        axes[0, 1].text(0.5, 0.5, 'Need ≥3 corners\nfor polar plot', 
                       ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('Distance Profile (Polar)')
        axes[0, 1].axis('off')
    
    # 3. Distance bar chart
    if corners:
        distances = [np.sqrt((corner[0] - centroid[0])**2 + (corner[1] - centroid[1])**2) 
                    for corner in corners]
        corner_labels = [f'C{i+1}' for i in range(len(corners))]
        
        bars = axes[1, 0].bar(corner_labels, distances, color=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99'][:len(distances)])
        axes[1, 0].set_title('Corner Distances from Centroid')
        axes[1, 0].set_ylabel('Distance (pixels)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, dist in zip(bars, distances):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                           f'{dist:.1f}', ha='center', va='bottom', fontweight='bold')
    else:
        axes[1, 0].text(0.5, 0.5, 'No corners detected', ha='center', va='center', 
                       transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Corner Distances from Centroid')
    
    # 4. Corner angle analysis
    if len(corners) >= 3:
        # Calculate internal angles between consecutive corners
        corner_angles = []
        for i in range(len(corners)):
            prev_corner = corners[i-1]
            curr_corner = corners[i]
            next_corner = corners[(i+1) % len(corners)]
            
            # Vectors
            v1 = np.array([prev_corner[0] - curr_corner[0], prev_corner[1] - curr_corner[1]])
            v2 = np.array([next_corner[0] - curr_corner[0], next_corner[1] - curr_corner[1]])
            
            # Angle between vectors
            if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                cos_angle = np.clip(cos_angle, -1, 1)
                angle_deg = np.degrees(np.arccos(cos_angle))
                corner_angles.append(angle_deg)
            else:
                corner_angles.append(0)
        
        # Plot angles
        angle_labels = [f'∠C{i+1}' for i in range(len(corner_angles))]
        bars = axes[1, 1].bar(angle_labels, corner_angles, 
                             color=['#ffeb3b', '#ff9800', '#f44336', '#9c27b0'][:len(corner_angles)])
        axes[1, 1].set_title('Internal Corner Angles')
        axes[1, 1].set_ylabel('Angle (degrees)')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axhline(y=90, color='red', linestyle='--', alpha=0.7, label='90°')
        
        # Add value labels
        for bar, angle in zip(bars, corner_angles):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                           f'{angle:.1f}°', ha='center', va='bottom', fontweight='bold')
        
        axes[1, 1].legend()
    else:
        axes[1, 1].text(0.5, 0.5, 'Need ≥3 corners\nfor angle analysis', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Internal Corner Angles')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'piece_{piece_idx+1}_corner_distance_analysis.png'), 
               dpi=150, bbox_inches='tight')
    plt.close()
    
    # Create detailed text report
    report_file = os.path.join(output_dir, f'piece_{piece_idx+1}_geometry_report.txt')
    with open(report_file, 'w') as f:
        f.write(f"Piece {piece_idx+1} - Geometric Analysis Report\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Centroid: ({centroid[0]:.1f}, {centroid[1]:.1f})\n")
        f.write(f"Number of corners detected: {len(corners)}\n\n")
        
        if corners:
            f.write("Corner Coordinates:\n")
            for i, corner in enumerate(corners):
                f.write(f"  C{i+1}: ({corner[0]:.1f}, {corner[1]:.1f})\n")
            
            f.write(f"\nCorner Distances from Centroid:\n")
            for i, corner in enumerate(corners):
                dist = np.sqrt((corner[0] - centroid[0])**2 + (corner[1] - centroid[1])**2)
                f.write(f"  C{i+1}: {dist:.1f} pixels\n")
            
            if len(corners) >= 3:
                f.write(f"\nInternal Corner Angles:\n")
                for i, angle in enumerate(corner_angles):
                    f.write(f"  ∠C{i+1}: {angle:.1f}°\n")
                
                avg_angle = np.mean(corner_angles)
                f.write(f"\nAverage corner angle: {avg_angle:.1f}°\n")
                f.write(f"Expected for regular polygon: {180 * (len(corners) - 2) / len(corners):.1f}°\n")


def create_edge_classification_visualization(piece_results: List[Dict[str, Any]], output_dir: str):
    """Create edge classification summary visualization.
    
    Args:
        piece_results: List of processed piece results
        output_dir: Output directory
    """
    edge_types_dir = os.path.join(output_dir, 'edge_types')
    os.makedirs(edge_types_dir, exist_ok=True)
    
    # Collect edge type statistics
    edge_type_counts = {}
    total_edges = 0
    
    for result in piece_results:
        edge_types = result.get('edge_types', [])
        for edge_type in edge_types:
            edge_type_counts[edge_type] = edge_type_counts.get(edge_type, 0) + 1
            total_edges += 1
    
    # Create pie chart
    if edge_type_counts:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Pie chart
        labels = list(edge_type_counts.keys())
        sizes = list(edge_type_counts.values())
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc']
        
        ax1.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors[:len(labels)], startangle=90)
        ax1.set_title('Edge Type Distribution')
        
        # Bar chart
        ax2.bar(labels, sizes, color=colors[:len(labels)])
        ax2.set_title('Edge Type Counts')
        ax2.set_ylabel('Number of Edges')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'edge_classification_summary.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    # Create detailed report
    report_file = os.path.join(output_dir, 'classification_report.txt')
    with open(report_file, 'w') as f:
        f.write("Edge Classification Report\n")
        f.write("=" * 40 + "\n")
        f.write(f"Total pieces analyzed: {len(piece_results)}\n")
        f.write(f"Total edges analyzed: {total_edges}\n\n")
        
        f.write("Edge type distribution:\n")
        for edge_type, count in edge_type_counts.items():
            percentage = (count / total_edges * 100) if total_edges > 0 else 0
            f.write(f"  {edge_type}: {count} ({percentage:.1f}%)\n")
        
        f.write("\nPiece details:\n")
        for i, result in enumerate(piece_results):
            edge_types = result.get('edge_types', [])
            f.write(f"  Piece {i+1}: {edge_types}\n")


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
        f.write(f"  📁 05_geometry/ - Geometric analysis\n")
        f.write(f"  📁 06_colors/ - Color feature analysis\n") 
        f.write(f"  📁 07_edges/ - Edge feature extraction\n")
        f.write(f"  📁 08_classification/ - Piece classification\n")
        f.write(f"  📁 09_matching/ - Edge matching results\n")
        f.write(f"  📁 10_assembly/ - Final assembly output\n")


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