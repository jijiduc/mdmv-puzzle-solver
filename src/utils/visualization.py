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









def create_edge_classification_visualization(piece_results: List[Dict[str, Any]], output_dir: str):
    """Create edge classification summary visualization (legacy function)."""
    # Simple edge type summary - functionality moved to shape analysis
    os.makedirs(output_dir, exist_ok=True)
    
    # Create simple summary file
    with open(os.path.join(output_dir, 'edge_summary.txt'), 'w') as f:
        f.write("Edge Classification Summary\n")
        f.write("=" * 30 + "\n")
        f.write(f"Total pieces processed: {len(piece_results)}\n")
        f.write("Note: Detailed edge analysis available in 06_features/shape/\n")


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


# Shape analysis visualization functions

def create_shape_analysis_visualization(piece: Any, piece_idx: int, output_dir: str):
    """Create comprehensive shape analysis visualization for a single piece.
    
    Args:
        piece: Piece object with edges containing shape analysis
        piece_idx: Index of the piece
        output_dir: Output directory for visualizations
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from src.features.shape_analysis import calculate_curvature_profile, calculate_turning_angles
    
    piece_shape_dir = os.path.join(output_dir, f'piece_{piece_idx+1:02d}')
    os.makedirs(piece_shape_dir, exist_ok=True)
    
    # Create figure with subplots for all 4 edges
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Piece {piece_idx+1} - Shape Analysis', fontsize=16, fontweight='bold')
    
    axes = axes.flatten()
    edge_names = ['Top', 'Right', 'Bottom', 'Left']
    
    for edge_idx, (edge, edge_name) in enumerate(zip(piece.edges, edge_names)):
        ax = axes[edge_idx]
        
        if edge.points and len(edge.points) > 2:
            # Extract edge data
            points = np.array(edge.points)
            edge_type = getattr(edge, 'edge_type', 'unknown')
            sub_type = getattr(edge, 'sub_type', None)
            confidence = getattr(edge, 'confidence', 0.0)
            
            # Calculate curvature profile
            try:
                curvature = calculate_curvature_profile(points)
                
                # Plot curvature profile
                x_positions = np.linspace(0, 1, len(curvature))
                ax.plot(x_positions, curvature, 'b-', linewidth=2, label='Curvature')
                ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
                
                # Color code by edge type
                if edge_type == 'convex':
                    color = 'red'
                elif edge_type == 'concave':
                    color = 'blue'
                else:  # flat
                    color = 'green'
                
                ax.fill_between(x_positions, curvature, alpha=0.3, color=color)
                
                # Add classification info
                title = f'{edge_name} Edge: {edge_type}'
                if sub_type:
                    title += f' ({sub_type})'
                title += f'\nConfidence: {confidence:.2f}'
                
                ax.set_title(title, fontsize=12, fontweight='bold')
                ax.set_xlabel('Position along edge')
                ax.set_ylabel('Curvature')
                ax.grid(True, alpha=0.3)
                ax.legend()
                
            except Exception as e:
                ax.text(0.5, 0.5, f'Error calculating curvature:\n{str(e)}', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{edge_name} Edge: Analysis Error')
        else:
            ax.text(0.5, 0.5, 'Insufficient edge points', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{edge_name} Edge: No Data')
    
    plt.tight_layout()
    plt.savefig(os.path.join(piece_shape_dir, 'edge_profiles.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Create classification visualization
    create_piece_classification_visualization(piece, piece_idx, piece_shape_dir)
    
    # Create shape metrics visualization
    create_shape_metrics_visualization(piece, piece_idx, piece_shape_dir)


def create_piece_classification_visualization(piece: Any, piece_idx: int, output_dir: str):
    """Create visual classification results for a piece."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # Draw piece outline if available
    if hasattr(piece, 'contour') and piece.contour is not None:
        contour_points = piece.contour.squeeze()
        if len(contour_points.shape) == 2:
            ax.plot(contour_points[:, 0], contour_points[:, 1], 'k-', linewidth=2, alpha=0.7)
    
    # Draw corners and edges with classification colors
    edge_colors = {'convex': 'red', 'concave': 'blue', 'flat': 'green', 'unknown': 'gray'}
    edge_names = ['Top', 'Right', 'Bottom', 'Left']
    
    for edge_idx, (edge, edge_name) in enumerate(zip(piece.edges, edge_names)):
        if edge.points and len(edge.points) > 2:
            points = np.array(edge.points)
            edge_type = getattr(edge, 'edge_type', 'unknown')
            sub_type = getattr(edge, 'sub_type', None)
            
            color = edge_colors.get(edge_type, 'gray')
            linewidth = 4 if sub_type == 'symmetric' else 2
            linestyle = '-' if sub_type == 'symmetric' else '--'
            
            ax.plot(points[:, 0], points[:, 1], color=color, 
                   linewidth=linewidth, linestyle=linestyle, 
                   label=f'{edge_name}: {edge_type}' + (f' ({sub_type})' if sub_type else ''))
    
    # Draw corners
    if hasattr(piece, 'corners') and piece.corners:
        corners = np.array(piece.corners)
        ax.scatter(corners[:, 0], corners[:, 1], c='orange', s=100, marker='o', 
                  edgecolor='black', linewidth=2, label='Corners', zorder=5)
    
    ax.set_title(f'Piece {piece_idx+1} - Edge Classification', fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'classification.png'), dpi=150, bbox_inches='tight')
    plt.close()


def create_shape_metrics_visualization(piece: Any, piece_idx: int, output_dir: str):
    """Create shape metrics visualization for a piece."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Piece {piece_idx+1} - Shape Metrics', fontsize=16, fontweight='bold')
    
    edge_names = ['Top', 'Right', 'Bottom', 'Left']
    metrics = {'symmetry': [], 'compactness': [], 'confidence': [], 'deviation': []}
    
    # Ensure we have edges to work with
    if not hasattr(piece, 'edges') or not piece.edges:
        # Create placeholder data for 4 edges
        for _ in range(4):
            metrics['symmetry'].append(0.0)
            metrics['confidence'].append(0.0)
            metrics['compactness'].append(0.0)
            metrics['deviation'].append(0.0)
    else:
        for edge in piece.edges:
            # Calculate symmetry score (1.0 for symmetric, 0.0 for asymmetric)
            symmetry = 1.0 if getattr(edge, 'sub_type', None) == 'symmetric' else 0.0
            metrics['symmetry'].append(symmetry)
            metrics['confidence'].append(getattr(edge, 'confidence', 0.0))
            metrics['compactness'].append(0.5)  # Placeholder
            metrics['deviation'].append(abs(getattr(edge, 'deviation', 0.0)))
        
        # Ensure we have exactly 4 edges (pad or truncate)
        for key in metrics:
            while len(metrics[key]) < 4:
                metrics[key].append(0.0)
            metrics[key] = metrics[key][:4]
    
    # Symmetry scores
    axes[0,0].bar(edge_names, metrics['symmetry'], color=['red' if s > 0.5 else 'blue' for s in metrics['symmetry']])
    axes[0,0].set_title('Edge Symmetry Scores')
    axes[0,0].set_ylabel('Symmetry (1=symmetric, 0=asymmetric)')
    axes[0,0].set_ylim(0, 1.1)
    axes[0,0].grid(True, alpha=0.3)
    
    # Confidence scores
    axes[0,1].bar(edge_names, metrics['confidence'], color='green', alpha=0.7)
    axes[0,1].set_title('Classification Confidence')
    axes[0,1].set_ylabel('Confidence Score')
    axes[0,1].set_ylim(0, 1.1)
    axes[0,1].grid(True, alpha=0.3)
    
    # Deviation values
    axes[1,0].bar(edge_names, metrics['deviation'], color='orange', alpha=0.7)
    axes[1,0].set_title('Edge Deviations')
    axes[1,0].set_ylabel('Absolute Deviation')
    axes[1,0].grid(True, alpha=0.3)
    
    # Summary radar chart
    angles = np.linspace(0, 2*np.pi, len(edge_names), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    confidence_values = metrics['confidence'] + [metrics['confidence'][0]]
    
    axes[1,1] = plt.subplot(2, 2, 4, projection='polar')
    axes[1,1].plot(angles, confidence_values, 'o-', linewidth=2, color='purple')
    axes[1,1].fill(angles, confidence_values, alpha=0.25, color='purple')
    axes[1,1].set_xticks(angles[:-1])
    axes[1,1].set_xticklabels(edge_names)
    axes[1,1].set_ylim(0, 1)
    axes[1,1].set_title('Confidence Radar')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'shape_metrics.png'), dpi=150, bbox_inches='tight')
    plt.close()


def create_shape_summary_visualization(pieces: List[Any], output_dir: str):
    """Create summary visualization of all pieces' shape analysis."""
    summary_dir = os.path.join(output_dir, 'summary')
    os.makedirs(summary_dir, exist_ok=True)
    
    # Collect edge type statistics with sub-types
    edge_stats = {}
    total_edges = 0
    
    for piece in pieces:
        if hasattr(piece, 'edges') and piece.edges:
            for edge in piece.edges:
                edge_type = getattr(edge, 'edge_type', 'unknown')
                sub_type = getattr(edge, 'sub_type', None)
                
                full_type = edge_type
                if sub_type:
                    full_type += f'_{sub_type}'
                
                edge_stats[full_type] = edge_stats.get(full_type, 0) + 1
                total_edges += 1
    
    # Create edge gallery grouped by type
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Shape Analysis Summary', fontsize=16, fontweight='bold')
    
    # Distribution pie chart
    if edge_stats:
        labels = list(edge_stats.keys())
        sizes = list(edge_stats.values())
        colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
        
        axes[0,0].pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
        axes[0,0].set_title('Edge Type Distribution')
    
    # Bar chart with counts
    if edge_stats:
        axes[0,1].bar(labels, sizes, color=colors)
        axes[0,1].set_title('Edge Type Counts')
        axes[0,1].set_ylabel('Number of Edges')
        axes[0,1].tick_params(axis='x', rotation=45)
        axes[0,1].grid(True, alpha=0.3)
    
    # Confidence distribution
    all_confidences = []
    for piece in pieces:
        if hasattr(piece, 'edges') and piece.edges:
            for edge in piece.edges:
                confidence = getattr(edge, 'confidence', 0.0)
                all_confidences.append(confidence)
    
    if all_confidences:
        axes[1,0].hist(all_confidences, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1,0].set_title('Classification Confidence Distribution')
        axes[1,0].set_xlabel('Confidence Score')
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].grid(True, alpha=0.3)
    
    # Statistics summary text
    axes[1,1].axis('off')
    stats_text = f"Shape Analysis Statistics\n"
    stats_text += f"{'='*30}\n"
    stats_text += f"Total pieces analyzed: {len(pieces)}\n"
    stats_text += f"Total edges classified: {total_edges}\n"
    if all_confidences:
        stats_text += f"Average confidence: {np.mean(all_confidences):.3f}\n"
        stats_text += f"Min confidence: {min(all_confidences):.3f}\n"
        stats_text += f"Max confidence: {max(all_confidences):.3f}\n"
    
    stats_text += f"\nEdge Type Breakdown:\n"
    for edge_type, count in edge_stats.items():
        percentage = (count / total_edges * 100) if total_edges > 0 else 0
        stats_text += f"  {edge_type}: {count} ({percentage:.1f}%)\n"
    
    axes[1,1].text(0.1, 0.9, stats_text, fontsize=10, fontfamily='monospace',
                   verticalalignment='top', transform=axes[1,1].transAxes)
    
    plt.tight_layout()
    plt.savefig(os.path.join(summary_dir, 'edge_gallery.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Create statistics file
    stats_file = os.path.join(summary_dir, 'statistics.txt')
    with open(stats_file, 'w') as f:
        f.write(stats_text)