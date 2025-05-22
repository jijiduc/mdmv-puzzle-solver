"""Detailed corner detection analysis and debugging tools."""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from typing import List, Tuple, Dict, Any
import os


def analyze_corner_detection_method(piece_data: Dict[str, Any], piece_img: np.ndarray, 
                                   piece_idx: int, output_dir: str):
    """Create comprehensive analysis of the corner detection method.
    
    Args:
        piece_data: Piece processing result data
        piece_img: Piece image
        piece_idx: Piece index
        output_dir: Output directory for analysis
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract edge points and calculate centroid
    piece_mask = np.array(piece_data.get('mask', []), dtype=np.uint8) if 'mask' in piece_data else None
    if piece_mask is None:
        return
    
    # Detect edges
    edges = cv2.Canny(piece_mask, 50, 150)
    edge_points = np.where(edges > 0)
    y_edge, x_edge = edge_points[0], edge_points[1]
    edge_coordinates = np.column_stack((x_edge, y_edge))
    
    if len(edge_coordinates) == 0:
        return
    
    # Calculate centroid
    moments = cv2.moments(piece_mask)
    if moments["m00"] != 0:
        centroid_x = int(moments["m10"] / moments["m00"])
        centroid_y = int(moments["m01"] / moments["m00"])
    else:
        centroid_x = piece_mask.shape[1] // 2
        centroid_y = piece_mask.shape[0] // 2
    centroid = (centroid_x, centroid_y)
    
    # Calculate distances and angles from centroid
    distances = []
    angles = []
    coords = []
    
    for x, y in edge_coordinates:
        # Distance from centroid
        dist = np.sqrt((x - centroid_x)**2 + (y - centroid_y)**2)
        distances.append(dist)
        
        # Angle from centroid
        angle = np.arctan2(y - centroid_y, x - centroid_x)
        angles.append(angle)
        coords.append((x, y))
    
    # Sort by angle for proper ordering
    if angles:
        sorted_data = sorted(zip(angles, distances, coords))
        sorted_angles, sorted_distances, sorted_coords = zip(*sorted_data)
        sorted_angles = list(sorted_angles)
        sorted_distances = list(sorted_distances)
        sorted_coords = list(sorted_coords)
    else:
        return
    
    # Apply the corner detection algorithm step by step
    corners_analysis = detailed_corner_detection_analysis(
        sorted_distances, sorted_coords, sorted_angles
    )
    
    # Create comprehensive visualization
    create_corner_detection_breakdown(
        piece_img, sorted_coords, sorted_distances, sorted_angles, 
        centroid, corners_analysis, piece_idx, output_dir
    )
    
    # Create detailed report
    create_corner_detection_report(
        corners_analysis, centroid, piece_idx, output_dir
    )


def detailed_corner_detection_analysis(distances: List[float], coords: List[Tuple[int, int]], 
                                     angles: List[float]) -> Dict[str, Any]:
    """Analyze the corner detection algorithm step by step.
    
    Args:
        distances: Sorted distances from centroid
        coords: Sorted coordinates
        angles: Sorted angles from centroid
        
    Returns:
        Dictionary with detailed analysis results
    """
    analysis = {
        'input_data': {
            'num_edge_points': len(distances),
            'distance_range': (min(distances), max(distances)),
            'mean_distance': np.mean(distances),
            'std_distance': np.std(distances)
        },
        'steps': {}
    }
    
    if len(distances) < 4:
        analysis['result'] = coords[:4] if len(coords) >= 4 else coords
        analysis['method_used'] = 'insufficient_data'
        return analysis
    
    # Step 1: Smoothing
    if len(distances) > 5:
        window_length = min(len(distances)//4*2+1, 11)
        smoothed_distances = savgol_filter(distances, window_length, 2)
        analysis['steps']['smoothing'] = {
            'applied': True,
            'window_length': window_length,
            'polynomial_order': 2,
            'original_distances': distances.copy(),
            'smoothed_distances': smoothed_distances.tolist()
        }
    else:
        smoothed_distances = distances
        analysis['steps']['smoothing'] = {
            'applied': False,
            'reason': 'too_few_points'
        }
    
    # Step 2: Peak detection
    mean_dist = np.mean(smoothed_distances)
    peaks, peak_properties = find_peaks(smoothed_distances, height=mean_dist)
    
    analysis['steps']['peak_detection'] = {
        'threshold': mean_dist,
        'peaks_found': len(peaks),
        'peak_indices': peaks.tolist(),
        'peak_heights': [smoothed_distances[i] for i in peaks],
        'target_peaks': 4
    }
    
    # Step 3: Method selection and corner finding
    if len(peaks) == 4:
        # Primary method: use detected peaks
        final_corners = [coords[i] for i in peaks]
        analysis['method_used'] = 'peak_detection'
        analysis['steps']['method_selection'] = {
            'method': 'peak_detection',
            'reason': 'exactly_4_peaks_found'
        }
    else:
        # Alternative method: angular sectors
        sectors = []
        sector_analysis = []
        
        for i in range(4):
            start_angle = -np.pi + i * np.pi/2
            end_angle = -np.pi + (i+1) * np.pi/2
            
            sector_indices = [j for j, angle in enumerate(angles) 
                            if start_angle <= angle < end_angle]
            
            sector_info = {
                'sector': i+1,
                'angle_range': (start_angle, end_angle),
                'points_in_sector': len(sector_indices),
                'indices': sector_indices
            }
            
            if sector_indices:
                max_dist_idx = max(sector_indices, key=lambda j: distances[j])
                sectors.append(max_dist_idx)
                sector_info['max_distance_idx'] = max_dist_idx
                sector_info['max_distance'] = distances[max_dist_idx]
            else:
                sector_info['max_distance_idx'] = None
                
            sector_analysis.append(sector_info)
        
        analysis['steps']['angular_sectors'] = {
            'sectors': sector_analysis,
            'valid_sectors': len(sectors)
        }
        
        if len(sectors) == 4:
            peaks = sorted(sectors)
            analysis['method_used'] = 'angular_sectors'
        else:
            # Fallback method: uniform distribution
            peaks = list(range(0, len(distances), len(distances)//4))[:4]
            analysis['method_used'] = 'uniform_distribution'
        
        analysis['steps']['method_selection'] = {
            'method': analysis['method_used'],
            'reason': f'peak_detection_found_{len(peaks)}_peaks'
        }
    
    # Step 4: Peak refinement
    original_peaks = peaks.copy()
    
    if len(peaks) > 4:
        # Keep 4 highest peaks
        peak_heights = [distances[i] for i in peaks]
        top_peaks = sorted(zip(peak_heights, peaks), reverse=True)[:4]
        peaks = sorted([p[1] for p in top_peaks])
        
        analysis['steps']['peak_refinement'] = {
            'action': 'reduced_to_4_highest',
            'original_count': len(original_peaks),
            'removed_peaks': len(original_peaks) - 4
        }
        
    elif len(peaks) < 4:
        # Add more peaks by subdividing gaps
        refinement_steps = []
        
        while len(peaks) < 4 and len(distances) > len(peaks):
            gaps = []
            for i in range(len(peaks)):
                next_i = (i + 1) % len(peaks)
                gap_size = (peaks[next_i] - peaks[i]) % len(distances)
                gaps.append((gap_size, i))
            
            largest_gap = max(gaps)
            gap_idx = largest_gap[1]
            mid_point = (peaks[gap_idx] + peaks[(gap_idx + 1) % len(peaks)]) // 2
            peaks.insert(gap_idx + 1, mid_point)
            
            refinement_steps.append({
                'iteration': len(refinement_steps) + 1,
                'largest_gap_size': largest_gap[0],
                'gap_position': gap_idx,
                'added_peak': mid_point,
                'current_peaks': peaks.copy()
            })
        
        analysis['steps']['peak_refinement'] = {
            'action': 'added_peaks_by_subdivision',
            'original_count': len(original_peaks),
            'final_count': len(peaks),
            'subdivision_steps': refinement_steps
        }
    else:
        analysis['steps']['peak_refinement'] = {
            'action': 'no_refinement_needed'
        }
    
    # Final result
    final_corners = [coords[i] for i in peaks[:4]]
    analysis['result'] = final_corners
    analysis['final_peak_indices'] = peaks[:4]
    
    return analysis


def create_corner_detection_breakdown(piece_img: np.ndarray, coords: List[Tuple[int, int]], 
                                    distances: List[float], angles: List[float],
                                    centroid: Tuple[int, int], analysis: Dict[str, Any],
                                    piece_idx: int, output_dir: str):
    """Create detailed visualization of corner detection process.
    
    Args:
        piece_img: Piece image
        coords: Edge coordinates
        distances: Distances from centroid
        angles: Angles from centroid
        centroid: Centroid coordinates
        analysis: Corner detection analysis results
        piece_idx: Piece index
        output_dir: Output directory
    """
    # Create comprehensive figure
    fig = plt.figure(figsize=(16, 12))
    
    # Main title
    fig.suptitle(f'Piece {piece_idx+1} - Corner Detection Method Breakdown', 
                fontsize=16, fontweight='bold')
    
    # 1. Original piece with all edge points
    ax1 = plt.subplot(3, 4, 1)
    ax1.imshow(cv2.cvtColor(piece_img, cv2.COLOR_BGR2RGB))
    
    # Plot all edge points
    coords_array = np.array(coords)
    ax1.scatter(coords_array[:, 0], coords_array[:, 1], c='red', s=1, alpha=0.5)
    ax1.scatter(centroid[0], centroid[1], c='blue', s=100, marker='x', linewidth=3)
    ax1.set_title('Edge Points & Centroid')
    ax1.axis('off')
    
    # 2. Distance profile (polar)
    ax2 = plt.subplot(3, 4, 2, projection='polar')
    ax2.plot(angles, distances, 'b-', linewidth=1, alpha=0.7)
    ax2.scatter(angles, distances, c='red', s=2)
    ax2.set_title('Distance Profile (Polar)')
    
    # 3. Distance profile (linear)
    ax3 = plt.subplot(3, 4, 3)
    ax3.plot(distances, 'b-', linewidth=1, alpha=0.7)
    ax3.axhline(y=np.mean(distances), color='red', linestyle='--', alpha=0.7, label='Mean')
    ax3.set_title('Distance Profile (Linear)')
    ax3.set_xlabel('Point Index')
    ax3.set_ylabel('Distance from Centroid')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Smoothed distances (if applied)
    ax4 = plt.subplot(3, 4, 4)
    if analysis['steps']['smoothing']['applied']:
        smoothed = analysis['steps']['smoothing']['smoothed_distances']
        ax4.plot(distances, 'b-', alpha=0.5, label='Original')
        ax4.plot(smoothed, 'r-', linewidth=2, label='Smoothed')
        ax4.set_title('Distance Smoothing')
    else:
        ax4.plot(distances, 'b-', linewidth=1)
        ax4.text(0.5, 0.5, 'No smoothing applied\n(too few points)', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Distance Profile (No Smoothing)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Peak detection results
    ax5 = plt.subplot(3, 4, 5)
    smoothed_distances = (analysis['steps']['smoothing']['smoothed_distances'] 
                         if analysis['steps']['smoothing']['applied'] else distances)
    ax5.plot(smoothed_distances, 'b-', linewidth=1)
    
    peak_data = analysis['steps']['peak_detection']
    if peak_data['peaks_found'] > 0:
        peaks = peak_data['peak_indices']
        ax5.scatter(peaks, [smoothed_distances[i] for i in peaks], 
                   c='red', s=50, marker='^', label=f"{len(peaks)} peaks")
    
    ax5.axhline(y=peak_data['threshold'], color='orange', linestyle='--', 
               alpha=0.7, label='Threshold')
    ax5.set_title(f'Peak Detection ({peak_data["peaks_found"]} peaks)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Angular sectors (if used)
    ax6 = plt.subplot(3, 4, 6, projection='polar')
    if 'angular_sectors' in analysis['steps']:
        # Color code by sector
        colors = ['red', 'green', 'blue', 'orange']
        for i, sector in enumerate(analysis['steps']['angular_sectors']['sectors']):
            if sector['indices']:
                sector_angles = [angles[j] for j in sector['indices']]
                sector_distances = [distances[j] for j in sector['indices']]
                ax6.scatter(sector_angles, sector_distances, c=colors[i], 
                           s=10, alpha=0.6, label=f'Sector {i+1}')
                
                if sector['max_distance_idx'] is not None:
                    max_idx = sector['max_distance_idx']
                    ax6.scatter(angles[max_idx], distances[max_idx], 
                               c=colors[i], s=100, marker='*', edgecolors='black')
    
    ax6.set_title('Angular Sectors')
    ax6.legend()
    
    # 7. Final corners on piece
    ax7 = plt.subplot(3, 4, 7)
    ax7.imshow(cv2.cvtColor(piece_img, cv2.COLOR_BGR2RGB))
    
    final_corners = analysis['result']
    if final_corners:
        corners_array = np.array(final_corners)
        ax7.scatter(corners_array[:, 0], corners_array[:, 1], 
                   c='lime', s=100, marker='o', edgecolors='black', linewidth=2)
        
        # Label corners
        for i, corner in enumerate(final_corners):
            ax7.annotate(f'C{i+1}', corner, xytext=(5, 5), 
                        textcoords='offset points', fontsize=10, 
                        color='white', fontweight='bold')
    
    ax7.scatter(centroid[0], centroid[1], c='red', s=80, marker='x', linewidth=3)
    ax7.set_title(f'Final Corners ({len(final_corners)})')
    ax7.axis('off')
    
    # 8. Method used text summary
    ax8 = plt.subplot(3, 4, 8)
    ax8.axis('off')
    
    summary_text = f"""Method Used: {analysis['method_used']}
    
Edge Points: {analysis['input_data']['num_edge_points']}
Distance Range: {analysis['input_data']['distance_range'][0]:.1f} - {analysis['input_data']['distance_range'][1]:.1f}
Mean Distance: {analysis['input_data']['mean_distance']:.1f}
Std Distance: {analysis['input_data']['std_distance']:.1f}

Smoothing: {'Yes' if analysis['steps']['smoothing']['applied'] else 'No'}
Peaks Found: {analysis['steps']['peak_detection']['peaks_found']}
Final Corners: {len(analysis['result'])}"""
    
    ax8.text(0.1, 0.9, summary_text, transform=ax8.transAxes, 
            fontsize=10, verticalalignment='top', fontfamily='monospace')
    ax8.set_title('Detection Summary')
    
    # Lower row: detailed step analysis
    # 9. Distance histogram
    ax9 = plt.subplot(3, 4, 9)
    ax9.hist(distances, bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax9.axvline(x=np.mean(distances), color='red', linestyle='--', label='Mean')
    ax9.set_title('Distance Distribution')
    ax9.set_xlabel('Distance from Centroid')
    ax9.set_ylabel('Frequency')
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    
    # 10. Angle distribution
    ax10 = plt.subplot(3, 4, 10)
    ax10.hist(angles, bins=30, alpha=0.7, color='green', edgecolor='black')
    ax10.set_title('Angle Distribution')
    ax10.set_xlabel('Angle (radians)')
    ax10.set_ylabel('Frequency')
    ax10.grid(True, alpha=0.3)
    
    # 11. Corner quality metrics
    ax11 = plt.subplot(3, 4, 11)
    ax11.axis('off')
    
    if len(final_corners) >= 3:
        # Calculate corner metrics
        corner_distances = [np.sqrt((c[0] - centroid[0])**2 + (c[1] - centroid[1])**2) 
                           for c in final_corners]
        distance_variance = np.var(corner_distances)
        distance_cv = np.std(corner_distances) / np.mean(corner_distances) if np.mean(corner_distances) > 0 else 0
        
        quality_text = f"""Corner Quality Metrics:
        
Distance Variance: {distance_variance:.1f}
Distance CV: {distance_cv:.3f}
Distance Range: {max(corner_distances) - min(corner_distances):.1f}

Quality Assessment:
Distance CV < 0.1: {'✓' if distance_cv < 0.1 else '✗'} (Uniform)
All corners found: {'✓' if len(final_corners) == 4 else '✗'}"""
        
        ax11.text(0.1, 0.9, quality_text, transform=ax11.transAxes, 
                 fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    ax11.set_title('Quality Assessment')
    
    # 12. Issues and recommendations
    ax12 = plt.subplot(3, 4, 12)
    ax12.axis('off')
    
    issues = []
    recommendations = []
    
    if analysis['input_data']['num_edge_points'] < 50:
        issues.append("Too few edge points")
        recommendations.append("Improve edge detection")
    
    if analysis['steps']['peak_detection']['peaks_found'] != 4:
        issues.append(f"Peak detection found {analysis['steps']['peak_detection']['peaks_found']} peaks")
        recommendations.append("Adjust peak detection threshold")
    
    if analysis['method_used'] == 'uniform_distribution':
        issues.append("Fallback method used")
        recommendations.append("Improve primary methods")
    
    if not issues:
        issues.append("No major issues detected")
        recommendations.append("Detection appears successful")
    
    issues_text = "Issues:\n" + "\n".join(f"• {issue}" for issue in issues[:3])
    issues_text += "\n\nRecommendations:\n" + "\n".join(f"• {rec}" for rec in recommendations[:3])
    
    ax12.text(0.1, 0.9, issues_text, transform=ax12.transAxes, 
             fontsize=9, verticalalignment='top')
    ax12.set_title('Issues & Recommendations')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'piece_{piece_idx+1}_corner_detection_breakdown.png'), 
               dpi=150, bbox_inches='tight')
    plt.close()


def create_corner_detection_report(analysis: Dict[str, Any], centroid: Tuple[int, int],
                                 piece_idx: int, output_dir: str):
    """Create detailed text report of corner detection analysis.
    
    Args:
        analysis: Corner detection analysis results
        centroid: Centroid coordinates
        piece_idx: Piece index
        output_dir: Output directory
    """
    report_file = os.path.join(output_dir, f'piece_{piece_idx+1}_corner_detection_analysis.txt')
    
    with open(report_file, 'w') as f:
        f.write(f"Piece {piece_idx+1} - Corner Detection Method Analysis\n")
        f.write("=" * 60 + "\n\n")
        
        # Input data summary
        f.write("INPUT DATA SUMMARY:\n")
        f.write("-" * 20 + "\n")
        input_data = analysis['input_data']
        f.write(f"Number of edge points: {input_data['num_edge_points']}\n")
        f.write(f"Distance range: {input_data['distance_range'][0]:.1f} - {input_data['distance_range'][1]:.1f} pixels\n")
        f.write(f"Mean distance: {input_data['mean_distance']:.1f} pixels\n")
        f.write(f"Standard deviation: {input_data['std_distance']:.1f} pixels\n")
        f.write(f"Centroid: ({centroid[0]}, {centroid[1]})\n\n")
        
        # Method used
        f.write("DETECTION METHOD:\n")
        f.write("-" * 17 + "\n")
        f.write(f"Primary method: {analysis['method_used']}\n")
        if 'method_selection' in analysis['steps']:
            f.write(f"Reason: {analysis['steps']['method_selection']['reason']}\n")
        f.write("\n")
        
        # Step-by-step analysis
        f.write("STEP-BY-STEP ANALYSIS:\n")
        f.write("-" * 22 + "\n")
        
        # Smoothing
        smoothing = analysis['steps']['smoothing']
        f.write("1. Distance Smoothing:\n")
        if smoothing['applied']:
            f.write(f"   ✓ Applied Savitzky-Golay filter\n")
            f.write(f"   ✓ Window length: {smoothing['window_length']}\n")
            f.write(f"   ✓ Polynomial order: {smoothing['polynomial_order']}\n")
        else:
            f.write(f"   ✗ Not applied - {smoothing.get('reason', 'unknown reason')}\n")
        f.write("\n")
        
        # Peak detection
        peak_detection = analysis['steps']['peak_detection']
        f.write("2. Peak Detection:\n")
        f.write(f"   Threshold (mean distance): {peak_detection['threshold']:.1f}\n")
        f.write(f"   Peaks found: {peak_detection['peaks_found']} (target: {peak_detection['target_peaks']})\n")
        if peak_detection['peaks_found'] > 0:
            f.write(f"   Peak indices: {peak_detection['peak_indices']}\n")
            f.write(f"   Peak heights: {[f'{h:.1f}' for h in peak_detection['peak_heights']]}\n")
        f.write("\n")
        
        # Alternative method (if used)
        if 'angular_sectors' in analysis['steps']:
            f.write("3. Angular Sectors (Alternative Method):\n")
            sectors = analysis['steps']['angular_sectors']
            f.write(f"   Valid sectors found: {sectors['valid_sectors']}/4\n")
            for sector in sectors['sectors']:
                f.write(f"   Sector {sector['sector']}: {sector['points_in_sector']} points")
                if sector['max_distance_idx'] is not None:
                    f.write(f", max distance: {sector['max_distance']:.1f}")
                f.write("\n")
            f.write("\n")
        
        # Peak refinement
        if 'peak_refinement' in analysis['steps']:
            refinement = analysis['steps']['peak_refinement']
            f.write("4. Peak Refinement:\n")
            f.write(f"   Action: {refinement['action']}\n")
            if 'original_count' in refinement:
                f.write(f"   Original peaks: {refinement['original_count']}\n")
                if 'final_count' in refinement:
                    f.write(f"   Final peaks: {refinement['final_count']}\n")
                if 'subdivision_steps' in refinement:
                    f.write(f"   Subdivision steps: {len(refinement['subdivision_steps'])}\n")
            f.write("\n")
        
        # Final results
        f.write("FINAL RESULTS:\n")
        f.write("-" * 14 + "\n")
        f.write(f"Corners found: {len(analysis['result'])}\n")
        f.write(f"Final peak indices: {analysis.get('final_peak_indices', 'N/A')}\n")
        f.write("Corner coordinates:\n")
        for i, corner in enumerate(analysis['result']):
            distance = np.sqrt((corner[0] - centroid[0])**2 + (corner[1] - centroid[1])**2)
            f.write(f"  C{i+1}: ({corner[0]:.1f}, {corner[1]:.1f}) - Distance: {distance:.1f}\n")
        
        # Quality assessment
        if len(analysis['result']) >= 3:
            corner_distances = [np.sqrt((c[0] - centroid[0])**2 + (c[1] - centroid[1])**2) 
                               for c in analysis['result']]
            distance_variance = np.var(corner_distances)
            distance_cv = np.std(corner_distances) / np.mean(corner_distances)
            
            f.write("\nQUALITY ASSESSMENT:\n")
            f.write("-" * 19 + "\n")
            f.write(f"Distance variance: {distance_variance:.1f}\n")
            f.write(f"Distance coefficient of variation: {distance_cv:.3f}\n")
            f.write(f"Distance uniformity: {'Good' if distance_cv < 0.1 else 'Poor'} (CV < 0.1)\n")
            f.write(f"Corner count: {'Correct' if len(analysis['result']) == 4 else 'Incorrect'} (4 expected)\n")
        
        f.write("\nMETHOD STRENGTHS & WEAKNESSES:\n")
        f.write("-" * 31 + "\n")
        f.write("Strengths:\n")
        f.write("+ Uses distance from centroid (geometrically sound)\n")
        f.write("+ Applies smoothing to reduce noise\n")
        f.write("+ Has fallback methods for difficult cases\n")
        f.write("+ Handles irregular piece shapes\n")
        f.write("\nWeaknesses:\n")
        f.write("- Assumes corners are at maximum distances\n")
        f.write("- May fail on pieces with rounded corners\n")
        f.write("- Sensitive to edge detection quality\n")
        f.write("- Angular sector method may miss close corners\n")
        f.write("- No validation of corner angles\n")