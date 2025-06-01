"""Edge matching visualization functions for puzzle assembly."""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Any
import json
from datetime import datetime

from ..core.piece import Piece, EdgeSegment
from ..features.edge_matching import EdgeMatch, GlobalMatchRegistry, EdgeSpatialIndex
from ..features.shape_analysis import calculate_curvature_profile
from ..features.color_analysis import color_distance


def create_edge_match_visualization(piece1: Piece, edge1_idx: int, 
                                  piece2: Piece, edge2_idx: int, 
                                  match: EdgeMatch, output_dir: str) -> str:
    """Create detailed visualization of a single edge match.
    
    Args:
        piece1: First puzzle piece
        edge1_idx: Edge index on first piece
        piece2: Second puzzle piece  
        edge2_idx: Edge index on second piece
        match: EdgeMatch object with scores
        output_dir: Output directory
        
    Returns:
        Path to saved visualization
    """
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    # Title
    fig.suptitle(f'Edge Match: Piece {piece1.index} Edge {edge1_idx} ↔ Piece {piece2.index} Edge {edge2_idx}',
                 fontsize=16, fontweight='bold')
    
    # Get edges
    edge1 = piece1.get_edge(edge1_idx)
    edge2 = piece2.get_edge(edge2_idx)
    
    if not edge1 or not edge2:
        plt.close(fig)
        return ""
    
    # 1. Edge images side by side
    ax1 = fig.add_subplot(gs[0, 0:2])
    ax2 = fig.add_subplot(gs[0, 2:4])
    
    # Extract edge regions
    edge1_img = _extract_edge_region(piece1.image, edge1.points)
    edge2_img = _extract_edge_region(piece2.image, edge2.points)
    
    ax1.imshow(cv2.cvtColor(edge1_img, cv2.COLOR_BGR2RGB))
    ax1.set_title(f'Piece {piece1.index} - Edge {edge1_idx}\n'
                  f'Type: {edge1.edge_type} ({edge1.sub_type or "none"})', fontsize=12)
    ax1.axis('off')
    
    ax2.imshow(cv2.cvtColor(edge2_img, cv2.COLOR_BGR2RGB))
    ax2.set_title(f'Piece {piece2.index} - Edge {edge2_idx}\n'
                  f'Type: {edge2.edge_type} ({edge2.sub_type or "none"})', fontsize=12)
    ax2.axis('off')
    
    # 2. Shape profile comparison
    ax3 = fig.add_subplot(gs[1, :])
    edge1_points = np.array(edge1.points)
    edge2_points = np.array(edge2.points)
    
    if len(edge1_points) > 3 and len(edge2_points) > 3:
        curv1 = calculate_curvature_profile(edge1_points)
        curv2 = calculate_curvature_profile(edge2_points)
        
        # Normalize to same length for comparison
        x1 = np.linspace(0, 1, len(curv1))
        x2 = np.linspace(0, 1, len(curv2))
        
        ax3.plot(x1, curv1, 'b-', label=f'Edge {edge1_idx}', linewidth=2, alpha=0.7)
        ax3.plot(x2, curv2, 'r-', label=f'Edge {edge2_idx}', linewidth=2, alpha=0.7)
        ax3.fill_between(x1, curv1, alpha=0.3, color='blue')
        ax3.fill_between(x2, curv2, alpha=0.3, color='red')
        
        ax3.set_xlabel('Normalized Position', fontsize=12)
        ax3.set_ylabel('Curvature', fontsize=12)
        ax3.set_title('Shape Profile Comparison', fontsize=14)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # 3. Color sequence comparison
    ax4 = fig.add_subplot(gs[2, 0:2])
    if edge1.color_sequence and edge2.color_sequence:
        _plot_color_sequences(ax4, edge1.color_sequence, edge2.color_sequence)
    ax4.set_title('Color Sequence Comparison', fontsize=12)
    
    # 4. Match scores breakdown
    ax5 = fig.add_subplot(gs[2, 2])
    scores = {
        'Shape': match.shape_score,
        'Color': match.color_score,
        'Total': match.similarity_score
    }
    
    bars = ax5.bar(scores.keys(), scores.values(), color=['skyblue', 'lightcoral', 'lightgreen'])
    ax5.set_ylim(0, 1.0)
    ax5.set_ylabel('Score', fontsize=12)
    ax5.set_title('Match Scores', fontsize=12)
    
    # Add value labels on bars
    for bar, (name, value) in zip(bars, scores.items()):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{value:.3f}', ha='center', va='bottom', fontsize=10)
    
    # 5. Validation flags
    ax6 = fig.add_subplot(gs[2, 3])
    ax6.axis('off')
    
    validation_text = "Validation Flags:\n\n"
    for flag, value in match.validation_flags.items():
        symbol = "✓" if value else "✗"
        color = "green" if value else "red"
        validation_text += f"{symbol} {flag.replace('_', ' ').title()}\n"
    
    validation_text += f"\nMatch Type: {match.match_type.upper()}"
    validation_text += f"\nConfidence: {match.confidence:.2%}"
    
    ax6.text(0.1, 0.9, validation_text, transform=ax6.transAxes,
             fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 
                              f'match_p{piece1.index}e{edge1_idx}_p{piece2.index}e{edge2_idx}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return output_path


def create_match_candidates_gallery(piece: Piece, edge_idx: int, 
                                   candidates: List[Tuple[EdgeMatch, Piece]], 
                                   registry: GlobalMatchRegistry,
                                   output_dir: str) -> str:
    """Create enhanced gallery view of match candidates for an edge.
    
    Args:
        piece: Source puzzle piece
        edge_idx: Edge index to show matches for
        candidates: List of (EdgeMatch, target_piece) tuples
        registry: Global match registry
        output_dir: Output directory
        
    Returns:
        Path to saved visualization
    """
    # Show more candidates for better overview
    n_candidates = min(len(candidates), 20)  # Show up to 20 candidates
    if n_candidates == 0:
        return ""
    
    # Create figure with source edge comparison
    fig = plt.figure(figsize=(24, 16))
    gs = GridSpec(5, 5, figure=fig, hspace=0.4, wspace=0.3,
                  height_ratios=[1.5, 1, 1, 1, 1])
    
    source_edge = piece.get_edge(edge_idx)
    if not source_edge:
        plt.close(fig)
        return ""
    
    # Title with source edge information
    fig.suptitle(f'Match Candidates for Piece {piece.index} - Edge {edge_idx}\n'
                 f'Source Edge Type: {source_edge.edge_type} ({source_edge.sub_type or "none"})',
                 fontsize=18, fontweight='bold')
    
    # 1. Show source edge prominently at the top
    source_ax = fig.add_subplot(gs[0, 1:4])
    source_img = _extract_edge_region(piece.image, source_edge.points)
    source_ax.imshow(cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB))
    source_ax.set_title(f'SOURCE EDGE\nLength: {source_edge.length:.0f} pixels', 
                       fontsize=14, fontweight='bold', color='darkblue')
    source_ax.axis('off')
    
    # Add golden border to source
    for spine in source_ax.spines.values():
        spine.set_edgecolor('gold')
        spine.set_linewidth(4)
    
    # 2. Show statistics box
    stats_ax = fig.add_subplot(gs[0, 0])
    stats_ax.axis('off')
    
    # Calculate statistics
    confirmed_count = sum(1 for match, _ in candidates 
                         if (piece.index, edge_idx, match.piece_idx, match.edge_idx) in registry.confirmed_matches)
    perfect_count = sum(1 for match, _ in candidates if match.match_type == 'perfect')
    good_count = sum(1 for match, _ in candidates if match.match_type == 'good')
    possible_count = sum(1 for match, _ in candidates if match.match_type == 'possible')
    
    stats_text = f'Total Candidates: {len(candidates)}\n\n'
    stats_text += f'✓ Confirmed: {confirmed_count}\n'
    stats_text += f'★ Perfect: {perfect_count}\n'
    stats_text += f'● Good: {good_count}\n'
    stats_text += f'○ Possible: {possible_count}\n\n'
    stats_text += f'Avg Score: {np.mean([m.similarity_score for m, _ in candidates]):.3f}'
    
    stats_ax.text(0.1, 0.9, stats_text, transform=stats_ax.transAxes,
                 fontsize=12, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # 3. Show candidates in grid below
    for idx, (match, target_piece) in enumerate(candidates[:n_candidates]):
        row = (idx // 5) + 1  # Start from row 1 (after source)
        col = idx % 5
        
        if row >= 5:  # Safety check
            break
            
        ax = fig.add_subplot(gs[row, col])
        
        target_edge = target_piece.get_edge(match.edge_idx)
        if not target_edge:
            ax.axis('off')
            continue
        
        # Extract and display edge region
        edge_img = _extract_edge_region(target_piece.image, target_edge.points)
        ax.imshow(cv2.cvtColor(edge_img, cv2.COLOR_BGR2RGB))
        
        # Check match status
        is_confirmed = (piece.index, edge_idx, target_piece.index, match.edge_idx) in registry.confirmed_matches
        
        # Enhanced border styling
        if is_confirmed:
            border_color = 'green'
            border_width = 4
        else:
            border_color = _get_match_quality_color(match.similarity_score)
            border_width = 3
        
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(border_width)
        
        # Enhanced title information
        title = f'P{target_piece.index}:E{match.edge_idx}'
        if is_confirmed:
            title = f'✓ {title}'
        
        # Add edge type compatibility indicator
        type_compat = "✓" if _are_types_compatible(source_edge.edge_type, target_edge.edge_type) else "✗"
        
        subtitle = f'{target_edge.edge_type} {type_compat}\n'
        subtitle += f'S:{match.similarity_score:.2f} '
        subtitle += f'({match.match_type})\n'
        subtitle += f'L:{target_edge.length:.0f}px'
        
        # Add rank number
        rank_text = f'#{idx + 1}'
        ax.text(0.05, 0.95, rank_text, transform=ax.transAxes,
                fontsize=10, fontweight='bold', color='white',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.text(0.5, -0.15, subtitle, transform=ax.transAxes,
                fontsize=9, ha='center', va='top')
        ax.axis('off')
    
    # Hide unused subplots
    total_positions = 20  # 4 rows * 5 cols
    for idx in range(n_candidates, total_positions):
        row = (idx // 5) + 1
        col = idx % 5
        if row < 5:
            ax = fig.add_subplot(gs[row, col])
            ax.axis('off')
    
    plt.tight_layout()
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'candidates_p{piece.index}e{edge_idx}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return output_path


def create_match_validation_dashboard(registry: GlobalMatchRegistry, 
                                     spatial_index: EdgeSpatialIndex,
                                     output_dir: str) -> str:
    """Create comprehensive match validation dashboard.
    
    Args:
        registry: Global match registry
        spatial_index: Edge spatial index
        output_dir: Output directory
        
    Returns:
        Path to saved visualization
    """
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    fig.suptitle('Edge Matching Validation Dashboard', fontsize=18, fontweight='bold')
    
    # 1. Match type distribution
    ax1 = fig.add_subplot(gs[0, 0])
    match_types = {}
    for matches_dict in registry.matches.values():
        for match in matches_dict.values():
            key = f"{match.match_type}"
            match_types[key] = match_types.get(key, 0) + 1
    
    if match_types:
        ax1.pie(match_types.values(), labels=match_types.keys(), autopct='%1.1f%%',
                colors=['lightgreen', 'yellow', 'lightcoral'])
        ax1.set_title('Match Type Distribution', fontsize=14)
    
    # 2. Score distribution histogram
    ax2 = fig.add_subplot(gs[0, 1])
    all_scores = []
    for matches_dict in registry.matches.values():
        for match in matches_dict.values():
            all_scores.append(match.similarity_score)
    
    if all_scores:
        ax2.hist(all_scores, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
        ax2.axvline(np.mean(all_scores), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(all_scores):.3f}')
        ax2.set_xlabel('Similarity Score', fontsize=12)
        ax2.set_ylabel('Count', fontsize=12)
        ax2.set_title('Score Distribution', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # 3. Edge type compatibility matrix
    ax3 = fig.add_subplot(gs[0, 2])
    edge_types = ['flat', 'convex', 'concave']
    compatibility_matrix = np.zeros((len(edge_types), len(edge_types)))
    
    for i, type1 in enumerate(edge_types):
        for j, type2 in enumerate(edge_types):
            compatibility_matrix[i, j] = registry.match_stats.get(type1, {}).get(type2, 0)
    
    if np.any(compatibility_matrix > 0):
        sns.heatmap(compatibility_matrix, annot=True, fmt='g', 
                    xticklabels=edge_types, yticklabels=edge_types,
                    cmap='YlOrRd', ax=ax3)
        ax3.set_title('Edge Type Match Frequency', fontsize=14)
    
    # 4. Confidence vs Score scatter
    ax4 = fig.add_subplot(gs[1, :])
    confidences = []
    scores = []
    colors = []
    
    for matches_dict in registry.matches.values():
        for match in matches_dict.values():
            confidences.append(match.confidence)
            scores.append(match.similarity_score)
            colors.append(_get_match_quality_color(match.similarity_score))
    
    if confidences and scores:
        scatter = ax4.scatter(scores, confidences, c=colors, alpha=0.6, s=50)
        ax4.set_xlabel('Similarity Score', fontsize=12)
        ax4.set_ylabel('Confidence', fontsize=12)
        ax4.set_title('Confidence vs Similarity Score', fontsize=14)
        ax4.grid(True, alpha=0.3)
        
        # Add regions
        ax4.axvspan(0.9, 1.0, alpha=0.1, color='green', label='Perfect')
        ax4.axvspan(0.7, 0.9, alpha=0.1, color='yellow', label='Good')
        ax4.axvspan(0, 0.7, alpha=0.1, color='red', label='Possible')
        ax4.legend()
    
    # 5. Problematic matches
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')
    
    problematic_text = "Problematic Matches (Low Confidence or Failed Validation):\n\n"
    problem_count = 0
    
    for (p1, e1), matches_dict in registry.matches.items():
        for (p2, e2), match in matches_dict.items():
            if match.confidence < 0.5 or not match.is_valid():
                problematic_text += f"• P{p1}:E{e1} ↔ P{p2}:E{e2} "
                problematic_text += f"(Score: {match.similarity_score:.3f}, "
                problematic_text += f"Conf: {match.confidence:.2f})\n"
                problem_count += 1
                if problem_count >= 10:  # Limit display
                    break
        if problem_count >= 10:
            break
    
    if problem_count == 0:
        problematic_text += "No problematic matches found!"
    
    ax5.text(0.05, 0.95, problematic_text, transform=ax5.transAxes,
             fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # 6. Match statistics summary
    ax6 = fig.add_subplot(gs[3, :])
    ax6.axis('off')
    
    total_matches = sum(len(m) for m in registry.matches.values())
    confirmed_matches = len(registry.confirmed_matches) // 2  # Bidirectional
    
    stats_text = f"Match Statistics Summary:\n\n"
    stats_text += f"• Total Matches Evaluated: {total_matches}\n"
    stats_text += f"• Confirmed Matches: {confirmed_matches}\n"
    stats_text += f"• Average Score: {np.mean(all_scores):.3f}\n" if all_scores else ""
    stats_text += f"• Score Std Dev: {np.std(all_scores):.3f}\n" if all_scores else ""
    stats_text += f"\nSpatial Index Statistics:\n"
    
    index_stats = spatial_index.get_statistics()
    stats_text += f"• Total Indexed Edges: {index_stats.get('total_edges', 0)}\n"
    stats_text += f"• Edge Types: {index_stats.get('edge_types', {})}\n"
    
    ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes,
             fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'match_validation_dashboard.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return output_path


def create_match_confidence_report(pieces: List[Piece], registry: GlobalMatchRegistry,
                                  output_dir: str) -> str:
    """Create detailed HTML report of match confidence.
    
    Args:
        pieces: List of puzzle pieces
        registry: Global match registry
        output_dir: Output directory
        
    Returns:
        Path to saved HTML report
    """
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, 'match_confidence_report.html')
    
    # Gather statistics
    piece_stats = {}
    orphan_edges = []
    low_confidence_matches = []
    
    for piece in pieces:
        piece_stats[piece.index] = {
            'total_edges': len(piece.edges),
            'matched_edges': 0,
            'avg_score': 0,
            'best_matches': []
        }
        
        scores = []
        for edge_idx, edge in enumerate(piece.edges):
            matches = registry.get_best_matches(piece.index, edge_idx, n=3)
            
            if matches:
                best_match = matches[0][1]
                scores.append(best_match.similarity_score)
                piece_stats[piece.index]['matched_edges'] += 1
                piece_stats[piece.index]['best_matches'].append({
                    'edge_idx': edge_idx,
                    'target': matches[0][0],
                    'score': best_match.similarity_score,
                    'confidence': best_match.confidence
                })
                
                if best_match.confidence < 0.5:
                    low_confidence_matches.append({
                        'source': (piece.index, edge_idx),
                        'target': matches[0][0],
                        'match': best_match
                    })
            else:
                orphan_edges.append((piece.index, edge_idx))
        
        if scores:
            piece_stats[piece.index]['avg_score'] = np.mean(scores)
    
    # Generate HTML
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Edge Match Confidence Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
            .header {{ background-color: #333; color: white; padding: 20px; text-align: center; }}
            .section {{ background-color: white; margin: 20px 0; padding: 20px; border-radius: 8px; 
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
                          gap: 15px; margin: 20px 0; }}
            .stat-box {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; 
                        text-align: center; border: 1px solid #dee2e6; }}
            .stat-value {{ font-size: 2em; font-weight: bold; color: #333; }}
            .stat-label {{ color: #666; margin-top: 5px; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
            th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #f8f9fa; font-weight: bold; }}
            tr:hover {{ background-color: #f5f5f5; }}
            .good {{ color: #28a745; }}
            .warning {{ color: #ffc107; }}
            .danger {{ color: #dc3545; }}
            .piece-card {{ border: 1px solid #ddd; padding: 15px; margin: 10px 0; 
                          border-radius: 5px; background-color: #fafafa; }}
            .progress-bar {{ width: 100%; height: 20px; background-color: #e0e0e0; 
                           border-radius: 10px; overflow: hidden; }}
            .progress-fill {{ height: 100%; background-color: #4CAF50; transition: width 0.3s; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Edge Match Confidence Report</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="section">
            <h2>Summary Statistics</h2>
            <div class="stats-grid">
                <div class="stat-box">
                    <div class="stat-value">{len(pieces)}</div>
                    <div class="stat-label">Total Pieces</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{sum(len(p.edges) for p in pieces)}</div>
                    <div class="stat-label">Total Edges</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{len(registry.confirmed_matches) // 2}</div>
                    <div class="stat-label">Confirmed Matches</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{len(orphan_edges)}</div>
                    <div class="stat-label">Orphan Edges</div>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>Low Confidence Matches</h2>
            <table>
                <tr>
                    <th>Source</th>
                    <th>Target</th>
                    <th>Score</th>
                    <th>Confidence</th>
                    <th>Type</th>
                </tr>
    """
    
    for item in low_confidence_matches[:20]:  # Limit to top 20
        source = f"P{item['source'][0]}:E{item['source'][1]}"
        target = f"P{item['target'][0]}:E{item['target'][1]}"
        match = item['match']
        
        score_class = 'good' if match.similarity_score > 0.8 else 'warning' if match.similarity_score > 0.6 else 'danger'
        conf_class = 'danger'  # Low confidence
        
        html_content += f"""
                <tr>
                    <td>{source}</td>
                    <td>{target}</td>
                    <td class="{score_class}">{match.similarity_score:.3f}</td>
                    <td class="{conf_class}">{match.confidence:.3f}</td>
                    <td>{match.match_type}</td>
                </tr>
        """
    
    html_content += """
            </table>
        </div>
        
        <div class="section">
            <h2>Orphan Edges</h2>
            <p>Edges with no suitable matches found:</p>
            <div style="column-count: 3; column-gap: 20px;">
    """
    
    for piece_idx, edge_idx in orphan_edges:
        html_content += f"<div>• Piece {piece_idx}, Edge {edge_idx}</div>"
    
    html_content += """
            </div>
        </div>
        
        <div class="section">
            <h2>Per-Piece Analysis</h2>
    """
    
    for piece_idx, stats in piece_stats.items():
        match_rate = stats['matched_edges'] / stats['total_edges'] if stats['total_edges'] > 0 else 0
        
        html_content += f"""
            <div class="piece-card">
                <h3>Piece {piece_idx}</h3>
                <div class="stats-grid">
                    <div>
                        <strong>Edges:</strong> {stats['total_edges']}<br>
                        <strong>Matched:</strong> {stats['matched_edges']}
                    </div>
                    <div>
                        <strong>Match Rate:</strong> {match_rate:.1%}<br>
                        <strong>Avg Score:</strong> {stats['avg_score']:.3f}
                    </div>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {match_rate * 100}%"></div>
                </div>
            </div>
        """
    
    html_content += """
        </div>
    </body>
    </html>
    """
    
    with open(report_path, 'w') as f:
        f.write(html_content)
    
    return report_path


def create_interactive_match_explorer(pieces: List[Piece], registry: GlobalMatchRegistry,
                                     output_dir: str) -> str:
    """Create interactive HTML explorer for match inspection.
    
    Args:
        pieces: List of puzzle pieces
        registry: Global match registry
        output_dir: Output directory
        
    Returns:
        Path to saved HTML file
    """
    os.makedirs(output_dir, exist_ok=True)
    explorer_path = os.path.join(output_dir, 'match_explorer.html')
    
    # Prepare match data for JavaScript
    match_data = {}
    for piece in pieces:
        match_data[piece.index] = {}
        for edge_idx in range(len(piece.edges)):
            matches = registry.get_best_matches(piece.index, edge_idx, n=10)
            match_data[piece.index][edge_idx] = [
                {
                    'target_piece': target[0],
                    'target_edge': target[1],
                    'score': match.similarity_score,
                    'shape_score': match.shape_score,
                    'color_score': match.color_score,
                    'confidence': match.confidence,
                    'type': match.match_type,
                    'is_confirmed': (piece.index, edge_idx, target[0], target[1]) in registry.confirmed_matches
                }
                for target, match in matches
            ]
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Interactive Match Explorer</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            .header {{ background-color: #333; color: white; padding: 20px; text-align: center; 
                      border-radius: 8px 8px 0 0; }}
            .controls {{ background-color: white; padding: 20px; border: 1px solid #ddd; 
                        margin-bottom: 20px; border-radius: 0 0 8px 8px; }}
            .piece-selector, .edge-selector {{ padding: 8px; margin: 0 10px; font-size: 16px; }}
            .threshold-slider {{ width: 300px; margin: 0 10px; }}
            .match-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); 
                          gap: 15px; }}
            .match-card {{ background-color: white; border: 2px solid #ddd; border-radius: 8px; 
                          padding: 15px; transition: all 0.3s; cursor: pointer; }}
            .match-card:hover {{ transform: translateY(-2px); box-shadow: 0 4px 8px rgba(0,0,0,0.1); }}
            .match-card.confirmed {{ border-color: #28a745; background-color: #f8fff9; }}
            .match-card.good {{ border-color: #ffc107; }}
            .match-card.poor {{ border-color: #dc3545; opacity: 0.7; }}
            .score-bar {{ height: 10px; background-color: #e0e0e0; border-radius: 5px; 
                         margin: 5px 0; overflow: hidden; }}
            .score-fill {{ height: 100%; transition: width 0.3s; }}
            .score-fill.shape {{ background-color: #2196F3; }}
            .score-fill.color {{ background-color: #FF9800; }}
            .score-fill.total {{ background-color: #4CAF50; }}
            .details {{ font-size: 14px; color: #666; margin-top: 10px; }}
            .hidden {{ display: none; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Interactive Edge Match Explorer</h1>
                <p>Click on pieces and edges to explore matches</p>
            </div>
            
            <div class="controls">
                <label>Piece: 
                    <select id="pieceSelector" class="piece-selector">
                        {' '.join(f'<option value="{p.index}">Piece {p.index}</option>' for p in pieces)}
                    </select>
                </label>
                
                <label>Edge: 
                    <select id="edgeSelector" class="edge-selector">
                        <option value="0">Edge 0</option>
                        <option value="1">Edge 1</option>
                        <option value="2">Edge 2</option>
                        <option value="3">Edge 3</option>
                    </select>
                </label>
                
                <label>Min Score: 
                    <input type="range" id="thresholdSlider" class="threshold-slider" 
                           min="0" max="1" step="0.05" value="0.5">
                    <span id="thresholdValue">0.50</span>
                </label>
                
                <button onclick="exportMatches()">Export Selections</button>
            </div>
            
            <div id="matchGrid" class="match-grid"></div>
        </div>
        
        <script>
            const matchData = {json.dumps(match_data)};
            let selectedMatches = new Set();
            
            function updateDisplay() {{
                const pieceIdx = parseInt(document.getElementById('pieceSelector').value);
                const edgeIdx = parseInt(document.getElementById('edgeSelector').value);
                const threshold = parseFloat(document.getElementById('thresholdSlider').value);
                
                const matches = matchData[pieceIdx]?.[edgeIdx] || [];
                const grid = document.getElementById('matchGrid');
                grid.innerHTML = '';
                
                matches.forEach((match, idx) => {{
                    if (match.score < threshold) return;
                    
                    const card = document.createElement('div');
                    card.className = 'match-card';
                    
                    if (match.is_confirmed) {{
                        card.className += ' confirmed';
                    }} else if (match.score >= 0.8) {{
                        card.className += ' good';
                    }} else if (match.score < 0.6) {{
                        card.className += ' poor';
                    }}
                    
                    const matchKey = `${{pieceIdx}}-${{edgeIdx}}-${{match.target_piece}}-${{match.target_edge}}`;
                    if (selectedMatches.has(matchKey)) {{
                        card.style.backgroundColor = '#e3f2fd';
                    }}
                    
                    card.innerHTML = `
                        <h3>Target: Piece ${{match.target_piece}}, Edge ${{match.target_edge}}</h3>
                        <div class="details">
                            <strong>Type:</strong> ${{match.type}} 
                            ${{match.is_confirmed ? '✓ CONFIRMED' : ''}}
                        </div>
                        
                        <div style="margin-top: 10px;">
                            <div>Total Score: ${{match.score.toFixed(3)}}</div>
                            <div class="score-bar">
                                <div class="score-fill total" style="width: ${{match.score * 100}}%"></div>
                            </div>
                        </div>
                        
                        <div style="margin-top: 5px;">
                            <div>Shape: ${{match.shape_score.toFixed(3)}}</div>
                            <div class="score-bar">
                                <div class="score-fill shape" style="width: ${{match.shape_score * 100}}%"></div>
                            </div>
                        </div>
                        
                        <div style="margin-top: 5px;">
                            <div>Color: ${{match.color_score.toFixed(3)}}</div>
                            <div class="score-bar">
                                <div class="score-fill color" style="width: ${{match.color_score * 100}}%"></div>
                            </div>
                        </div>
                        
                        <div class="details">
                            <strong>Confidence:</strong> ${{match.confidence.toFixed(3)}}
                        </div>
                    `;
                    
                    card.onclick = () => {{
                        if (selectedMatches.has(matchKey)) {{
                            selectedMatches.delete(matchKey);
                            card.style.backgroundColor = '';
                        }} else {{
                            selectedMatches.add(matchKey);
                            card.style.backgroundColor = '#e3f2fd';
                        }}
                    }};
                    
                    grid.appendChild(card);
                }});
            }}
            
            document.getElementById('pieceSelector').onchange = updateDisplay;
            document.getElementById('edgeSelector').onchange = updateDisplay;
            document.getElementById('thresholdSlider').oninput = function() {{
                document.getElementById('thresholdValue').textContent = this.value;
                updateDisplay();
            }};
            
            function exportMatches() {{
                const data = Array.from(selectedMatches).map(key => {{
                    const [p1, e1, p2, e2] = key.split('-').map(Number);
                    return {{ piece1: p1, edge1: e1, piece2: p2, edge2: e2 }};
                }});
                
                const blob = new Blob([JSON.stringify(data, null, 2)], {{type: 'application/json'}});
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'selected_matches.json';
                a.click();
            }}
            
            // Initial display
            updateDisplay();
        </script>
    </body>
    </html>
    """
    
    with open(explorer_path, 'w') as f:
        f.write(html_content)
    
    return explorer_path


def create_color_continuity_visualization(piece1: Piece, edge1_idx: int,
                                         piece2: Piece, edge2_idx: int,
                                         output_dir: str) -> str:
    """Visualize color continuity across matched edges.
    
    Args:
        piece1: First puzzle piece
        edge1_idx: Edge index on first piece
        piece2: Second puzzle piece
        edge2_idx: Edge index on second piece
        output_dir: Output directory
        
    Returns:
        Path to saved visualization
    """
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    edge1 = piece1.get_edge(edge1_idx)
    edge2 = piece2.get_edge(edge2_idx)
    
    if not edge1 or not edge2:
        plt.close(fig)
        return ""
    
    fig.suptitle(f'Color Continuity: P{piece1.index}:E{edge1_idx} ↔ P{piece2.index}:E{edge2_idx}',
                 fontsize=16, fontweight='bold')
    
    # 1. Edge color strips
    ax1 = fig.add_subplot(gs[0, :])
    if edge1.color_sequence and edge2.color_sequence:
        # Create color strips
        n_colors = max(len(edge1.color_sequence), len(edge2.color_sequence))
        
        # Resample to same length
        from ..features.edge_extraction import resample_sequence
        colors1 = resample_sequence(edge1.color_sequence, n_colors)
        colors2 = resample_sequence(edge2.color_sequence, n_colors)
        
        # Convert LAB to RGB for display
        rgb_colors1 = []
        rgb_colors2 = []
        
        for lab_color in colors1:
            lab_pixel = np.array(lab_color, dtype=np.uint8).reshape(1, 1, 3)
            rgb_pixel = cv2.cvtColor(lab_pixel, cv2.COLOR_LAB2RGB)
            rgb_colors1.append(rgb_pixel[0, 0])
        
        for lab_color in colors2:
            lab_pixel = np.array(lab_color, dtype=np.uint8).reshape(1, 1, 3)
            rgb_pixel = cv2.cvtColor(lab_pixel, cv2.COLOR_LAB2RGB)
            rgb_colors2.append(rgb_pixel[0, 0])
        
        # Display color strips
        strip1 = np.array(rgb_colors1).reshape(1, -1, 3) / 255.0
        strip2 = np.array(rgb_colors2).reshape(1, -1, 3) / 255.0
        
        combined_strip = np.vstack([strip1] * 20 + [np.ones((5, n_colors, 3))] + [strip2] * 20)
        
        ax1.imshow(combined_strip, aspect='auto')
        ax1.set_title('Color Sequences (Edge 1 top, Edge 2 bottom)', fontsize=14)
        ax1.axis('off')
    
    # 2. Color difference heatmap
    ax2 = fig.add_subplot(gs[1, :2])
    if edge1.color_sequence and edge2.color_sequence:
        # Calculate color distances
        distances = []
        for i, c1 in enumerate(colors1):
            row = []
            for j, c2 in enumerate(colors2):
                dist = color_distance(np.array(c1), np.array(c2))
                row.append(dist)
            distances.append(row)
        
        distances = np.array(distances)
        
        im = ax2.imshow(distances, cmap='RdYlGn_r', aspect='auto')
        ax2.set_xlabel('Edge 2 Position', fontsize=12)
        ax2.set_ylabel('Edge 1 Position', fontsize=12)
        ax2.set_title('Color Distance Matrix', fontsize=14)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax2)
        cbar.set_label('LAB Distance', fontsize=12)
    
    # 3. Color gradient analysis
    ax3 = fig.add_subplot(gs[1, 2])
    if edge1.color_sequence and edge2.color_sequence:
        # Calculate gradients
        grad1 = np.diff([c[0] for c in colors1])  # L channel gradient
        grad2 = np.diff([c[0] for c in colors2])
        
        x = np.linspace(0, 1, len(grad1))
        ax3.plot(x, grad1, 'b-', label='Edge 1', linewidth=2)
        ax3.plot(x, grad2, 'r-', label='Edge 2', linewidth=2)
        ax3.set_xlabel('Position', fontsize=12)
        ax3.set_ylabel('L* Gradient', fontsize=12)
        ax3.set_title('Lightness Gradient', fontsize=14)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # 4. LAB color space trajectory
    ax4 = fig.add_subplot(gs[2, :], projection='3d')
    if edge1.color_sequence and edge2.color_sequence:
        # Plot in LAB space
        l1 = [c[0] for c in colors1]
        a1 = [c[1] for c in colors1]
        b1 = [c[2] for c in colors1]
        
        l2 = [c[0] for c in colors2]
        a2 = [c[1] for c in colors2]
        b2 = [c[2] for c in colors2]
        
        ax4.plot(l1, a1, b1, 'b-', linewidth=3, label='Edge 1', alpha=0.8)
        ax4.plot(l2, a2, b2, 'r-', linewidth=3, label='Edge 2', alpha=0.8)
        
        # Mark start and end points
        ax4.scatter(l1[0], a1[0], b1[0], c='blue', s=100, marker='o', label='Edge 1 Start')
        ax4.scatter(l1[-1], a1[-1], b1[-1], c='blue', s=100, marker='s')
        ax4.scatter(l2[0], a2[0], b2[0], c='red', s=100, marker='o', label='Edge 2 Start')
        ax4.scatter(l2[-1], a2[-1], b2[-1], c='red', s=100, marker='s')
        
        ax4.set_xlabel('L*', fontsize=12)
        ax4.set_ylabel('a*', fontsize=12)
        ax4.set_zlabel('b*', fontsize=12)
        ax4.set_title('Color Trajectory in LAB Space', fontsize=14)
        ax4.legend()
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 
                              f'color_continuity_p{piece1.index}e{edge1_idx}_p{piece2.index}e{edge2_idx}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return output_path


def create_shape_compatibility_analysis(matches: List[Tuple[Tuple[int, int, int, int], EdgeMatch]],
                                       output_dir: str) -> str:
    """Analyze shape compatibility patterns across all matches.
    
    Args:
        matches: List of ((p1, e1, p2, e2), EdgeMatch) tuples
        output_dir: Output directory
        
    Returns:
        Path to saved visualization
    """
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    fig.suptitle('Shape Compatibility Analysis', fontsize=18, fontweight='bold')
    
    # Extract data
    shape_scores = []
    color_scores = []
    total_scores = []
    match_types = []
    confidences = []
    
    for (p1, e1, p2, e2), match in matches:
        shape_scores.append(match.shape_score)
        color_scores.append(match.color_score)
        total_scores.append(match.similarity_score)
        match_types.append(match.match_type)
        confidences.append(match.confidence)
    
    # 1. Shape vs Color scatter
    ax1 = fig.add_subplot(gs[0, 0])
    colors_map = {'perfect': 'green', 'good': 'yellow', 'possible': 'red'}
    colors = [colors_map.get(mt, 'gray') for mt in match_types]
    
    scatter = ax1.scatter(shape_scores, color_scores, c=colors, alpha=0.6, s=50)
    ax1.set_xlabel('Shape Score', fontsize=12)
    ax1.set_ylabel('Color Score', fontsize=12)
    ax1.set_title('Shape vs Color Scores', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Add diagonal line
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    
    # Add legend
    for match_type, color in colors_map.items():
        ax1.scatter([], [], c=color, label=match_type.title(), s=50)
    ax1.legend()
    
    # 2. Score distributions
    ax2 = fig.add_subplot(gs[0, 1])
    data = [shape_scores, color_scores, total_scores]
    labels = ['Shape', 'Color', 'Total']
    
    bp = ax2.boxplot(data, labels=labels, patch_artist=True)
    colors = ['lightblue', 'lightcoral', 'lightgreen']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax2.set_ylabel('Score', fontsize=12)
    ax2.set_title('Score Distributions', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Match type distribution by score range
    ax3 = fig.add_subplot(gs[0, 2])
    score_bins = [(0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0)]
    bin_labels = ['<0.5', '0.5-0.7', '0.7-0.9', '0.9-1.0']
    
    type_counts = {bt: {mt: 0 for mt in ['perfect', 'good', 'possible']} 
                   for bt in bin_labels}
    
    for score, mtype in zip(total_scores, match_types):
        for i, (low, high) in enumerate(score_bins):
            if low <= score < high:
                type_counts[bin_labels[i]][mtype] += 1
                break
    
    # Stacked bar chart
    x = np.arange(len(bin_labels))
    width = 0.6
    
    bottom = np.zeros(len(bin_labels))
    for mtype, color in colors_map.items():
        counts = [type_counts[bl][mtype] for bl in bin_labels]
        ax3.bar(x, counts, width, bottom=bottom, label=mtype.title(), color=color)
        bottom += counts
    
    ax3.set_xlabel('Score Range', fontsize=12)
    ax3.set_ylabel('Count', fontsize=12)
    ax3.set_title('Match Types by Score Range', fontsize=14)
    ax3.set_xticks(x)
    ax3.set_xticklabels(bin_labels)
    ax3.legend()
    
    # 4. Correlation heatmap
    ax4 = fig.add_subplot(gs[1, 0])
    correlation_data = np.array([shape_scores, color_scores, total_scores, confidences]).T
    correlation_matrix = np.corrcoef(correlation_data.T)
    
    im = ax4.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    ax4.set_xticks(range(4))
    ax4.set_yticks(range(4))
    ax4.set_xticklabels(['Shape', 'Color', 'Total', 'Confidence'])
    ax4.set_yticklabels(['Shape', 'Color', 'Total', 'Confidence'])
    
    # Add correlation values
    for i in range(4):
        for j in range(4):
            text = ax4.text(j, i, f'{correlation_matrix[i, j]:.2f}',
                           ha='center', va='center', color='black' if abs(correlation_matrix[i, j]) < 0.5 else 'white')
    
    ax4.set_title('Score Correlations', fontsize=14)
    plt.colorbar(im, ax=ax4)
    
    # 5. Outlier analysis
    ax5 = fig.add_subplot(gs[1, 1:])
    
    # Calculate z-scores for outlier detection
    from scipy import stats
    z_scores = np.abs(stats.zscore(total_scores))
    outliers = [(i, score) for i, (score, z) in enumerate(zip(total_scores, z_scores)) if z > 2]
    
    ax5.scatter(range(len(total_scores)), total_scores, alpha=0.5, s=30, c='blue')
    
    if outliers:
        outlier_indices, outlier_scores = zip(*outliers)
        ax5.scatter(outlier_indices, outlier_scores, c='red', s=100, 
                   label=f'Outliers (n={len(outliers)})', edgecolors='black', linewidth=2)
    
    ax5.set_xlabel('Match Index', fontsize=12)
    ax5.set_ylabel('Total Score', fontsize=12)
    ax5.set_title('Score Distribution with Outliers', fontsize=14)
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    
    # Add statistics text
    stats_text = f"Total Matches: {len(matches)}\n"
    stats_text += f"Mean Score: {np.mean(total_scores):.3f}\n"
    stats_text += f"Std Dev: {np.std(total_scores):.3f}\n"
    stats_text += f"Outliers: {len(outliers) if outliers else 0}"
    
    ax5.text(0.02, 0.98, stats_text, transform=ax5.transAxes,
             fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'shape_compatibility_analysis.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return output_path


# Helper functions
def _extract_edge_region(image: np.ndarray, edge_points: List[Tuple[int, int]], 
                        padding: int = 20) -> np.ndarray:
    """Extract region around edge points."""
    if not edge_points:
        return np.zeros((100, 100, 3), dtype=np.uint8)
    
    points = np.array(edge_points)
    x_min, y_min = points.min(axis=0) - padding
    x_max, y_max = points.max(axis=0) + padding
    
    # Ensure bounds are within image
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(image.shape[1], x_max)
    y_max = min(image.shape[0], y_max)
    
    return image[y_min:y_max, x_min:x_max].copy()


def _plot_color_sequences(ax, seq1: List[np.ndarray], seq2: List[np.ndarray]):
    """Plot color sequences as strips."""
    n_colors = max(len(seq1), len(seq2))
    
    # Resample to same length
    from ..features.edge_extraction import resample_sequence
    colors1 = resample_sequence(seq1, n_colors)
    colors2 = resample_sequence(seq2, n_colors)
    
    # Convert LAB to RGB for display
    rgb_colors1 = []
    rgb_colors2 = []
    
    for lab_color in colors1:
        lab_pixel = np.array(lab_color, dtype=np.uint8).reshape(1, 1, 3)
        rgb_pixel = cv2.cvtColor(lab_pixel, cv2.COLOR_LAB2RGB)
        rgb_colors1.append(rgb_pixel[0, 0] / 255.0)
    
    for lab_color in colors2:
        lab_pixel = np.array(lab_color, dtype=np.uint8).reshape(1, 1, 3)
        rgb_pixel = cv2.cvtColor(lab_pixel, cv2.COLOR_LAB2RGB)
        rgb_colors2.append(rgb_pixel[0, 0] / 255.0)
    
    # Create color strips
    for i, (c1, c2) in enumerate(zip(rgb_colors1, rgb_colors2)):
        ax.add_patch(patches.Rectangle((i, 0), 1, 1, facecolor=c1))
        ax.add_patch(patches.Rectangle((i, 1.2), 1, 1, facecolor=c2))
    
    ax.set_xlim(0, n_colors)
    ax.set_ylim(0, 2.2)
    ax.set_yticks([0.5, 1.7])
    ax.set_yticklabels(['Edge 1', 'Edge 2'])
    ax.set_xlabel('Position along edge')
    ax.set_aspect('equal')


def _get_match_quality_color(score: float) -> str:
    """Get color based on match score."""
    if score >= 0.9:
        return 'green'
    elif score >= 0.7:
        return 'yellow'
    else:
        return 'red'


def _are_types_compatible(type1: str, type2: str) -> bool:
    """Check if two edge types are compatible for matching."""
    if type1 == 'flat' or type2 == 'flat':
        return False
    if type1 == 'convex' and type2 == 'concave':
        return True
    if type1 == 'concave' and type2 == 'convex':
        return True
    return False