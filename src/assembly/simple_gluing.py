"""Simple gluing that doesn't require edge data."""

import numpy as np
import cv2
from typing import Dict, Tuple, List
import os

from ..core.piece import Piece
from ..features.edge_matching_rotation_aware import PuzzleAssembly


def create_glued_puzzle(assembly: PuzzleAssembly, pieces: List[Piece], output_path: str) -> np.ndarray:
    """Create final glued puzzle with tight connections."""
    if not assembly.grid:
        print("Error: Empty assembly grid")
        return None
    
    print(f"Creating glued puzzle from {len(assembly.grid)} pieces...")
    
    # Calculate actual piece dimensions
    row_heights = {}
    col_widths = {}
    
    for (row, col), (piece, rotation) in assembly.grid.items():
        h, w = piece.image.shape[:2]
        if rotation in [90, 270]:
            h, w = w, h
        row_heights[row] = max(row_heights.get(row, 0), h)
        col_widths[col] = max(col_widths.get(col, 0), w)
    
    # Calculate positions with overlap
    overlap = -20  # Negative for tighter connection
    
    x_positions = {0: 0}
    for col in range(1, assembly.cols):
        x_positions[col] = x_positions[col-1] + col_widths.get(col-1, 0) + overlap
    
    y_positions = {0: 0}
    for row in range(1, assembly.rows):
        y_positions[row] = y_positions[row-1] + row_heights.get(row-1, 0) + overlap
    
    # Create canvas
    padding = 50
    total_width = x_positions[assembly.cols-1] + col_widths.get(assembly.cols-1, 0)
    total_height = y_positions[assembly.rows-1] + row_heights.get(assembly.rows-1, 0)
    
    canvas_width = total_width + 2 * padding
    canvas_height = total_height + 2 * padding
    
    # Use accumulation for smooth blending
    accumulator = np.zeros((canvas_height, canvas_width, 3), dtype=np.float32)
    weight_map = np.zeros((canvas_height, canvas_width), dtype=np.float32)
    
    # Place pieces
    for (row, col), (piece, rotation) in assembly.grid.items():
        # Get rotated piece
        piece_img = piece.image.copy()
        piece_mask = piece.mask.copy()
        
        if rotation > 0:
            h, w = piece_img.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, -rotation, 1.0)
            
            if rotation in [90, 270]:
                M[0, 2] += (h - w) / 2
                M[1, 2] += (w - h) / 2
                new_size = (h, w)
            else:
                new_size = (w, h)
            
            piece_img = cv2.warpAffine(piece_img, M, new_size)
            piece_mask = cv2.warpAffine(piece_mask, M, new_size)
        
        # Position
        x_start = padding + x_positions[col]
        y_start = padding + y_positions[row]
        
        # Center in allocated space
        ph, pw = piece_img.shape[:2]
        x_offset = (col_widths[col] - pw) // 2
        y_offset = (row_heights[row] - ph) // 2
        x_start += max(0, x_offset)
        y_start += max(0, y_offset)
        
        x_end = min(x_start + pw, canvas_width)
        y_end = min(y_start + ph, canvas_height)
        
        if x_end <= x_start or y_end <= y_start:
            continue
        
        # Extract valid region
        valid_w = x_end - x_start
        valid_h = y_end - y_start
        piece_region = piece_img[:valid_h, :valid_w]
        mask_region = piece_mask[:valid_h, :valid_w].astype(np.float32) / 255.0
        
        # Feather edges
        kernel_size = 5
        mask_region = cv2.GaussianBlur(mask_region, (kernel_size, kernel_size), 2)
        
        # Accumulate
        accumulator[y_start:y_end, x_start:x_end] += piece_region * mask_region[:, :, np.newaxis]
        weight_map[y_start:y_end, x_start:x_end] += mask_region
    
    # Normalize
    weight_3ch = np.maximum(weight_map[:, :, np.newaxis], 1e-6)
    canvas = (accumulator / weight_3ch).astype(np.uint8)
    
    # Fill gaps
    valid_mask = (weight_map > 0.1).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    dilated = cv2.dilate(valid_mask, kernel, iterations=1)
    gaps = cv2.bitwise_and(dilated, cv2.bitwise_not(valid_mask))
    
    if np.any(gaps):
        canvas = cv2.inpaint(canvas, gaps, 3, cv2.INPAINT_TELEA)
    
    # Auto-crop
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        x_min = y_min = float('inf')
        x_max = y_max = 0
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x + w)
            y_max = max(y_max, y + h)
        
        margin = 20
        x_min = max(0, x_min - margin)
        y_min = max(0, y_min - margin)
        x_max = min(canvas_width, x_max + margin)
        y_max = min(canvas_height, y_max + margin)
        
        canvas = canvas[y_min:y_max, x_min:x_max]
    
    # Save
    cv2.imwrite(output_path, canvas)
    
    # Create comparison
    comparison_path = output_path.replace('final_puzzle.png', 'gluing_comparison.png')
    create_comparison(assembly, pieces, canvas, comparison_path)
    
    return canvas


def create_comparison(assembly: PuzzleAssembly, pieces: List[Piece], 
                     final_image: np.ndarray, output_path: str):
    """Create before/after comparison."""
    # Create grid view
    cell_size = 200
    grid_h = assembly.rows * cell_size
    grid_w = assembly.cols * cell_size
    grid_canvas = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 240
    
    for (row, col), (piece, rotation) in assembly.grid.items():
        piece_img = piece.image.copy()
        
        if rotation > 0:
            h, w = piece_img.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, -rotation, 1.0)
            if rotation in [90, 270]:
                new_size = (h, w)
            else:
                new_size = (w, h)
            piece_img = cv2.warpAffine(piece_img, M, new_size)
        
        # Resize to fit cell
        h, w = piece_img.shape[:2]
        scale = min(cell_size/w, cell_size/h) * 0.9
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(piece_img, (new_w, new_h))
        
        # Place in grid
        y = row * cell_size + (cell_size - new_h) // 2
        x = col * cell_size + (cell_size - new_w) // 2
        grid_canvas[y:y+new_h, x:x+new_w] = resized
        
        # Add piece number
        cv2.putText(grid_canvas, f"P{piece.index}", 
                   (col * cell_size + 10, row * cell_size + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    # Resize for comparison
    target_height = 600
    scale1 = target_height / grid_canvas.shape[0]
    grid_resized = cv2.resize(grid_canvas, None, fx=scale1, fy=scale1)
    
    scale2 = target_height / final_image.shape[0]
    final_resized = cv2.resize(final_image, None, fx=scale2, fy=scale2)
    
    # Create comparison
    gap = 20
    comparison_width = grid_resized.shape[1] + final_resized.shape[1] + gap
    comparison = np.ones((target_height, comparison_width, 3), dtype=np.uint8) * 255
    
    comparison[:, :grid_resized.shape[1]] = grid_resized
    comparison[:, -final_resized.shape[1]:] = final_resized
    
    # Labels
    cv2.putText(comparison, "Assembly Layout", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(comparison, "Final Glued Puzzle", 
               (grid_resized.shape[1] + gap + 10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    cv2.imwrite(output_path, comparison)