import cv2
import numpy as np
import os
import argparse
from image_processing import read_image, save_image
from pieces_cutting import PieceProcessor, Visualization, Config
from contours_metrics import ContourMetrics  # Fixed import name
from puzzle_piece import Piece  # This import should work once puzzle_piece.py is fixed


# Define color codes for terminal output
class Colors:
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    PURPLE = "\033[95m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def print_info(message):
    print(f"{Colors.BLUE}ℹ️  {message}{Colors.RESET}")


def print_success(message):
    print(f"{Colors.GREEN}✅{message}{Colors.RESET}")


def print_warning(message):
    print(f"{Colors.YELLOW}⚠️  {message}{Colors.RESET}")


def print_error(message):
    print(f"{Colors.RED}❌  {message}{Colors.RESET}")


def print_header(message):
    print(f"{Colors.PURPLE}{Colors.BOLD}🔷  {message}{Colors.RESET}")


def main():
    # Set up command line arguments
    parser = argparse.ArgumentParser(description="Puzzle piece analyzer")
    parser.add_argument(
        "--image",
        type=str,
        default="picture/puzzle_24-1/b-2.jpg",
        help="Path to the puzzle image",
    )
    parser.add_argument(
        "--pieces", type=int, default=24, help="Expected number of pieces in the puzzle"
    )
    parser.add_argument(
        "--debug-dir", type=str, default="debug", help="Directory to save debug outputs"
    )

    args = parser.parse_args()

    # Initialize configuration
    config = Config()

    # Setup directories
    os.makedirs(args.debug_dir, exist_ok=True)
    for f in os.listdir(args.debug_dir):
        if f.endswith(".jpg"):
            os.remove(f"{args.debug_dir}/{f}")

    # Load image
    print_info(f"Loading image from {args.image}")
    input_image = read_image(args.image)
    if input_image is None:
        print_error(f"Error: Failed to load input image from {args.image}")
        return []

    # Debug the threshold selection
    try:
        optimal_threshold = PieceProcessor.debug_adaptive_threshold(
            input_image, f"{args.debug_dir}/threshold_analysis.jpg"
        )
        print_info(f"Selected optimal threshold value: {optimal_threshold}")
    except Exception as e:
        print_warning(f"Threshold debugging failed: {str(e)}")
        optimal_threshold = 120  # Fallback value

    # Find contours with improved method
    print_info("Detecting puzzle piece contours...")
    contours = PieceProcessor.find_contour(input_image)
    if not contours:
        print_error("No contours detected. Check the image contrast.")
        return []

    print_info(f"Found {len(contours)} initial contours")

    # Refine contours for better accuracy
    print_info("Refining contours...")
    refined_contours = PieceProcessor.refine_contours(contours, input_image)
    print_info(f"After refinement: {len(refined_contours)} contours")

    # Calculate metrics - update expected_count with actual piece count
    print_info("Calculating metrics...")
    metrics = ContourMetrics.calculate_metrics(
        refined_contours, input_image, expected_count=args.pieces
    )

    # Visualize metrics
    metrics_vis_path = f"{args.debug_dir}/metrics_visualization.jpg"
    ContourMetrics.visualize_metrics(
        input_image, refined_contours, metrics, metrics_vis_path
    )

    # Generate metrics report
    metrics_report_path = f"{args.debug_dir}/metrics_report.txt"
    ContourMetrics.generate_report(metrics, metrics_report_path)

    print_info(f"Metrics calculated and saved to {metrics_report_path}")

    # Process contours to create Piece objects
    print_info("Processing individual pieces...")
    pieces = []
    for i, cnt in enumerate(refined_contours):
        try:
            piece = Piece(input_image, cnt, config)
            pieces.append(piece)
        except Exception as e:
            print_warning(f"Error processing piece #{i}: {str(e)}")

    valid_pieces = [p for p in pieces if p.is_valid]

    print_info(f"Valid pieces found: {len(valid_pieces)}/{len(pieces)}")

    # Visualize piece validation results
    contours_list = [p.contour for p in pieces]
    statuses = [p.validation_status for p in pieces]

    # NOTE: The original debug_contours function only takes 3 arguments
    # Either modify the function in pieces_cutting.py to accept a 4th parameter
    # or use the hardcoded default path
    Visualization.debug_contours(input_image, contours_list, statuses)

    # Process valid pieces
    print_header("\nProcessing puzzle pieces:")
    summary_image = input_image.copy()

    # Determine number of pieces to show in detail (up to 6)
    pieces_to_show = min(6, len(valid_pieces))

    # Calculate grid layout (2x3 or smaller if fewer pieces)
    grid_cols = min(2, pieces_to_show) if pieces_to_show > 0 else 1
    grid_rows = (
        (pieces_to_show + grid_cols - 1) // grid_cols if pieces_to_show > 0 else 1
    )

    for idx, piece in enumerate(valid_pieces[:pieces_to_show]):
        piece_num = idx + 1
        print_info(f"\n🧩 Analyzing piece #{piece_num}")

        if not piece.is_valid:
            print_error(f"  Failed to process piece #{piece_num}")
            continue

        print_success(f"  Border types: {piece.border_types}")

        # Save individual visualization
        vis_img = piece.draw(input_image)
        save_image(vis_img, f"{args.debug_dir}/piece_{idx}_borders.jpg")

        # Add to summary image if we have at least one valid piece
        if pieces_to_show > 0:
            x, y, w, h = cv2.boundingRect(piece.contour)
            padding = 20
            roi = vis_img[
                max(0, y - padding) : min(input_image.shape[0], y + h + padding),
                max(0, x - padding) : min(input_image.shape[1], x + w + padding),
            ]

            # Resize for grid layout
            cell_size = (
                summary_image.shape[1] // grid_cols,
                summary_image.shape[0] // grid_rows,
            )
            resized_roi = cv2.resize(roi, cell_size)

            # Position in grid
            row = idx // grid_cols
            col = idx % grid_cols
            summary_image[
                row * cell_size[1] : (row + 1) * cell_size[1],
                col * cell_size[0] : (col + 1) * cell_size[0],
            ] = resized_roi

    # Save summary visualization
    if pieces_to_show > 0:
        save_image(summary_image, f"{args.debug_dir}/processing_summary.jpg")
        print_info(
            f"Summary visualization saved to {args.debug_dir}/processing_summary.jpg"
        )
    else:
        print_warning("No valid pieces to visualize")

    return valid_pieces


if __name__ == "__main__":
    valid = main()
    print_success(f"\n🎉 Analysis complete. Valid pieces: {len(valid)}")
