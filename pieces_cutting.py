import cv2
import numpy as np
import os
import itertools
import math
from image_processing import save_image


class Config:
    """Centralized configuration parameters"""

    # Piece validation parameters (it's a flow in the system - should be systemized)
    MIN_AREA = 1000
    MIN_PERIMETER = 100
    CLOSED_THRESHOLD = 0.02

    # Corner detection
    CORNER_EPSILON = 0.02  # Approximation precision
    ANGLE_TOLERANCE = 15  # Degrees from 90° (±value)
    REQUIRED_CORNERS = 4  # Number of expected corners
    SIDE_TOLERANCE_RATIO = 0.4  # Allow 40% difference in opposite sides
    CONVEXITY_CHECK = True

    # Border classification
    COMPLEXITY_THRESHOLD = 1.2
    DEVIATION_THRESHOLD = 10
    SMOOTHING_KERNEL = (5, 5)  # For border smoothing


class PieceProcessor:
    @staticmethod
    def save_pieces(pieces, output_dir="pieces"):
        """
        Saves extracted puzzle pieces to specified directory.
        Creates directory if it doesn't exist.
        """
        if not pieces:
            return 0

        # Create output directory if needed
        os.makedirs(output_dir, exist_ok=True)

        valid_count = 0
        for i, piece in enumerate(pieces):
            if piece is not None:
                try:
                    path = os.path.join(output_dir, f"piece_{i}.jpg")
                    save_image(piece, path)
                    valid_count += 1
                except Exception as e:
                    print(f"Error saving piece {i}: {str(e)}")

        print(f"Saved {valid_count}/{len(pieces)} valid pieces to {output_dir}")
        return valid_count

    @staticmethod
    def find_piece(image, contour, config=Config()):
        """
        Extracts puzzle piece using contour with configurable validation thresholds.
        Returns piece image or None if invalid.
        """
        if len(contour) < 5:
            return None

        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        if area < config.MIN_AREA or perimeter < config.MIN_PERIMETER:
            return None

        compactness = 4 * np.pi * area / (perimeter**2)
        if compactness < config.CLOSED_THRESHOLD:
            return None

        # Extract piece using contour mask
        x, y, w, h = cv2.boundingRect(contour)
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], 0, 255, -1)
        return cv2.bitwise_and(
            image[y : y + h, x : x + w],
            image[y : y + h, x : x + w],
            mask=mask[y : y + h, x : x + w],
        )

    @staticmethod
    def adaptive_threshold(img):
        """Apply multiple thresholds and select the best one based on piece count stability"""
        thresholds = []
        contour_counts = []

        # Try multiple threshold values
        for thresh_val in range(50, 200, 15):
            _, binary = cv2.threshold(img, thresh_val, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(
                binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            filtered = [c for c in contours if cv2.contourArea(c) > 500]

            thresholds.append(thresh_val)
            contour_counts.append(len(filtered))

        # Find most stable region (least change in count)
        variations = [
            abs(contour_counts[i] - contour_counts[i - 1])
            for i in range(1, len(contour_counts))
        ]
        most_stable_idx = variations.index(min(variations))

        return thresholds[most_stable_idx]

    @staticmethod
    def debug_adaptive_threshold(image, output_path="debug/threshold_analysis.jpg"):
        """Visualize the adaptive threshold selection process"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(blurred)

        thresholds = []
        contour_counts = []
        images = []

        # Try multiple threshold values
        for thresh_val in range(50, 200, 15):
            _, binary = cv2.threshold(enhanced, thresh_val, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(
                binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            filtered = [c for c in contours if cv2.contourArea(c) > 500]

            # Draw contours for visualization
            vis_img = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(vis_img, filtered, -1, (0, 255, 0), 2)

            # Add text with threshold and count
            cv2.putText(
                vis_img,
                f"Thresh: {thresh_val}, Count: {len(filtered)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

            thresholds.append(thresh_val)
            contour_counts.append(len(filtered))
            images.append(vis_img)

        # Find optimal threshold
        variations = [
            abs(contour_counts[i] - contour_counts[i - 1])
            for i in range(1, len(contour_counts))
        ]
        most_stable_idx = variations.index(min(variations))
        optimal_threshold = thresholds[most_stable_idx]

        # Highlight the optimal threshold image
        cv2.rectangle(
            images[most_stable_idx],
            (0, 0),
            (images[most_stable_idx].shape[1], images[most_stable_idx].shape[0]),
            (0, 0, 255),
            3,
        )

        # Create grid visualization
        rows = 3
        cols = math.ceil(len(images) / rows)
        grid_h = rows * images[0].shape[0]
        grid_w = cols * images[0].shape[1]
        grid = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 255

        for i, img in enumerate(images):
            row = i // cols
            col = i % cols
            h, w = img.shape[:2]
            grid[row * h : (row + 1) * h, col * w : (col + 1) * w] = img

        # Add summary text
        summary = np.ones((100, grid_w, 3), dtype=np.uint8) * 255
        cv2.putText(
            summary,
            f"Optimal threshold: {optimal_threshold} (most stable piece count)",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            2,
        )

        # Combine and save
        result = np.vstack((grid, summary))
        cv2.imwrite(output_path, result)

        return optimal_threshold

    def find_contour(image):
        """
        Improved contour detection with better corner handling and validation
        """
        # Create debug directory
        os.makedirs("debug", exist_ok=True)

        # 1. Preprocessing with CLAHE for better contrast
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        blurred = cv2.GaussianBlur(enhanced, (7, 7), 0)
        cv2.imwrite("debug/01_enhanced.jpg", blurred)

        # 2. Adaptive thresholding with noise handling
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 7
        )
        cv2.imwrite("debug/02_binary.jpg", binary)

        # 3. Morphological processing
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        # Close small holes while preserving shape
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)
        cv2.imwrite("debug/03_closed.jpg", closed)

        # Remove small noise
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=2)
        cv2.imwrite("debug/04_opened.jpg", opened)

        # 4. Edge detection with hysteresis thresholds
        edges = cv2.Canny(opened, 50, 150)
        cv2.imwrite("debug/05_edges.jpg", edges)

        # 5. Contour detection with improved parameters
        contours, hierarchy = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS
        )

        # 6. Enhanced contour filtering
        filtered_contours = []
        # 0.03% of image area
        min_area = 0.0003 * (image.shape[0] * image.shape[1])
        max_area = 0.2 * (image.shape[0] * image.shape[1])

        for cnt in contours:
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)

            # Basic area filtering
            if area < min_area or area > max_area:
                continue

            # Compactness check (allow wider range for complex shapes)
            compactness = 4 * np.pi * area / (perimeter**2) if perimeter > 0 else 0
            if not (0.08 <= compactness <= 0.95):
                continue

            # Convexity check with tolerance
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            if solidity < 0.65:  # Allow more concave shapes
                continue

            filtered_contours.append(cnt)

        # 8. Validation with corner tolerance
        valid_contours = []
        for cnt in filtered_contours:
            # Get rotated rectangle as fallback
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.intp(box)

            # Check if we have at least 4 significant points
            if len(cv2.approxPolyDP(cnt, 0.02 * perimeter, True)) >= 4:
                valid_contours.append(cnt)
            elif cv2.contourArea(box) > min_area:  # Fallback to rotated rectangle
                valid_contours.append(box)

        # 9. Final visualization
        result_img = image.copy()
        for i, cnt in enumerate(valid_contours):
            color = (
                np.random.randint(0, 255),
                np.random.randint(0, 255),
                np.random.randint(0, 255),
            )
            cv2.drawContours(result_img, [cnt], -1, color, 3)

        cv2.imwrite("debug/06_final_contours.jpg", result_img)

        return valid_contours

    @staticmethod
    def watershed_contour_detection(image):
        """
        Fallback method using watershed algorithm for harder cases
        """
        import cv2
        import numpy as np

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply threshold
        _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV)

        # Noise removal with morphological operations
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

        # Sure background area
        sure_bg = cv2.dilate(opening, kernel, iterations=3)

        # Finding sure foreground area
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)

        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)

        # Marker labelling
        _, markers = cv2.connectedComponents(sure_fg)

        # Add one to all labels so that background is not 0, but 1
        markers = markers + 1

        # Mark the unknown region with zero
        markers[unknown == 255] = 0

        # Apply watershed
        cv2.watershed(image, markers)

        # Create a mask for each piece
        mask = np.zeros_like(gray)
        mask[markers > 1] = 255

        # Save the watershed mask
        cv2.imwrite("debug/watershed_mask.jpg", mask)

        # Find contours on the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter by area
        min_area = 500
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

        # Visualize watershed contours
        watershed_img = image.copy()
        for i, cnt in enumerate(filtered_contours):
            cv2.drawContours(watershed_img, [cnt], -1, (0, 255, 0), 2)

            # Add contour number
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.putText(
                    watershed_img,
                    f"{i}",
                    (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

        cv2.imwrite("debug/watershed_contours.jpg", watershed_img)

        print(f"Watershed method found {len(filtered_contours)} puzzle pieces")
        return filtered_contours

    @staticmethod
    def refine_contours(contours, image):
        """Refine detected contours using edge information"""
        refined = []
        edge_map = cv2.Canny(image, 50, 150)

        for cnt in contours:
            # Create mask from contour
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [cnt], 0, 255, 2)

            # Extract edge pixels within mask region (with dilation)
            kernel = np.ones((5, 5), np.uint8)
            dilated_mask = cv2.dilate(mask, kernel, iterations=2)
            masked_edges = cv2.bitwise_and(edge_map, dilated_mask)

            # Find new contours using edge information
            refined_cnts, _ = cv2.findContours(
                masked_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS
            )

            if refined_cnts:
                # Use the largest refined contour
                best_cnt = max(refined_cnts, key=cv2.contourArea)
                if cv2.contourArea(best_cnt) > 0.7 * cv2.contourArea(cnt):
                    refined.append(best_cnt)
                else:
                    refined.append(cnt)
            else:
                refined.append(cnt)

        return refined

    @staticmethod
    def filter_puzzle_contours(contours, config):
        """Filter contours based on expected puzzle piece characteristics"""
        filtered = []

        # Get median area of all contours
        areas = [cv2.contourArea(c) for c in contours]
        median_area = np.median(areas)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)

            # Typical puzzle piece validation criteria
            if area < config.MIN_AREA or area > median_area * 3:
                continue

            # Calculate compactness (circularity)
            compactness = 4 * np.pi * area / (perimeter**2)
            if (
                compactness < 0.1 or compactness > 0.7
            ):  # Puzzle pieces are not very circular
                continue

            # Check aspect ratio of bounding rect
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = max(w, h) / min(w, h)
            if aspect_ratio > 4:  # Filter very elongated shapes
                continue

            filtered.append(cnt)

        return filtered

    @staticmethod
    def remove_non_puzzle_shapes(contours):
        """Remove contours unlikely to be puzzle pieces based on characteristics"""
        valid_contours = []

        for cnt in contours:
            # Get convex hull and calculate solidity
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            cnt_area = cv2.contourArea(cnt)
            solidity = float(cnt_area) / hull_area if hull_area > 0 else 0

            # Most puzzle pieces have multiple concavities and moderate solidity
            if 0.4 < solidity < 0.9:
                valid_contours.append(cnt)

        return valid_contours

    @staticmethod
    def find_corners(contour, config=Config()):
        """Detects corners with angles close to 90° (± tolerance). Returns exactly 4 corners sorted clockwise or empty array."""
        perimeter = cv2.arcLength(contour, True)
        epsilon = config.CORNER_EPSILON * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        points = approx.reshape(-1, 2)

        print(f"\nInitial approximation points: {points.tolist()}")

        if len(points) < 4:
            print("Rejected: Not enough initial points")
            return np.array([])

        # Calculate angles and filter by 90° criteria
        valid_points = []
        angle_diffs = []

        for i in range(len(points)):
            prev_idx = (i - 1) % len(points)
            next_idx = (i + 1) % len(points)

            a = points[prev_idx]
            b = points[i]
            c = points[next_idx]

            ba = a - b
            bc = c - b

            norm_ba = np.linalg.norm(ba)
            norm_bc = np.linalg.norm(bc)

            if norm_ba == 0 or norm_bc == 0:
                continue

            cos_theta = np.dot(ba, bc) / (norm_ba * norm_bc)
            angle = np.degrees(np.arccos(np.clip(cos_theta, -1, 1)))
            diff = abs(angle - 90)

            if diff <= config.ANGLE_TOLERANCE:
                valid_points.append(b)
                angle_diffs.append(diff)
                print(f"Point {b} has valid angle: {angle:.1f}°")

        # Check if there are enough valid points
        if len(valid_points) < config.REQUIRED_CORNERS:
            print(f"Rejected: Only {len(valid_points)} valid angles found")
            return np.array([])

        # Generate all combinations of 4 points and find the best one
        best_quad = None
        min_score = float("inf")

        for indices in itertools.combinations(range(len(valid_points)), 4):
            # Extract the points for this combination
            quad_points = [valid_points[i] for i in indices]

            # Sort points clockwise around centroid
            centroid = np.mean(quad_points, axis=0)
            rel_points = np.array(quad_points) - centroid
            angles = np.arctan2(rel_points[:, 1], rel_points[:, 0])
            sorted_indices = np.argsort(angles)
            sorted_quad = [quad_points[i] for i in sorted_indices]

            # Check convexity
            quad_array = np.array(sorted_quad)
            hull = cv2.convexHull(quad_array)
            if len(hull) != 4:
                continue

            # Check side ratios
            sides = [
                np.linalg.norm(sorted_quad[i] - sorted_quad[(i + 1) % 4])
                for i in range(4)
            ]
            max_side_1 = max(sides[0], sides[2]) if max(sides[0], sides[2]) != 0 else 1
            max_side_2 = max(sides[1], sides[3]) if max(sides[1], sides[3]) != 0 else 1
            ratio1 = abs(sides[0] - sides[2]) / max_side_1
            ratio2 = abs(sides[1] - sides[3]) / max_side_2

            if (
                ratio1 > config.SIDE_TOLERANCE_RATIO
                or ratio2 > config.SIDE_TOLERANCE_RATIO
            ):
                continue

            # Calculate the score as the sum of angle differences
            score = sum(angle_diffs[i] for i in indices)

            # Update best quad if this is the best score so far
            if score < min_score:
                min_score = score
                best_quad = sorted_quad

        if best_quad is None:
            print("Rejected: No valid quadrilateral found in combinations")
            return np.array([])

        print(f"Best quad selected with score {min_score:.2f}: {best_quad}")

        # Log side lengths and ratios for the best quad
        sides = [
            np.linalg.norm(best_quad[i] - best_quad[(i + 1) % 4]) for i in range(4)
        ]
        print(f"Side lengths: {[f'{s:.1f}' for s in sides]}")
        ratio_1 = abs(sides[0] - sides[2]) / max(sides[0], sides[2])
        ratio_2 = abs(sides[1] - sides[3]) / max(sides[1], sides[3])
        print(f"Opposite side ratios: {ratio_1:.2f}, {ratio_2:.2f}")

        # Final convexity check (shouldn't fail)
        hull = cv2.convexHull(np.array(best_quad))
        if len(hull) != 4:
            print("Unexpected convex hull failure")
            return np.array([])

        print("Valid rectangle-like shape confirmed")
        return np.array(best_quad)

    @staticmethod
    def classify_border(border, config=Config()):
        """
        Classifies border segment into: 'bump', 'cavity', or 'straight'.
        Uses complexity and deviation metrics for classification.
        """
        if len(border) < 2:
            return "straight"

        start, end = border[0], border[-1]
        direct_dist = np.linalg.norm(end - start)

        if direct_dist < 1e-5:
            return "straight"

        # Calculate path complexity
        path_length = sum(
            np.linalg.norm(border[i + 1] - border[i]) for i in range(len(border) - 1)
        )
        complexity = path_length / direct_dist

        # Calculate maximum deviation from straight line
        deviations = [
            np.abs(np.cross(end - start, start - p)) / direct_dist for p in border
        ]
        max_deviation = max(deviations)

        if (
            complexity > config.COMPLEXITY_THRESHOLD
            and max_deviation > config.DEVIATION_THRESHOLD
        ):
            # Determine bulge direction
            midpoint = (start + end) / 2
            centroid = np.mean(border, axis=0)
            dir_vec = end - start
            perp_vec = np.array([-dir_vec[1], dir_vec[0]])
            return "bump" if np.dot(centroid - midpoint, perp_vec) > 0 else "cavity"

        return "straight"

    @staticmethod
    def process_borders(contour, corners, config=Config()):
        """
        Extracts and processes border segments between corners.
        Applies Gaussian smoothing to reduce noise.
        """
        points = contour.reshape(-1, 2)
        borders = []

        for i in range(len(corners)):
            start_idx = np.where((points == corners[i]).all(axis=1))[0][0]
            end_idx = np.where((points == corners[(i + 1) % len(corners)]).all(axis=1))[
                0
            ][0]

            if start_idx < end_idx:
                segment = points[start_idx : end_idx + 1]
            else:
                segment = np.vstack((points[start_idx:], points[: end_idx + 1]))

            # Apply Gaussian smoothing
            if len(segment) > 5:
                segment = cv2.GaussianBlur(
                    segment.astype(np.float32), config.SMOOTHING_KERNEL, 0
                )

            borders.append(segment.astype(np.int32))

        return borders

    @staticmethod
    def analyze_piece(image, contour, config=Config()):
        """
        Full processing pipeline for a puzzle piece contour.
        Returns dictionary with piece data or None if invalid.
        """
        piece = PieceProcessor.find_piece(image, contour, config)
        if piece is None:
            return None

        corners = PieceProcessor.find_corners(contour, config)
        if len(corners) < 3:
            print("Only found {} corners".format(len(corners)))
            return None

        borders = PieceProcessor.process_borders(contour, corners, config)
        border_types = [PieceProcessor.classify_border(b, config) for b in borders]

        return {
            "piece": piece,
            "corners": corners,
            "borders": borders,
            "types": border_types,
            "contour": contour,
        }


class Visualization:
    @staticmethod
    def debug_contours(
        image, contours, statuses, output_path="debug/contour_validation.jpg"
    ):
        """
        Visualizes contours with color coding by validation status

        Parameters:
        image (numpy.ndarray): The image to draw on
        contours (list): List of contours to visualize
        statuses (list): List of status strings for each contour
        output_path (str, optional): Path to save the output image

        Returns:
        numpy.ndarray: Visualization image
        """
        debug_img = image.copy()
        colors = {"valid": (0, 255, 0), "small": (0, 0, 255), "open": (255, 0, 0)}

        for cnt, status in zip(contours, statuses):
            color = colors.get(status, (255, 255, 255))
            cv2.drawContours(debug_img, [cnt], 0, color, 2)

        save_image(debug_img, output_path)
        return debug_img

    @staticmethod
    def draw_borders(image, piece_info, config=None):
        """
        Draws color-coded borders and corners on the image.
        Color scheme:
        - Bump: Green
        - Cavity: Blue
        - Straight: Red
        """
        vis_img = image.copy()
        colors = {"bump": (0, 255, 0), "cavity": (255, 0, 0), "straight": (0, 0, 255)}

        for border, b_type in zip(piece_info["borders"], piece_info["types"]):
            pts = border.reshape((-1, 1, 2)).astype(np.int32)
            cv2.polylines(vis_img, [pts], False, colors.get(b_type, (0, 0, 0)), 2)

        for i, corner in enumerate(piece_info["corners"]):
            x, y = corner.astype(int)
            cv2.circle(vis_img, (x, y), 5, (0, 255, 255), -1)
            cv2.putText(
                vis_img,
                str(i),
                (x + 5, y + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )

        return vis_img
