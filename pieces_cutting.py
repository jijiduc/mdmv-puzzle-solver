import cv2
import numpy as np
import os
import itertools
from image_processing import save_image

class Config:
    """Centralized configuration parameters"""
    # Piece validation parameters (it's a flow in the system - should be systemized)
    MIN_AREA = 1000
    MIN_PERIMETER = 100
    CLOSED_THRESHOLD = 0.02
    
    #Contour detection
    RESIZE_SCALE = 30  # Percentage of original size
    THRESHOLD_VALUE = 100  # Fixed threshold value
    USE_ADAPTIVE_THRESHOLD = False
    ADAPTIVE_BLOCK_SIZE = 11
    MIN_CONTOUR_AREA = 2000
    MAX_CONTOUR_AREA = 100000
    
    # Corner detection
    CORNER_EPSILON = 0.02  # Approximation precision
    ANGLE_TOLERANCE = 15  # Degrees from 90° (±value)
    REQUIRED_CORNERS = 4   # Number of expected corners
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

        compactness = 4 * np.pi * area / (perimeter ** 2)
        if compactness < config.CLOSED_THRESHOLD:
            return None

        # Extract piece using contour mask
        x, y, w, h = cv2.boundingRect(contour)
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], 0, 255, -1)
        return cv2.bitwise_and(image[y:y+h, x:x+w], image[y:y+h, x:x+w], mask=mask[y:y+h, x:x+w])
    
    @staticmethod
    def find_contour(image):
        """Wrapper function for contour detection"""
        # Pre-process image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)
        
        # Thresholding and morphology
        _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        
        # Find contours
        contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [cnt for cnt in contours if cv2.contourArea(cnt) > 500]

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
            prev_idx = (i-1) % len(points)
            next_idx = (i+1) % len(points)
            
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
        min_score = float('inf')

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
                np.linalg.norm(sorted_quad[i] - sorted_quad[(i+1)%4])
                for i in range(4)
            ]
            max_side_1 = max(sides[0], sides[2]) if max(sides[0], sides[2]) != 0 else 1
            max_side_2 = max(sides[1], sides[3]) if max(sides[1], sides[3]) != 0 else 1
            ratio1 = abs(sides[0] - sides[2]) / max_side_1
            ratio2 = abs(sides[1] - sides[3]) / max_side_2

            if ratio1 > config.SIDE_TOLERANCE_RATIO or ratio2 > config.SIDE_TOLERANCE_RATIO:
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
            np.linalg.norm(best_quad[i] - best_quad[(i+1)%4])
            for i in range(4)
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
            return 'straight'

        start, end = border[0], border[-1]
        direct_dist = np.linalg.norm(end - start)
        
        if direct_dist < 1e-5:
            return 'straight'

        # Calculate path complexity
        path_length = sum(np.linalg.norm(border[i+1]-border[i]) for i in range(len(border)-1))
        complexity = path_length / direct_dist

        # Calculate maximum deviation from straight line
        deviations = [np.abs(np.cross(end-start, start-p)) / direct_dist for p in border]
        max_deviation = max(deviations)

        if complexity > config.COMPLEXITY_THRESHOLD and max_deviation > config.DEVIATION_THRESHOLD:
            # Determine bulge direction
            midpoint = (start + end) / 2
            centroid = np.mean(border, axis=0)
            dir_vec = end - start
            perp_vec = np.array([-dir_vec[1], dir_vec[0]])
            return 'bump' if np.dot(centroid - midpoint, perp_vec) > 0 else 'cavity'
        
        return 'straight'

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
            end_idx = np.where((points == corners[(i+1)%len(corners)]).all(axis=1))[0][0]
            
            if start_idx < end_idx:
                segment = points[start_idx:end_idx+1]
            else:
                segment = np.vstack((points[start_idx:], points[:end_idx+1]))
            
            # Apply Gaussian smoothing
            if len(segment) > 5:
                segment = cv2.GaussianBlur(segment.astype(np.float32), config.SMOOTHING_KERNEL, 0)
            
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
            'piece': piece,
            'corners': corners,
            'borders': borders,
            'types': border_types,
            'contour': contour
        }

class Visualization:
    @staticmethod
    def draw_borders(image, piece_info, config=Config()):
        """
        Draws color-coded borders and corners on the image.
        Color scheme:
        - Bump: Green
        - Cavity: Blue
        - Straight: Red
        """
        vis_img = image.copy()
        colors = {'bump': (0, 255, 0), 'cavity': (255, 0, 0), 'straight': (0, 0, 255)}

        for border, b_type in zip(piece_info['borders'], piece_info['types']):
            pts = border.reshape((-1, 1, 2)).astype(np.int32)
            cv2.polylines(vis_img, [pts], False, colors.get(b_type, (0, 0, 0)), 2)

        for i, corner in enumerate(piece_info['corners']):
            x, y = corner.astype(int)
            cv2.circle(vis_img, (x, y), 5, (0, 255, 255), -1)
            cv2.putText(vis_img, str(i), (x+5, y+5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        return vis_img

    @staticmethod
    def debug_contours(image, contours, statuses):
        """Visualizes contours with color coding by validation status"""
        debug_img = image.copy()
        colors = {'valid': (0, 255, 0), 'small': (0, 0, 255), 'open': (255, 0, 0)}
        
        for cnt, status in zip(contours, statuses):
            color = colors.get(status, (255, 255, 255))
            cv2.drawContours(debug_img, [cnt], 0, color, 2)
        
        save_image(debug_img, "debug/contour_validation.jpg")
        return debug_img