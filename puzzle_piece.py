import cv2
import numpy as np
import itertools

# Import Config at module level instead of inside methods
from pieces_cutting import Config


class Piece:
    """
    Class representing a puzzle piece with its properties and behaviors.
    """
    
    def __init__(self, image, contour, config=None):
        """
        Initialize a puzzle piece from an image and contour.
        
        Parameters:
        image (numpy.ndarray): The source image
        contour (numpy.ndarray): The contour of the piece
        config (Config, optional): Configuration parameters
        """
        self.image = None  # The extracted piece image
        self.contour = contour
        self.corners = None
        self.borders = None
        self.border_types = None
        self.is_valid = False
        self.validation_status = None
        self.config = config or Config()
        
        # Process the piece
        self._extract_image(image)
        if self.image is not None:
            self._find_corners()
            if self.corners is not None and len(self.corners) >= 4:
                self._process_borders()
                self.is_valid = True
    
    def _extract_image(self, image):
        """Extract the piece image from the full image using contour."""
        if len(self.contour) < 5:
            self.validation_status = "too_few_points"
            return

        area = cv2.contourArea(self.contour)
        perimeter = cv2.arcLength(self.contour, True)

        if area < self.config.MIN_AREA:
            self.validation_status = "small_area"
            return
            
        if perimeter < self.config.MIN_PERIMETER:
            self.validation_status = "short_perimeter"
            return

        compactness = 4 * np.pi * area / (perimeter ** 2)
        if compactness < self.config.CLOSED_THRESHOLD:
            self.validation_status = "low_compactness"
            return

        # Extract piece using contour mask
        x, y, w, h = cv2.boundingRect(self.contour)
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [self.contour], 0, 255, -1)
        self.image = cv2.bitwise_and(image[y:y+h, x:x+w], image[y:y+h, x:x+w], mask=mask[y:y+h, x:x+w])
        self.validation_status = "valid"
    
    def _find_corners(self):
        """Find corners of the piece."""
        perimeter = cv2.arcLength(self.contour, True)
        epsilon = self.config.CORNER_EPSILON * perimeter
        approx = cv2.approxPolyDP(self.contour, epsilon, True)
        points = approx.reshape(-1, 2)
        
        print(f"\nInitial approximation points: {points.tolist()}")

        if len(points) < 4:
            print("Rejected: Not enough initial points")
            return None

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
            
            if diff <= self.config.ANGLE_TOLERANCE:
                valid_points.append(b)
                angle_diffs.append(diff)
                print(f"Point {b} has valid angle: {angle:.1f}°")

        # Check if there are enough valid points
        if len(valid_points) < self.config.REQUIRED_CORNERS:
            print(f"Rejected: Only {len(valid_points)} valid angles found")
            return None

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
            if len(hull) != 4 and self.config.CONVEXITY_CHECK:
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

            if ratio1 > self.config.SIDE_TOLERANCE_RATIO or ratio2 > self.config.SIDE_TOLERANCE_RATIO:
                continue

            # Calculate the score as the sum of angle differences
            score = sum(angle_diffs[i] for i in indices)

            # Update best quad if this is the best score so far
            if score < min_score:
                min_score = score
                best_quad = sorted_quad

        if best_quad is None:
            print("Rejected: No valid quadrilateral found in combinations")
            return None

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
        if len(hull) != 4 and self.config.CONVEXITY_CHECK:
            print("Unexpected convex hull failure")
            return None

        print("Valid rectangle-like shape confirmed")
        self.corners = np.array(best_quad)
    
    def _classify_border(self, border):
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

        if complexity > self.config.COMPLEXITY_THRESHOLD and max_deviation > self.config.DEVIATION_THRESHOLD:
            # Determine bulge direction
            midpoint = (start + end) / 2
            centroid = np.mean(border, axis=0)
            dir_vec = end - start
            perp_vec = np.array([-dir_vec[1], dir_vec[0]])
            return 'bump' if np.dot(centroid - midpoint, perp_vec) > 0 else 'cavity'
        
        return 'straight'
    
    def _process_borders(self):
        """Process borders between corners and classify them."""
        points = self.contour.reshape(-1, 2)
        borders = []
        
        for i in range(len(self.corners)):
            start_idx = np.where((points == self.corners[i]).all(axis=1))[0][0]
            end_idx = np.where((points == self.corners[(i+1)%len(self.corners)]).all(axis=1))[0][0]
            
            if start_idx < end_idx:
                segment = points[start_idx:end_idx+1]
            else:
                segment = np.vstack((points[start_idx:], points[:end_idx+1]))
            
            # Apply Gaussian smoothing
            if len(segment) > 5:
                segment = cv2.GaussianBlur(segment.astype(np.float32), self.config.SMOOTHING_KERNEL, 0)
            
            borders.append(segment.astype(np.int32))
        
        self.borders = borders
        self.border_types = [self._classify_border(b) for b in borders]
    
    def draw(self, image=None):
        """
        Draws color-coded borders and corners on an image.
        
        Parameters:
        image (numpy.ndarray, optional): The image to draw on. If None, uses the original image.
        
        Returns:
        numpy.ndarray: Image with piece visualization
        """
        if not self.is_valid:
            return None
            
        if image is None:
            # Create a blank image with the same size as the piece
            x, y, w, h = cv2.boundingRect(self.contour)
            vis_img = np.zeros((h, w, 3), dtype=np.uint8)
        else:
            vis_img = image.copy()
            
        colors = {'bump': (0, 255, 0), 'cavity': (255, 0, 0), 'straight': (0, 0, 255)}

        for border, b_type in zip(self.borders, self.border_types):
            pts = border.reshape((-1, 1, 2)).astype(np.int32)
            cv2.polylines(vis_img, [pts], False, colors.get(b_type, (0, 0, 0)), 2)

        for i, corner in enumerate(self.corners):
            x, y = corner.astype(int)
            cv2.circle(vis_img, (x, y), 5, (0, 255, 255), -1)
            cv2.putText(vis_img, str(i), (x+5, y+5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        return vis_img
    
    def to_dict(self):
        """Convert piece to dictionary representation."""
        return {
            'piece': self.image,
            'corners': self.corners,
            'borders': self.borders,
            'types': self.border_types,
            'contour': self.contour,
            'is_valid': self.is_valid,
            'validation_status': self.validation_status
        }
        
    @classmethod
    def from_dict(cls, data, config=None):
        """Create piece from dictionary representation."""
        piece = cls.__new__(cls)
        piece.image = data['piece']
        piece.contour = data['contour']
        piece.corners = data['corners']
        piece.borders = data['borders']
        piece.border_types = data['types']
        piece.is_valid = data.get('is_valid', True)
        piece.validation_status = data.get('validation_status', 'valid')
        piece.config = config or Config()
        return piece