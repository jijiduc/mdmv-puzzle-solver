# mdmv-puzzle-solver

Project of Model Driven Machine Vision (Course 206.2) course, lectured by Professor Louis Lettry.

## Puzzle Solver Pipeline

This project implements a complete pipeline for puzzle piece detection, edge extraction, feature analysis, and puzzle assembly. The pipeline consists of five main phases:

1. **Segmentation**: Detection and isolation of individual puzzle pieces from the input image
2. **Edge Extraction**: Identification and extraction of the edges of each puzzle piece
3. **Feature Extraction**: Analysis of both shape and color features of the extracted edges
4. **Edge Matching**: Computation of compatibility scores between edges of different pieces
5. **Puzzle Assembly**: Assembly of pieces based on edge compatibility scores

## Implementation Details

### 1. Segmentation

The segmentation phase processes the input image to isolate individual puzzle pieces:

- Converts the image to grayscale and applies thresholding
- Uses morphological operations (closing and dilation) to clean up the binary mask
- Detects contours in the mask and filters out noise based on contour area
- Creates bounding boxes around valid contours to extract individual pieces
- Applies a mask to each piece to isolate it from the background

```python
# Core segmentation process
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, binary_mask = cv2.threshold(gray_img, threshold_value, 255, cv2.THRESH_BINARY)
closed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, closing_kernel)
processed_mask = cv2.dilate(closed_mask, dilation_kernel, iterations=1)
contours, _ = cv2.findContours(processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
```

### 2. Edge Extraction

Once pieces are segmented, the edge extraction phase identifies the corners and edges of each piece:

- Computes the contour of each piece using Canny edge detection
- Calculates distances and angles from the centroid to all contour points
- Uses peak detection on the distance function to find corner candidates
- Selects the best 4 corners to form a quadrilateral representing the piece
- Extracts edge points between consecutive corners

```python
# Corner detection approach
peaks, _ = find_peaks(
    sorted_distances_smooth, 
    prominence=5,
    distance=len(sorted_distances_smooth)/15
)

# Edge extraction between corners
def extract_edge_between_corners(corners, corner_idx1, corner_idx2, edge_coords, centroid):
    # Calculate angles from centroid to corners
    angle1 = math.atan2(corner1[1] - centroid_y, corner1[0] - centroid_x)
    angle2 = math.atan2(corner2[1] - centroid_y, corner2[0] - centroid_x)
    
    # Extract points between the angular range
    angle_mask = (all_angles_normalized >= angle1) & (all_angles_normalized <= angle2)
    filtered_points = edge_coords[angle_mask]
    
    # Sort points by angle
    sorted_indices = np.argsort([math.atan2(y - centroid_y, x - centroid_x) for x, y in filtered_points])
    sorted_points = filtered_points[sorted_indices]
    
    return sorted_points
```

#### Distance Transforms for Corner Detection

A key component of the edge extraction process is the use of distance transforms to enhance corner detection. Distance transforms are used to:

1. **Emphasize Corner Features**: 
   - The distance transform calculates how far each pixel is from the piece boundary
   - This creates a gradient map where corners appear as distinctive local maxima
   - Transforms make corner detection more robust against noise and irregularities

2. **Implementation Details**:
   - A binary mask of each puzzle piece is created
   - The distance transform is applied to this mask using `cv2.distanceTransform`
   - The transform creates a distance field where each pixel's value corresponds to its distance from the nearest contour edge
   - This field is normalized and analyzed for distinctive patterns
   
   ```python
   # Create binary mask for the piece
   piece_mask = np.zeros(piece_img.shape[:2], dtype=np.uint8)
   cv2.drawContours(piece_mask, [piece_contour], 0, 255, -1)
   
   # Apply distance transform
   dist_transform = cv2.distanceTransform(piece_mask, cv2.DIST_L2, 5)
   
   # Normalize for visualization
   dist_transform_vis = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX)
   dist_transform_vis = np.uint8(dist_transform_vis)
   
   # Create colormap for better visualization
   dist_transform_color = cv2.applyColorMap(dist_transform_vis, cv2.COLORMAP_JET)
   ```

3. **Analytical Applications**:
   - Identifies prominent features like tabs and slots by examining local maxima in the distance field
   - Helps classify edge types by analyzing distance field gradients
   - Provides additional data for shape characterization
   - Creates useful visualizations for debugging and analysis (stored in the debug/transforms directory)

4. **Benefits for Edge Matching**:
   - More accurate corner detection leads to better edge extraction
   - Improved shape feature extraction through complementary analysis methods
   - Enhanced robustness against variations in piece sizes and shapes
   - Provides additional metrics that help differentiate similar looking edges

### 3. Features Extraction

Each edge is analyzed to extract both shape and color features:

#### Shape Features:

The edge shape analysis provides detailed geometric characterization of each puzzle piece edge:

1. **Edge Classification System**:
   - **Straight**: Edges with minimal deviation from a straight line between corners
   - **Intrusion**: Edges that curve inward toward the center of the piece (concave)
   - **Extrusion**: Edges that curve outward away from the center of the piece (convex)

2. **Mathematical Approach**:
   - Creates a reference line connecting adjacent corners
   - Projects each edge point onto this reference line
   - Calculates perpendicular distance from each point to the reference line
   - Determines sign of deviation (positive = outward, negative = inward)
   - Uses an adaptive threshold based on edge length (5px or 5% of length)

3. **Statistical Analysis**:
   - Calculates mean deviation across all edge points
   - Identifies maximum absolute deviation
   - Counts significant deviations exceeding the threshold
   - Analyzes distribution pattern of deviations

4. **Classification Logic**:
   - Calculates the percentage of significant deviations
   - Examines the balance between positive and negative deviations
   - Determines maximum deviation magnitude for matching purposes
   - Considers the overall shape pattern of the edge

```python
def classify_edge(edge_points, corner1, corner2, centroid):
    # Step a: Create vector for straight line between corners
    x1, y1 = corner1
    x2, y2 = corner2
    line_vec = (x2-x1, y2-y1)
    line_length = math.sqrt(line_vec[0]**2 + line_vec[1]**2)
    
    # Step b: Determine outward normal vector (perpendicular to line, pointing away from centroid)
    normal_vec = (-line_vec[1]/line_length, line_vec[0]/line_length)
    mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
    centroid_to_mid = (mid_x - centroid[0], mid_y - centroid[1])
    normal_direction = centroid_to_mid[0]*normal_vec[0] + centroid_to_mid[1]*normal_vec[1]
    outward_normal = normal_vec if normal_direction > 0 else (-normal_vec[0], -normal_vec[1])
    
    # Step c: Calculate deviations for all edge points
    deviations = []
    for x, y in edge_points:
        # Vector from corner1 to the current point
        point_vec = (x-x1, y-y1)
        
        # Project point onto reference line
        line_dot = (point_vec[0]*line_vec[0] + point_vec[1]*line_vec[1]) / line_length
        proj_x = x1 + line_dot * line_vec[0] / line_length
        proj_y = y1 + line_dot * line_vec[1] / line_length
        
        # Calculate deviation vector and magnitude
        dev_vec = (x-proj_x, y-proj_y)
        deviation = math.sqrt(dev_vec[0]**2 + dev_vec[1]**2)
        
        # Determine sign (positive = outward/extrusion, negative = inward/intrusion)
        sign = 1 if (dev_vec[0]*outward_normal[0] + dev_vec[1]*outward_normal[1]) > 0 else -1
        deviations.append(sign * deviation)
    
    # Step d: Calculate statistics for classification
    mean_deviation = sum(deviations) / len(deviations)
    abs_deviations = [abs(d) for d in deviations]
    max_abs_deviation = max(abs_deviations)
    
    # Adaptive threshold based on edge length
    straight_threshold = max(5, line_length * 0.05)  # 5px or 5% of line length
    
    # Count significant deviations
    significant_positive = sum(1 for d in deviations if d > straight_threshold)
    significant_negative = sum(1 for d in deviations if d < -straight_threshold)
    portion_significant = (significant_positive + significant_negative) / len(deviations)
    
    # Step e: Classify the edge type
    if max_abs_deviation < straight_threshold or portion_significant < 0.2:
        # Small deviations or few significant points = straight edge
        edge_type = "straight"
        max_deviation = mean_deviation
    elif abs(mean_deviation) < straight_threshold * 0.5:
        # Mean close to zero but mixed deviations
        if significant_positive > significant_negative * 2:
            edge_type = "extrusion"
            max_deviation = max([d for d in deviations if d > 0], default=0)
        elif significant_negative > significant_positive * 2:
            edge_type = "intrusion"
            max_deviation = min([d for d in deviations if d < 0], default=0)
        else:
            # Balanced deviations - likely a straight edge with noise
            edge_type = "straight"
            max_deviation = mean_deviation
    elif mean_deviation > 0:
        # Overall positive deviation = extrusion
        edge_type = "extrusion"
        max_deviation = max(deviations)
    else:
        # Overall negative deviation = intrusion
        edge_type = "intrusion"
        max_deviation = min(deviations)
    
    return edge_type, max_deviation
```

When matching edges, the shape features are used to identify complementary edges:

```python
def calculate_shape_compatibility(edge1_type, edge1_deviation, edge2_type, edge2_deviation):
    # Intrusion should match with extrusion (tab and slot)
    if edge1_type == "intrusion" and edge2_type == "extrusion":
        # Check if deviations are complementary (similar magnitude)
        deviation_match = 1.0 - min(1.0, abs(abs(edge1_deviation) - abs(edge2_deviation)) / 
                                    max(abs(edge1_deviation), abs(edge2_deviation), 1))
        return 0.9 * deviation_match  # High score for complementary edges
    
    # Same check for opposite pairing
    if edge1_type == "extrusion" and edge2_type == "intrusion":
        deviation_match = 1.0 - min(1.0, abs(abs(edge1_deviation) - abs(edge2_deviation)) / 
                                   max(abs(edge1_deviation), abs(edge2_deviation), 1))
        return 0.9 * deviation_match
    
    # Straight edges can match with other straight edges (e.g., at the puzzle border)
    if edge1_type == "straight" and edge2_type == "straight":
        return 0.7  # Moderate score for straight-straight
    
    # All other combinations are unlikely matches
    return 0.1  # Low compatibility score
```

This detailed shape analysis enables accurate matching of puzzle pieces by finding complementary edge geometries, with particular attention to the magnitude and pattern of deviations that help identify matching tab and slot edges.

#### Color Features:

The color feature extraction implements three key enhancements for improved matching accuracy:

1. **Spatial Awareness**:
   - Divides each edge into 5 segments to preserve spatial color information
   - Tracks color patterns along the edge, from one corner to the other
   - Enables matching edges with specific color transitions or patterns

2. **Gradient/Transition Analysis**:
   - Detects and quantifies color changes along the edge
   - Calculates gradient statistics (mean, max) for each HSV channel
   - Counts significant color transitions to identify distinctive edges

3. **Higher Resolution Histograms**:
   - Uses 32 bins (instead of 16) for finer color differentiation
   - Captures more subtle color variations in each edge
   - Improves matching accuracy for complex or similar colored pieces

```python
def extract_edge_color_features(piece_img, edge_points, corner1, corner2, edge_index):
    # Basic sampling (same as before)
    mask = np.zeros(piece_img.shape[:2], dtype=np.uint8)
    for x, y in edge_points:
        cv2.circle(mask, (int(x), int(y)), 3, 255, -1)
    
    samples_bgr = piece_img[mask == 255]
    samples_hsv = cv2.cvtColor(samples_bgr.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    
    # 1. Higher resolution histograms (32 bins)
    bins = 32
    h_hist = np.histogram(samples_hsv[:, 0], bins=bins, range=(0, 180))[0]
    
    # 2. Spatial awareness - divide edge into segments
    num_segments = 5
    segment_length = max(1, len(sorted_points) // num_segments)
    
    # Calculate histogram for each segment
    spatial_h_hists = []
    for i in range(num_segments):
        segment_points = sorted_points[start_idx:end_idx]
        # [create and store segment histograms]
    
    # 3. Gradient/Transition features
    # Calculate gradients (differences between adjacent points)
    h_diffs = np.abs(np.diff(edge_hsv[:, 0]))
    # [calculate gradient statistics]
    
    transition_features = {
        'mean_h_gradient': float(np.mean(h_diffs)),
        'max_h_gradient': float(np.max(h_diffs)),
        'gradient_count': int(np.sum(h_diffs > 20))
    }
    
    # Create enhanced feature vector with all features
    color_feature = {
        'h_hist': h_hist.tolist(),
        'spatial_h_hists': spatial_h_hists,
        'transition_features': transition_features
    }
```

These color features are combined during matching with appropriate weights to provide more accurate edge compatibility scores:

```python
def calculate_color_compatibility(color_feature1, color_feature2):
    # 1. Global histogram comparison
    global_color_score = compare_histograms()
    
    # 2. Spatial histogram comparison (segment by segment)
    spatial_score = compare_spatial_segments()
    
    # 3. Compare color transition/gradient features
    gradient_score = compare_transitions()
    
    # 4. Compare basic color statistics
    mean_score = compare_means()
    
    # Combine all scores with appropriate weights
    final_score = (
        0.3 * global_color_score +  # Base color distribution
        0.3 * spatial_score +       # Spatial color patterns
        0.25 * gradient_score +     # Color transitions
        0.15 * mean_score           # Basic color similarity
    )
```

### 4. Edge Matching

The edge matching phase computes compatibility scores between edges of different pieces:

- Compares all possible pairs of edges between different pieces
- Calculates shape compatibility based on complementary edge types (intrusion matches with extrusion)
- Computes color compatibility by comparing histograms and color statistics
- Combines shape and color scores with appropriate weights (70% shape, 30% color)
- Selects matches above a certain threshold score

```python
def calculate_shape_compatibility(edge1_type, edge1_deviation, edge2_type, edge2_deviation):
    # Complementary types get high scores
    if edge1_type == "intrusion" and edge2_type == "extrusion":
        deviation_match = 1.0 - min(1.0, abs(abs(edge1_deviation) - abs(edge2_deviation)) / max(abs(edge1_deviation), abs(edge2_deviation), 1))
        return 0.9 * deviation_match
    
    # Straight-straight gets moderate score
    if edge1_type == "straight" and edge2_type == "straight":
        return 0.7
    
    # Default low compatibility
    return 0.1

def calculate_color_compatibility(color_feature1, color_feature2):
    # Compare histograms using correlation method
    h_corr = cv2.compareHist(np.array(color_feature1['h_hist'], dtype=np.float32), 
                           np.array(color_feature2['h_hist'], dtype=np.float32), 
                           cv2.HISTCMP_CORREL)
    
    # Compare mean HSV values
    mean_diff_h = abs(color_feature1['mean_hsv'][0] - color_feature2['mean_hsv'][0]) / 180.0
    mean_score = 1.0 - (0.5 * mean_diff_h + 0.3 * mean_diff_s + 0.2 * mean_diff_v)
    
    # Combine histogram and mean scores
    final_score = 0.6 * color_score + 0.4 * mean_score
```

### 5. Puzzle Assembly

The final phase assembles the puzzle based on the matched edges:

- Represents the puzzle as a grid where pieces can be placed
- Starts with a seed piece (the one with most high-quality matches)
- Maintains a frontier of candidate pieces that could be placed next
- Iteratively places the piece with the highest match score
- Handles the spatial arrangement of pieces based on edge orientations

```python
class PuzzleAssembler:
    def start_assembly(self):
        # Choose seed piece with most high-quality matches
        seed_piece_idx = max(seed_candidates.items(), key=lambda x: x[1])[0]
        self.place_piece(seed_piece_idx, 0, 0)
        self.update_frontier()
    
    def assemble_next_piece(self):
        # Find best piece to place next
        for piece_idx in self.frontier:
            position, edge_match = self.determine_piece_position(piece_idx)
            # Find best match score
            
        # Place the piece with highest score
        if best_piece and best_position:
            row, col = best_position
            self.place_piece(best_piece, row, col)
            self.update_frontier()
            return True
```

## Performance Optimizations

The implementation includes several optimizations for performance:

- **Caching**: Results of expensive operations are cached to avoid redundant calculations
- **Parallel Processing**: Piece processing is distributed across multiple CPU cores
- **Numba Acceleration**: Critical numerical operations are accelerated using Numba JIT compilation
- **Memory Management**: Large data structures are handled efficiently with appropriate cleanup

## Current Results

The system currently achieves:

- Accurate segmentation of individual puzzle pieces
- Reliable edge classification (straight, intrusion, extrusion)
- Good quality edge matching with 1500+ potential matches identified
- Partial assembly with ~16% of pieces correctly placed (4 out of 24 pieces in the test case)

## Future Improvements

Potential areas for improvement include:

- Enhanced edge matching by incorporating more geometric features
- Improved assembly strategy with backtracking to escape local optima
- Implementation of machine learning for feature extraction and matching
- Support for rotation and flipping of pieces during assembly

## Usage

```bash
# Install requirements
pip install -r requirements.txt

# Run the edge matching and assembly
python edges_matching.py
```

## Team Members

[Jeremy Duc](https://github.com/jijiduc) & [Alexandre Venturi](https://github.com/mastermeter)

## References

[Project description](https://isc.hevs.ch/learn/pluginfile.php/5191/mod_resource/content/0/Project.pdf), the provided PDF document in the course.