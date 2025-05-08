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

The edge matching phase computes compatibility scores between edges of different pieces using a sophisticated multi-stage approach:

#### Basic Edge Matching

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

#### Advanced Edge Matching

To improve edge matching accuracy, we've implemented advanced shape analysis techniques that provide more sophisticated geometric matching beyond simple edge type classification:

1. **Procrustes Analysis**
   - Determines the optimal alignment between two edge contours
   - Finds the translation, rotation, and scaling that minimizes the difference between edges
   - Provides a similarity score based on how well edges align after transformation
   - Particularly effective for matching complex shape features

```python
def calculate_procrustes_similarity(points1, points2):
    # Center both point sets
    centroid1 = np.mean(points1, axis=0)
    centroid2 = np.mean(points2, axis=0)
    centered1 = points1 - centroid1
    centered2 = points2 - centroid2
    
    # Find optimal rotation using SVD
    correlation_matrix = centered1.T @ centered2
    U, s, Vt = linalg.svd(correlation_matrix)
    rotation = U @ Vt
    
    # Calculate residual after alignment
    aligned = centered1 @ rotation
    residual = np.sum((aligned - centered2)**2)
    
    # Convert to similarity score
    similarity_score = 1.0 - min(1.0, residual / max_possible_residual)
    return similarity_score
```

2. **Hausdorff Distance**
   - Measures the maximum minimum distance between two edge contours
   - Quantifies how far two shapes differ from each other
   - Robust to local shape variations and minor misalignments
   - Provides a detailed measure of shape differences that categorical classification misses

```python
def calculate_hausdorff_distance(points1, points2):
    # Calculate pairwise distances
    distances = np.sqrt(d1 + d2[:, np.newaxis] - 2 * np.dot(points2, points1.T))
    
    # Forward Hausdorff: min distance from each point in points1 to any point in points2
    d1_to_2 = np.max(np.min(distances.T, axis=1))
    
    # Backward Hausdorff: min distance from each point in points2 to any point in points1
    d2_to_1 = np.max(np.min(distances, axis=1))
    
    # Hausdorff distance is the maximum of the two
    return max(d1_to_2, d2_to_1)
```

3. **Adaptive Color Weighting**
   - Analyzes color distinctiveness to determine appropriate shape/color balance
   - Gives more weight to color when edges have distinctive color patterns
   - Relies more on shape when colors are uniform or similar
   - Adapts to each edge pair's characteristics for optimal matching

4. **Multi-stage Matching Pipeline**
   - Uses fast type-based compatibility as an initial filter
   - Applies computationally intensive geometric analysis only to promising matches
   - Combines multiple matching strategies with weighted scoring
   - Gracefully falls back to basic matching when advanced analysis isn't applicable

```python
def enhanced_edge_compatibility(edge1_points, edge2_points, edge1_type, edge1_deviation, 
                              edge2_type, edge2_deviation, edge1_colors, edge2_colors):
    # Basic shape and color scores
    basic_shape_score = calculate_shape_compatibility(edge1_type, edge1_deviation, edge2_type, edge2_deviation)
    color_score = calculate_color_compatibility(edge1_colors, edge2_colors)
    
    # Skip advanced analysis for unlikely matches
    if basic_shape_score < 0.2:
        return basic_shape_score * 0.7 + color_score * 0.3
    
    # Advanced shape analysis
    procrustes_score = calculate_procrustes_similarity(edge1_normalized, edge2_normalized)
    hausdorff_dist = calculate_hausdorff_distance(edge1_normalized, edge2_normalized)
    hausdorff_score = 1.0 - min(1.0, hausdorff_dist / max_distance)
    
    # Adaptive color weighting based on color distinctiveness
    color_distinctiveness = analyze_color_distinctiveness(edge1_colors)
    color_weight = adaptive_weight_based_on_distinctiveness(color_distinctiveness)
    
    # Combine shape scores from different methods
    combined_shape_score = (0.4 * basic_shape_score + 
                           0.35 * procrustes_score + 
                           0.25 * hausdorff_score)
    
    # Final weighted score
    return (1 - color_weight) * combined_shape_score + color_weight * color_score
```

These advanced techniques significantly improve matching accuracy, particularly for complex edge shapes and pieces with distinctive color patterns.

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

### 6. DTW-based Color Matching

To improve color matching precision, we've implemented Dynamic Time Warping (DTW) for comparing color sequences along edges:

1. **Direct Color Sequence Extraction**:
   - Instead of aggregating colors into histograms, we extract the precise sequence of colors along each edge
   - Colors are sampled at regular intervals from one corner to the other
   - Convert to LAB color space for perceptually accurate color difference measurement
   - Each color sample is tagged with a confidence score based on local variance

2. **DTW Algorithm for Sequence Matching**:
   - Uses a modified Dynamic Time Warping algorithm to find optimal alignment between color sequences
   - Handles variations in edge lengths and color sampling rates
   - Incorporates confidence weighting to emphasize reliable color samples
   - Allows for partial matches at sequence ends to handle corner imperfections
   - Adds penalties for excessive warping to preserve edge proportions

3. **Bidirectional Matching**:
   - Performs matching in both directions (normal and reversed)
   - Takes the best matching direction since edges may connect in either orientation

4. **Implementation Features**:
   - Robust color sampling with local averaging to reduce noise
   - Confidence scoring based on local color variance
   - Color normalization to handle lighting differences
   - Sequence visualization for debugging
   - Graceful fallback to simpler methods for edge-case scenarios

```python
def dtw_color_matching(sequence1, sequence2, confidence1=None, confidence2=None):
    """Match color sequences using Dynamic Time Warping with confidence weighting."""
    # Initialize DTW matrix
    dtw_matrix = np.ones((n+1, m+1)) * float('inf')
    dtw_matrix[0, 0] = 0
    
    # Allow partial matching at ends
    for i in range(1, n+1): dtw_matrix[i, 0] = 0
    for j in range(1, m+1): dtw_matrix[0, j] = 0
    
    # Define constraining band to prevent excessive warping
    band_width = int(max(n, m) * 0.3)
    
    # Fill DTW matrix
    for i in range(1, n+1):
        j_start = max(1, i - band_width)
        j_end = min(m+1, i + band_width)
        
        for j in range(j_start, j_end):
            # Get colors and confidence values
            color1, color2 = sequence1[i-1], sequence2[j-1]
            conf1, conf2 = confidence1[i-1], confidence2[j-1]
            
            # Weight cost by confidence
            conf_weight = (conf1 + conf2) / 2.0
            base_cost = color_distance(color1, color2)
            weighted_cost = base_cost * (2.0 - conf_weight)
            
            # Step pattern with penalty for insertions/deletions
            diag_cost = dtw_matrix[i-1, j-1]
            horiz_cost = dtw_matrix[i, j-1] + 0.5  # Penalty
            vert_cost = dtw_matrix[i-1, j] + 0.5   # Penalty
            
            # Find minimum cost path
            dtw_matrix[i, j] = weighted_cost + min(diag_cost, horiz_cost, vert_cost)
            
    # Find minimum cost in last row or column (for subsequence matching)
    best_cost = min(np.min(dtw_matrix[n, 1:]), np.min(dtw_matrix[1:, m]))
    
    # Convert to similarity score
    similarity = 1.0 - min(1.0, best_cost / max_possible_cost)
    return similarity
```

DTW color matching significantly improves matching accuracy by preserving the spatial arrangement of colors along puzzle piece edges, leading to more precise matching than histogram-based approaches.

## Enhanced Puzzle Assembly

The puzzle assembly phase now includes several advanced features for more robust and accurate assembly:

### 1. Backtracking Assembly

Backtracking capability allows the assembler to recover from suboptimal placements and explore alternative solution paths:

- Maintains history of piece placements for potential reversal
- Intelligently removes pieces when reaching a dead end
- Tracks dead-end configurations to avoid repeating mistakes
- Limits backtracking depth to control exploration vs. exploitation
- Reset and restart capability for optimal solution search

```python
def backtrack(self):
    """Backtrack to a previous state in the assembly process."""
    if not self.history:
        return False
    
    # Get the last placement
    last_placement = self.history.pop()
    piece_idx = last_placement['piece_idx']
    
    # Mark this placement as a dead end to avoid trying it again
    placement_signature = (piece_idx, *last_placement['position'])
    self.dead_ends.add(placement_signature)
    
    # Remove the piece
    self.remove_piece(piece_idx)
    
    # Restore the used edges set to the state before this placement
    self.used_edges = last_placement['used_edges']
    
    # Update frontier after removing the piece
    self.update_frontier()
    
    return True
```

### 2. Dynamic Threshold Adjustment

The system now includes dynamic threshold adjustment to balance precision with coverage:

- Starts with high threshold for matching quality to prioritize confident matches
- Automatically lowers threshold when progress stalls
- Progressively explores less confident matches when needed
- Returns to higher thresholds when high-quality matches are found
- Preserves minimum quality standards to prevent nonsensical assemblies

```python
def adjust_threshold(self, lower=True):
    """Adjust the matching threshold dynamically."""
    if lower:
        # Don't go below minimum threshold
        if self.current_match_threshold > self.min_match_threshold:
            # Reduce by 0.05 at a time
            self.current_match_threshold = max(
                self.current_match_threshold - 0.05, 
                self.min_match_threshold
            )
            return True
    else:
        # Don't go above initial threshold
        if self.current_match_threshold < self.initial_match_threshold:
            # Increase by 0.05 at a time
            self.current_match_threshold = min(
                self.current_match_threshold + 0.05,
                self.initial_match_threshold
            )
            return True
            
    return False
```

### 3. Multiple Starting Points

Multiple starting point strategy helps find better initial configurations:

- Automatically identifies promising candidate pieces for starting the assembly
- Runs multiple assembly attempts with different starting pieces
- Tracks the best assembly configuration achieved
- Compares results to find the optimal solution
- Retains the best solution when complete

```python
def assemble_puzzle(self, try_multiple_starts=True, max_start_attempts=3):
    """Assemble the complete puzzle using multiple starting points."""
    print("Starting puzzle assembly...")
    
    best_assembly = None
    best_pieces_placed = 0
    
    # Get seed candidates if using multiple starts
    seed_candidates = [None]  # Default will use automatic selection
    if try_multiple_starts:
        seed_candidates = self.find_seed_candidates(num_candidates=max_start_attempts)
        print(f"Will try {len(seed_candidates)} different starting pieces")
    
    # Try each seed candidate
    for attempt, seed_piece in enumerate(seed_candidates):
        # Reset for a new attempt
        self.reset_assembly()
        
        # Place the first piece (automatic or specified)
        success = self.start_assembly(seed_piece)
        
        # [Assembly process...]
        
        # Save this assembly if it's the best so far
        if pieces_placed > best_pieces_placed:
            best_pieces_placed = pieces_placed
            best_assembly = {
                "success": pieces_placed > 0,
                "pieces_placed": pieces_placed,
                "total_pieces": self.num_pieces,
                "grid": dict(self.grid),
                "placed_positions": dict(self.placed_positions)
                # [Additional data stored...]
            }
```

### 4. Rotation Support

The assembler now handles piece rotation for more flexible assembly:

- Automatically calculates necessary rotation for proper piece alignment
- Supports all four possible orientations (0°, 90°, 180°, 270°)
- Tracks rotation information with each placement
- Applies slight scoring penalty for rotations to prefer non-rotated solutions when possible
- Visualizes rotated pieces in the assembly output

```python
# Calculate rotation needed for proper orientation
rotation_needed = (required_edge - new_edge) % 4  # Clockwise rotation steps needed

# Apply a rotation penalty to the score (slightly prefer non-rotated pieces)
adjusted_score = score
if rotation_needed > 0:
    # Small penalty for rotation (5% per step of rotation)
    rotation_penalty = 0.05 * rotation_needed
    adjusted_score = score * (1 - rotation_penalty)
```

### 5. Global Constraints

Global constraint checking ensures consistent multi-edge arrangements:

- Verifies that all edges of a piece have valid connections to neighbors
- Checks neighbor consistency during placement decisions
- Applies score penalties for conflicts with existing neighbors
- Provides bonuses for multiple consistent neighbors
- Rejects placements with severe conflicts to ensure coherent assembly

```python
# Apply global constraints to ensure consistent assembly
# Check neighbor consistency - pieces should have matching neighbors on all sides
neighbor_count = 0
conflict_count = 0

# Check all 4 adjacent positions
for neighbor_dir in range(4):
    # [Direction-specific calculations...]
    
    # Check if there's a neighbor in this direction
    neighbor_idx = self.grid.get((neighbor_row, neighbor_col))
    if neighbor_idx is not None:
        neighbor_count += 1
        
        # Check if there's a good match between this piece and the neighbor
        has_good_match = False
        for match in self.edge_matches:
            # [Match checking logic...]
        
        # If there's a neighbor but no good match, it's a conflict
        if not has_good_match:
            conflict_count += 1

# Adjust score based on global constraints
constraint_adjusted_score = adjusted_score

# Penalty for conflicts with existing neighbors
if conflict_count > 0:
    # Severe penalty for each conflict (50% per conflict)
    constraint_adjusted_score *= (1 - 0.5 * conflict_count)

# Bonus for having multiple consistent neighbors
consistent_neighbors = neighbor_count - conflict_count
if consistent_neighbors > 0:
    constraint_adjusted_score *= (1 + 0.05 * consistent_neighbors)
```

### 6. Enhanced Visualization

The system now generates multiple complementary visualizations for better understanding of the assembly:

- **Grid View**: Simple grid representation of the assembled puzzle
- **Realistic View**: Shows actual piece shapes and how they connect
- **Exploded View**: Displays pieces with spacing and connection lines
- **Edge-Highlighted View**: Shows connection lines between matched pieces
- **Combined View**: Presents all views side-by-side for comparison

Connection lines in the exploded view are color-coded by match quality:
- Green: High confidence matches
- Orange: Medium confidence matches
- Red: Lower confidence matches

```python
# Create visualizations:
# 1. Grid-based simple view (for reference)
# 2. Real-shape assembly view (more realistic)
# 3. Interactive exploded view (showing pieces with connections)

# Create a combined view with all visualizations
combined_output_path = output_path.replace('.png', '_all_views.png')
cv2.imwrite(combined_output_path, combined_canvas)
```

## Current Results

The system now achieves:

- Complete assembly of 24-piece puzzles (100% pieces placed in test cases)
- Accurate edge matching with DTW color sequence alignment
- Rotation handling for flexible piece orientation
- Intelligent backtracking to escape suboptimal configurations
- Adaptive threshold management for optimal precision vs. coverage balance
- Multiple starting points to find the best assembly configuration
- Enhanced visualizations showing realistic piece connections

The combination of these features has significantly improved assembly performance, allowing the system to handle complex puzzles that previously could only be partially assembled.

## Future Improvements

While the current system is much more robust, further improvements could include:

- Support for piece flipping (in addition to rotation) for double-sided puzzles
- Parallel hypothesis testing for even faster assembly

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