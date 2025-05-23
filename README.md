# MDMV Puzzle Solver

A computer vision-based jigsaw puzzle solver that automatically detects, analyzes, and classifies puzzle pieces using model-driven machine vision techniques.

## Overview

This project implements a complete pipeline for analyzing jigsaw puzzle pieces from images:
1. **Piece Detection**: Automatically detects individual puzzle pieces from a photograph
2. **Corner Detection**: Uses polar distance profile analysis to find the 4 corners of each piece
3. **Edge Extraction**: Extracts edge segments between corners for feature analysis
4. **Shape Classification**: Advanced edge shape analysis using curvature profiles and mathematical classification
5. **Color Feature Extraction**: Captures LAB color sequences along edges for matching
6. **Piece Classification**: Classifies pieces as corner, edge, or middle pieces based on their edges
7. **Visualization**: Provides comprehensive visualization tools for debugging and analysis

## Features

- **Robust Piece Detection**: Uses adaptive thresholding and morphological operations to handle various lighting conditions
- **Accurate Corner Detection**: Implements polar distance profile analysis to find true puzzle piece corners
- **Advanced Shape Classification**: Multi-metric edge analysis combining curvature profiles and distance measurements
- **Mathematical Edge Types**: Uses proper mathematical terminology (flat, convex, concave) with sub-type classification
- **Color Feature Extraction**: LAB color space edge descriptors with confidence scoring for robust matching
- **Object-Oriented Design**: Clean architecture with `Piece` and `EdgeSegment` classes
- **Parallel Processing**: Utilizes multiprocessing for efficient analysis of multiple pieces
- **Comprehensive Visualization**: Debug views for each processing step including detailed shape and color analysis

## Installation

1. Clone the repository:
```bash
git clone https://github.com/jijiduc/mdmv-puzzle-solver.git
cd mdmv-puzzle-solver
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
python main.py --image <image path file>
```

### Command Line Options

- `--image`: Path to the input image containing puzzle pieces
- `--debug`: Enable debug visualizations (default: True)
- `--output`: Output directory for results (default: debug/)
- `--parallel`: Enable parallel processing (default: True)

### Example with Custom Settings

```bash
python main.py --image <image path file> --output results/ --debug
```

## Project Structure

```
mdmv-puzzle-solver/
├── main.py                 # Main entry point
├── requirements.txt        # Python dependencies
├── picture/               # Sample puzzle images
│   ├── puzzle_6/
│   ├── puzzle_24-1/
│   ├── puzzle_24-2/
│   └── puzzle_49-1/
├── debug/                 # Debug output directory
└── src/
    ├── config/
    │   └── settings.py    # Configuration settings
    ├── core/
    │   ├── piece.py       # Piece and EdgeSegment classes
    │   ├── image_processing.py      # Image preprocessing
    │   ├── piece_detection.py       # Piece detection and analysis
    │   ├── corner_detection_proper.py # Corner detection algorithm
    │   └── geometry.py              # Geometric utilities
    ├── features/
    │   ├── color_analysis.py        # Color feature extraction
    │   ├── edge_extraction.py       # Edge extraction utilities
    │   └── shape_analysis.py        # Advanced shape analysis and classification
    └── utils/
        ├── visualization.py         # Visualization tools
        ├── corner_analysis.py       # Corner analysis utilities
        ├── io_operations.py         # File I/O operations
        └── parallel.py              # Parallel processing utilities
```

## Algorithm Details

### 1. Piece Detection

The piece detection algorithm (`src/core/piece_detection.py`) uses:
- Adaptive thresholding to handle varying lighting conditions
- Morphological operations (closing) to fill gaps
- Contour detection with area filtering
- Bounding box extraction for individual pieces

### 2. Corner Detection

The corner detection algorithm (`src/core/corner_detection_proper.py`) implements:
- Contour to polar coordinate conversion
- Distance profile smoothing
- Peak detection to find potential corners
- Rectangular pattern evaluation to select the best 4 corners
- Scoring based on:
  - 90° angle spacing between corners
  - Distance similarity for opposite sides
  - Overall rectangular shape quality

### 3. Edge Extraction & Shape Classification

Edges are extracted between detected corners and analyzed using advanced shape classification:
- Traces contour points between corner positions
- Calculates curvature profiles using discrete geometry
- Performs multi-metric classification combining:
  - Perpendicular distance analysis from reference line
  - Curvature-based feature extraction with corner artifact filtering
  - Weighted scoring system (60% distance, 40% curvature)
- Classifies edges into mathematical categories:
  - **Flat**: Low deviation and curvature (< 0.5% threshold)
  - **Convex**: Outward bulging edges (puzzle tabs)
  - **Concave**: Inward curving edges (puzzle sockets)
- Determines sub-types:
  - **Symmetric**: Regular, balanced shapes
  - **Asymmetric**: Irregular, unbalanced shapes
- Provides confidence scoring (0-1) for classification reliability

### 4. Edge Color Feature Extraction

Color descriptors are extracted for each edge to support future matching:
- Samples colors along edge points with configurable radius
- Converts BGR to LAB color space for perceptual accuracy
- Calculates confidence based on local color variance
- Normalizes color sequences for consistent matching
- Stores both color sequences and confidence scores per edge
- Provides comprehensive visualizations:
  - Color strips showing actual edge colors
  - Polar color wheel visualization
  - Statistical analysis of color features

### 5. Piece Classification

Pieces are classified based on their edge characteristics:
- **Corner pieces**: 2 flat edges
- **Edge pieces**: 1 flat edge
- **Middle pieces**: 0 flat edges (all interlocking)

## Output

The program generates comprehensive debug visualizations organized by processing stage:

### Core Processing
1. **01_input/**: Original image and metadata
2. **02_preprocessing/**: Binary thresholding and morphological operations
3. **03_detection/**: Contour detection and piece extraction
4. **04_pieces/**: Individual piece images with metadata
5. **05_geometry/**: Corner detection analysis and geometric features

### Advanced Analysis
6. **06_features/**: Feature extraction and analysis
   - **shape/**: Detailed shape analysis for each piece:
     - **Edge profiles**: Curvature plots for all 4 edges
     - **Classification**: Visual edge type classification with color coding
     - **Shape metrics**: Symmetry scores, confidence levels, and radar charts
     - **Summary**: Overall statistics and edge type distribution
   - **color/pieces/**: Edge color descriptor visualizations:
     - **Color strips**: Color sequences along each edge with confidence scores
     - **Color wheel**: Polar visualization of colors around piece perimeter
     - **Summary card**: Comprehensive color analysis including mean colors, variance, and matching hints

### Final Results
7. **07_piece_classification/**: Piece type classification (corner/edge/middle)
8. **08_matching/**: Edge matching results (future implementation)
9. **09_assembly/**: Final assembly output (future implementation)

## Data Model

### Piece Class

```python
class Piece:
    index: int                    # Unique identifier
    image: np.ndarray            # Cropped piece image
    mask: np.ndarray             # Binary mask
    corners: List[Tuple[int, int]]  # 4 corner coordinates
    edges: List[EdgeSegment]     # Edge segments between corners
    piece_type: str              # 'corner', 'edge', or 'middle'
    bbox: Tuple[int, int, int, int]  # Bounding box in original image
```

### EdgeSegment Class

```python
@dataclass
class EdgeSegment:
    points: List[Tuple[int, int]]    # Contour points
    corner1: Tuple[int, int]         # Start corner
    corner2: Tuple[int, int]         # End corner
    edge_type: str                   # 'flat', 'convex', 'concave'
    sub_type: Optional[str]          # 'symmetric', 'asymmetric', or None
    confidence: float                # Classification confidence (0-1)
    deviation: float                 # Maximum deviation from reference line
    length: int                      # Number of edge points
    curvature: Optional[float]       # Average curvature measure
    color_sequence: List[List[float]] # LAB color values along edge
    confidence_sequence: List[float]  # Color sampling confidence scores
```

## Future Enhancements

- [ ] Edge matching using Dynamic Time Warping (DTW) with shape and color descriptors
- [ ] Automatic puzzle assembly algorithm
- [ ] Real-time piece detection from camera feed
- [ ] Support for irregular puzzle shapes

## Authors

[Jeremy Duc](https://github.com/jijiduc) & [Alexandre Venturi](https://github.com/mastermeter)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Model-Driven Machine Vision course at HEI
- OpenCV community for computer vision tools
- NumPy and SciPy for numerical computations