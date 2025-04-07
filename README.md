# mdmv-puzzle-solver

Project of Model Driven Machine Vision (Course 206.2) course, lectured by Professor Louis Lettry.

## Project Context

This project aims to develop a robust solution for outputting a complete image file from a photograph containing dispersed puzzle pieces. From a broader perspective, such a task could arise in real-life situations when aiming to develop a solution for reassembling broken 2D artifacts by matching individual pieces together.

## Problem Analysis

To approach this problem, we consider several key questions:

- **Image Acquisition**: How should we capture images of the pieces to ensure consistency and accuracy?
- **Piece Definition**: What constitutes a piece in our system? How do we define a complete puzzle?
- **Differentiation**: What characteristics distinguish one piece from another?
- **Assembly Logic**: What algorithms and methods can we use to reassemble pieces into a correct puzzle?
- **Project Boundaries**: What constraints exist regarding number of pieces, piece shapes, colors, etc.?
- **Solution Scope**: What are the limitations and capabilities of our proposed solution?
- **Constraints**: What implicit/explicit constraints must we consider?

## Problem Formalization

When capturing images of puzzle pieces, we should:

- Minimize variability in photovolumetric parameters
- Use a unified background color that contrasts with the puzzle pieces (especially important for monochrome puzzles): our current choice is a black piece of fabric
- Ensure all pieces are visible and well-separated in the input image

The device chosen for taking the images is a Samsung Galaxy S24 Ultra.

Key definitions:

- **Piece**: A 2D form bounded by a defined perimeter
- **Contour**: A continuous and finite line forming the perimeter of a piece
- **Border**: An element formed from an n-cutting from the perimeter of a piece
- **straight border**: A border where the shape is a straight line
- **tab border**: A border where there is a bump coming out of a straight border
- **pocket border**: A border where there is a cavity into a straight border
- **Piece Differentiation**: Pieces can be distinguished by analyzing their boundaries and colors
- **Puzzle**: A finite ordered group of pieces whose boundaries match perfectly with each other without gaps
- **Classic Puzzle**: A puzzle with pieces of varied forms, bounded by 4 segments forming a rectangle, with the assembled pieces depicting a colored image
- **Monochrome Puzzle**: A puzzle with a unicolor motif created from the assembled pieces
- **Form Puzzle**: A puzzle whose pieces all have the same shape

## Enhanced Solution Approach

Our enhanced solution implements a robust, adaptive detection system:

1. **Multi-Channel Preprocessing**  
   1.1. Analyze image characteristics to determine optimal processing  
   1.2. Process multiple color channels (RGB, HSV, LAB) in parallel  
   1.3. Select the best channel representation for piece detection  
   1.4. Apply adaptive contrast enhancement based on image properties  

2. **Adaptive Parameter Optimization**  
   2.1. Dynamically test multiple parameter combinations  
   2.2. Optimize settings based on image characteristics  
   2.3. Use statistical clustering to determine optimal thresholds  

3. **Enhanced Contour Detection**  
   3.1. Use multiple methods for finding contours  
   3.2. Apply statistical filtering with robust metrics  
   3.3. Implement two-stage filtering to recover valid pieces  
   3.4. Validate contours using shape characteristics  

4. **Multi-Pass Detection**  
   4.1. Run multiple detection passes with different settings  
   4.2. Combine results from different passes  
   4.3. Remove duplicate detections between passes  

5. **Advanced Corner and Border Analysis**  
   5.1. Detect corners using adaptive algorithms  
   5.2. Classify borders as straight, tab, or pocket  
   5.3. Calculate validation scores for piece quality assessment  

6. **Image Reconstruction**  
   6.1. Analyze potential matches between pieces  
   6.2. Realign pieces based on orientation  
   6.3. Generate final complete image  

## Implementation Tools

- OpenCV for image processing and computer vision tasks
- Python 3.13.2 as the primary programming language
- NumPy and SciPy for numerical computations and statistical analysis
- Multiprocessing for parallel execution of detection tasks
- Additional libraries as needed for specific algorithms (under the approval of Prof. Lettry)

## Evaluation Metrics

- Accuracy of piece matching
- Completeness of puzzle reconstruction
- Quality of the final generated image
- Processing time
- Robustness across different puzzle types and piece configurations
- Ability to handle puzzles with varying numbers of pieces

## Usage

Here is how to use the enhanced program:

### Install requirements

```bash
pip install -r requirements.txt
```

### Basic usage

```bash
# Basic usage
python main.py --image picture\puzzle_24-1\b-2.jpg 

# Enhanced usage with all optimizations
python main.py --image picture\puzzle_24-1\b-2.jpg --pieces 24 --adaptive-preprocessing --optimize-parameters --multi-pass --extract

# Analysis of image characteristics 
python main.py --image picture\puzzle_24-1\b-2.jpg --analyze-image

# Adjust mean filtering threshold
python main.py --image puzzle.jpg --mean-threshold 2.0
```

### Command-line options

```bash
Required parameters:
  --image           Path to the puzzle image (required)

Optional parameters:
  --pieces          Expected number of pieces in the puzzle
  --debug-dir       Directory to save debug outputs (default: "debug")
  --extract         Extract individual pieces to separate files
  --extract-dir     Directory to save extracted pieces (default: "extracted_pieces")
  --use-multiprocessing  Use multiprocessing for faster detection
  --view            View results in image windows
  --auto-threshold  Apply both Otsu and adaptive thresholding
  --no-mean-filter  Disable filtering by mean area
  --mean-threshold  Standard deviation threshold for mean filtering (default: 1.5)
  
Enhanced detection options:
  --adaptive-preprocessing  Use adaptive multi-channel preprocessing
  --optimize-parameters     Test multiple parameter combinations and select the best
  --multi-pass             Run multiple detection passes with different settings
  --analyze-image          Analyze image characteristics and optimize settings accordingly
```

## Advanced Features

### Adaptive Preprocessing

The system analyzes the image characteristics and selects the optimal preprocessing approach from multiple color spaces and channels (RGB, HSV, LAB). This enables robust detection across varying lighting conditions and piece colors.

### Parameter Optimization

Instead of fixed parameters, the system can automatically test multiple parameter combinations to find the optimal settings for a specific image. This adapts to different puzzle types and image conditions.

### Multi-Pass Detection

By running multiple detection passes with different settings and combining the results, the system can detect pieces that might be missed by a single approach. This significantly improves detection rates.

### Statistical Filtering

The system uses robust statistical measures (median absolute deviation instead of standard deviation) for more accurate filtering of contours, reducing false negatives and false positives.

### Enhanced Validation

Each detected piece is assigned a validation score based on multiple criteria, allowing for quality assessment and potential filtering of invalid detections.

## Debug Output

The system generates comprehensive debug output in the debug directory:
- Visualizations of each processing stage
- Detected piece visualizations with border type classification
- Detailed metrics and analysis
- Parameter optimization results

## Team Members

[Jeremy Duc](https://github.com/jijiduc) & [Alexandre Venturi](https://github.com/mastermeter)

## References

[Project description](https://isc.hevs.ch/learn/pluginfile.php/5191/mod_resource/content/0/Project.pdf), the provided PDF document in the course.