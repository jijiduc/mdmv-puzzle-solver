# mdmv-puzzle-solver

Project of Model Driven Machine Vision (Course 206.2) course, lectured by Professor  Louis Lettry.

## Project Context

This project aims to develop a solution for outputting a complete image file from a photograph containing dispersed puzzle pieces.\
From a broader perspective, such a task could arise in real-life situation for example when aiming to develop a solution for reassembling broken 2D sculptures by matching individual pieces together.

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
- Use a unified background color that contrasts with the puzzle pieces (especially important for monochrome puzzles) : our current choice is a black piece of fabric
- Ensure all pieces are visible and well-separated in the input image

The device chosen for taking the images is a Samsung Galaxy S24 Ultra.

Key definitions:

- **Piece**: A 2D form bounded by a defined perimeter
- **Contour** : A contour is a continuus and finite line forming the perimeter of a piece
- **Border**: An element formed from an n-cutting from the perimeter of a piece
- **straight border** : A border where the shape of is a straight line
- **bump border** : A border where there is a bump coming out of a straight border
- **cavity border** : A border where there is a cavity into a straight border
- **Piece Differentiation**: Pieces can be distinguished by analyzing their boundaries and colors
- **Puzzle**: A finite ordered group of pieces whose boundaries match perfectly with each other without gaps
- **Classic Puzzle**: A puzzle with pieces of varied forms, bounded by 4 segments forming a rectangle, with the assembled pieces depicting a colored image
- **Monochrome Puzzle**: A puzzle with a unicolor motif created from the assembled pieces
- **Form Puzzle**: A puzzle whose pieces all have the same shape

## System Modeling

1. **Image Acquisition**\
   1.1. Ensure pieces don't touch or overlap in the captured image\
   1.2. Determine optimal number of images needed\
   1.3. Control environmental factors: uniform luminosity and unified background\

2. **Keypoints Detection** (using OpenCV)\
   2.1. Compare form, color, and piece orientation\
      2.1.1. Determine optimal n-value for the n-cutting of pieces\

3. **Keypoints Description**\
   3.1. Develop color and form descriptors\

4. **Matching**\
   4.1. Implement algorithms to identify the best piece matches\

5. **Reconstruction**\
   5.1. Algorithmically find best matching pieces\
   5.2. Develop method for combining matched pieces into larger units\
   5.3. Implement search algorithms (DFS/backtracking) to evaluate reconstructed puzzles\
   5.4. Develop scoring system for evaluating reconstruction quality\

6. **Image Generation**\
   6.1. Realign pieces based on orientation determined in step 2.1.1\
   6.2. Digitally stitch pieces together at matching boundaries\
   6.3. Generate final complete image file of the assembled puzzle\
   6.4. Apply post-processing to smooth transitions between pieces if necessary\

## Project Timeline

### Week 1

- Set up project repository and documentation
- Do points 1 & 2

### Week 2

- Do point 3

### Week 3

- Do point 4 & 5

### Week 4

- Peer presentation
- Do point 6

### Week 5

- At disposition if necessary

### Week 6

- At disposition if necessary

## Implementation Tools

- OpenCV for image processing and computer vision tasks
- Python 3.13.2 as the primary programming language
- Additional libraries as needed for specific algorithms (under the aproval of Prof.Lettry)

## Evaluation Metrics

- Accuracy of piece matching
- Completeness of puzzle reconstruction
- Quality of the final generated image
- Processing time
- Robustness across different puzzle types and piece configurations
- Ability to handle puzzles with varying numbers of pieces

## Team Members

[Jeremy Duc](https://github.com/jijiduc) & [Alexandre Venturi](https://github.com/mastermeter)

## References

[Project description](https://isc.hevs.ch/learn/pluginfile.php/5191/mod_resource/content/0/Project.pdf), the provided PDF document in the course
