# mdmv-puzzle-solver
In course 206.2 Model Driven Machine Vision

## Contextualisation
  In the case of a scupture which is broken, be able to match each pieces together. In our case, a 2D sculpture

## Analysis
  How to take the picture ?
  What is a piece ? What is a puzzle ?
  What make differences between 2 pieces ?
  How to reassemble the pieces to form a correct puzzle ?
  What are the boundaries of the project (nunmber of pieces / form of pieces / color )
  What is the scope of the solution ?
  What implicit/explicit constrains are present ?

## Formalization 
  When taking the picture, we should thrive to limit the variabilities into photovolumetrics parameters, the background sould be a unified color (not the same as a the color in a case of a monochrome puzzle)
  A piece is a 3D form which is bounded by a limit
  A border is a element form an n-cutting from the border of a piece
  To make difference between 2 pieces we can analyse thier boundaries, in a case the boundary is the same, analyse the color.
  A puzzle is finite ordered group of pieces whose bound matche perfectly with each other without having space inbetween.
  A classic puzzle is a puzzle with pieces of any forms and is bounded by 4 segments that form a rectangle and the assembled pieces described a image colored.
  A monochrome puzzle is puzzle with an unicolor motif made from assembling the piece.
  A form puzzle is a puzzle whose pieces have the same shape.

## Modelization
1 Image acquisition
  1.1 piece doesn't touch or overlap
  1.2 how many picture
  1.2 environment : uniformity of luminosity / unified background     
2 Keypoints detection : How to recognise piece (OPENCV)
  2.1 comparison on the form and color and piece orientation
    2.1.1  finding a right n value for the n-cutting of pieces 
3 Keypoints description :
  3.1 Color and form descriptor
4 Matching
  4.1 best piece matching to find the corectness
5 Reconstruction
     2.2. Algorithmicaly finding best matching pieces
     2.3. Idea of the best matching pieces together, giving a set of new pieces size the continuing 
     2.4 DFS / backtracking / grading all reconstructed puzzle
6 Image generation
  6.1 Realigning pieces on base of orientation 2.1.1


## Planification
Grading the time of each step
semaine 1
via the built-in tool of github
