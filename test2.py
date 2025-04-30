import cv2
from matplotlib import pyplot as plt
import numpy as np

# keep in mind that open CV loads images as BGR not RGB
image = cv2.imread("picture/puzzle_49-1/b-3.jpg")

def find_center(image, contours):
    list_center = []
    for c in contours:
        # compute the center of the contour
        m = cv2.moments(c)
        cx = int(m["m10"] / m["m00"])
        cy = int(m["m01"] / m["m00"])
        list_center.append((cx, cy))
    return list_center


## RESIZE IMAGE
scale = 30
newWidth = int(image.shape[1] * scale / 100)
newHeight = int(image.shape[0] * scale / 100)
newDimension = (newWidth, newHeight)
# resize image
resizedImage = cv2.resize(image, newDimension, interpolation=cv2.INTER_AREA)

## CONVERT TO GRAYSCALE
# convert image to grayscale
grayImage=cv2.cvtColor(resizedImage, cv2.COLOR_BGR2GRAY)

# THRESHOLD
estimatedThreshold, thresholdImage=cv2.threshold(grayImage,30,255,cv2.THRESH_BINARY)


## DETERMINE CONTOURS AND FILTER THEM
contours, _ = cv2.findContours(thresholdImage, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

centers = find_center(resizedImage, contours)


# make a copy of the resized image since we are going to draw contours on the resized image
resizedImageCopy = np.copy(resizedImage)

for i, c in enumerate(contours):
    areaContour = cv2.contourArea(c)
    if areaContour < 2000 or 100000 < areaContour:
        continue
    cv2.drawContours(resizedImageCopy, contours, i, (255, 10, 255), 5)

# display the original image with contours
cv2.imshow('Image', resizedImageCopy)
cv2.waitKey(0)
cv2.destroyAllWindows()
