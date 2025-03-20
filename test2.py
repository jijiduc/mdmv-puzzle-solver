import cv2
from matplotlib import pyplot as plt
import numpy as np

# keep in mind that open CV loads images as BGR not RGB
image = cv2.imread("picture/puzzle_49-1/b-2.jpg")
cv2.imshow('Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

## RESIZE IMAGE
# scale in percentage
scale = 30
newWidth = int(image.shape[1] * scale / 100)
newHeight = int(image.shape[0] * scale / 100)
newDimension = (newWidth, newHeight)
# resize image
resizedImage = cv2.resize(image, newDimension, interpolation=cv2.INTER_AREA)
cv2.imshow('Image', resizedImage)
cv2.waitKey(0)
cv2.destroyAllWindows()
# save the resized image
cv2.imwrite("pieces/resizedParts.png", resizedImage, [cv2.IMWRITE_PNG_COMPRESSION, 0])

## CONVERT TO GRAYSCALE
# convert image to grayscale
grayImage=cv2.cvtColor(resizedImage, cv2.COLOR_BGR2GRAY)
# display converted image
cv2.imshow('Image', grayImage)
cv2.waitKey(0)
cv2.destroyAllWindows()
# save the transformed image
cv2.imwrite("pieces/resizedPartsGray.png", grayImage, [cv2.IMWRITE_PNG_COMPRESSION, 0])

# THRESHOLD
# https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
#estimatedThreshold, thresholdImage=cv2.threshold(grayImage,50,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
estimatedThreshold, thresholdImage=cv2.threshold(grayImage,100,255,cv2.THRESH_BINARY)
# display converted image
cv2.imshow('Image', thresholdImage)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("pieces/resizedPartsThreshold.png", thresholdImage, [cv2.IMWRITE_PNG_COMPRESSION, 0])

"""
# PLOT HISTOGRAM OF THRESHOLDED AND GRAYSCALE IMAGES
plt.figure(figsize=(14, 12))
plt.subplot(2,2,1), plt.imshow(grayImage,'gray'), plt.title('Grayscale Image')
plt.subplot(2,2,2), plt.hist(grayImage.ravel(), 256), plt.title('Color Histogram of Grayscale Image')
plt.subplot(2,2,3), plt.imshow(thresholdImage,'gray'), plt.title('Binary (Thresholded)  Image')
plt.subplot(2,2,4), plt.hist(thresholdImage.ravel(),256), plt.title('Color Histogram of Binary (Thresholded) Image')
plt.savefig('fig1.png')
plt.show()
"""

## DETERMINE CONTOURS AND FILTER THEM
contours, hierarchy = cv2.findContours(thresholdImage, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

# make a copy of the resized image since we are going to draw contours on the resized image
resizedImageCopy = np.copy(resizedImage)

# draw all contours with setting the parameter to -1
# but if you use this function, you should comment the for loop below
# cv2.drawContours(resizedImageCopy,contours,-1,(0,0,255),2)
# filter contours
for i, c in enumerate(contours):
    areaContour = cv2.contourArea(c)
    if areaContour < 2000 or 100000 < areaContour:
        continue
    cv2.drawContours(resizedImageCopy, contours, i, (255, 10, 255), 1)

# display the original image with contours
cv2.imshow('Image', resizedImageCopy)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("pieces/resizedPartsContours.png", resizedImageCopy, [cv2.IMWRITE_PNG_COMPRESSION, 0])