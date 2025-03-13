"""This file is the main file for the project. It is responsible for running the program and handling the pictures inputs."""

import sys
import os
import time
import random
import cv2
import numpy as np
import matplotlib

def read_image(image_path):
    """Reads an image from the given path and returns the image."""
    image = cv2.imread(image_path)
    return image

def show_image(image):
    """Displays the given image."""
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def resize_image(image, width, height):
    """Resizes the given image to the given width and height."""
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

def save_image(image, image_path):
    """Saves the given image to the given path."""
    cv2.imwrite(image_path, image)

# function to find the contour of the pieces on the unified background, using otsu thresholding
def find_contour(image):
    """Finds the contours of the pieces on the image."""
    # grayscale the image
    # cv2.cvtColor(src, code[, dst[, dstCn]]) → dst
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # apply otsu thresholding
    # cv2.threshold(src, thresh, maxval, type[, dst]) → retval, dst
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # find contours
    # cv2.findContours(image, mode, method[, contours[, hierarchy[, offset]]]) → contours, hierarchy
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def show_contour(image, contours):
    """Shows the contour of the pieces on the image."""
    cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
    show_image(image)



def main():
    """Main function for the program."""
    w_1 = read_image("picture/puzzle_24-1/w-1.jpg")
    w_1 = resize_image(w_1, 1250, 1250)
    w_1_contour = find_contour(w_1)
    show_contour(w_1, w_1_contour)


if __name__ == '__main__':
    main()