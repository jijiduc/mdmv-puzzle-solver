import cv2


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

# function to pre-process the image, to enhance the contrast
def pre_process_image(image):
    """Pre-processes the image to enhance the contrast."""
    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)

# function to find the contour of the pieces on the unified background, mixing otsu thresholding and canny edge detection and using pre-processed image function and external contour retrieval mode
def find_contour(image):
    """Finds the contour of the pieces on the unified background."""
    # pre-process the image
    pre_processed = pre_process_image(image)
    # apply Otsu thresholding
    _, threshold = cv2.threshold(pre_processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # apply Canny edge detection
    edges = cv2.Canny(threshold, 30, 200)
    # find the contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def show_contour(image, contours):
    """Shows the contour of the pieces on the image."""
    cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
    show_image(image)
