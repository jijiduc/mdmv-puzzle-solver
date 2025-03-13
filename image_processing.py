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
    # Convertir en niveaux de gris
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Appliquer un flou gaussien pour réduire le bruit
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Appliquer CLAHE pour améliorer le contraste
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(blurred)

# function to find the contour of the pieces on the unified background, mixing otsu thresholding and canny edge detection and using pre-processed image function and external contour retrieval mode
def find_contour(image):
    """Finds the contour of the pieces on the unified background."""
    # Prétraitement
    pre_processed = pre_process_image(image)
    
    # Seuillage d'Otsu
    _, threshold = cv2.threshold(pre_processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Opérations morphologiques pour éliminer le bruit et consolider les contours
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morphed = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)
    
    # Détection de contours avec des paramètres moins sensibles
    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filtrage préliminaire des contours trop petits
    min_contour_area = 500  # Ajustez selon vos besoins
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
    
    return filtered_contours

def show_contour(image, contours):
    """Shows the contour of the pieces on the image."""
    cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
    show_image(image)
