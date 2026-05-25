import cv2
import numpy as np

def resize_image(image, size=(224, 224)):
    """Resizes image to target dimensions."""
    return cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)

def reduce_noise(image, sigma=1.0):
    """Applies Gaussian smoothing for noise reduction."""
    return cv2.GaussianBlur(image, (0, 0), sigma)

def normalize_pixels(image):
    """Normalizes pixel values to [0, 1] range."""
    return image.astype(np.float32) / 255.0

def full_preprocess(image_path):
    """Full preprocessing pipeline for packaging images."""
    img = cv2.imread(image_path)
    img = resize_image(img)
    img = reduce_noise(img)
    img = normalize_pixels(img)
    return img
