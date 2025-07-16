import cv2
import numpy as np

def generate_saliency_map(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    saliency = cv2.Laplacian(gray, cv2.CV_64F)
    saliency = np.absolute(saliency)
    saliency = (saliency / saliency.max() * 255).astype(np.uint8)
    return saliency
