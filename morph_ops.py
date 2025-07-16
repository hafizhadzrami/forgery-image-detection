import cv2
import numpy as np

def apply_morph_ops(mask):
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(mask, kernel, iterations=2)
    eroded = cv2.erode(dilated, kernel, iterations=1)
    return eroded
