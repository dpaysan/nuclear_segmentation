import cv2
import numpy as np
from scipy import ndimage as ndi


def apply_morphology_transformation(
    img: np.ndarray, mode: str = "opening", kernel_size: int = 3, iterations: int = 1
):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    if mode == "opening":
        return cv2.morphologyEx(
            img, cv2.MORPH_OPEN, kernel=kernel, iterations=iterations
        )
    elif mode == "closing":
        return cv2.morphologyEx(
            img, cv2.MORPH_CLOSE, kernel=kernel, iterations=iterations
        )
    elif mode == "dilate":
        return cv2.morphologyEx(
            img, cv2.MORPH_DILATE, kernel=kernel, iterations=iterations
        )
    elif mode == "erode":
        return cv2.morphologyEx(
            img, cv2.MORPH_ERODE, kernel=kernel, iterations=iterations
        )
    else:
        raise RuntimeError("Unknown transformation mode: {} .".format(mode))


def binary_fill_holes_layerwise(img: np.ndarray):
    if len(img.shape) == 2:
        filled = ndi.binary_fill_holes(img)
    else:
        filled = []
        for i in range(len(img)):
            filled.append(ndi.binary_fill_holes(img[i]))
        filled = np.ndarray(filled)
    return filled
