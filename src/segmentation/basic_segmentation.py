import cv2
import numpy as np
from skimage.filters import threshold_otsu
from skimage.segmentation import watershed
from skimage.measure import label


def get_watershed_labels(
    img: np.ndarray, sure_fg: np.ndarray, sure_bg: np.ndarray, mask: np.ndarray = None
):
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)

    # Watershed will only fill 0 labels and we want it to fill the unknown areas.
    markers = markers + 1

    markers[unknown == 255] = 0
    labels = watershed(img, markers, mask=mask)

    # To remove potential artifacts we ensure that we do not get disconnected areas for the same label
    labels = label(labels)
    return labels


def label_single_object_image(img:np.ndarray, kernel_size:int=3, iterations:int=2):
    kernel = np.ones((kernel_size, kernel_size))
    thresh = threshold_otsu(img)
    binary = np.uint8(img > thresh)
    closing = cv2.morphologyEx(binary.transpose(), cv2.MORPH_CLOSE, kernel=kernel, iterations=iterations)
    opening = cv2.morphologyEx(closing.transpose(), cv2.MORPH_OPEN, kernel=kernel, iterations=iterations)
    labeled_img = label(opening)
    return labeled_img
