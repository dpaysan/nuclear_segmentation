import cv2
import numpy as np
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
    # labels = label(labels, connectivity=2)
    return labels
