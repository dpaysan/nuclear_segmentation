import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage import segmentation
from skimage.filters import threshold_otsu
from skimage.feature import canny
from skimage.segmentation import watershed
from skimage.measure import label
from scipy import ndimage as ndi


def get_watershed_labels_with_fg_bg_masks(
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


def get_watershed_labels_distance_transform(binary_img:np.ndarray, relative_threshold:float=0.0)-> np.ndarray:
    distance = distance_transform_edt(binary_img)
    markers = distance > relative_threshold * distance.max()
    markers = label(markers)
    labels = watershed(distance, markers=markers, mask=binary_img)
    return labels


def label_single_object_image(
        img: np.ndarray, kernel_size: int = 3, iterations: int = 2
):
    img = np.squeeze(img)
    kernel = np.ones((kernel_size, kernel_size))
    thresh = threshold_otsu(img)
    binary = np.uint8(img > thresh)
    closing = cv2.morphologyEx(
        binary.transpose(), cv2.MORPH_CLOSE, kernel=kernel, iterations=iterations
    )
    opening = cv2.morphologyEx(
        closing.transpose(), cv2.MORPH_OPEN, kernel=kernel, iterations=iterations
    )
    labeled_img = label(opening)
    return labeled_img


def get_edge_based_segmentation(
        img: np.ndarray,
        sigma: float = 3.0,
        low_threshold: float = None,
        high_threshold: float = None,
):
    edges = canny(
        img, sigma=sigma, low_threshold=low_threshold, high_threshold=high_threshold
    )
    filled_objects = ndi.binary_fill_holes(edges)
    return filled_objects


def get_chan_vese_based_object_mask_2d(
        img: np.ndarray, max_iter: int = 500, fill_holes: bool = True,
):
    object_mask = segmentation.chan_vese(img, max_iter=max_iter)
    if fill_holes:
        object_mask = ndi.binary_fill_holes(object_mask)
    return object_mask


def label_objects(img: np.ndarray):
    labeled_img = label(img)
    return labeled_img


def get_chan_vese_based_object_mask_3d(
        img: np.ndarray,
        smoothing: float = 1.0,
        iterations: int = 500,
        init_level_set: str = "circle",
        lambda1: float = 1.0,
        lambda2: float = 1.0,
) -> np.ndarray:
    object_mask = segmentation.morphological_chan_vese(
        image=img,
        smoothing=smoothing,
        iterations=iterations,
        init_level_set=init_level_set,
        lambda1=lambda1,
        lambda2=lambda2,
    )
    return object_mask
