import copy
from typing import Tuple, Iterable

from numpy import ma
import numpy as np
import SimpleITK as sitk


def get_structure_by_thresholding(
    img: np.ndarray, threshold, object_mask: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray]:
    # Required due to the memory management of python
    img_copy = copy.deepcopy(img)
    if object_mask is not None:
        img_copy = ma.masked_array(img_copy, ~object_mask)
    structure_mask = img_copy > threshold
    return structure_mask, img


def canny_based_surface_detection(
    img: np.ndarray,
    sigma: Iterable,
    lower_threshold: float = 0.0,
    upper_threshold: float = 20.0,
) -> np.ndarray:
    img = sitk.GetImageFromArray(img)
    surface_boundaries = sitk.CannyEdgeDetection(
        img,
        lowerThreshold=lower_threshold,
        upperThreshold=upper_threshold,
        variance=sigma,
    )
    surface_boundaries = sitk.GetArrayFromImage(surface_boundaries)
    return surface_boundaries


def blob_detection(img: np.ndarray):
    pass
