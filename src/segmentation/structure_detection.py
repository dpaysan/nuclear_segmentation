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


def get_hc_ec_structure_maps_by_thresholding(img, object_mask, k=0.6):
    tmp_img = copy.deepcopy(img)
    masked_img = ma.masked_array(tmp_img, ~(object_mask.astype(bool)))
    threshold = masked_img.min() + k * (masked_img.max() - masked_img.min())
    hc_mask = masked_img > threshold
    ec_mask = masked_img <= threshold
    structure_dict = {'cell_img':img, 'masked_img':masked_img, 'hc_mask' : hc_mask,
                          'ec_mask' : ec_mask}
    return structure_dict
