import copy

from skimage import filters, restoration, feature
import numpy as np


def remove_noise_layer_from_3d_img_by_canny(img, sigma: float) -> np.ndarray:
    filtered = []
    for i in range(len(img)):
        if feature.canny(img[i], sigma=sigma).any():
            filtered.append(img[i])
    filtered = np.ndarray(filtered)
    return filtered


def remove_layers_outside_roi(img, object_mask):
    masked_img= copy.deepcopy(img)
    for j in range(len(object_mask)):
        if not object_mask[j].astype(bool).any():
            masked_img[j] = np.zeros_like(masked_img[j])
    return masked_img


def denoise_img_bilateral(
    img: np.ndarray,
    window_size: int = None,
    sigma_spatial: float = 1,
    sigma_color: float = None,
    multichannel=False,
) -> np.ndarray:
    if len(img.shape) == 3 and not multichannel:
        denoised_img = []
        for i in range(len(img)):
            denoised_img.append(
                restoration.denoise_bilateral(
                    img[i],
                    win_size=window_size,
                    sigma_spatial=sigma_spatial,
                    sigma_color=sigma_color,
                    multichannel=multichannel,
                )
            )
        denoised_img = np.array(denoised_img)
    else:
        denoised_img = restoration.denoise_bilateral(
            img,
            win_size=window_size,
            sigma_spatial=sigma_spatial,
            sigma_color=sigma_color,
            multichannel=multichannel,
        )
    return denoised_img


def filter_img_median(img) -> np.ndarray:
    filtered = filters.median(img)
    return filtered


def filter_img_gaussian(img, sigma: float = 1.0) -> np.ndarray:
    filtered = filters.gaussian(img, sigma=sigma)
    return filtered
