import numpy as np
from skimage import exposure


def rescale_intensities(img, qs=None, out_range=None):
    if qs is None:
        qs = [0.05, 99.5]
    if out_range is None:
        out_range = np.float32
    vmin, vmax = np.percentile(img, q=qs)
    rescaled_img = exposure.rescale_intensity(img, in_range=(vmin, vmax),
                                                             out_range=out_range)

    return rescaled_img