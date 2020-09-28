import numpy as np
from typing import List
from skimage.measure import regionprops


def get_conservative_nuclear_crops(
    labels: np.ndarray,
    intensity_image: np.ndarray,
    xbuffer: int = 0,
    ybuffer: int = 0,
    zbuffer: int = 1,
) -> List[dict]:
    nuclear_properties = regionprops(labels, intensity_image=intensity_image)
    nuclei_dicts = []

    for properties in nuclear_properties:
        depth, width, height = intensity_image.shape
        zmin, xmin, ymin, zmax, xmax, ymax = properties.bbox
        zmin = max(0, zmin - zbuffer)
        xmin = max(0, xmin - xbuffer)
        ymin = max(0, ymin - ybuffer)
        zmax = min(zmax + zbuffer + 1, depth)
        xmax = min(xmax + xbuffer + 1, width)
        ymax = min(ymax + ybuffer + 1, height)
        nuclei_dicts.append(
            {
                "image": intensity_image[zmin:zmax, xmin:xmax, ymin:ymax],
                "props": properties,
            }
        )
    return nuclei_dicts
