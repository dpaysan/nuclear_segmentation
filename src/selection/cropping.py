import numpy as np
from typing import List, Tuple
from skimage.measure import regionprops

from src.selection.filtering import PropertyFilter


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


def get_3d_nuclear_crops_from_2d_segmentation(
    labeled_projection: np.ndarray,
    intensity_projection: np.ndarray,
    intensity_image: np.ndarray,
    xbuffer: int = 0,
    ybuffer: int = 0,
    filter_object: PropertyFilter = None,
):
    nuclear_properties = regionprops(
        label_image=labeled_projection, intensity_image=intensity_projection
    )
    nuclei_dicts = []
    for properties in nuclear_properties:
        depth, width, height = intensity_image.shape
        xmin, ymin, xmax, ymax = properties.bbox
        xmin = max(0, xmin - xbuffer)
        ymin = max(0, ymin - ybuffer)
        xmax = min(xmax + xbuffer + 1, width)
        ymax = min(ymax + ybuffer + 1, height)

        # Filter artifact labels by a simple size filter of 2 pixels
        if filter_object is None:
            nuclei_dicts.append(
                {
                    "image": intensity_image[:, xmin:xmax, ymin:ymax],
                    "props": properties,
                }
            )
        else:
            filter_object.set_properties(properties=properties)
            if filter_object.filter(input=intensity_image):
                nuclei_dicts.append(
                    {
                        "image": intensity_image[:, xmin:xmax, ymin:ymax],
                        "props": properties,
                    }
                )

    return nuclei_dicts
