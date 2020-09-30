import cv2
import numpy as np


def get_sure_foreground_from_distance(
    img, threshold: float, distance_type: str = "l2", mask_size: int = 3
):

    if distance_type == "l1":
        distance = cv2.DIST_L1
    elif distance_type == "l2":
        distance = cv2.DIST_L2
    elif distance_type == "c":
        distance = cv2.DIST_C
    else:
        raise RuntimeError("Unknown distance type: {}".format(distance_type))
    dist_transform = cv2.distanceTransform(img, distance, maskSize=mask_size)
    ret2, sure_fg = cv2.threshold(
        dist_transform, threshold * dist_transform.max(), 255, 0
    )
    return np.uint8(sure_fg)
