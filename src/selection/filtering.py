from typing import Tuple, List, Any, Iterable
from abc import ABC
import numpy as np
from src.utils.general import intersection
import cv2


class Filter(object):
    def __init__(self):
        super().__init__()

    def filter(self, **kwargs) -> bool:
        raise NotImplementedError


class PropertyFilter(Filter, ABC):
    def __init__(self, properties: Any):
        super().__init__()
        self.properties = properties

    def set_properties(self, properties: Any):
        self.properties = properties

    def filter(self, **kwargs) -> bool:
        raise NotImplementedError


class AreaFilter(PropertyFilter, ABC):
    def __init__(self, properties: Any, thresholds: Any, threshold_unit_pp: float):
        super().__init__(properties)
        if isinstance(thresholds, float) or isinstance(thresholds, int):
            self.min_area = thresholds
            self.max_area = np.infty
        elif isinstance(thresholds, Iterable):
            self.min_area = thresholds[0]
            self.max_area = thresholds[1]
        else:
            raise RuntimeError("Thresholds must be Iterable of size 2 or a float.")
        self.threshold_unit_pp = threshold_unit_pp

    def set_properties(self, properties: Any):
        super().set_properties(properties)

    def filter(self, **kwargs) -> bool:
        if (
            self.properties.area * self.threshold_unit_pp < self.min_area
            or self.properties.area * self.threshold_unit_pp > self.max_area
        ):
            return False
        else:
            return True


class AspectRatioFilter(PropertyFilter, ABC):
    def __init__(self, properties: Any, thresholds: Any):
        super().__init__(properties)
        if isinstance(thresholds, float):
            self.min_ar = thresholds
            self.max_ar = 1.0
        elif isinstance(thresholds, Iterable):
            self.min_ar = thresholds[0]
            self.max_ar = thresholds[1]
        else:
            raise RuntimeError("Thresholds must be Iterable of size 2 or a float.")

    def set_properties(self, properties: Any):
        super().set_properties(properties)

    def filter(self, **kwargs) -> bool:
        min_dim = min(self.properties.image.shape[-1], self.properties.image.shape[-2])
        max_dim = max(self.properties.image.shape[-1], self.properties.image.shape[-2])
        ar = min_dim / max_dim
        if ar < self.min_ar or ar > self.max_ar:
            return False
        else:
            return True


class SolidityFilter(PropertyFilter):
    def __init__(self, properties: Any, thresholds: Any):
        super().__init__(properties)
        if isinstance(thresholds, float):
            self.min_solidity = thresholds
            self.max_solidity = 1.0
        elif isinstance(thresholds, Iterable):
            self.min_solidity = thresholds[0]
            self.max_solidity = thresholds[1]
        else:
            raise RuntimeError("Thresholds must be Iterable of size 2 or a float.")

    def set_properties(self, properties: Any):
        super().set_properties(properties)

    def filter(self, **kwargs) -> bool:
        solidity = self.properties.solidity
        if solidity < self.min_solidity or solidity > self.max_solidity:
            return False
        else:
            return True


class EccentricityFilter(PropertyFilter):
    def __init__(self, properties: Any, thresholds: Any):
        super().__init__(properties)
        if isinstance(thresholds, float):
            self.min_eccentricity = thresholds
            self.max_eccentricity = 1.0
        elif isinstance(thresholds, Iterable):
            self.min_eccentricity = thresholds[0]
            self.max_eccentricity = thresholds[1]
        else:
            raise RuntimeError("Thresholds must be Iterable of size 2 or a float.")

    def set_properties(self, properties: Any):
        super().set_properties(properties)

    def filter(self, **kwargs) -> bool:
        solidity = self.properties.solidity
        if solidity < self.min_eccentricity or solidity > self.max_eccentricity:
            return False
        else:
            return True


class ConfocalShiftFilter(Filter):
    """ Class to detect a confocal shift in the data."""

    def __init__(self, threshold: float):
        super().__init__()
        self.threshold = threshold

    def filter(self, input: np.ndarray, axis: int = 0, **kwargs) -> bool:
        idx = np.zeros_like(input.shape)
        hist1 = cv2.calcHist(input[idx])
        for i in range(1, len(input.shape[axis])):
            idx[i] = i
            hist2 = cv2.calcHist(input[idx])
            chi2 = cv2.compareHist(hist1, hist2)
            if chi2 > self.threshold:
                return False
            else:
                hist1 = hist2


class PropertyFilterPipeline(PropertyFilter):
    def __init__(self, filter_list: Iterable[Filter]):
        super().__init__(None)
        self.filter_list = filter_list

    def set_properties(self, properties):
        for filter in self.filter_list:
            filter.set_properties(properties)

    def filter(self, **kwargs) -> bool:
        for filter in self.filter_list:
            if not filter.filter(**kwargs):
                return False
        return True


class DeadCellFilter(Filter):
    def __init__(self):
        super().__init__()

    def filter(self, **kwargs) -> bool:
        raise NotImplementedError

# def filter_nuclei_by_size(
#     nuclear_crops,
#     microns_per_pixel: float,
#     min_volume: float = None,
#     max_volume: float = None,
#     **kwargs
# ) -> Tuple[list, list]:
#     filtered_idc = []
#     filtered_out_idc = []
#     for i in range(len(nuclear_crops)):
#         volume = nuclear_crops[i]["props"].filled_area * (microns_per_pixel) ** 3
#         if min_volume is not None and volume < min_volume:
#             filtered_out_idc.append(i)
#         elif max_volume is not None and volume > max_volume:
#             filtered_out_idc.append(i)
#         else:
#             filtered_idc.append(i)
#     return filtered_idc, filtered_out_idc
#
#
# def filter_nuclei(nuclear_crops: List[dict], mode: str, **kwargs):
#     if mode == "size":
#         return filter_nuclei_by_size(nuclear_crops=nuclear_crops, **kwargs)
#     elif mode == "solidity":
#         return filter_nuclei_by_solidity(nuclear_crops=nuclear_crops, **kwargs)
#     elif mode == "aspect_ratio":
#         return filter_nuclei_by_aspect_ratio(nuclear_crops, **kwargs)
#     elif mode == "all":
#         return intersection(
#             filter_nuclei_by_solidity(nuclear_crops=nuclear_crops, **kwargs),
#             filter_nuclei_by_size(nuclear_crops=nuclear_crops, **kwargs),
#         )
#     else:
#         raise RuntimeError("Unknown filter mode: {}".format(mode))
#
#
# def filter_nuclei_by_solidity(
#     nuclear_crops: List[dict], threshold: float = 0.8, **kwargs
# ):
#     filtered_idc = []
#     filtered_out_idc = []
#     for i in range(len(nuclear_crops)):
#         solidity = nuclear_crops[i]["props"].solidity
#         if solidity > threshold:
#             filtered_idc.append(i)
#         else:
#             filtered_out_idc.append(i)
#     return filtered_idc, filtered_out_idc
#
#
# def filter_nuclei_by_aspect_ratio(
#     nuclear_crops: List[dict], threshold: float = 0.8, **kwargs
# ):
#     filtered_idc = []
#     filtered_out_idc = []
#     for i in range(len(nuclear_crops)):
#         _, width, height = nuclear_crops[i]["image"].shape
#         if min(width, height) / max(width, height) < threshold:
#             filtered_out_idc.append(i)
#         else:
#             filtered_idc.append(i)
#     return filtered_idc, filtered_out_idc
