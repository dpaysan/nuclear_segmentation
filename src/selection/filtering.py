from typing import Tuple, List


def filter_nuclei_by_size(
    nuclear_crops,
    microns_per_pixel: float,
    min_volume: float = None,
    max_volume: float = None,
) -> Tuple[list, list]:
    filtered_idc = []
    filtered_out_idc = []
    for i in range(len(nuclear_crops)):
        volume = nuclear_crops[i]["props"].filled_area * (microns_per_pixel) ** 3
        if min_volume is not None and volume < min_volume:
            filtered_out_idc.append(i)
        elif max_volume is not None and volume > max_volume:
            filtered_out_idc.append(i)
        else:
            filtered_idc.append(i)
    return filtered_idc, filtered_out_idc


def filter_nuclei(nuclear_crops: List[dict], mode: str, **kwargs):
    if mode == "size":
        return filter_nuclei_by_size(nuclear_crops=nuclear_crops, **kwargs)
    else:
        raise RuntimeError('Unknown filter mode: {}'.format(mode))
