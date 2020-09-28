def filter_nuclei_by_size(
    nuclei_dicts,
    microns_per_pixel: float,
    min_volume: float = None,
    max_volume: float = None,
) -> Tuple[List]:
    filtered = []
    filtered_out = []
    for i in range(len(nuclei_dicts)):
        volume = nuclei_dicts[i]["props"].filled_area * (microns_per_pixel) ** 3
        print(volume)
        if min_volume is not None and volume < min_volume:
            filtered_out.append(nuclei_dicts[i])
        elif volume is not None and volume > max_volume:
            filtered_out.append(nuclei_dicts[i])
        else:
            filtered.append(nuclei_dicts[i])
    return filtered, filtered_out
