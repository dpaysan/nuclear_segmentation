import os
import numpy as np
import tifffile
from nd2reader import ND2Reader
from typing import List


def get_file_list(
    root_dir: str,
    absolute_path: bool = True,
    file_ending: bool = True,
    file_type_filter: str = None,
) -> List:

    assert os.path.exists(root_dir)
    list_of_data_locs = []
    for (root_dir, dirname, filename) in os.walk(root_dir):
        for file in filename:
            if file_type_filter is not None and file_type_filter not in file:
                continue
            else:
                if not file_ending:
                    file = file[: file.index(".")]
                if absolute_path:
                    list_of_data_locs.append(os.path.join(root_dir, file))
                else:
                    list_of_data_locs.append(file)
    return sorted(list_of_data_locs)


def get_image_data_from_bioformat(file: str, file_type: str):
    if file_type == "nd2":
        return nd2_to_npy(file)
    else:
        raise NotImplementedError("Unknown file type: {}".format(file_type))


def nd2_to_npy(nd2_file: str) -> dict:
    """Function that returns a dictionary that includes the images for each channel of the .nd2 file
    and the meta-information"""
    with ND2Reader(nd2_file) as reader:
        metadata = reader.metadata
        width, height, depth = reader.sizes["x"], reader.sizes["y"], reader.sizes["z"]
        channels = metadata["channels"]
        image = np.zeros([depth, height, width, depth], dtype=np.float64)
        for idx, channel in enumerate(channels):
            reader.default_coords["c"] = idx
            for i in range(depth):
                image[i, :, :, idx] = reader[i]
        data_dict = {"channels": channels, "image": image, "metadata": metadata}
        return data_dict


def save_npy_as_tiff(frames: np.ndarray, path: str):
    tifffile.imsave(path, frames)
