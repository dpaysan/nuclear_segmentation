import logging
import os
import numpy as np
import tifffile
from nd2reader import ND2Reader
from typing import List
import pickle as cPickle
import skimage

from src.utils.general import sorted_nicely


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
    return sorted_nicely(list_of_data_locs)


def get_image_data_from_bioformat(file: str, file_type: str):
    if file_type == "nd2":
        return nd2_to_npy(file)
    elif file_type == "pkl":
        return load_data_dict_from_pickle(path=file)
    else:
        raise NotImplementedError("Unknown file type: {}".format(file_type))


def get_image_from_disk(file, file_type: str) -> np.ndarray:
    if file_type in ["tif", "tiff"]:
        return tifffile.imread(file)
    else:
        raise RuntimeError("Unknown file type: ".format(file_type))


def nd2_to_npy(nd2_file: str) -> List[dict]:
    """Function that returns a dictionary that includes the images for each channel of the .nd2 file
    and the meta-information"""
    try:
        data_dicts = []
        with ND2Reader(nd2_file) as reader:
            metadata = reader.metadata
            channels = metadata["channels"]
            reader.bundle_axes = "zyxc"
            if "v" in reader.axes:
                reader.iter_axes = "v"
                for frame in reader:
                    image = frame
                    data_dict = {
                        "channels": channels,
                        "image": image,
                        "metadata": metadata,
                    }
                    data_dicts.append(data_dict)
            else:
                logging.debug(
                    "No image series but a single image found for {}.".format(nd2_file)
                )
                image = reader[0]
                data_dict = {"channels": channels, "image": image, "metadata": metadata}
                data_dicts.append(data_dict)

        return data_dicts

    except Exception:
        logging.debug("File not readable: {}".format(nd2_file))
        return []


def lsm_to_npy(lsm_file: str, channels=None) -> List[dict]:
    try:
        data_dicts = []
        # Expect TZCXY shape but reshapes it to ZYXC
        lsm_images = tifffile.imread(lsm_file)
        for i in range(len(lsm_images)):
            image = lsm_images[i]
            z,c,x,y = image.shape
            image = np.transpose(image, [0,2,3,1])
            data_dict = {"channels": channels, "image": image, "meta_data": None}
            data_dicts.append(data_dict)
        return data_dicts
    except Exception:
        logging.debug("File not readable: {}".format(lsm_file))
        return []


def split_nd2_series_save_as_pickle(nd2_file: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    file_name = os.path.split(nd2_file)[1]
    file_name = file_name[: file_name.index(".")]
    data_dicts = nd2_to_npy(nd2_file=nd2_file)
    for i in range(len(data_dicts)):
        path = str(os.path.join(output_dir, file_name)) + "_s{}.pkl".format(i)
        save_pickle(data_dicts[i], path)


def split_lsm_series_save_as_pickle(
    lsm_file: str, output_dir: str, channels: List[str] = None
):
    os.makedirs(output_dir, exist_ok=True)
    parent_dir, file_name = os.path.split(lsm_file)
    file_name = file_name[: file_name.index(".")]
    file_name = os.path.basename(parent_dir) + '_' + file_name
    data_dicts = lsm_to_npy(lsm_file=lsm_file, channels=channels)
    for i in range(len(data_dicts)):
        path = str(os.path.join(output_dir, file_name)) + "_s{}.pkl".format(i)
        save_pickle(data_dicts[i], path)


def load_data_dict_from_pickle(path: str):
    return load_pickle(filename=path)


def split_nd2_series_save_as_tif(nd2_file: str, output_dir: str):
    file_name = os.path.split(nd2_file)[1]
    file_name = file_name[: file_name.index(".")]
    data_dicts = nd2_to_npy(nd2_file=nd2_file)
    for i in range(len(data_dicts)):
        image = data_dicts[i]["image"]
        path = str(os.path.join(output_dir, file_name)) + "_s{}.tif".format(i)
        # Add dimension to reach format conform with the expected output
        tifffile.imsave(path, np.expand_dims(image, axis=0), metadata={"axes": "TZYXC"})


def save_npy_as_tiff(frame: np.ndarray, path: str):
    # expects np.ndarray in shape ZYXC or ZYX
    frame = np.expand_dims(frame, axis=0)
    if len(frame.shape) == 4:
        frame = np.expand_dims(frame, axis=2)
    else:
        frame = frame.transpose([0,1,4,2,3])
    tifffile.imsave(path, frame, metadata = {'axes':'TZCYX'}, imagej=True)


def save_pickle(obj, filename, protocol=-1):
    r""" Function to save an object as a zip-compressed pickle file.
    Parameters
    ----------
    obj : object
        The object that is serialized
    filename : str
        The path of the file that should hold the serialized object.
    protocol : int
        The protocol used to compress the object. See :py:meth:`cPickle.dump` for more information.
    """
    with open(filename, "wb") as f:
        cPickle.dump(obj, f, protocol)


def load_pickle(filename):
    r""" Function to load an object from disk that was serialized as a zip-compressed pickle file.
    Parameters
    ----------
    filename : str
        The path of the file that is the serialized object.
    Returns
    -------
    loaded_object : object
        The deserialized object.
    """
    with open(filename, "rb") as f:
        loaded_object = cPickle.load(f)
        return loaded_object
