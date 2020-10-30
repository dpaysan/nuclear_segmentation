import logging
import os
from typing import List, Any
import numpy as np

from src.segmentation.basic_segmentation import label_single_object_image
from src.selection.filtering import (
    ConservativeDeadCellFilter,
    AreaFilter,
    AspectRatioFilter,
)
from src.utils.io import get_file_list, get_image_from_disk
from skimage.measure import regionprops
import tifffile

from src.utils.visualization import plot_3d_images_as_map


class FilterPipeline(object):
    def __init__(self, input_dir: str, output_dir: str, multi_channel:bool=False):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.filters = []
        self.file_list = get_file_list(self.input_dir)
        self.file_name = None
        self.image = None
        self.filtered_images = []
        self.filtered_out_images = []
        self.object_properties = None
        self.multi_channel = multi_channel

    def read_in_image(self, index:int):
        # If multi-channel data is read - the first one is assumed to be the DAPI channel
        file = self.file_list[index]
        file_name = os.path.split(file)[1]
        file_ending_idx = file_name.index(".")
        file_type = file_name[file_ending_idx + 1 :]
        file_name = file_name[:file_ending_idx]
        self.file_name = file_name
        self.image = get_image_from_disk(file=file, file_type=file_type)
        if self.multi_channel:
            # Input shape is assumed to be ZCYX
            self.dapi_image = self.image[:, 0, :, :]
        else:
            self.dapi_image = self.image

    def get_object_properties(self, kernel_size: int = 3, iterations: int = 3):
        labeled_image = label_single_object_image(
            self.dapi_image, kernel_size=kernel_size, iterations=iterations
        )
        object_properties = regionprops(
            labeled_image, intensity_image=np.squeeze(self.dapi_image)
        )
        if len(object_properties) != 1:
            logging.debug("No or multiple objects detected in the image")

    def run_filtering(self):
        filtered_output_dir = os.path.join(self.output_dir, "filtered")
        filtered_out_output_dir = os.path.join(self.output_dir, "filtered_out")

        os.makedirs(filtered_output_dir, exist_ok=True)
        os.makedirs(filtered_out_output_dir, exist_ok=True)

        for i in range(len(self.file_list)):
            filtered = True
            self.read_in_image(i)
            for filter in self.filters:
                filtered = filtered and filter.filter(input=self.dapi_image)
                if not filtered:
                    break
            if filtered:
                self.filtered_images.append(self.image)
                path = str(filtered_output_dir) + "/" + self.file_name + ".tiff"
            else:
                self.filtered_out_images.append(self.image)
                path = str(filtered_out_output_dir) + "/" + self.file_name + ".tiff"
            tifffile.imsave(path, np.expand_dims(self.image,axis=0), imagej=True)

    def add_area_filter(self, thresholds: Any, threshold_unit_pp: Any):
        self.filters.append(
            AreaFilter(thresholds=thresholds, threshold_unit_pp=threshold_unit_pp)
        )

    def add_aspect_ratio_filter(self, thresholds: Any):
        self.filters.append(AspectRatioFilter(thresholds=thresholds))

    def add_conservative_dead_cell_filter(
        self, intensity_threshold: float = 250, portion_threshold: float = 0.05
    ):
        self.filters.append(
            ConservativeDeadCellFilter(
                intensity_threshold=intensity_threshold,
                portion_threshold=portion_threshold,
            )
        )

    def plot_images(self, image_type: str = "all"):
        save_path = str(os.path.join(self.output_dir, "summary/" + image_type))
        if image_type == "filtered":
            os.makedirs(save_path)
            plot_3d_images_as_map(self.filtered_images, save_path=save_path)
        elif image_type == "filtered_out":
            os.makedirs(save_path)
            plot_3d_images_as_map(self.filtered_out_images, save_path=save_path)
        elif image_type == "all":
            self.plot_images(image_type="filtered")
            self.plot_images(image_type="filtered_out")
        else:
            raise RuntimeError("Unknown image type: {}.".format(image_type))
