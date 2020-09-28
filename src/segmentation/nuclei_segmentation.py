import os
import numpy as np
import cv2 as cv
from skimage.filters import threshold_otsu
from skimage.morphology import label
from skimage.measure import regionprops

from src.selection.cropping import get_conservative_nuclear_crops
from src.selection.filtering import filter_nuclei
from src.utils.io import get_file_list, get_image_data_from_bioformat, save_npy_as_tiff
from src.utils.visualization import get_colored_label_image_for_3d
from tqdm import tqdm


class SegmentationPipeline(object):
    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = input_dir
        self.output_dir = output_dir


class Simple3dSegmentationPipeline(SegmentationPipeline):
    def __init__(self, input_dir: str, output_dir: str):
        super().__init__(input_dir=input_dir, output_dir=output_dir)
        self.file_list = get_file_list(self.input_dir)
        self.file_name = None
        self.image_data = None
        self.raw_image = None
        self.processed_image = None
        self.labeled_image = None
        self.segmentation_visualization = None
        self.nuclei_properties = None
        self.nuclear_crops = None
        self.filtered_idc = None
        self.filtered_out_idc = None

    def read_in_image(self, index: int):
        file = self.file_list[index]
        file_name = os.path.split(file)[1]
        file_ending_idx = file_name.index(".")
        file_type = file_name[file_ending_idx + 1 :]
        file_name = file_name[:file_ending_idx]
        self.file_name = file_name
        self.image_data = get_image_data_from_bioformat(file=file, file_type=file_type)

    def select_image_by_channel(self, channel: str = "DAPI", normalize: bool = True):
        raw_image = self.image_data["image"][
            :, :, :, self.image_data["channels"].index(channel)
        ]
        if normalize:
            raw_image /= raw_image.max()
            raw_image *= 255
            raw_image = raw_image.astype(np.uint8)
        self.raw_image = raw_image
        self.processed_image = self.raw_image

    def otsu_thresholding(self):
        tresh = threshold_otsu(self.processed_image)
        binary = self.processed_image > tresh
        self.processed_image = np.uint8(binary)

    def remove_small_holes(self, kernel_size: int = 3, iterations: int = 10):
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        self.processed_image = cv.morphologyEx(
            self.processed_image.transpose(),
            cv.MORPH_CLOSE,
            kernel=kernel,
            iterations=iterations,
        ).transpose()

    def label_connected_objects(self, connectivity: int = 3):
        self.labeled_image = label(self.processed_image, connectivity=connectivity)

    def visualize_segmentation_performance(self, posfix: str = "_segmentation"):
        colored_label_image = get_colored_label_image_for_3d(
            self.raw_image, self.labeled_image
        )
        save_npy_as_tiff(colored_label_image, self.output_dir + '/' + self.file_name + posfix + ".tiff")

    def get_nuclei_properties(self):
        self.nuclei_properties = regionprops(
            label_image=self.labeled_image, intensity_image=self.raw_image
        )

    def get_nuclear_crops(self):
        self.nuclear_crops = get_conservative_nuclear_crops(
            labels=self.labeled_image, intensity_image=self.raw_image
        )

    def filter_nuclei_crops(self, mode: str = "size", **kwargs):
        self.filtered_idc, self.filtered_out_idc = filter_nuclei(
            nuclear_crops=self.nuclear_crops, mode=mode, **kwargs
        )

    def save_nuclei_crops(self):
        if self.filtered_idc is None:
            output_dir = self.output_dir + "/segmentations_all/"
            os.makedirs(output_dir, exist_ok=True)
            for i in range(len(self.nuclear_crops)):
                save_npy_as_tiff(
                    self.nuclear_crops[i]["image"],
                    output_dir + self.file_name + "_{}.tiff".format(i),
                )
        else:
            output_dir = self.output_dir + "/segmentations_filtered/"
            os.makedirs(output_dir, exist_ok=True)
            for i in self.filtered_idc:
                save_npy_as_tiff(
                    self.nuclear_crops[i]["image"],
                    output_dir + self.file_name + "_{}.tiff".format(i),
                )

            output_dir = self.output_dir + "/segmentations_filtered_out/"
            os.makedirs(output_dir, exist_ok=True)
            for i in self.filtered_out_idc:
                save_npy_as_tiff(
                    self.nuclear_crops[i]["image"],
                    output_dir + self.file_name + "_{}.tiff".format(i),
                )

    def run_default_pipeline(self):
        for i in tqdm(range(len(self.file_list))):
            self.read_in_image(i)
            self.select_image_by_channel()
            self.otsu_thresholding()
            self.remove_small_holes()
            self.label_connected_objects()
            self.visualize_segmentation_performance()
            self.get_nuclear_crops()
            self.filter_nuclei_crops(mode='size', microns_per_pixel=0.1, min_volume=5, max_volume=200)
            self.save_nuclei_crops()

