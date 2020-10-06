import os
import numpy as np
import cv2

from src.segmentation.basic_segmentation import (
    get_watershed_labels,
    get_edge_based_labels,
)
from src.selection.cropping import get_3d_nuclear_crops_from_2d_segmentation
from src.selection.filtering import Filter
from src.utils.io import get_file_list, get_image_data_from_bioformat, save_npy_as_tiff
from src.transformations.distances import get_sure_foreground_from_distance
from src.transformations.morphology import apply_morphology_transformation


class SegmentationPipeline(object):
    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.file_list = get_file_list(self.input_dir)
        self.file_name = None
        self.image_data = None

    def read_in_image(self, index: int):
        file = self.file_list[index]
        file_name = os.path.split(file)[1]
        file_ending_idx = file_name.index(".")
        file_type = file_name[file_ending_idx + 1 :]
        file_name = file_name[:file_ending_idx]
        self.file_name = file_name
        self.image_data = get_image_data_from_bioformat(file=file, file_type=file_type)


class ProjectedWatershed3dSegmentationPipeline(SegmentationPipeline):
    def __init__(self, input_dir: str, output_dir: str):
        super().__init__(input_dir=input_dir, output_dir=output_dir)
        self.file_name = None
        self.image_data = None
        self.raw_image = None
        self.z_projection = None
        self.processed_projection = None
        self.sure_fg_projection = None
        self.sure_bg_projection = None
        self.labeled_projection = None
        self.labeled_image = None
        self.segmentation_visualization = None
        self.nuclear_crops = None
        self.filtered_idc = None
        self.filtered_out_idc = None

    def read_in_image(self, index: int):
        super().read_in_image(index=index)

    def select_image_by_channel(self, channel: str = "DAPI", normalize: bool = True):
        raw_image = self.image_data["image"][
            :, :, :, self.image_data["channels"].index(channel)
        ]
        self.raw_image = raw_image
        if normalize:
            raw_image /= raw_image.max()
            raw_image *= 255
            raw_image = raw_image.astype(np.uint8)
        #self.raw_image = raw_image
        # Requires first axis to be the z axis.
        self.z_projection = np.max(raw_image, axis=0)
        self.processed_projection = self.z_projection

    def apply_morphological_transformation(
        self, mode: str = "opening", kernel_size: int = 3, iterations: int = 3
    ):
        self.processed_projection = apply_morphology_transformation(
            self.processed_projection,
            mode=mode,
            kernel_size=kernel_size,
            iterations=iterations,
        )

    def otsu_thresholding(self):
        _, thresh = cv2.threshold(
            self.processed_projection, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        self.processed_projection = np.uint8(thresh)

    def set_sure_fg_projection_by_distance_transform(
        self, threshold: float = 0.4, distance_type: str = "l2", mask_size: int = 0
    ):
        self.sure_fg_projection = get_sure_foreground_from_distance(
            self.processed_projection,
            threshold=threshold,
            distance_type=distance_type,
            mask_size=mask_size,
        )

    def set_sure_bg_projection_by_morphological_transformation(
        self, mode: str = "dilate", kernel_size: int = 3, iterations: int = 10
    ):
        self.sure_bg_projection = apply_morphology_transformation(
            self.processed_projection,
            mode=mode,
            kernel_size=kernel_size,
            iterations=iterations,
        )

    def segment_projection_by_watershed(self):
        self.labeled_projection = get_watershed_labels(
            img=self.z_projection,
            sure_fg=self.sure_fg_projection,
            sure_bg=self.sure_bg_projection,
            mask=self.processed_projection,
        )

    def get_nuclear_crops_from_labeled_projection(
        self, xbuffer: int = 0, ybuffer: int = 0, filter: Filter = None
    ):
        self.nuclear_crops = get_3d_nuclear_crops_from_2d_segmentation(
            labeled_projection=self.labeled_projection,
            intensity_projection=self.z_projection,
            intensity_image=self.raw_image,
            xbuffer=xbuffer,
            ybuffer=ybuffer,
            filter_object=filter,
        )

    def save_nuclear_crops(self):
        output_dir = self.output_dir + "/segmentations_all/"
        os.makedirs(output_dir, exist_ok=True)
        for i in range(len(self.nuclear_crops)):
            save_npy_as_tiff(
                self.nuclear_crops[i]["image"],
                output_dir + self.file_name + "_n{}.tiff".format(i),
            )

    def segment_by_canny_edge_detection(
        self,
        sigma: float = 3.0,
        low_threshold: float = None,
        high_threshold: float = None,
    ):
        self.labeled_projection = get_edge_based_labels(
            img=self.processed_projection,
            sigma=sigma,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
        )

    def run_default_pipeline(self):
        filter_pipeline = None
        for i in range(len(self.file_list)):
            self.read_in_image(i)
            self.select_image_by_channel()
            # self.otsu_thresholding()
            # self.apply_morphological_transformation(
            #    mode="closing", kernel_size=3, iterations=2
            # )
            # self.apply_morphological_transformation(iterations=2)
            # self.set_sure_fg_projection_by_distance_transform()
            # self.set_sure_bg_projection_by_morphological_transformation()
            # self.segment_projection_by_watershed()
            self.segment_by_canny_edge_detection(sigma=2.5)
            self.get_nuclear_crops_from_labeled_projection(
                filter=filter_pipeline, xbuffer=2, ybuffer=2
            )
            self.save_nuclear_crops()
