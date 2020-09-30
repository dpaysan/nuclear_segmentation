import os
from typing import List

from src.segmentation.basic_segmentation import label_single_object_image
from src.selection.filtering import PropertyFilter
from src.utils.filter import get_filter_from_config
from src.utils.io import get_file_list, get_image_from_disk
from skimage.measure import regionprops
import tifffile


class FilterPipeline(object):
    def __init__(self, input_dir:str, output_dir:str, filter_configs:List[dict]):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.filter_configs = filter_configs
        self.filters = []
        self.file_list = get_file_list(self.input_dir)
        self.file_name = None
        self.image = None
        self.object_properties = None

    def init_filter(self):
        for filter_config in self.filter_configs:
            self.filters.append(get_filter_from_config(filter_config))

    def read_in_image(self, index):
        file = self.file_list[index]
        file_name = os.path.split(file)[1]
        file_ending_idx = file_name.index(".")
        file_type = file_name[file_ending_idx + 1:]
        file_name = file_name[:file_ending_idx]
        self.file_name = file_name
        self.image = get_image_from_disk(file=file, file_type=file_type)

    def get_object_properties(self, kernel_size:int=3, iterations:int=3):
        labeled_image = label_single_object_image(self.image, kernel_size=kernel_size, iterations=iterations)
        self.object_properties = regionprops(labeled_image, intensity_image=self.image)

    def run_filtering(self, kernel_size:int=3, iterations:int=3):
        filtered_output_dir = os.path.join(self.output_dir, 'filtered')
        filtered_out_output_dir = os.path.join(self.output_dir, 'filtered_out')

        os.makedirs(filtered_output_dir, exist_ok=True)
        os.makedirs(filtered_out_output_dir, exist_ok=True)

        self.init_filter()

        for i in range(len(self.file_list)):
            filtered = True
            self.read_in_image(i)
            self.get_object_properties(kernel_size=kernel_size, iterations=iterations)
            for filter in self.filters:
                if isinstance(filter, PropertyFilter):
                    filter.set_properties(self.object_properties)
                filtered = filtered and filter.filter(input=self.image)
                if not filtered:
                    break
            if filtered:
                path = str(filtered_output_dir) + self.file_name + '.tiff'
            else:
                path = str(filtered_out_output_dir) + self.file_name + '.tiff'
            tifffile.imsave(path, self.image)

