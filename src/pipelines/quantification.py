import os
from abc import ABC

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
import tifffile
from skimage.exposure import rescale_intensity
from tqdm import tqdm
import multiprocessing
from joblib import Parallel, delayed

from src.quantification.nuclei_quantification import get_basic_properties, get_hc_ec_properties
from src.segmentation.basic_segmentation import get_chan_vese_based_object_mask_3d
from src.segmentation.structure_detection import get_hc_ec_structure_maps_by_thresholding
from src.transformations.noise_reduction import remove_layers_outside_roi, denoise_img_bilateral
from src.utils.io import get_file_list


class QuantificationPipeline:
    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.file_list = get_file_list(self.input_dir)
        self.images = self.read_in_data(self.file_list)

        self.statistics = None

    def read_in_data(self, file_list):
        images = []
        for file in file_list:
            images.append(np.squeeze(tifffile.imread(file)))
        return images

    def run(self):
        raise NotImplementedError


class HumanTCellPipeline(QuantificationPipeline, ABC):
    def __init__(self, input_dir, output_dir):
        super().__init__(input_dir=input_dir, output_dir=output_dir)

    def run_quantification(self):
        cpu_count = int(multiprocessing.cpu_count())
        self.statistics = Parallel(n_jobs=cpu_count)(
            delayed(self.get_statistics_for_single_image)(i) for i in tqdm(range(len(self.images)))
        )

    def save_statistics(self):
        statistics = pd.DataFrame(self.statistics)
        statistics.to_csv(os.path.join(self.output_dir, 'statistics.csv'))

    def get_statistics_for_single_image(self, index):
        img = self.images[index]
        #img = (img-img.min())/(img.max()-img.min())
        object_mask = get_chan_vese_based_object_mask_3d(img, smoothing=1)
        masked_img = remove_layers_outside_roi(img, object_mask)
        filtered_img = denoise_img_bilateral(masked_img)
        normalized_img = rescale_intensity(filtered_img)
        hc_ec_structure_dict = get_hc_ec_structure_maps_by_thresholding(normalized_img, object_mask, k=0.6)
        hc_mask = hc_ec_structure_dict['hc_mask']
        ec_mask = hc_ec_structure_dict['ec_mask']

        hc_ec_properties = get_hc_ec_properties(hc_mask=hc_mask, ec_mask=ec_mask)
        basic_properties = get_basic_properties(object_mask, img)
        basic_properties.update(hc_ec_properties)
        basic_properties['file'] = self.file_list[index]
        return basic_properties

    def run_default_pipeline(self):
        self.run_quantification()
        self.save_statistics()

