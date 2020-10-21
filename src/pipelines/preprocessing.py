from typing import List

from tqdm import tqdm

from src.utils.io import (
    get_file_list,
    split_nd2_series_save_as_tif,
    split_nd2_series_save_as_pickle,
    split_lsm_series_save_as_pickle,
)


class PreprocessingPipeline(object):
    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = input_dir
        self.output_dir = output_dir


class SimplePreprocessingPipeline(PreprocessingPipeline):
    def __init__(self, input_dir: str, output_dir: str):
        super().__init__(input_dir=input_dir, output_dir=output_dir)

        self.file_list = get_file_list(self.input_dir)

    def split_nd2_file_save_as_tif(self, index):
        nd2_file = self.file_list[index]
        split_nd2_series_save_as_tif(nd2_file, output_dir=self.output_dir)

    def split_nd2_file_save_as_compressed_dict(self, index):
        nd2_file = self.file_list[index]
        split_nd2_series_save_as_pickle(nd2_file=nd2_file, output_dir=self.output_dir)

    def split_lsm_file_save_as_compressed_dict(self, index, channels: List[str] = None):
        lsm_file = self.file_list[index]
        split_lsm_series_save_as_pickle(
            lsm_file=lsm_file, output_dir=self.output_dir, channels=channels
        )

    def run_default_pipeline_nd2(self):
        for i in tqdm(range(len(self.file_list))):
            self.split_nd2_file_save_as_compressed_dict(index=i)

    def run_default_pipeline_lsm(self, channels: List[str] = None):
        for i in tqdm(range(len(self.file_list))):
            self.split_lsm_file_save_as_compressed_dict(index=i, channels=channels)
