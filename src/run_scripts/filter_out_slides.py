import pandas as pd
import os
from src.utils.io import get_file_list
from shutil import copyfile
import sys


output_dir = '../../data/tcell_project/split_raw_data/filtered_splits/tzyxc_pkl'
os.makedirs(output_dir, exist_ok=True)
data_infos = pd.read_csv('../../data/tcell_project/data_list.csv')
nan_rows = data_infos.iloc[:,1:].isnull().all(axis=1)
filtered_slice_names = data_infos.loc[nan_rows].iloc[:,0]

all_files = get_file_list('../../data/tcell_project/split_raw_data/unfiltered_splits/tzyxc_pkl/pkl_files')
filtered_files = []
for file in all_files:
    short_file_name = os.path.split(file)[1]
    short_file_name = short_file_name[:short_file_name.index('.')]
    if short_file_name in list(filtered_slice_names):
        copyfile(file, os.path.join(output_dir, os.path.split(file)[1]))
sys.exit(0)
