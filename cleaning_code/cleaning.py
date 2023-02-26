import os 
from utils import *

if __name__ == "__main__":
    

    dataset_path = '/media/stavros/WD_2TB/PhD Datasets/Cleaned/Medical/Lung Cancer/CT/NSCLC-Radiomics'
    raw_data_path   = f'{dataset_path}/raw_data'
    clean_data_path = f'{dataset_path}/clean_data'

    try:
        os.mkdir(clean_data_path)
    except IOError:
        # already exists
        pass

    create_folder_structure(raw_data_path,clean_data_path)
    move_data_to_clean(raw_data_path,clean_data_path)