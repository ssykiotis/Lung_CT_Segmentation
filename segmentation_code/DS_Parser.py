import os
from Patient import *
from LungDataset import *
import random
import cv2
from lungmask import mask
import pandas as pd


class DataParser:

    def __init__(self,config):
        self.config = config

        self.patient_paths = self.get_patient_paths()
        # self.patient_paths = self.get_patient_paths(self.config["vendor"])
        
        self.patient_paths = self.patient_paths
        self.train_patients,self.val_patients,self.test_patients = self.train_test_split()

    def get_dataset(self,mode):
        """
        returns parsed dataset that corresponds to the specified mode
        input: mode: str

        output: dataset: Dataset 
        """

        assert mode in ["train","val","test"], 'Mode not supported'

        if mode == "train":

            x, y, img_names, flags = self.parse_patients(self.train_patients, keeponly = True)
            dataset         = Dataset(self.config,
                                      x,
                                      y,
                                      img_names,
                                      flags
                                     )

            self.x_min, self.x_max = dataset.get_minmax()
        else:
            if mode=='val':
                x, y, img_names, flags = self.parse_patients(self.val_patients, keeponly = False)
            else:
                x, y, img_names, flags = self.parse_patients(self.test_patients, keeponly = False)

            x_train, _, _, _ = self.parse_patients(self.train_patients, keeponly = True)
            self.x_min = np.min(x_train)
            self.x_max = np.max(x_train)

            dataset   = Dataset(self.config,
                                x,
                                y,
                                img_names,
                                flags,
                                self.x_min,
                                self.x_max
                                )
        return dataset
        
    def parse_patients(self,patients_path,keeponly = False):
        """
        parses patients included in patient_path and returnes the processed patient images:
        input:  patients_path: list  List with all patient paths that need to be parsed

        output: x:np.ndarray    CT scan images, cropped and resized
                y:np.ndarray    segmentation masks
                img_names: list image names
        """

        instance_seg_model = mask.get_model('unet','R231CovidWeb')
        instance_seg_model = instance_seg_model.to('cuda:0')

        x, y ,img_names, flags = [], [], [], []
        for patient in patients_path:
            try:
                pat = Patient(patient,instance_seg_model)
                x_i,y_i,img_names_i,flag_i = self.format(pat,keeponly)
                x.append(x_i)
                y.append(y_i)
                img_names.append(img_names_i)
                flags.append(flag_i)
            except InvalidPatientError:
                print(f'Patient {patient} excluded: not consistent data lengths')
                continue
        x         = np.concatenate(x)
        y         = np.concatenate(y) 
        img_names = np.concatenate(img_names)
        flags     = np.concatenate(flags)

        return x, y, img_names,flags


    def format(self,pat,keeponly = False):
        """
        selects frames, applies lung mask, resize.
        input: pat      : Patient

        output x        : np.ndarray
               y        : np.ndarray
               img_names: list
        """
        x         = []
        y         = []
        img_names = []
        flags     = []

        if keeponly:
            idxs_to_keep = np.where(pat.lesion_seg.sum(axis = (1,2)))[0]
        else:
            idxs_to_keep = list(range(pat.imgs.shape[0]))

        for idx in idxs_to_keep:
           
            lung_seg = pat.lung_seg[idx]
            if keeponly:
                flag = 1
            else:
                flag = (np.unique(lung_seg).tolist()==[0,1,2])*1
            lung_seg[lung_seg>0] = 1
            max_values = np.amax(lung_seg)
            
            result = np.where(lung_seg == max_values)
            x1 = np.min(result[0]) - self.config["crop_buffer"]
            x2 = np.max(result[0]) + self.config["crop_buffer"]
            y1 = np.min(result[1]) - self.config["crop_buffer"]
            y2 = np.max(result[1]) + self.config["crop_buffer"]

            img_cropped = pat.imgs[idx,x1:x2,y1:y2]
            img_cropped = img_cropped.astype(np.float32)

            dim = (self.config["img_size"],self.config["img_size"])

            img_resized = cv2.resize(img_cropped,
                                     dsize = dim,
                                     interpolation = cv2.INTER_CUBIC)
            
            img_resized = img_resized.astype(np.float16)
            
            mask_cropped = pat.lesion_seg[idx,x1:x2,y1:y2]
            mask_cropped = mask_cropped.astype(np.float32)

            mask_resized = cv2.resize(mask_cropped,
                                      dsize = dim,
                                      interpolation = cv2.INTER_NEAREST)
            
            mask_resized[mask_resized>0] = 1
            mask_resized = mask_resized.astype(np.int8)

            x.append(np.expand_dims(img_resized,  axis = (0, 1)))
            y.append(np.expand_dims(mask_resized, axis = (0, 1)))
            img_names.append(pat.img_names[idx])
            flags.append(flag)

        x = np.concatenate(x, axis = 0)
        y = np.concatenate(y, axis = 0)

        return x, y, img_names, flags

    
    def train_test_split(self):
        """
        splits dataset in training, validation and test set according to the split ratio defined in the config file
        """

        num_patients = len(self.patient_paths)
        random.shuffle(self.patient_paths)

        first_test_idx = int(num_patients*self.config["training_percent"])
        first_val_idx  = int(first_test_idx*self.config["training_percent"])

        training_patients = self.patient_paths[:first_val_idx]
        val_patients      = self.patient_paths[first_val_idx:first_test_idx]
        test_patients     = self.patient_paths[first_test_idx:]

        return training_patients,val_patients,test_patients
    

    def get_patient_paths(self,vendor = None):
        if vendor:
            metadata = pd.read_csv(self.config['metadata_location'])
            mod = 'CT'
            metadata = metadata.query('Modality==@mod and Manufacturer == @vendor')
            eligible_patients  = metadata['File Location'].apply(lambda x: x.split('/')[2]).tolist()
            available_patients = [file for file in sorted(os.listdir(self.config["data_location"]))\
                                  if not file.startswith('.')]
            eligible_patients  = [pat for pat in eligible_patients if pat in available_patients]

            patient_paths = sorted([f'{self.config["data_location"]}/{pat}' for pat in eligible_patients])
        else:
            patient_paths = [f'{self.config["data_location"]}/{file}'\
                            for file in sorted(os.listdir(self.config["data_location"]))\
                            if not file.startswith('.')]
            
        return patient_paths
    


