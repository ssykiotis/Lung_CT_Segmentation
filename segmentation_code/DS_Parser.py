import os
from Patient import *
from LungDataset import *
import random
import cv2
from lungmask import mask


class DataParser:

    def __init__(self,config):
        self.config = config

        self.patient_paths = [f'{config["data_location"]}/{file}'\
                              for file in sorted(os.listdir(config["data_location"]))\
                              if not file.startswith('.')]
        
        self.patient_paths = self.patient_paths[:10]
        self.train_patients,self.val_patients,self.test_patients = self.train_test_split()

    def get_dataset(self,mode):

        assert mode in ["train","val","test"],'Mode not supported'

        if mode == "train":

            x, y, img_names = self.parse_patients(self.train_patients, keeponly = True)
            dataset         = Dataset(self.config,
                                      x,
                                      y,
                                      img_names
                                     )

            self.x_min, self.x_max = dataset.get_minmax()
        else:
            if mode=='val':
                x, y, img_names = self.parse_patients(self.val_patients, keeponly = False)
            else:
                x, y, img_names = self.parse_patients(self.test_patients, keeponly = False)

            dataset   = Dataset(self.config,
                                x,
                                y,
                                img_names,
                                self.x_min,
                                self.x_max
                                )
        return dataset
        
    def parse_patients(self,patients_path,keeponly = False):

        instance_seg_model = mask.get_model('unet','LTRCLobes')
        instance_seg_model = instance_seg_model.to('cuda:0')

        x, y ,img_names = [], [], []
        l = len(patients_path)
        for patient in patients_path:
            pat = Patient(patient,instance_seg_model)
            x_i,y_i,img_names_i = self.format(pat,keeponly)
            x.append(x_i)
            y.append(y_i)
            img_names.append(img_names_i)

        x         = np.concatenate(x)
        y         = np.concatenate(y) 
        img_names = np.concatenate(img_names)

        return x, y, img_names


    def format(self,pat,keeponly = False):
        """
        selects frames, applies lung mask, resize.

        input: pat: Patient
        output x    : np.ndarray
               y    : np.ndarray
               names: list
        """
        x         = []
        y         = []
        img_names = []

        if keeponly:
            idxs_to_keep = np.where(pat.lesion_seg.sum(axis = (1,2)))[0]
        else:
            idxs_to_keep = list(range(pat.imgs.shape[0]))

        for idx in idxs_to_keep:
            max_values = np.amax(pat.lung_seg[idx])
            result = np.where(pat.lung_seg[idx] == max_values)
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

            x.append(np.expand_dims(img_resized, axis = 0))
            y.append(np.expand_dims(mask_resized,axis = 0))
            img_names.append(pat.img_names[idx])

        x = np.concatenate(x, axis = 0)
        y = np.concatenate(y, axis = 0)

        return x, y, img_names

    
    def train_test_split(self):
        num_patients = len(self.patient_paths)
        random.shuffle(self.patient_paths)

        first_test_idx = int(num_patients*self.config["training_percent"])
        first_val_idx  = int(first_test_idx*self.config["training_percent"])

        training_patients = self.patient_paths[:first_val_idx]
        val_patients      = self.patient_paths[first_val_idx:first_test_idx]
        test_patients     = self.patient_paths[first_test_idx:]

        return training_patients,val_patients,test_patients
    


