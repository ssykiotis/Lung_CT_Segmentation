import os
from Patient import *
import random
import cv2
from lungmask import mask


class DataParser:

    def __init__(self,config):
        self.config = config

        self.patient_paths = [f'{config["data_location"]}/{file}'\
                              for file in sorted(os.listdir(config["data_location"]))\
                              if not file.startswith('.')]
        
        self.patient_paths = self.patient_paths[:50]
        self.train_patients,self.val_patients,self.test_patients = self.train_test_split()



    def get_dataset(self,mode):

        assert mode in ["train","test"],'Mode not supported'

        instance_seg_model = mask.get_model('unet','LTRCLobes')
        instance_seg_model = instance_seg_model.to('cuda:0')


        if mode == "train":

            x_train,y_train  = self.parse_patients(self.train_patients,instance_seg_model)
            x_val, y_val     = self.parse_patients(self.val_patients,instance_seg_model)
            training_dataset = Dataset(self.config,
                                       x_train,
                                       y_train)

            self.x_min, self.x_max = training_dataset.get_minmax()

            val_dataset      = Dataset(self.config,
                                       x_val,
                                       y_val,
                                       self.x_min,
                                       self.x_max)
            
            return training_dataset,val_dataset
        else:

            x_test, y_test = self.parse_patients(self.test_patients,instance_seg_model)
            test_dataset   = Dataset(self.config,
                                     x_test,
                                     y_test,
                                     self.x_min,
                                     self.x_max)
            return test_dataset
        
    def parse_patients(self,patients_path,instance_seg_model):

        x, y = [], []
        l = len(patients_path)
        for idx,patient in enumerate(patients_path):
            print(f'{idx}/{l}')
            pat = Patient(patient,instance_seg_model)
            x_i,y_i = self.format(pat,keeponly = True)
            x.append(x_i)
            y.append(y_i)

        x = np.concatenate(x)
        y = np.concatenate(y) 

        return x,y


    def format(self,pat,keeponly = False):
        """
        selects frames. applies lung mask, resize.

        input: pat: Patient
        output x : np.ndarray
               y : np.ndarray
        """
        x = []
        y = []

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

        x = np.concatenate(x, axis = 0)
        y = np.concatenate(y, axis = 0)

        return x,y

    
    def train_test_split(self):
        num_patients = len(self.patient_paths)
        random.shuffle(self.patient_paths)

        first_test_idx = int(num_patients*self.config["training_percent"])
        first_val_idx  = int(first_test_idx*self.config["training_percent"])

        training_patients = self.patient_paths[:first_val_idx]
        val_patients      = self.patient_paths[first_val_idx:first_test_idx]
        test_patients     = self.patient_paths[first_test_idx:]

        return training_patients,val_patients,test_patients
    

class Dataset:

    def __init__(self,config,x,y, x_min = None, x_max = None):

        self.config = config

        self.x = x
        self.y = y

        self.x_min = x_min
        self.x_max = x_max


    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self,index):

        x = self.x[index].copy()
        y = self.y[index].copy()

        if self.x_min and self.x_max:
            x = (x-self.x_min)/(self.x_max-self.x_min)

        return torch.tensor(x).to(self.config["device"]), torch.tensor(y).to(self.config["device"])
    
    def get_minmax(self):

        x_min = np.min(self.x)
        x_max = np.max(self.x)

        return x_min,x_max




