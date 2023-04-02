import os
import numpy   as np
import pydicom as pdcm
import pydicom_seg
import numpy as np

from pydicom.pixel_data_handlers.util import apply_modality_lut
from lungmask import mask

class Patient:

    def __init__(self,path,instance_seg = None):
        self.path = path
        self.images_loc = f'{self.path}/images/'
        self.seg_loc    = f'{self.path}/seg/'

        self.lung_seg_model = instance_seg

        self.dcm_paths   = self.parse_images()
        self.seg_path    = self.parse_segmentation()

        self.metadata    = self.get_exam_metadata()
        self.imgs        = self.read_images()
        self.img_names   = self.get_image_names()

        self.lung_seg,self.lesion_seg = self.read_segmentations()

    
    def parse_images(self):
        """
        returns paths of dicom imaging files
        input: 
        output: dcm_paths: list
        """
        dcm_paths = [f'{self.images_loc}/{file}' for file in os.listdir(self.images_loc) if not file.startswith('.')]
        return dcm_paths

    def parse_segmentation(self):
        """
        returns paths of dicom seg files
        input: 
        output: seg_file: str
        """
        seg_file = [f'{self.seg_loc}/{file}' for file in os.listdir(self.seg_loc) if not file.startswith('.')]
        return seg_file[0]


    def get_exam_metadata(self):
        """
        returns a dictionary with examination metadata
        input: 
        output: metadata: dict
        """
        with open(self.dcm_paths[0],'rb') as f:
            dcm_obj = pdcm.dcmread(f)

        metadata = {'vendor':      (dcm_obj.Manufacturer).lower(),
                    'orientation': dcm_obj.PatientPosition}
        return metadata



    def read_images(self):
        """
        parses dicom imaging files, applies lut_sequence or hu transformation and returns all images of the examination
        input:
        output: images: np.ndarray of size (num_images,512,512)
        """


        imgs = np.zeros(shape = (len(self.dcm_paths),512,512))

        for dcm_path in self.dcm_paths:
            with open(dcm_path,'rb') as f:
                dcm_obj = pdcm.dcmread(f)

            pixel_data = dcm_obj.pixel_array
            pixel_data = apply_modality_lut(pixel_data,dcm_obj)
            idx = dcm_obj.InstanceNumber

            imgs[idx-1] = pixel_data

        return imgs.astype(np.float16)

        

    def read_segmentations(self):

        """
        reads the dicom seg file and returns the lung segmentation and the lesion segmentation
        input:
        output: lung_seg: np.ndarray of size (num_images,512,512)
                lesion_seg: np.ndarray of size (num_images,512,512)
        """

        reader    = pydicom_seg.SegmentReader()
        with open(self.seg_path,'rb') as f:
            seg_dcm   = pdcm.dcmread(f)
        res       = reader.read(seg_dcm)
        
        lung_idxs = []
        for item in res.segment_infos:
            if res.segment_infos[item].to_json_dict()['00620006']['Value'][0] =='GTV-1':
                lesion_seg = res.segment_data(item)
            if 'Lung' in res.segment_infos[item].to_json_dict()['00620006']['Value'][0]:
                   lung_idxs.append(item) 

        if len(lung_idxs)>1:
            lung_seg = res.segment_data(lung_idxs[0])+res.segment_data(lung_idxs[1])
        elif len(lung_idxs)==1:
            lung_seg = res.segment_data(lung_idxs[0]) 
        else:
            lung_seg = mask.apply(self.imgs.astype(np.float32), self.lung_seg_model, batch_size = 32)

        lung_seg[lung_seg!=0] = 1

        if self.metadata["orientation"] == 'HFS':
            lesion_seg = lesion_seg[::-1,:,:]
            lung_seg   = lung_seg[::-1,:,:]

        return lung_seg,lesion_seg
    

    def get_image_names(self):
        patient_name = self.path.split('/')[-1]
        image_names = [f'{patient_name}/{i+1}' for i in range(self.imgs.shape[0])]

        return image_names

