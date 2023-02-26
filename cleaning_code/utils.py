import os
import pydicom as pdcm
import shutil


def create_folder_structure(raw_path,clean_path):

    patients = os.listdir(raw_path)

    for patient in patients:
        clean_patient_path = f'{clean_path}/{patient}'
        try:
            os.mkdir(clean_patient_path)
        except FileExistsError:
            pass #folder already exists

        try:
            os.mkdir(f'{clean_patient_path}/images')
        except FileExistsError:
            pass #folder already exists

        try:
            os.mkdir(f'{clean_patient_path}/seg')
        except FileExistsError:
            pass #folder already exists


def move_data_to_clean(raw_path,clean_path):
    
    patient_paths = [f'{raw_path}/{patient}' for patient in sorted(os.listdir(raw_path))]
    patient_names = sorted(os.listdir(raw_path))

    patients = list(zip(patient_paths,patient_names))


    for patient_data in patients:
        patient_path,patient_name = patient_data
        patient = f'{patient_path}/{os.listdir(patient_path)[0]}'
        print(patient_name)
        subfolders = [f'{patient}/{subf}' for subf in os.listdir(patient)]

        for subf in subfolders:
            files = [f'{subf}/{file}' for file in os.listdir(subf) if not file.startswith('._')]

            for file in files:
                try:
                    with open(file,'rb') as f:
                        dcm_obj = pdcm.dcmread(file)
                except pdcm.errors.InvalidDicomError:
                    print(file)
                    return None


                if dcm_obj.Modality == 'CT':
                    fname = file.split('/')[-1]
                    copy_path = f'{clean_path}/{patient_name}/images/{fname}'
                    if not os.path.exists(copy_path):
                        shutil.copy(file,copy_path)

                    
                elif dcm_obj.Modality == 'SEG':
                    fname = file.split('/')[-1]
                    copy_path = f'{clean_path}/{patient_name}/seg/{fname}'
                    if not os.path.exists(copy_path):
                        shutil.copy(file,copy_path)
                else:
                    pass
