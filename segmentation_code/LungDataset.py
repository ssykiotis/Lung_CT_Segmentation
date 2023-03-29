import torch.utils.data as data_utils
import numpy as np
import torch


class Dataset(data_utils.Dataset):

    def __init__(self, config, x, y, mode, lung_seg = None, img_names = None, x_min = None, x_max = None):

        assert mode in ['train','val','test'], "Mode not supported"

        self.config = config

        self.x = x
        self.y = y
        
        self.lung_seg  = lung_seg
        self.img_names = img_names

        self.x_min = x_min if x_min else np.min(self.x)
        self.x_max = x_max if x_max else np.max(self.x)

        self.mode = mode


    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self,index):

        x = self.x[index].copy()
        y = self.y[index].copy()


        if self.x_min and self.x_max:
            x = (x-self.x_min)/(self.x_max-self.x_min)

        if self.mode =='train':
            return torch.tensor(x), torch.tensor(y)
        
        else:
            lung_mask = self.lung_seg[index].copy()
            name      = self.img_names[index]

            return torch.tensor(x), torch.tensor(y), torch.tensor(lung_mask), name


    
    def get_minmax(self):

        x_min = np.min(self.x)
        x_max = np.max(self.x)

        return x_min,x_max