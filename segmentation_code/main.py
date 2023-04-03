
import json
import matplotlib
import torch
import matplotlib.pyplot as plt
from time import time
import psutil


from DS_Parser import *
from Trainer   import *

def setup_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False  
    random.seed(seed)                          
    np.random.seed(seed)                       
    torch.manual_seed(seed)                    
    torch.cuda.manual_seed(seed)               
    torch.cuda.manual_seed_all(seed)  


if __name__ == "__main__":

    #read config file
    with open('config.json','rb') as f:
        config = json.load(f)

    config["device"] = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    setup_seed(config["seed"])


    #parse dataset
    ds_parser = DataParser(config)

    print(len(ds_parser.train_patients))
    print(len(ds_parser.val_patients))
    print(len(ds_parser.test_patients))

    start = time()
    train_dataset = ds_parser.get_dataset("train")
    end   = time()
    total = end-start

    print(train_dataset.x.shape)

    print(f'Parsing train_dataset took {total} seconds')

    print('RAM memory % used:', psutil.virtual_memory()[2])
    print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)

    print(train_dataset.img_names)
    
    start = time()
    val_dataset = ds_parser.get_dataset("val")
    end   = time()
    total = end-start

    print(val_dataset.x.shape)

  
    print(f'Parsing val_dataset took {total} seconds')

    print('RAM memory % used:', psutil.virtual_memory()[2])
    print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)
    print(val_dataset.img_names)



    start = time()
    test_dataset = ds_parser.get_dataset("test")
    end   = time()
    total = end-start

    print(test_dataset.x.shape)


    print(f'Parsing test_dataset took {total} seconds')

    print('RAM memory % used:', psutil.virtual_memory()[2])
    print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)
    print(test_dataset.img_names)


    # trainer = Trainer(config,ds_parser)


