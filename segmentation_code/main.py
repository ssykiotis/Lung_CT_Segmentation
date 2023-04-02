
import json
import matplotlib
import torch
import matplotlib.pyplot as plt
from time import time


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

    start = time()
    train_dataset = ds_parser.get_dataset("train")
    end   = time()
    total = end-start

    print(len(train_dataset))
    print(f'Parsing train_dataset took {total} seconds')
    
    start = time()
    train_dataset = ds_parser.get_dataset("val")
    end   = time()
    total = end-start

    print(len(train_dataset))
    print(f'Parsing val_dataset took {total} seconds')

    start = time()
    train_dataset = ds_parser.get_dataset("test")
    end   = time()
    total = end-start

    print(len(train_dataset))
    print(f'Parsing test_dataset took {total} seconds')


    # trainer = Trainer(config,ds_parser)


