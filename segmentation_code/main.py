
import json
import matplotlib
import torch
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')


from DS_Parser import *
from Trainer   import *

#TODO: SET SEED


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

    train_dataset, val_dataset = ds_parser.get_dataset("train")

    
    trainer   = Trainer(config,ds_parser)


