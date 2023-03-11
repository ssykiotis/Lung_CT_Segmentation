
import json
import matplotlib
import torch
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')


from DS_Parser import *
from Trainer   import *

#TODO: SET SEED


if __name__ == "__main__":

    #read config file
    with open('config.json','rb') as f:
        config = json.load(f)

    config["device"] = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    #parse dataset
    ds_parser = DataParser(config)

    train_dataset, val_dataset = ds_parser.get_dataset("train")

    
    trainer   = Trainer(config)


