
import json
import matplotlib
import torch
import matplotlib.pyplot as plt
from time import time
import psutil
import os 



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
    os.environ['NUMEXPR_MAX_THREADS'] = '32'


    #read config file
    with open('config.json','rb') as f:
        config = json.load(f)

    config["device"] = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    setup_seed(config["seed"])

    #parse dataset
    ds_parser = DataParser(config)

    
    trainer = Trainer(config,ds_parser)
    start = time()
    trainer.train()
    end = time()
    total = (end-start)/60

    print('Training took {:.2f} hours'.format(total))
    print(f'Best Epoch: {trainer.best_epoch}')

    trainer.test()


