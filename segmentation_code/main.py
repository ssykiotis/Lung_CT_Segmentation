
import json


from DS_Parser import *
from Trainer   import *

if __name__ == "__main__":

    #read config file
    with open('config.json','rb') as f:
        config = json.load(f)

    #parse dataset
    ds_parser = DataParser(config)

    trainer = Trainer(config)


