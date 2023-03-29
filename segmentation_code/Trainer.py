from Model import *
import torch.utils.data as data_utils
from loss import BCEDiceLoss

class Trainer:

    #TODO
    def __init__(self,config,ds_parser):
        self.config = config

        self.ds_parser = ds_parser

        self.train_dl,self.val_dl = self.get_dataloaders("train")

        self.epochs = self.config["num_epochs"]

        self.model = ResUnetPlusPlus(1)
        self.model = self.model.to(self.config["device"]).to(torch.float16)

        self.bce_dice = BCEDiceLoss()


        self.optimizer = self.create_optimizer()
        if args.enable_lr_schedule:
            self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer,
                                                          step_size=args.decay_step,
                                                          gamma=args.gamma
                                                         )


    def train(self):
        best_f1 = self.validate()

        for epoch in range(self.num_epochs):
            self.train_one_epoch(epoch+1)
            f1 = self.validate()

            if f1>best_f1:
                best_f1 = f1
                self.save_model()


    
    #TODO
    def train_one_epoch(self):
        pass
    
    #TODO
    def validate(self):
        pass


    #TODO
    def test(self):
        pass


    #TODO
    def loss_fn(self):
        pass


    def get_dataloaders(self,mode):

        if mode == "train":

            train_dataset,val_dataset = self.ds_parser.get_dataset("train")

            self.x_min = train_dataset.x_min
            self.x_max = train_dataset.x_max

            train_dataloader = data_utils.DataLoader(train_dataset,
                                                    batch_size = self.batch_size,
                                                    shuffle    = True,
                                                    pin_memory = True
                                                    )
            
            val_dataloader  = data_utils.DataLoader(val_dataset,
                                                    batch_size = self.batch_size,
                                                    shuffle = False,
                                                    pin_memory = True
                                                   )
            
            return train_dataloader, val_dataloader
        
        else:
            test_dataset = self.ds_parser.get_dataset("test")

            test_dataloader = data_utils.DataLoader(test_dataset,
                                                    batch_size = 1,
                                                    shuffle = False,
                                                    pin_memory = True
                                                    )
            return test_dataloader




    #TODO        
    def save_model(self):
        pass


    def create_optimizer(self):
        pass

    def export_results(self):
        pass
        