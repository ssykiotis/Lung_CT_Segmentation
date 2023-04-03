from Model import *
import os
import numpy as np
import torch.utils.data as data_utils
import torch.optim      as optim

from tqdm               import tqdm
from loss               import BCEDiceLoss

class Trainer:

    #TODO
    def __init__(self,config,ds_parser):
        self.config = config

        self.ds_parser = ds_parser

        self.train_dl = self.get_dataloader("train")
        self.val_dl   = self.get_dataloader("val")

        self.epochs   = self.config["num_epochs"]

        self.model    = ResUnetPlusPlus(1)
        self.model    = self.model.to(self.config["device"]).to(torch.float16)
        self.model    = torch.compile(self.model)

        self.export_root = self.config['export_root']

        self.bce_dice = BCEDiceLoss()
        self.optimizer = self.create_optimizer()

        if config['enable_lr_schedule']:
            self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer,
                                                          step_size = self.config['decay_step'],
                                                          gamma     = self.config['gamma']
                                                         )


    def train(self):
        best_f1 = self.validate()
        self.save_model()

        for epoch in range(self.num_epochs):
            self.train_one_epoch(epoch+1)
            f1 = self.validate()

            if f1>best_f1:
                best_f1 = f1
                self.save_model()


    
    #TODO
    def train_one_epoch(self,epoch):
        loss_values     = []
        tqdm_dataloader = tqdm(self.train_dl)
        self.model.train()

        for _,batch in enumerate(tqdm_dataloader):

            x, y, _ = batch
            x       = x.to(self.config["device"])
            y       = y.to(self.config["device"]).to(torch.float16)

            self.optimizer.zero_grad()

            y_hat = self.model(x)
            loss  = self.loss_fn(y_hat,y)
            
            loss.backward()
            self.optimizer.step()

            loss_values.append(loss.item())
            average_loss = np.mean(np.array(loss_values))
            
            tqdm_dataloader.set_description('Epoch {}, loss {:.2f}'.format(epoch, average_loss))
    
    #TODO
    def validate(self):
        pass


    #TODO
    def test(self):
        pass


    #TODO
    def loss_fn(self):
        pass


    def get_dataloader(self,mode):

        assert mode in ["train", "val", "test"], 'Mode not supported'

        if mode == "train":
            dataset    = self.ds_parser.get_dataset(mode)
            dataloader = data_utils.DataLoader(dataset,
                                               batch_size = self.batch_size,
                                               shuffle    = True,
                                               pin_memory = True
                                               )
            self.x_min, self.x_max = dataset.get_minmax()
        else:
            dataset    = self.ds_parser.get_dataset(mode)
            dataloader = data_utils.DataLoader(dataset,
                                               batch_size = self.batch_size,
                                               shuffle    = False,
                                               pin_memory = True
                                               )
        return dataloader




            
    def save_model(self):
        if not os.path.exists(self.export_root):
            os.makedirs(self.export_root)
        print('Saving best model...')
        torch.save(self.model.state_dict(), self.export_root.joinpath('best_model.pth'))


    def create_optimizer(self):
        param_optimizer = list(self.model.named_parameters())
        no_decay        = ['bias', 'layer_norm']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay': self.config['weight_decay'],
            },
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        ]
        if self.config['optimizer'].lower() == 'adamw':
            return optim.AdamW(optimizer_grouped_parameters, lr=self.lr)
        elif self.config['optimizer'].lower() == 'adam':
            return optim.Adam(optimizer_grouped_parameters, lr=self.lr)
        elif self.config['optimizer'].lower() == 'sgd':
            return optim.SGD(optimizer_grouped_parameters, lr=self.lr, momentum=self.config['momentum'])
        else:
            raise ValueError

    def export_results(self):
        pass
        