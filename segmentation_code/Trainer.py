import os
import pandas as pd
import numpy  as np


import torch.utils.data as data_utils
import torch.optim      as optim

from Model                       import *
from PIL                         import Image
from tqdm                        import tqdm
from loss                        import BCEDiceLoss
from torchmetrics.classification import BinaryF1Score
from torchmetrics.classification import BinaryConfusionMatrix

class Trainer:

    def __init__(self,config,ds_parser):
        self.config = config

        self.ds_parser = ds_parser

        self.train_dl = self.get_dataloader("train")
        self.val_dl   = self.get_dataloader("val")

        self.epochs   = self.config["num_epochs"]
        self.lr       = self.config["learning_rate"]

        self.model    = ResUnetPlusPlus(1)
        self.model    = self.model.to(self.config["device"]).to(torch.float32)
        # self.model    = torch.compile(self.model)

        self.export_root = self.config['export_root']

        self.loss_fn = BCEDiceLoss()
        self.optimizer = self.create_optimizer()

        if config['enable_lr_schedule']:
            self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer,
                                                          step_size = self.config['decay_step'],
                                                          gamma     = self.config['gamma']
                                                         )

    def train(self):
        best_f1 = self.validate()
        self.save_model()

        for epoch in range(self.epochs):
            self.train_one_epoch(epoch+1)
            f1 = self.validate()

            if f1>best_f1:
                best_f1 = f1
                self.save_model()

    def train_one_epoch(self,epoch):
        loss_values     = []
        tqdm_dataloader = tqdm(self.train_dl)
        self.model.train()

        for _,batch in enumerate(tqdm_dataloader):

            x, y, _ = batch
            x       = x.to(self.config["device"])
            y       = y.to(self.config["device"]).to(torch.float32)

            self.optimizer.zero_grad()

            y_hat = self.model(x)
            loss  = self.loss_fn(y_hat,y)
            
            loss.backward()
            self.optimizer.step()

            loss_values.append(loss.item())
            average_loss = np.mean(np.array(loss_values))

            tqdm_dataloader.set_description('Epoch {}, loss {:.2f}'.format(epoch, average_loss))
    
    
    def validate(self):
        self.model.eval()
        F1_Score = BinaryF1Score().to(self.config["device"])
        with torch.no_grad():
            tqdm_dataloader = tqdm(self.val_dl)
            for _,batch in enumerate(tqdm_dataloader):
                x, y, _ = batch 
                x       = x.to(self.config["device"])
                y       = y.to(self.config["device"]).to(torch.float32)                
            
                y_hat   = self.model(x)

                f1      = F1_Score.update(y_hat, y)
                f1_mean = F1_Score.compute()

                tqdm_dataloader.set_description('Validation, F1 {:.2f}'.format(f1_mean))

        return f1_mean

    def test(self):

        self.model.eval()
        F1_Score = BinaryF1Score().to(self.config["device"])
        Confusion_Matrix = BinaryConfusionMatrix().to(self.config["device"])
        self.test_dl = self.get_dataloader("test")
        img_names = self.test_dl.dataset.img_names
        results = pd.DataFrame(data    = None,
                               index   = img_names,
                               columns = ['tn', 'fp', 'fn', 'tp']
                               )

        tqdm_dataloader = tqdm(self.test_dl)

        with torch.no_grad():
            for _,batch in enumerate(tqdm_dataloader):
                x, y, names = batch 
                x       = x.to(self.config["device"])
                y       = y.to(self.config["device"]).to(torch.float32)                
            
                y_hat   = self.model(x)

                f1      = F1_Score.update(y_hat, y)
                f1_mean = F1_Score.compute()
                for i in range(y_hat.shape[0]):
                    c  = Confusion_Matrix(y_hat[i],y[i])
                    tn = c[0,0].detach().cpu().numpy().item()
                    fp = c[0,1].detach().cpu().numpy().item()
                    fn = c[1,0].detach().cpu().numpy().item()
                    tp = c[1,1].detach().cpu().numpy().item()

                    results.loc[names[i]] = (tn, fp, fn, tp)

                    self.export_images(x,y_hat,y,names)

                tqdm_dataloader.set_description('Test, F1 {:.2f}'.format(f1_mean))

            

        results.to_csv(f'{self.export_path}/results.csv')

        return f1_mean



    def get_dataloader(self,mode):

        assert mode in ["train", "val", "test"], 'Mode not supported'

        if mode == "train":
            dataset    = self.ds_parser.get_dataset(mode)
            dataloader = data_utils.DataLoader(dataset,
                                               batch_size = self.config['batch_size'],
                                               shuffle    = True,
                                               pin_memory = True,
                                               drop_last  = False
                                               )
            self.x_min, self.x_max = dataset.get_minmax()
        else:
            dataset    = self.ds_parser.get_dataset(mode)
            dataloader = data_utils.DataLoader(dataset,
                                               batch_size = self.config['batch_size'],
                                               shuffle    = False,
                                               pin_memory = True,
                                               drop_last  = False
                                               )
        return dataloader
            
    def save_model(self):
        if not os.path.exists(self.export_root):
            os.makedirs(self.export_root)
        print('Saving best model...')
        torch.save(self.model.state_dict(), f'{self.export_root}/best_model.pth')


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
        

    def export_images(self,x,y_hat,y,img_names):

        imgs_path = f'{self.export_root}/imgs'
        if not os.path.exists(imgs_path):
            os.makedirs(imgs_path)

        for i in range(x.shape[0]):
            img_norm = (x[i]-x[i].min())/(x[i].max()-x[i].min())
            img_norm = (img_norm* 255).astype('uint8')

            png_image = Image.fromarray(img_norm)
            png_path  = f'{imgs_path}/{img_names[i]}_original.png'
            png_image.save(png_path)

            mask    = y_hat[i].astype('uint8')
            mask_gt = y[i].astype('uint8')

            R,G,B = mask.copy(), mask.copy(), mask_gt.copy()
            R       = mask*255
            G[mask] = 0
            B[mask] = mask_gt*255
            mask_rgb = np.stack((R,G,B), axis=2)

            png_mask = Image.fromarray(mask_rgb,mode = 'RGB')
            mask_path = f'{imgs_path}/{img_names[i]}_masks.png'
            png_mask.save(mask_path)

            png_image_rgb = png_image.convert(mode = 'RGB')
            image_overlay = Image.blend(png_image_rgb, png_mask, 0.3)
            png_overlay_path = f'{imgs_path}/{img_names[i]}_overlay.png'
            image_overlay.save(png_overlay_path)
            