import os
import pandas as pd
import numpy  as np


import torch.utils.data as data_utils
import torch.optim      as optim

from lungmask.utils import postprocessing

from Model                       import *
from PIL                         import Image
from tqdm                        import tqdm
from loss                        import DiceLoss, FocalLoss, FocalTverskyLoss
from torchmetrics.classification import BinaryF1Score
from torchmetrics.classification import BinaryConfusionMatrix

from ptflops import get_model_complexity_info

class Trainer:

    def __init__(self,config,ds_parser):
        self.config = config

        self.ds_parser = ds_parser

        self.train_dl = self.get_dataloader("train")
        print("Training Set Length:",len(self.train_dl.dataset))
        print('First image:', self.train_dl.dataset.img_names[0])
 
        self.val_dl   = self.get_dataloader("val")

        self.epochs   = self.config["num_epochs"]
        self.lr       = self.config["learning_rate"]

        self.model    = ResUnetPlusPlus()
        self.model    = self.model.to(self.config["device"])
        # self.model    = self.model.to(self.config["device"]).to(torch.float32)
        flops, params = get_model_complexity_info(self.model, input_res=(1, 256, 256), as_strings=True, print_per_layer_stat=False)
        print('      - Flops:  ' + flops)
        print('      - Params: ' + params)

        # self.model    = torch.compile(self.model)

        self.export_root = self.config['export_root']

        # alpha = self.train_dl.dataset.y.sum()/torch.prod(torch.tensor(self.train_dl.dataset.y.shape))

        # self.loss_fn = FocalLoss(alpha = 0.25,gamma = 5)
        self.loss_fn = DiceLoss()
        # print(f'Alpha: {alpha}')
        self.optimizer = self.create_optimizer()

        if config['enable_lr_schedule']:
            self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer,
                                                          step_size = self.config['decay_step'],
                                                          gamma     = self.config['gamma']
                                                         )

    def train(self):
        loss_monitoring = dict()
        best_f1,_ = self.validate()
        self.save_model()
        self.best_epoch = 0

        for epoch in range(self.epochs):
            train_loss = self.train_one_epoch(epoch+1)
            f1, val_loss = self.validate()
            loss_monitoring[epoch+1] = [train_loss,val_loss]
            if self.lr_scheduler:
                self.lr_scheduler.step()

            if f1>best_f1:
                best_f1 = f1
                self.best_epoch = epoch+1
                self.save_model()

        losses = pd.DataFrame.from_dict(loss_monitoring, orient = 'index')
        losses.to_csv(f'{self.export_root}/loss_monitoring.csv')

    def train_one_epoch(self,epoch):
        loss_values     = []
        tqdm_dataloader = tqdm(self.train_dl)
        self.model.train()

        for _,batch in enumerate(tqdm_dataloader):

            x, y, _,_ = batch
            x       = x.to(self.config["device"])
            y       = y.to(self.config["device"]).to(torch.float32)

            y_hat = self.model(x)
            loss  = self.loss_fn(y_hat,y)

            
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            loss_values.append(loss.item())
            average_loss = np.mean(np.array(loss_values))

            tqdm_dataloader.set_description('Epoch {}, loss {:.2f}'.format(epoch, average_loss))
        return average_loss
    
    
    def validate(self):
        self.model.eval()
        loss_values = []
        F1_Score = BinaryF1Score().to(self.config["device"])
        with torch.no_grad():
            tqdm_dataloader = tqdm(self.val_dl)
            for _,batch in enumerate(tqdm_dataloader):
                x, y, _ ,flag = batch 
                x       = x.to(self.config["device"])
                y       = y.to(self.config["device"]).to(torch.float32)   
          
                y_hat   = self.model(x)
                y_hat   = torch.round(y_hat)
                loss  = self.loss_fn(y_hat,y)

                # y_hat[flag!=1] = 0 

                f1      = F1_Score.update(y_hat, y)
                f1_mean = F1_Score.compute()

                if not torch.isnan(loss):
                    loss_values.append(loss.item())
                
                loss_mean = np.mean(np.array(loss_values))
                tqdm_dataloader.set_description('Validation, F1 {:.2f}, Loss {:.2f}'.format(f1_mean,loss_mean))

        return f1_mean, loss_mean
    
    def test_per_patient(self):

        self.load_best_model()
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

        patients, num_frames = self.get_patient_frames(img_names)
        img_size = self.config['img_size']

        images_per_patient       = [np.zeros((nframe,img_size,img_size)) for nframe in num_frames]
        predictions_per_patient  = [np.zeros((nframe,img_size,img_size)) for nframe in num_frames]
        ground_truth_per_patient = [np.zeros((nframe,img_size,img_size)) for nframe in num_frames]

        with torch.no_grad():
            for _,batch in enumerate(tqdm_dataloader):  
                x, y, names,flag = batch      
                x       = x.to(self.config["device"])
                y       = y.to(self.config["device"]).to(torch.float32)                
                
                y_hat   = self.model(x)
                y_hat   = torch.round(y_hat)

                for i, name in enumerate(names):
                    patient = patients.index(name.split('/')[0])
                    frame   = int(name.split('/')[1]) - 1

                    images_per_patient[patient][frame]       = x[i].detach().cpu().numpy()
                    predictions_per_patient[patient][frame]  = y_hat[i].detach().cpu().numpy()
                    ground_truth_per_patient[patient][frame] = y[i].detach().cpu().numpy()
            
            for batch in zip(patients, images_per_patient, predictions_per_patient, ground_truth_per_patient):
                patient, x, y_hat, y = batch

                names = [f'{patient}/{i+1}' for i in range(x.shape[0])]

                y_hat_post = postprocessing(y_hat.astype(np.uint8))
                f1      = F1_Score.update(torch.tensor(y_hat_post).to(torch.float32), torch.tensor(y).to(torch.float32))
                f1_mean = F1_Score.compute()

                self.export_images(x, y_hat_post, y, names)

                print(' Test, F2 {:.2f}'.format(f1_mean))



    def get_patient_frames(self,img_names:list):
        images = pd.DataFrame(data = img_names,
                                 columns = ['img_names'])

        images['patient'] = images['img_names'].apply(lambda x:x.split('/')[0])
        images_grouped    = images.groupby('patient').count().reset_index()

        return images_grouped['patient'].values.tolist(), images_grouped['img_names'].values.tolist()

    def test(self):

        self.load_best_model()

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
                x, y, names,flag = batch 
                x       = x.to(self.config["device"])
                y       = y.to(self.config["device"]).to(torch.float32)                
                
                y_hat   = self.model(x)
                y_hat   = torch.round(y_hat)
                
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

        results.to_csv(f'{self.export_root}/results.csv')

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

        x     = x.detach().cpu().numpy().squeeze() if torch.is_tensor(x) else x
        y_hat = y_hat.detach().cpu().numpy().squeeze() if torch.is_tensor(y_hat) else y_hat
        y     = y.detach().cpu().numpy().squeeze() if torch.is_tensor(y) else y

        imgs_path = f'{self.export_root}/imgs'
        if not os.path.exists(imgs_path):
            os.makedirs(imgs_path)

        for i in range(x.shape[0]):
            img_norm = (x[i]-x[i].min())/(x[i].max()-x[i].min()+1e-9)

            img_norm = (img_norm* 255).astype('uint8')

            patient_name   = img_names[i].split('/')[0]
            img_num        = img_names[i].split('/')[1]
            patient_folder = f'{imgs_path}/{patient_name}'
            if not os.path.exists(patient_folder):
                os.makedirs(patient_folder)

            png_image = Image.fromarray(img_norm)
            png_path  = f'{patient_folder}/{img_num}_original.png'
            # png_image.save(png_path)

            mask    = y_hat[i].astype('uint8')
            mask_gt = y[i].astype('uint8')

            R       = mask*255
            G       = mask*0
            B       = mask_gt*255
            mask_rgb = np.stack((R,G,B), axis=2)

            png_mask = Image.fromarray(mask_rgb,mode = 'RGB')
            mask_path = f'{patient_folder}/{img_num}_masks.png'
            # png_mask.save(mask_path)

            png_image_rgb = png_image.convert(mode = 'RGB')
            image_overlay = Image.blend(png_image_rgb, png_mask, 0.3)
            png_overlay_path = f'{patient_folder}/{img_num}_overlay.png'
            image_overlay.save(png_overlay_path)

    def load_best_model(self):
        try:
            self.model.load_state_dict(torch.load(f'{self.export_root}/best_model.pth'))
            self.model.to(self.config["device"])
            print('Model loaded successfully')
        except:
            print('Failed to load best model, continue testing with current model...')