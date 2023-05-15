import torch
from torch import nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, input, target):
        pred = input.view(-1)
        truth = target.view(-1)

        # BCE loss
        # bce_loss = nn.BCELoss()(pred, truth)

        # Dice Loss
        dice_coef = (2.0 * (pred * truth).sum() + 1) / (pred.sum() + truth.sum() + 1)

        # return bce_loss + (1 - dice_coef)
        return 1-dice_coef
    

class FocalLoss(nn.Module):

    def __init__(self,alpha = 0,gamma = 2, reduction = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self,inputs,targets):
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = inputs * targets + (1 - inputs) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        # Check reduction option and return loss accordingly
        if self.reduction == "none":
            pass
        elif self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        else:
            raise ValueError(
                f"Invalid Value for arg 'reduction': '{self.reduction} \n Supported reduction modes: 'none', 'mean', 'sum'"
            )
        return loss
        
class FocalTverskyLoss(nn.Module):
    def __init__(self,alpha = 0.7, gamma = 3/4, smooth = 1):
        super().__init__()
        self.alpha  = alpha
        self.gamma  = gamma
        self.smooth = smooth
    
    def tversky_index(self,y_true, y_pred):
        y_true_pos = y_true.reshape(-1)
        y_pred_pos = ((y_pred>=0.5)*1).reshape(-1)

        tp = (y_true_pos * y_pred_pos).sum()
        fn = (y_true_pos * (1 - y_pred_pos)).sum()
        fp = ((1 - y_true_pos) * y_pred_pos).sum()

        nom = tp + self.smooth
        den = tp + self.alpha*fn +(1-self.alpha)*fp +self.smooth

        return nom/den
        
    def forward(self,y_true, y_pred):
        tversky_index = self.tversky_index(y_true,y_pred)

        return (1-tversky_index)**self.gamma

