import torch
from torch import nn

class BCEDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, input, target):
        pred = input.view(-1)
        truth = target.view(-1)

        # BCE loss
        bce_loss = nn.BCELoss()(pred.double(), truth.double())

        # Dice Loss
        dice_coef = (2.0 * (pred * truth).sum() + 1) / (pred.sum() + truth.sum() + 1)

        return bce_loss + (1 - dice_coef)