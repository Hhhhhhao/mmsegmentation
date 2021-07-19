import torch
import torch.nn as nn
import torch.nn.functional as F

from .cross_entropy_loss import CrossEntropyLoss
from .focal_loss import FocalLoss
from .dice_loss import DiceLoss
from .lovasz_loss import LovaszLoss
from ..builder import LOSSES

@LOSSES.register_module()
class CombinedLoss(nn.Module):
    def __init__(self,
                 # TODO: add loss arguments
                 losses=['DiceLoss', 'FocalLoss'],
                 lambdas=[1.0, 1.0],
                 class_weight=None,
                 loss_weight=1.0):
        super(CombinedLoss, self).__init__()
        assert len(losses) == len(lambdas)

        self.lambdas = lambdas
        self.losses = []
        for loss_name in losses:
            if loss_name == 'LovaszLoss':
                loss_fn = eval(loss_name)(loss_weight=loss_weight)
            else:
                loss_fn = eval(loss_name)(class_weight=class_weight, loss_weight=loss_weight)
            self.losses.append(loss_fn)
    

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        loss = 0.0
        for loss_fn, loss_lambda in zip(self.losses, self.lambdas):
            loss += loss_fn(
                cls_score=cls_score, 
                label=label, 
                weight=weight, 
                avg_factor=avg_factor, 
                reduction_override=reduction_override, 
                **kwargs) * loss_lambda
        return loss

