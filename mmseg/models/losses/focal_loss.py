
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from ..builder import LOSSES


@LOSSES.register_module()
class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True, loss_weight=1.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average
        self.loss_weight = loss_weight

    def forward(self, cls_score, label, **kwargs):
        if cls_score.dim()>2:
            cls_score = cls_score.view(cls_score.size(0),cls_score.size(1),-1)  # N,C,H,W => N,C,H*W
            cls_score = cls_score.transpose(1,2)    # N,C,H*W => N,H*W,C
            cls_score = cls_score.contiguous().view(-1,cls_score.size(2))   # N,H*W,C => N*H*W,C
        label = label.view(-1, 1)

        logpt = F.log_softmax(cls_score, dim=-1)
        logpt = logpt.gather(1, label)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=cls_score.data.type():
                self.alpha = self.alpha.type_as(cls_score.data)
            at = self.alpha.gather(0,label.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean() * self.loss_weight
        else: return loss.sum() * self.loss_weight