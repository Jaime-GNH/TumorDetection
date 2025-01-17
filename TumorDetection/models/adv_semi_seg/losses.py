import torch
import torch.nn.functional as torch_fun
import torch.nn as nn
from torch.autograd import Variable


class CrossEntropy2d(nn.Module):

    def __init__(self, ignore_label=255, reduction='mean'):
        super(CrossEntropy2d, self).__init__()
        self.ignore_label = ignore_label
        self.reduction = reduction

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict: (n, c, h, w)
                target: (n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))
        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask].to(device=predict.device)
        if not target.data.dim():
            return Variable(torch.zeros(1))
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        loss = torch_fun.cross_entropy(predict, target, weight=weight,
                                       reduction=self.reduction)
        return loss


class BCEWithLogitsLoss2d(nn.Module):

    def __init__(self, ignore_label=255, reduction='mean'):
        super(BCEWithLogitsLoss2d, self).__init__()
        self.ignore_label = ignore_label
        self.reduction = reduction

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict: (n, 1, h, w)
                target: (n, 1, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 4
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(2), "{0} vs {1} ".format(predict.size(2), target.size(2))
        assert predict.size(3) == target.size(3), "{0} vs {1} ".format(predict.size(3), target.size(3))
        # n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask].to(device=predict.device)
        if not target.data.dim():
            return Variable(torch.zeros(1))
        predict = predict[target_mask]
        loss = torch_fun.binary_cross_entropy_with_logits(predict, target, weight=weight,
                                                          reduction=self.reduction)
        return loss
