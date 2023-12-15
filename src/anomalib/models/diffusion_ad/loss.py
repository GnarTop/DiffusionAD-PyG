import torch
import torch.nn as nn
import torch.nn.functional as F

class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=4, logits=False, reduce=True): 
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


class DiffusionADLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.focal = BinaryFocalLoss()
        self.smL1 = nn.SmoothL1Loss()

    def forward(self, pred_mask, anomaly_mask, noise_loss):
        focal_loss = self.focal(pred_mask, anomaly_mask)
        smL1_loss = self.smL1(pred_mask, anomaly_mask)
        total_loss = noise_loss + 5*focal_loss + smL1_loss
        return total_loss
