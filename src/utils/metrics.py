import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, is_3d=False):
        super().__init__()
        self.dims = (1, 2, 3, 4) if is_3d else (1, 2, 3)

    def forward(self, logits, target):
        probs = torch.sigmoid(logits)
        probs = probs.view(probs.size(0), -1)
        target = target.view(target.size(0), -1)
        inter = (probs * target).sum(1)
        union = probs.sum(1) + target.sum(1)
        return 1 - ((2 * inter + 1e-6) / (union + 1e-6)).mean()

def dice_metric(pred, target, is_3d=False):
    dims = (1, 2, 3, 4) if is_3d else (1, 2, 3)
    pred = (torch.sigmoid(pred) > 0.5).float()
    intersection = (pred * target).sum(dims)
    union = pred.sum(dims) + target.sum(dims)
    return ((2 * intersection + 1e-6) / (union + 1e-6)).mean().item()

def binary_metrics(pred, target):
    pred = (torch.sigmoid(pred) > 0.5).float()
    tp = (pred * target).sum()
    fp = (pred * (1 - target)).sum()
    fn = ((1 - pred) * target).sum()
    tn = ((1 - pred) * (1 - target)).sum()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    return {
        "accuracy": accuracy.item(),
        "precision": precision.item(),
        "recall": recall.item(),
        "f1": f1.item()
    }
