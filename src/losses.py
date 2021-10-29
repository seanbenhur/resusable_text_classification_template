import torch.nn as nn

def bcewithlogits_loss_fn(outputs, targets, reduction=None):
    return nn.BCEWithLogitsLoss(reduction)(outputs, targets.view(-1, 1))


def crossentropy_loss_fn(outputs, targets, reduction=None):
    targets = targets.long()
    return nn.CrossEntropyLoss(reduction)(outputs, targets)


def mse_loss_fn(outputs, targets, reduction=None):
    return nn.MSELoss(reduction)(outputs, targets.view(-1, 1))
