"""Learning-rate schedulers for TransCIF training."""

import numpy as np
import torch


def get_cosine_warmup_scheduler(optimizer, warmup_epochs, total_epochs):
    """Cosine annealing with linear warmup.

    Args:
        optimizer      : torch optimizer
        warmup_epochs  : number of linear-warmup epochs
        total_epochs   : total number of epochs
    """

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch) / float(max(1, warmup_epochs))
        progress = float(epoch - warmup_epochs) / float(
            max(1, total_epochs - warmup_epochs))
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
