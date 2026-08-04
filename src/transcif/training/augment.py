"""Missing-pattern augmentation for robust TransCIF training."""

import numpy as np
import torch


class MissingMaskAugmentor:
    """Randomly drop out time steps during training to improve robustness.

    Supports two modes:
        point: independent Bernoulli per time step
        block: contiguous block missing (simulates sensor outage)
    """

    def __init__(self, prob=0.05, mode='point', min_block=1, max_block=12):
        self.prob = prob
        self.mode = mode
        self.min_block = min_block
        self.max_block = max_block

    def __call__(self, x):
        """x : (B, L) numpy or torch tensor -> augmented x, mask"""
        is_np = isinstance(x, np.ndarray)
        if is_np:
            x = torch.from_numpy(x).float()
        B, L = x.shape
        mask = torch.ones(B, L)
        if self.mode == 'point':
            mask = torch.bernoulli(torch.full((B, L), 1.0 - self.prob))
        elif self.mode == 'block':
            for i in range(B):
                j = 0
                while j < L:
                    if torch.rand(1).item() < self.prob:
                        blen = np.random.randint(self.min_block, self.max_block + 1)
                        mask[i, j:j + blen] = 0.0
                        j += blen
                    else:
                        j += 1
        x_aug = x * mask
        if is_np:
            return x_aug.numpy(), mask.numpy()
        return x_aug, mask
