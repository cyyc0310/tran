"""Supervised PatchTST baseline training."""

import numpy as np
import torch
import torch.nn as nn

from transcif.config import SEQ_LEN, HORIZON, BATCH_SIZE, EPOCHS_SUPERVISED
from transcif.models.base import PatchTSTFixed
from transcif.training.schedulers import get_cosine_warmup_scheduler
from transcif.training.progress import TrainProgress


def train_patchtst(x_train, y_train, epochs=EPOCHS_SUPERVISED, lr=3e-4, device=None,
                   pbar=None):
    """Train a supervised PatchTST baseline."""
    model = PatchTSTFixed(seq_len=SEQ_LEN, horizon=HORIZON)
    if device:
        model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = get_cosine_warmup_scheduler(optimizer, 30, epochs)
    x_t = torch.tensor(x_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)
    if device:
        x_t, y_t = x_t.to(device), y_t.to(device)
    n = len(x_t)
    model.train()
    for epoch in range(epochs):
        idx = torch.randperm(n)[:min(BATCH_SIZE, n)]
        pred = model(x_t[idx])
        loss = nn.functional.l1_loss(pred, y_t[idx])
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if pbar is not None:
            pbar(epoch, epochs, loss.item())
    model.eval()
    if pbar is not None:
        pbar.finish()
    return model
