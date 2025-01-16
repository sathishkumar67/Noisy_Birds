from __future__ import annotations
from typing import Tuple
import gin
import torch
import torch.nn as nn
import lightning as L
from dataclasses import dataclass
from mae.encoder import *
from mae.decoder import *

@gin.configurable
@dataclass
class MAEWrapperConfig:
    lr: float
    batch_size: int
    num_epochs: int
    weight_decay: float
    eps: float
    seed: int
    betas: Tuple[float, float]
    gpu_count: int
    device: str

class MAE(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        encoder_op, mask, ids_restore = self.encoder(x)
        decoder_op = self.decoder((encoder_op, mask, ids_restore), x)
        return decoder_op

class MAEWrapper(L.LightningModule):
    def __init__(self, mae: nn.Module, config: MAEWrapperConfig):
        super().__init__()
        self.config = config
        self.model = mae
        self.optimizer = self.configure_optimizers()

    def training_step(self, batch, batch_idx):
        self.model.train()
        optimizer = self.optimizers()
        optimizer.zero_grad()
        batch, _ = batch
        _, loss = self.model(batch)
        self.log("Train_Loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.lr, betas=self.config.betas, eps=self.config.eps, weight_decay=self.config.weight_decay)
        return optimizer            