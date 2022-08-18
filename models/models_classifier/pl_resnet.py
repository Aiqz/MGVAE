from turtle import forward
import torch
import torchmetrics
from torch import nn
from torch import optim
from torch.nn import functional as F
import pytorch_lightning as pl
from ..utils import indices
from .resnet import resnet32, resnet20

class pl_resnet(pl.LightningModule):
    def __init__(self, num_classes=2, criterion=None):
        super(pl_resnet, self).__init__()

        self.model = resnet20(num_classes=num_classes)
        self.loss_func = criterion
        self.indices = indices
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        input, label = batch
        # print(label)
        output = self.forward(input)
        # train_loss = self.loss_func(output, label)
        train_loss = self.loss_func(output, label).mean()

        acsa, gm, _, _, accuracy = self.indices(torch.argmax(output, dim=1).cpu(), label.cpu())
        self.log('train_acc', accuracy, on_epoch=True)
        self.log('train_acsa', acsa, on_epoch=True)
        self.log('train_gm', gm, on_epoch=True)
        self.log('train_loss', train_loss, on_step=True, on_epoch=True)
        return train_loss

    def validation_step(self, batch, batch_idx):
        input, label = batch
        output = self.forward(input)
        # val_loss = self.loss_func(output, label)
        val_loss = self.loss_func(output, label).mean()

        # self.acc(output, label)
        acsa, gm, _, _, accuracy = self.indices(torch.argmax(output, dim=1).cpu(), label.cpu())
        self.log('valid_acc', accuracy, on_step=True, on_epoch=True)
        self.log('valid_acsa', acsa, on_epoch=True)
        self.log('valid_gm', gm, on_epoch=True)
        self.log('valid_loss', val_loss, on_step=True, on_epoch=True)
        # self.log('valid_acc', self.acc, on_step=True, on_epoch=True)
        return val_loss
    
    
    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(), 
            lr=0.01,
            weight_decay=0.0002)
        schedule = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        # schedule = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        return ([optimizer], [schedule])
