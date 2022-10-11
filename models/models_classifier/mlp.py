import torch
import torchmetrics
from torch import nn
from torch import optim
from torch.nn import functional as F
import pytorch_lightning as pl
from ..utils import indices

class NormedLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return out

class simple_mlp(pl.LightningModule):

    def __init__(self, input_dim=784, output_dim=10, criterion=None, use_norm=False):
        super(simple_mlp, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim


        self.layer_1 = nn.Linear(self.input_dim, 256)
        self.layer_2 = nn.Linear(256, 128)
        if use_norm:
            self.layer_3 = NormedLinear(128, self.output_dim)
        else:
            self.layer_3 = nn.Linear(128, self.output_dim)
        self.activation_1 = nn.LeakyReLU(0.1)
        self.activation_2 = nn.Softmax(dim=1)

        self.acc = torchmetrics.Accuracy()
        # self.loss_func = nn.BCELoss()
        # self.loss_func = nn.CrossEntropyLoss()
        self.loss_func = criterion
        self.indices = indices

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        x = self.layer_1(x)
        x = self.activation_1(x)

        x = self.layer_2(x)
        x = self.activation_1(x)

        x = self.layer_3(x)
        x = self.activation_2(x)
        return x

    def training_step(self, batch, batch_idx):
        input, label = batch
        # print(label)
        output = self.forward(input)
        train_loss = self.loss_func(output, label).mean()

        acsa, gm, _, _, accuracy = self.indices(torch.argmax(output, dim=1).cpu(), label.cpu())
        self.log('train_acc', accuracy, on_epoch=True)
        self.log('train_acsa', acsa, on_epoch=True)
        self.log('train_gm', gm, on_epoch=True)
        self.log('train_loss', train_loss, on_step=True, on_epoch=True)
        return train_loss

    def validation_step(self, batch, batch_idx):
        input, label = batch

        # output = self.forward(input).squeeze()
        output = self.forward(input)
        # print(torch.argmax(output, dim=1))
        # print(label)
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
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        schedule = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        return ([optimizer], [schedule])
