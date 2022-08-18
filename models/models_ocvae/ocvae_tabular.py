import os
from ast import Raise
from logging import raiseExceptions
import pytorch_lightning as pl
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.nn.utils import weight_norm
from torch.autograd import grad as torch_grad
import torchvision.utils as vutils
from ..utils import *
from typing import List, Callable, Union, Any, TypeVar, Tuple
Tensor = TypeVar('Tensor')


class Condition_VAE_Tabular(pl.LightningModule):
    def __init__(self,
                 input_dim: int = 166,
                 latent_dim: int = 40,
                 num_classes: int = 2,
                 hidden_size: int = 300,
                 cur_device = 'cuda', 
                 params:dict = None,
                 **kwargs) -> None:
        super(Condition_VAE_Tabular, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.cur_device = cur_device
        self.params = params

        # Build Encoder
        self.encoder = nn.Sequential(
                nn.Linear(self.input_dim + num_classes, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),
                )

        # self.hidden_size = 1536
        self.fc_mu = nn.Linear(self.hidden_size, self.latent_dim)
        self.fc_logvar = nn.Linear(self.hidden_size, self.latent_dim)

        # Build Decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim + num_classes, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.input_dim),
            # [0-1]
            # nn.Sigmoid(),
            # [-1, 1]
            nn.Tanh(),
            )
        
        # self.save_hyperparameters(ignore=["exemplar_data"])


    def encode(self, input):
        # input_size: [B, C, H, W]
        h = self.encoder(input)
        h = h.view(input.size(0), -1)

        mu = self.fc_mu(h)
        log_var = self.fc_logvar(h)
        # log_var = self.prior_log_variance * torch.ones((mu.shape[0], self.latent_dim)).to(self.cur_device)

        return [mu, log_var]
    
    def decode(self, z):

        result = self.decoder(z)
        result = result.view(-1, self.input_dim)

        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        y = kwargs['labels']
        y = F.one_hot(y, num_classes=self.num_classes).float()

        x = torch.cat([input.view(-1, self.input_dim), y], dim = 1)
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        z_and_y = torch.cat([z, y], dim = 1)

        x_mean = self.decode(z_and_y)

        return  [x_mean, input, z, mu, log_var]
    
    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        recons = args[0]
        input = args[1]
        z = args[2]
        mu = args[3]
        log_var = args[4]

        kld_weight = kwargs['M_N']

        # KLD
        KL = -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1)

        # Recons
        RE = -F.mse_loss(input.view(-1, self.input_dim), recons.view(-1, self.input_dim), reduction='none')
        RE = torch.sum(RE, dim=1)

        RE = torch.mean(RE)
        KL = torch.mean(KL)

        loss = -RE + kld_weight * KL

        return {'loss': loss, 'Reconstruction_Loss':RE.detach(), 'KLD':-KL.detach()}

    def sample(self,
               num_samples:int,
               current_device: int,
               **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        y = kwargs['labels']
        y = F.one_hot(y, num_classes=self.num_classes).float()

        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        z = torch.cat([z, y], dim=1)
        samples = self.decode(z)
        return samples

    
    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x, **kwargs)[0]

    def training_step(self, batch, batch_idx, optimizer_idx = 0):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels = labels)
        train_loss = self.loss_function(*results,
                                              M_N = self.params['kld_weight'], #al_img.shape[0]/ self.num_train_imgs,
                                              optimizer_idx=optimizer_idx,
                                              batch_idx = batch_idx)

        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)

        return train_loss['loss']

    def validation_step(self, batch, batch_idx, optimizer_idx = 0):

        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels = labels)
        val_loss = self.loss_function(*results,
                                            M_N = 1.0,
                                            optimizer_idx = optimizer_idx,
                                            batch_idx = batch_idx)

        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)

    def on_validation_end(self) -> None:
        self.sample_images()
        
    def sample_images(self):
        # Get sample reconstruction image            
        test_input, test_label = next(iter(self.trainer.datamodule.test_dataloader()))
        test_input = test_input.to('cuda')
        test_label = test_label.to('cuda')

        # test_input, test_label = batch
        recons = self.generate(test_input, labels = test_label)

        # vutils.save_image(recons.data,
        #                   os.path.join(self.logger.log_dir , 
        #                                "Reconstructions", 
        #                                f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png"),
        #                   normalize=True,
        #                   nrow=12)

        try:
            samples = self.sample(100,
                                self.cur_device,
                                labels = test_label)
            # vutils.save_image(samples.cpu().data,
            #                   os.path.join(self.logger.log_dir , 
            #                                "Samples",      
            #                                f"{self.logger.name}_Epoch_{self.current_epoch}.png"),
            #                   normalize=True,
            #                   nrow=10)
        except Warning:
            pass

    def configure_optimizers(self):

        optims = []
        scheds = []

        optimizer = optim.Adam(self.parameters(),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])
        optims.append(optimizer)
        # Check if more than 1 optimizer is required (Used for adversarial training)
        try:
            if self.params['LR_2'] is not None:
                optimizer2 = optim.Adam(getattr(self.model,self.params['submodel']).parameters(),
                                        lr=self.params['LR_2'])
                optims.append(optimizer2)
        except:
            pass

        try:
            if self.params['scheduler_gamma'] is not None:
                scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                             gamma = self.params['scheduler_gamma'])
                scheds.append(scheduler)

                # Check if another scheduler is required for the second optimizer
                try:
                    if self.params['scheduler_gamma_2'] is not None:
                        scheduler2 = optim.lr_scheduler.ExponentialLR(optims[1],
                                                                      gamma = self.params['scheduler_gamma_2'])
                        scheds.append(scheduler2)
                except:
                    pass
                return optims, scheds
        except:
            return optims

