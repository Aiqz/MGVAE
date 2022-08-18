import os
import math
from pyexpat import model
from random import sample
import torch
from torch import optim
from .mgvae_mlp import *
import pytorch_lightning as pl
from torchvision import transforms
import torchvision.utils as vutils
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader
from torch.autograd import grad as torch_grad
from .mgvae_mlp_experiment import MGVAE_MLP_Experiment

class MGVAE_MLP_Experiment_EWC(MGVAE_MLP_Experiment):
    def __init__(self, vae_model: MGVAE_MLP, params: dict, ewc_weight=5e4) -> None:
        super().__init__(vae_model, params)
        self.ewc_weight = ewc_weight

        self.init_params = [p.clone().detach() for p in self.model.parameters()]

        def get_fisher_info(n_samples=100):
            n_params = len(self.init_params)
            # sums = [torch.zeros(p.shape).cuda() if self.use_cuda
            #         else torch.zeros(p.shape) for p in self.model.parameters()]
            sums = [torch.zeros(p.shape).cuda() for p in self.model.parameters()]
            
            for i in range(n_samples):
                # sample
                sampled_data = self.model.index_sample(i, current_device='cuda')
                recons_sampled_data = self.model(sampled_data)
                log_probs = self.model.loss_function(*recons_sampled_data, M_N = self.params['kld_weight'])
                loss_grads = torch_grad(outputs=log_probs['loss'],
                                        inputs=list(self.model.parameters()))
                for j in range(n_params):
                    sums[j] = sums[j] + loss_grads[j]**2

            return [s / n_samples for s in sums]

        self.fisher = get_fisher_info(n_samples=100)

    def _ewc_loss(self):
        loss = 0
        n_params = len(self.init_params)
        params = list(self.model.parameters())
        for i in range(n_params):
            loss += torch.sum(self.fisher[i] * (params[i] - self.init_params[i])**2)
        return self.ewc_weight * loss

    def training_step(self, batch, batch_idx, optimizer_idx = 0):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels = labels)
        train_loss = self.model.loss_function(*results,
                                              M_N = self.params['kld_weight'], #al_img.shape[0]/ self.num_train_imgs,
                                              optimizer_idx=optimizer_idx,
                                              batch_idx = batch_idx)
        ewc_loss = self._ewc_loss()

        loss = train_loss['loss'] + ewc_loss
        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)

        return loss
    
    def validation_step(self, batch, batch_idx, optimizer_idx = 0):

        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels = labels)
        val_loss = self.model.loss_function(*results,
                                            M_N = 1.0, #real_img.shape[0]/ self.num_val_imgs,
                                            optimizer_idx = optimizer_idx,
                                            batch_idx = batch_idx)
        val_ewc_loss = self._ewc_loss()
        loss = val_loss['loss'] + val_ewc_loss
        self.log("val_ewc_loss", loss)
        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)

