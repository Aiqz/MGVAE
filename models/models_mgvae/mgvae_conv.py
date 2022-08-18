import os
from ast import Raise
from logging import raiseExceptions
from weakref import ref
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

class block(nn.Module):
    def __init__(self, input_size, output_size, stride=1, kernel=3, padding=1):
        super(block, self).__init__()
        # self.normalization = nn.BatchNorm2d(input_size)
        self.conv1 = weight_norm(nn.Conv2d(input_size, output_size, kernel_size=kernel, stride=stride, padding=padding, bias=True))
        self.activation = nn.ELU()
        self.f = nn.Sequential(self.activation, self.conv1)
    
    def forward(self, x):
        return x + self.f(x)


class MGVAE_Conv(pl.LightningModule):
    def __init__(self,
                 latent_dim: int = 40,
                 hidden_size: int = 300,
                 number_components: int = 500,
                 exemplar_data = None,
                 cur_device = 'cuda', 
                 params:dict = None,
                 ewc_params:dict = None,
                 **kwargs) -> None:
        super(MGVAE_Conv, self).__init__()
    
        self.prior_log_variance = torch.nn.Parameter(torch.randn((1)))

        self.latent_dim = latent_dim
        self.hidden_size = hidden_size
        self.cur_device = cur_device
        self.number_components = number_components
        if exemplar_data == None:
            raise TypeError("Dataset is None!")
        else:
            self.exemplar_data = exemplar_data

        self.params = params
        self.ewc_params = ewc_params

        self.hold_graph = False
        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass

        self.cs = 48
        self.bottleneck = 6

        # Build Encoder
        self.encoder = nn.Sequential(
            weight_norm(nn.Conv2d(in_channels=3, out_channels=self.cs, kernel_size=3, stride=2, padding=1)),
            nn.ELU(),
            block(input_size=self.cs, output_size=self.cs, stride=1, kernel=3, padding=1),
            block(input_size=self.cs, output_size=self.cs, stride=1, kernel=3, padding=1),
            block(input_size=self.cs, output_size=self.cs, stride=1, kernel=3, padding=1),
            block(input_size=self.cs, output_size=self.cs, stride=1, kernel=3, padding=1),
            block(input_size=self.cs, output_size=self.cs, stride=1, kernel=3, padding=1),
            block(input_size=self.cs, output_size=self.cs, stride=1, kernel=3, padding=1),

            weight_norm(nn.Conv2d(in_channels=self.cs, out_channels=self.cs*2, kernel_size=3, stride=2, padding=1)),
            nn.ELU(),
            block(input_size=self.cs*2, output_size=self.cs*2, stride=1, kernel=3, padding=1),
            block(input_size=self.cs*2, output_size=self.cs*2, stride=1, kernel=3, padding=1),
            block(input_size=self.cs*2, output_size=self.cs*2, stride=1, kernel=3, padding=1),
            block(input_size=self.cs*2, output_size=self.cs*2, stride=1, kernel=3, padding=1),
            block(input_size=self.cs*2, output_size=self.cs*2, stride=1, kernel=3, padding=1),
            block(input_size=self.cs*2, output_size=self.cs*2, stride=1, kernel=3, padding=1),

            weight_norm(nn.Conv2d(in_channels=self.cs * 2, out_channels=self.bottleneck, kernel_size=3, stride=1, padding=1)),
            nn.ELU(),
        )

        # self.hidden_size = 1536
        self.fc_mu = weight_norm(nn.Linear(self.hidden_size, self.latent_dim))
        # self.fc_logvar = weight_norm(nn.Linear(self.hidden_size, self.latent_dim))

        self.pre_decoder = nn.Sequential(weight_norm(nn.Linear(self.latent_dim, self.hidden_size)))

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2),
            weight_norm(nn.Conv2d(in_channels=self.bottleneck, out_channels=self.cs*2, kernel_size=3, stride=1, padding=1)),
            nn.ELU(),
            block(input_size=self.cs*2, output_size=self.cs*2, stride=1, kernel=3, padding=1),
            block(input_size=self.cs*2, output_size=self.cs*2, stride=1, kernel=3, padding=1),
            block(input_size=self.cs*2, output_size=self.cs*2, stride=1, kernel=3, padding=1),
            block(input_size=self.cs*2, output_size=self.cs*2, stride=1, kernel=3, padding=1),
            block(input_size=self.cs*2, output_size=self.cs*2, stride=1, kernel=3, padding=1),
            block(input_size=self.cs*2, output_size=self.cs*2, stride=1, kernel=3, padding=1),
            nn.Upsample(scale_factor=2),
            weight_norm(nn.Conv2d(in_channels=self.cs*2, out_channels=self.cs, kernel_size=3, stride=1, padding=1)),
            nn.ELU(),
            block(input_size=self.cs, output_size=self.cs, stride=1, kernel=3, padding=1),
            block(input_size=self.cs, output_size=self.cs, stride=1, kernel=3, padding=1),
            block(input_size=self.cs, output_size=self.cs, stride=1, kernel=3, padding=1),
            block(input_size=self.cs, output_size=self.cs, stride=1, kernel=3, padding=1),
            block(input_size=self.cs, output_size=self.cs, stride=1, kernel=3, padding=1),
            block(input_size=self.cs, output_size=self.cs, stride=1, kernel=3, padding=1),
        )
        
        self.x_mean = weight_norm(nn.Conv2d(in_channels=self.cs, out_channels=3, kernel_size=3, stride=1, padding=1))
        # self.x_logvar = nn.Conv2d(in_channels=self.cs, out_channels=3, kernel_size=3, stride=1, padding=1)
        # self.save_hyperparameters(ignore=["exemplar_data"])

    def ewc_pare(self):
        self.ewc_weight = self.ewc_params['ewc_weight']
        self.init_params = [p.clone().detach() for p in self.parameters()]

        def get_fisher_info(n_samples=100):
            print("Getting fisher information...")
            n_params = len(self.init_params)
            # sums = [torch.zeros(p.shape).cuda() if self.use_cuda
            #         else torch.zeros(p.shape) for p in self.parameters()]
            sums = [torch.zeros(p.shape).cuda() for p in self.parameters()]
            
            for i in range(n_samples):
                # sample
                sampled_data = self.index_sample(i, current_device='cuda')
                recons_sampled_data = self.forward(sampled_data)
                log_probs = self.loss_function(*recons_sampled_data, M_N = self.params['kld_weight'])
                loss_grads = torch_grad(outputs=log_probs['loss'],
                                        inputs=list(self.parameters()),allow_unused=True)
                for j in range(n_params):
                    sums[j] = sums[j] + loss_grads[j]**2

            return [s / n_samples for s in sums]

        self.fisher = get_fisher_info(n_samples=30)

    def _ewc_loss(self):
        loss = 0
        n_params = len(self.init_params)
        params = list(self.parameters())
        for i in range(n_params):
            loss += torch.sum(self.fisher[i] * (params[i] - self.init_params[i])**2)
        return self.ewc_weight * loss

    def encode(self, input):
        # input_size: [B, C, H, W]
        h = self.encoder(input)
        h = h.view(input.size(0), -1)

        mu = self.fc_mu(h)
        log_var = self.prior_log_variance * torch.ones((mu.shape[0], self.latent_dim)).to(self.cur_device)

        return [mu, log_var]
    
    def decode(self, z):
        z = self.pre_decoder(z)
        z = z.reshape(-1, self.bottleneck, 16, 16)
        z = self.decoder(z)
        x_mean = self.x_mean(z)
        # x_logvar = self.x_logvar(z)

        return x_mean

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:

        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        x_mean = self.decode(z)

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
        # get exemplar set
        exempalr_embedding = self.get_exemplar_set()
        log_p_z = self.log_p_z(z=z, exemplars_embedding=exempalr_embedding)
        log_q_z = log_normal_diag(z, mu, log_var, dim=1)
        KL = -(log_p_z - log_q_z)
        # print(KL.shape)


        # Recons
        # RE = -F.binary_cross_entropy(input.view(-1, 3*64*64), recons.view(-1, 3*64*64), reduction='none')
        # RE = torch.sum(RE, dim=1)
        # RE = -F.mse_loss(input.view(-1, 3*64*64), recons.view(-1, 3*64*64))
        # RE = F.mse_loss(recons, input)
        RE = -F.mse_loss(input.view(-1, 3*64*64), recons.view(-1, 3*64*64), reduction='none')
        RE = torch.sum(RE, dim=1)
        # print(RE.shape)

        # RE = log_bernoulli(input.view(-1, 3*64*64), recons.view(-1, 3*64*64), dim=1)
        # print(RE.shape)

        # loss = -RE + kld_weight * KL
        # loss = torch.mean(loss)
        RE = torch.mean(RE)
        KL = torch.mean(KL)
        # loss = RE + kld_weight * KL
        loss = -RE + kld_weight * KL
        # print(RE)
        # print(KL)
        return {'loss': loss, 'Reconstruction_Loss':RE.detach(), 'KLD':-KL.detach()}

    def get_exemplar_set(self):
        exemplars_indices = torch.randint(low=0, high=len(self.exemplar_data), size=(self.number_components, ))
        exemplars_z, log_variance = self.encode(self.exemplar_data[exemplars_indices].float().to(self.cur_device))
        exemplar_set = (exemplars_z, log_variance, exemplars_indices.to(self.cur_device))
        return exemplar_set

    def log_p_z_exemplar(self, z, exemplars_embedding):
        centers, center_log_variance, center_indices = exemplars_embedding
        denominator = torch.tensor(len(centers)).expand(len(z)).float().to(self.cur_device)
        center_log_variance = center_log_variance[0, :].unsqueeze(0)
        prob, _ = log_normal_diag_vectorized(z, centers, center_log_variance)  # MB x C
        prob -= torch.log(denominator).unsqueeze(1)
        return prob

    def log_p_z(self, z, exemplars_embedding, sum=True, test=None):
        if test is None:
            prob = self.log_p_z_exemplar(z, exemplars_embedding)

        if sum:
            prob_max, _ = torch.max(prob, 1)  # MB x 1
            log_prior = prob_max + torch.log(torch.sum(torch.exp(prob - prob_max.unsqueeze(1)), 1))  # MB x 1
        else:
            return prob
        return log_prior

    def sample(self,
               num_samples:int,
               current_device: int,
                **kwargs) -> Tensor:
        # exemplar_data = self.exemplar_data
        # select reference samples
        selected_indices = torch.randint(low=0, high=len(self.exemplar_data), size=(num_samples,))
        reference_images = self.exemplar_data[selected_indices].float().to(self.cur_device)
        # generated samples for every sample
        per_exemplar = 1
        pseudo, log_var = self.encode(reference_images)
        pseudo = pseudo.unsqueeze(1).expand(-1, per_exemplar, -1).reshape(-1, pseudo.shape[-1])
        log_var = log_var[0].unsqueeze(0).expand(len(pseudo), -1)
        z_sample_rand = self.reparameterize(pseudo, log_var)
        z_sample_rand = z_sample_rand.reshape(-1, per_exemplar, pseudo.shape[1])

        # z to x
        generated_samples = self.decode(z_sample_rand)
        # print(generated_samples.shape)
        # refer:[num_samples, 3, 64, 64] to [num_samples, 1, 3, 64, 64]
        # generated: [num_samples * per_exemplar, 1, 28, 28] to [num_samples , per_exemplar, 1, 28, 28]
        # reference_images = reference_images.unsqueeze(1)
        # generated_samples = generated_samples.reshape(num_samples , per_exemplar, 3, 64, 64)

        # results_samples = torch.cat((reference_images, generated_samples), dim=1)
        # results_samples = results_samples.reshape(num_samples*(per_exemplar+1), 3, 64, 64)
        
        return generated_samples
    
    def data_sample(self, exemplar_data, per_exemplar, cur_device='cuda'):
        reference_images = exemplar_data.float().to(cur_device)
        pseudo, log_var = self.encode(reference_images)
        pseudo = pseudo.unsqueeze(1).expand(-1, per_exemplar, -1).reshape(-1, pseudo.shape[-1])

        log_var = log_var[0].unsqueeze(0).expand(len(pseudo), -1)
        z_sample_rand = self.reparameterize(pseudo, log_var)
        z_sample_rand = z_sample_rand.reshape(-1, per_exemplar, pseudo.shape[1])
        generated_samples = self.decode(z_sample_rand)

        reference_images = reference_images.unsqueeze(1).unsqueeze(1)
        generated_samples = generated_samples.reshape(exemplar_data.shape[0] , per_exemplar, 3, 64, 64)

        results_samples = torch.cat((reference_images, generated_samples), dim=1)
        results_samples = results_samples.reshape((exemplar_data.shape[0])*(per_exemplar+1), 3, 64, 64)

        return results_samples

    def index_sample_with_input(self,
               index: List,
               current_device: int, **kwargs) -> Tensor:
        # select reference samples
        # selected_indices = torch.randint(low=0, high=len(exemplar_data), size=(num_samples,))
        selected_indices = index
        num_samples = len(selected_indices)
        reference_images = self.exemplar_data[selected_indices].float().to(current_device)
        if reference_images.dim() == 3:
            reference_images = reference_images.unsqueeze(0)
        # generated samples for every sample
        per_exemplar = 1
        pseudo, log_var = self.encode(reference_images)
        pseudo = pseudo.unsqueeze(1).expand(-1, per_exemplar, -1).reshape(-1, pseudo.shape[-1])
        log_var = log_var[0].unsqueeze(0).expand(len(pseudo), -1)
        z_sample_rand = self.reparameterize(pseudo, log_var)
        z_sample_rand = z_sample_rand.reshape(-1, per_exemplar, pseudo.shape[1])

        # z to x
        generated_samples = self.decode(z_sample_rand)
        # print(reference_images.shape)
        # print(generated_samples.shape)
        
        # refer:[num_samples, 3, 64, 64] to [num_samples, 1, 3, 64, 64]
        # generated: [num_samples * per_exemplar, 1, 28, 28] to [num_samples , per_exemplar, 1, 28, 28]
        # reference_images = reference_images.unsqueeze(1)
        # generated_samples = generated_samples.reshape(num_samples , per_exemplar, 3, 64, 64)
        # # print(reference_images.shape)
        # # print(generated_samples.shape)

        # results_samples = torch.cat((reference_images, generated_samples), dim=1)
        # results_samples = results_samples.reshape(num_samples*(per_exemplar+1), 3, 64, 64)
        # print(results_samples.shape)
        return reference_images, generated_samples


    def index_sample(self,
               index: List,
               current_device: int, **kwargs) -> Tensor:
        # select reference samples
        # selected_indices = torch.randint(low=0, high=len(exemplar_data), size=(num_samples,))
        selected_indices = index
        reference_images = self.exemplar_data[selected_indices].float().to(current_device)
        if reference_images.dim() == 3:
            reference_images = reference_images.unsqueeze(0)
        # generated samples for every sample
        per_exemplar = 1
        pseudo, log_var = self.encode(reference_images)
        pseudo = pseudo.unsqueeze(1).expand(-1, per_exemplar, -1).reshape(-1, pseudo.shape[-1])
        log_var = log_var[0].unsqueeze(0).expand(len(pseudo), -1)
        z_sample_rand = self.reparameterize(pseudo, log_var)
        z_sample_rand = z_sample_rand.reshape(-1, per_exemplar, pseudo.shape[1])

        # z to x
        generated_samples = self.decode(z_sample_rand)
        
        return generated_samples
    
    def generate(self, x: Tensor, **kwargs) -> Tensor:
        return self.forward(x)[0]

    def training_step(self, batch, batch_idx, optimizer_idx = 0):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels = labels)
        train_loss = self.loss_function(*results,
                                              M_N = self.params['kld_weight'], #al_img.shape[0]/ self.num_train_imgs,
                                              optimizer_idx=optimizer_idx,
                                              batch_idx = batch_idx)
        if self.ewc_params is not None and self.ewc_params['ewc'] == True:
            ewc_loss = self._ewc_loss()
            loss = train_loss['loss'] + ewc_loss
            self.log("train_ewc_loss", ewc_loss)
        else:
            loss = train_loss['loss']

        self.log("train_loss", loss)
        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)

        return loss

    # def validation_step(self, batch, batch_idx, optimizer_idx = 0):
    #     pass
        # real_img, labels = batch
        # self.curr_device = real_img.device

        # results = self.forward(real_img, labels = labels)
        # val_loss = self.loss_function(*results,
        #                                     M_N = 1.0, #real_img.shape[0]/ self.num_val_imgs,
        #                                     optimizer_idx = optimizer_idx,
        #                                     batch_idx = batch_idx)
        # if self.ewc_params is not None and self.ewc_params['ewc'] == True:
        #     ewc_loss = self._ewc_loss()
        #     loss = val_loss['loss'] + ewc_loss
        #     self.log("val_ewc_loss", ewc_loss)
        # else:
        #     loss = val_loss['loss']

        # self.log("val_loss", loss)
        # self.log_dict({f"vae_val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)

    # def on_validation_end(self) -> None:
    #     self.sample_images()

    def on_train_epoch_end(self) -> None:
        if (self.current_epoch + 1) % 5 == 0:
            self.sample_images()
        
    def sample_images(self):
        # Get sample reconstruction image            
        test_input, test_label = next(iter(self.trainer.datamodule.test_dataloader()))
        test_input = test_input.to('cuda')
        test_label = test_label.to('cuda')

        # test_input, test_label = batch
        recons = self.generate(test_input, labels = test_label)
        vutils.save_image(test_input,
                          os.path.join(self.logger.log_dir , 
                                       "Reconstructions", 
                                       f"input_{self.logger.name}_Epoch_{self.current_epoch}.png"),
                          normalize=True,
                          nrow=12)

        vutils.save_image(recons.data,
                          os.path.join(self.logger.log_dir , 
                                       "Reconstructions", 
                                       f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png"),
                          normalize=True,
                          nrow=12)

        try:
            samples = self.sample(144,
                                self.cur_device,
                                labels = test_label)
            vutils.save_image(samples.cpu().data,
                              os.path.join(self.logger.log_dir , 
                                           "Samples",      
                                           f"{self.logger.name}_Epoch_{self.current_epoch}.png"),
                              normalize=True,
                              nrow=12)
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

