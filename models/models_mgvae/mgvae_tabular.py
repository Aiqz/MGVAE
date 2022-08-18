from ast import Raise
from logging import raiseExceptions
import torch
from torch import device, nn
from torch.nn import functional as F
from ..utils import *
from typing import List, Callable, Union, Any, TypeVar, Tuple
Tensor = TypeVar('Tensor')

class MGVAE_Tabular(nn.Module):

    def __init__(self,
                 input_dim: int = 166,
                 latent_dim: int = 40,
                 hidden_size: int = 300,
                 number_components: int = 500,
                 dataset_loader = None,
                 cur_device = 'cuda',
                 **kwargs) -> None:
        super(MGVAE_Tabular, self).__init__()

        self.prior_log_variance = torch.nn.Parameter(torch.randn((1)))
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_size = hidden_size
        self.cur_device = cur_device
        self.number_components = number_components
        if dataset_loader == None:
            raise TypeError("Dataset is None!")
        else:
            self.dataset_loader = dataset_loader

        # Build Encoder
        self.encoder = nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),
                )
        self.fc_mu = nn.Linear(self.hidden_size, self.latent_dim)
        # self.fc_var = nn.Linear(self.hidden_size, self.latent_dim)

        # Build Decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.input_dim),
            # [0-1]
            # nn.Sigmoid(),
            # [-1, 1]
            nn.Tanh(),
            )

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input.view(-1, self.input_dim))
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        # log_var = self.fc_var(result)
        log_var = self.prior_log_variance * torch.ones((mu.shape[0], self.latent_dim)).to(self.cur_device)

        return [mu, log_var]
        
    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder(z)
        result = result.view(-1, self.input_dim)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        
        z = self.reparameterize(mu, log_var)

        return  [self.decode(z), input, z, mu, log_var]

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
        
        # Recons
        # RE = -F.binary_cross_entropy(input.view(-1, 784), recons.view(-1, 784), reduction='none')
        # RE = torch.sum(RE, dim=1)

        RE = -F.mse_loss(input.view(-1, self.input_dim), recons.view(-1, self.input_dim), reduction='none')
        RE = torch.sum(RE, dim=1)

        # RE = log_bernoulli(input.view(-1, 784), recons.view(-1, 784), dim=1)
        # print(RE.shape)

        # loss = -RE + kld_weight * KL
        # loss = torch.mean(loss)
        RE = torch.mean(RE)
        KL = torch.mean(KL)
        loss = -RE + kld_weight * KL
        # print(RE)
        # print(KL)
        return {'loss': loss, 'Reconstruction_Loss':RE.detach(), 'KLD':-KL.detach()}

    def get_exemplar_set(self):
        exemplar_data = self.dataset_loader.dataset.tensors[0]
        # print(exemplar_data.shape)

        exemplars_indices = torch.randint(low=0, high=len(exemplar_data), size=(self.number_components, ))
        exemplars_z, log_variance = self.encode(exemplar_data[exemplars_indices].float().to(self.cur_device))
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
        exemplar_data = self.dataset_loader.dataset.tensors[0]
        # select reference samples
        selected_indices = torch.randint(low=0, high=len(exemplar_data), size=(num_samples,))
        reference_images = exemplar_data[selected_indices].float().to(self.cur_device)
        # generated samples for every sample
        per_exemplar = 11
        pseudo, log_var = self.encode(reference_images)
        pseudo = pseudo.unsqueeze(1).expand(-1, per_exemplar, -1).reshape(-1, pseudo.shape[-1])
        log_var = log_var[0].unsqueeze(0).expand(len(pseudo), -1)
        z_sample_rand = self.reparameterize(pseudo, log_var)
        z_sample_rand = z_sample_rand.reshape(-1, per_exemplar, pseudo.shape[1])

        # z to x
        generated_samples = self.decode(z_sample_rand)
        # refer:[num_samples, 166] to [num_samples, 1, 166]
        # generated: [num_samples * per_exemplar, 166] to [num_samples , per_exemplar, 166]
        reference_images = reference_images.unsqueeze(1)
        generated_samples = generated_samples.reshape(num_samples , per_exemplar, self.input_dim)

        results_samples = torch.cat((reference_images, generated_samples), dim=1)
        results_samples = results_samples.reshape(num_samples*(per_exemplar+1), self.input_dim)
        
        return results_samples
    
    def data_sample(self, exemplar_data, per_exemplar, cur_device='cuda'):
        reference_images = exemplar_data.float().to(cur_device)
        pseudo, log_var = self.encode(reference_images)
        pseudo = pseudo.unsqueeze(1).expand(-1, per_exemplar, -1).reshape(-1, pseudo.shape[-1])

        log_var = log_var[0].unsqueeze(0).expand(len(pseudo), -1)
        z_sample_rand = self.reparameterize(pseudo, log_var)
        z_sample_rand = z_sample_rand.reshape(-1, per_exemplar, pseudo.shape[1])
        generated_samples = self.decode(z_sample_rand)

        reference_images = reference_images.unsqueeze(1).unsqueeze(1)
        generated_samples = generated_samples.reshape(exemplar_data.shape[0] , per_exemplar, 1, 28, 28)

        results_samples = torch.cat((reference_images, generated_samples), dim=1)
        results_samples = results_samples.reshape((exemplar_data.shape[0])*(per_exemplar+1), 1, 28, 28)

        return results_samples

    def index_sample(self,
               index: List,
               current_device: int, **kwargs) -> Tensor:
        exemplar_data = self.dataset_loader.dataset.tensors[0]
        # select reference samples
        # selected_indices = torch.randint(low=0, high=len(exemplar_data), size=(num_samples,))
        selected_indices = index
        reference_images = exemplar_data[selected_indices].float().to(self.cur_device)
        # generated samples for every sample
        per_exemplar = 1
        pseudo, log_var = self.encode(reference_images)
        pseudo = pseudo.unsqueeze(1).expand(-1, per_exemplar, -1).reshape(-1, pseudo.shape[-1])
        log_var = log_var[0].unsqueeze(0).expand(len(pseudo), -1)
        z_sample_rand = self.reparameterize(pseudo, log_var)
        z_sample_rand = z_sample_rand.reshape(-1, per_exemplar, pseudo.shape[1])

        # z to x
        generated_samples = self.decode(z_sample_rand)

        # refer:[num_samples, 28, 28] to [num_samples, 1, 1, 28, 28]
        # generated: [num_samples * per_exemplar, 1, 28, 28] to [num_samples , per_exemplar, 1, 28, 28]
        # reference_images = reference_images.unsqueeze(1).unsqueeze(1)
        # generated_samples = generated_samples.reshape(num_samples , per_exemplar, 1, 28, 28)

        # results_samples = torch.cat((reference_images, generated_samples), dim=1)
        # results_samples = results_samples.reshape(num_samples*(per_exemplar+1), 1, 28, 28)
        
        return generated_samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        return self.forward(x)[0]
