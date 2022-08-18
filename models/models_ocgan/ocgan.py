from typing import *
import os
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision
import torchvision.utils as vutils

class Generator(nn.Module):

    def __init__(self,
                 classes: int,
                 latent_dim: int,
                 img_shape: Optional[Tuple[int, int, int]] = None):
        super(Generator, self).__init__()
        if img_shape is None:
            img_shape = (3, 64, 64)

        self.latent_dim = latent_dim
        self.img_shape = img_shape

        self.g_fill = nn.Embedding(classes, classes)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim + classes, 256, normalize=False),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )
        self._initialize_weights()

    def forward(self, noise, labels):
        # Concatenate label embedding and image to produce input
        gen_input = torch.cat((noise, self.g_fill(labels)), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *self.img_shape)
        return img

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
                torch.nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                m.weight.data *= 0.1
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
                torch.nn.init.constant_(m.bias, 0.0)


class Discriminator(nn.Module):
    def __init__(self,
                 classes: int,
                 img_shape: Optional[Tuple[int, int, int]] = None):
        super(Discriminator, self).__init__()
        if img_shape is None:
            img_shape = (3, 64, 64)

        self.d_fill = nn.Embedding(classes, classes)

        self.model = nn.Sequential(
            nn.Linear(classes + int(np.prod(img_shape)), 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        self._initialize_weights()

    def forward(self, img, labels):
        # Concatenate label embedding and image to produce input
        d_in = torch.cat((img.view(img.size(0), -1), self.d_fill(labels)), -1)
        validity = self.model(d_in)
        return validity

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
                torch.nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                m.weight.data *= 0.1
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
                torch.nn.init.constant_(m.bias, 0.0)


class CGAN(pl.LightningModule):

    def __init__(self,
                 input_dim: int,
                 amount_classes: int,
                 color_channels: int,
                 image_size: int,
                 device: torch.device,
                 image_intervall=10,
                 tensorboard_image_rows=10,
                 batch_size: int = 128,
                 **kwargs):
        super().__init__()
        self.tensorboard_images_rows = tensorboard_image_rows
        self.image_intervall = image_intervall
        self.used_device = device
        self.image_size = image_size
        self.amount_classes = amount_classes
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.loc_scale = (0, 1)
        image_shape = (color_channels, image_size, image_size)

        self.generator = Generator(
            img_shape=image_shape,
            classes=amount_classes,
            latent_dim=input_dim,
        )
        self.discriminator = Discriminator(
            classes=amount_classes,
            img_shape=image_shape
        )
        self.criterion = nn.BCELoss()
        # self.save_hyperparameters()

        # generating fixed noise
        fixed_noise = torch.tensor(
            np.random.normal(self.loc_scale[0], self.loc_scale[1],
                             (self.tensorboard_images_rows * self.amount_classes, self.generator.latent_dim)),
            device=self.used_device, dtype=torch.float)
        # generating fixed labels
        fixed_labels = torch.tensor(
            [i % self.amount_classes for i in range(self.tensorboard_images_rows * self.amount_classes)],
            device=self.used_device, dtype=torch.long)
        # sample noise used for tensorboard images
        self.sample_noise: Tuple[torch.Tensor, torch.Tensor] = (fixed_noise, fixed_labels)

    def forward(self, z, labels):
        """
        Generates an image using the generator
        given input noise z and labels y
        """
        return self.generator(z, labels)

    def generator_step(self, y):
        """
        Training step for generator
        1. Sample random noise and labels
        2. Pass noise and labels to generator to
           generate images
        3. Classify generated images using
           the discriminator
        4. Backprop loss
        """

        # Sample random noise and labels

        z = torch.tensor(
            np.random.normal(self.loc_scale[0], self.loc_scale[1], (y.shape[0], self.generator.latent_dim)),
            device=self.used_device, dtype=torch.float)

        # Generate images
        generated_imgs = self(z, y)
        # Classify generated image using the discriminator
        d_g_z: torch.tensor = self.discriminator(generated_imgs,
                                                 y)

        d_output = d_g_z.reshape(-1)
        d_ref = torch.ones(y.shape[0], device=self.used_device)
        g_loss = self.criterion(d_output,
                                d_ref)
        return g_loss

    def discriminator_step(self, x, y):
        """
        Training step for discriminator
        1. Get actual images and labels
        2. Predict probabilities of actual images and get BCE loss
        3. Get fake images from generator
        4. Predict probabilities of fake images and get BCE loss
        5. Combine loss from both and backprop
        """

        # Real images
        d_ref_r = torch.ones((x.shape[0]), device=self.used_device)
        d_i_y = self.discriminator(x, y).reshape(-1)
        loss_real = self.criterion(d_i_y,
                                   d_ref_r)

        # Fake images
        z = torch.tensor(
            np.random.normal(self.loc_scale[0], self.loc_scale[1], (x.shape[0], self.generator.latent_dim)),
            device=self.used_device, dtype=torch.float)

        generated_imgs = self(z, y)
        d_g_z_y = self.discriminator(generated_imgs, y)
        d_output = d_g_z_y.reshape(-1)
        d_zeros = torch.zeros((x.shape[0]), device=self.used_device)
        loss_fake = self.criterion(d_output,
                                   d_zeros)

        return loss_real + loss_fake

    def training_step(self, batch, batch_idx, optimizer_idx):
        X, y = batch
        loss = None
        # train generator
        if optimizer_idx == 0:
            loss = self.generator_step(y)

        # train discriminator
        if optimizer_idx == 1:
            loss = self.discriminator_step(X, y)

        return loss

    def configure_optimizers(self):
        g_optimizer = torch.optim.Adam(self.generator.parameters(), betas=(0.5, 0.999), lr=0.0002)
        d_optimizer = torch.optim.Adam(self.discriminator.parameters(), betas=(0.5, 0.999), lr=0.0002)
        return [g_optimizer, d_optimizer], []

    def on_epoch_end(self) -> None:
        imgs = self(self.sample_noise[0], self.sample_noise[1])
        d_g_z_y_y = self.discriminator(imgs, self.sample_noise[1]).reshape(-1)
        g_loss = self.criterion(d_g_z_y_y, torch.ones(d_g_z_y_y.shape[0], device=self.device))
        self.log("g_loss_fixed_noise", g_loss)
        if self.current_epoch % self.image_intervall == 0:
            vutils.save_image(imgs.cpu().data,
                              os.path.join(self.logger.log_dir , 
                                           "Samples",      
                                           f"{self.logger.name}_Epoch_{self.current_epoch}.png"),
                              normalize=False,
                              nrow=2)
    
    def cond_sample(self, y):
        # Conditional generate based on label y
        z = torch.tensor(
            np.random.normal(self.loc_scale[0], self.loc_scale[1], (y.shape[0], self.generator.latent_dim)),
            device=self.used_device, dtype=torch.float)
        # Generate images
        generated_imgs = self(z, y)
        return generated_imgs
