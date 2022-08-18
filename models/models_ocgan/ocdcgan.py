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
                 input_dim: int, # 100
                 label_dim: int, # 5
                 filter_sizes: List[int],
                 output_dim: int,
                 used_device: torch.device,
                 kernel_size: Union[int, Tuple[int, int]] = 4,
                 stride: Union[int, Tuple[int, int]] = 2,
                 padding: Union[int, Tuple[int, int]] = 1):
        super().__init__()
        self.latent_dim = input_dim
        self.__kernel_size = (kernel_size, kernel_size) if type(kernel_size) is int else kernel_size
        self.__stride = (stride, stride) if type(stride) is int else stride
        self.__padding = (padding, padding) if type(padding) is int else padding
        # Hidden layers
        self.image_layer = torch.nn.Sequential()
        self.label_layer = torch.nn.Sequential()
        self.hidden_layer = torch.nn.Sequential()
        self.output_layer = torch.nn.Sequential()

        # generator representation
        self.g_fill = torch.zeros([label_dim, label_dim, 1, 1], device=used_device)
        # each class has it's own 1 layer within this tensor.
        for i in range(label_dim):
            self.g_fill[i, i, :] = 1
        for i in range(len(filter_sizes)):
            # Deconvolutional layer
            if i == 0:
                # For input
                input_deconv = torch.nn.ConvTranspose2d(input_dim,
                                                        int(filter_sizes[i] / 2),
                                                        kernel_size=self.__kernel_size,
                                                        stride=(1, 1),
                                                        padding=(0, 0))
                self.image_layer.add_module('input_deconv', input_deconv)

                # Initializer
                torch.nn.init.normal_(input_deconv.weight, mean=0.0, std=0.02)
                torch.nn.init.constant_(input_deconv.bias, 0.0)

                # Batch normalization
                self.image_layer.add_module('input_bn', torch.nn.BatchNorm2d(int(filter_sizes[i] / 2)))

                # Activation
                self.image_layer.add_module('input_act', torch.nn.ReLU())

                # For label
                label_deconv = torch.nn.ConvTranspose2d(label_dim, int(filter_sizes[i] / 2),
                                                        kernel_size=self.__kernel_size,
                                                        stride=self.__stride)
                self.label_layer.add_module('label_deconv', label_deconv)

                # Initializer
                torch.nn.init.normal_(label_deconv.weight, mean=0.0, std=0.02)
                torch.nn.init.constant_(label_deconv.bias, 0.0)

                # Batch normalization
                self.label_layer.add_module('label_bn', torch.nn.BatchNorm2d(int(filter_sizes[i] / 2)))

                # Activation
                self.label_layer.add_module('label_act', torch.nn.ReLU())
            else:
                deconv = torch.nn.ConvTranspose2d(filter_sizes[i - 1], filter_sizes[i],
                                                  kernel_size=self.__kernel_size,
                                                  stride=self.__stride,
                                                  padding=self.__padding)

                deconv_name = 'deconv' + str(i + 1)
                self.hidden_layer.add_module(deconv_name, deconv)

                # Initializer
                torch.nn.init.normal_(deconv.weight, mean=0.0, std=0.02)
                torch.nn.init.constant_(deconv.bias, 0.0)

                # Batch normalization
                bn_name = 'bn' + str(i + 1)
                self.hidden_layer.add_module(bn_name, torch.nn.BatchNorm2d(filter_sizes[i]))

                # Activation
                act_name = 'act' + str(i + 1)
                self.hidden_layer.add_module(act_name, torch.nn.ReLU())

        # Deconvolutional layer
        out = torch.nn.ConvTranspose2d(filter_sizes[-1], output_dim,
                                       kernel_size=self.__kernel_size,
                                       stride=self.__stride,
                                       padding=self.__padding)
        self.output_layer.add_module('out', out)
        # Initializer
        torch.nn.init.normal_(out.weight, mean=0.0, std=0.02)
        torch.nn.init.constant_(out.bias, 0.0)
        # Activation
        self.output_layer.add_module('act', torch.nn.Tanh())

    def forward(self, noise, labels):
        h1 = self.image_layer(noise)
        h2 = self.label_layer(self.g_fill[labels])
        x = torch.cat([h1, h2], 1)
        h = self.hidden_layer(x)
        out = self.output_layer(h)
        return out


class Discriminator(nn.Module):
    def __init__(self,
                 input_dim,
                 label_dim,
                 filter_sizes: List[int],
                 output_dim: int,
                 image_size: int,
                 device: torch.device,
                 kernel_size: Union[int, Tuple[int, int]] = 4,
                 stride: Union[int, Tuple[int, int]] = 2,
                 padding: Union[int, Tuple[int, int]] = 1):
        super().__init__()
        self.__kernel_size = (kernel_size, kernel_size) if type(kernel_size) is int else kernel_size
        self.__stride = (stride, stride) if type(stride) is int else stride
        self.__padding = (padding, padding) if type(padding) is int else padding

        self.image_layer = torch.nn.Sequential()
        self.label_layer = torch.nn.Sequential()
        self.hidden_layer = torch.nn.Sequential()
        self.output_layer = torch.nn.Sequential()
        self.fill = torch.zeros([label_dim, label_dim, image_size, image_size], device=device)

        # each class has it's own 1 layer within this tensor.
        for i in range(label_dim):
            self.fill[i, i, :, :] = 1

        for i in range(len(filter_sizes)):
            # Convolutional layer
            if i == 0:
                # For input
                input_conv = torch.nn.Conv2d(input_dim, int(filter_sizes[i] / 2),
                                             kernel_size=self.__kernel_size,
                                             stride=self.__stride,
                                             padding=self.__padding)
                self.image_layer.add_module('input_conv', input_conv)

                # Initializer
                torch.nn.init.normal_(input_conv.weight, mean=0.0, std=0.02)
                torch.nn.init.constant_(input_conv.bias, 0.0)

                # Activation
                self.image_layer.add_module('input_act', torch.nn.LeakyReLU(0.2))

                # For label
                label_conv = torch.nn.Conv2d(label_dim, int(filter_sizes[i] / 2),
                                             kernel_size=self.__kernel_size,
                                             stride=self.__stride,
                                             padding=self.__padding)
                self.label_layer.add_module('label_conv', label_conv)

                # Initializer
                torch.nn.init.normal_(label_conv.weight, mean=0.0, std=0.02)
                torch.nn.init.constant_(label_conv.bias, 0.0)

                # Activation
                self.label_layer.add_module('label_act', torch.nn.LeakyReLU(0.2))
            else:
                conv = torch.nn.Conv2d(filter_sizes[i - 1], filter_sizes[i],
                                       kernel_size=self.__kernel_size,
                                       stride=self.__stride,
                                       padding=self.__padding)

                conv_name = 'conv' + str(i + 1)
                self.hidden_layer.add_module(conv_name, conv)

                # Initializer
                torch.nn.init.normal_(conv.weight, mean=0.0, std=0.02)
                torch.nn.init.constant_(conv.bias, 0.0)

                # Batch normalization
                bn_name = 'bn' + str(i + 1)
                self.hidden_layer.add_module(bn_name, torch.nn.BatchNorm2d(filter_sizes[i]))

                # Activation
                act_name = 'act' + str(i + 1)
                self.hidden_layer.add_module(act_name, torch.nn.LeakyReLU(0.2))

        # Output layer
        # Convolutional layer
        out = torch.nn.Conv2d(filter_sizes[-1],
                              output_dim,
                              kernel_size=self.__kernel_size, stride=(1, 1), padding=(0, 0))
        self.output_layer.add_module('out', out)
        # Initializer
        torch.nn.init.normal_(out.weight, mean=0.0, std=0.02)
        torch.nn.init.constant_(out.bias, 0.0)
        # Activation
        self.output_layer.add_module('act', torch.nn.Sigmoid())

    def forward(self, images, labels):
        h1 = self.image_layer(images)
        h2 = self.label_layer(self.fill[labels])
        images = torch.cat([h1, h2], 1)
        h = self.hidden_layer(images)
        out = self.output_layer(h)
        return out


class CDCGAN(pl.LightningModule):

    def __init__(self,
                 input_dim: int,
                 amount_classes: int,
                 filter_sizes: List[int],
                 color_channels: int,
                 image_size: int,
                 device: torch.device,
                 image_intervall=10,
                 tensorboard_image_rows=10,
                 batch_size: int = 128,
                 **kwargs):
        super().__init__()
        # self.save_hyperparameters()
        self.tensorboard_images_rows = tensorboard_image_rows
        self.image_intervall = image_intervall
        self.used_device = device
        self.image_size = image_size
        self.amount_classes = amount_classes
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.loc_scale = (0, 1)
        self.filter_sizes = filter_sizes
        self.g_optimizer: torch.optim.Optimizer
        self.d_optimizer: torch.optim.Optimizer
        self.d_lr = 2e-4
        self.g_lr = 2e-4

        self.generator = Generator(

            input_dim=input_dim,
            label_dim=amount_classes,
            filter_sizes=filter_sizes,
            output_dim=color_channels,
            used_device=device
        )
        self.discriminator = Discriminator(
            input_dim=color_channels,
            label_dim=amount_classes,
            filter_sizes=filter_sizes[::-1],
            output_dim=1,
            device=self.used_device,
            image_size=image_size
        )
        self.criterion = nn.BCELoss()
        # self.confusion_matrix = torch.zeros((amount_classes, amount_classes), device=self.used_device)
        self.validation_z = torch.rand(batch_size, input_dim)
        self.sample_noise: Union[Tuple[torch.Tensor, torch.Tensor], None] = None

        # setting up classes trick
        # discriminator representation

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

        if self.sample_noise is None:
            # saving noise and lables for
            fixed_noise = torch.tensor(
                np.random.normal(self.loc_scale[0], self.loc_scale[1],
                                 (self.tensorboard_images_rows * self.amount_classes, self.generator.latent_dim, 1, 1)),
                device=self.used_device, dtype=torch.float)
            fixed = torch.tensor(
                [i % self.amount_classes for i in range(self.tensorboard_images_rows * self.amount_classes)],
                device=self.used_device, dtype=torch.long)
            self.sample_noise = (fixed_noise, fixed)

        z = torch.tensor(
            np.random.normal(self.loc_scale[0], self.loc_scale[1], (y.shape[0], self.generator.latent_dim, 1, 1)),
            device=self.used_device, dtype=torch.float)

        # Generate images
        generated_imgs = self(z, y)
        # Classify generated image using the discriminator
        d_g_z: torch.tensor = self.discriminator(generated_imgs,
                                                 y)

        d_output = d_g_z.reshape(-1)

        # Backprop loss. We want to maximize the discriminator's
        # loss, which is equivalent to minimizing the loss with the true
        # labels flipped (i.e. y_true=1 for fake images). We do this
        # as PyTorch can only minimize a function instead of maximizing
        d_ref = torch.ones(y.shape[0], device=self.used_device)
        g_loss = self.criterion(d_output,
                                d_ref)
        self.log("g_loss", g_loss)
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
            np.random.normal(self.loc_scale[0], self.loc_scale[1], (x.shape[0], self.generator.latent_dim, 1, 1)),
            device=self.used_device, dtype=torch.float)

        generated_imgs = self(z, y)
        d_g_z_y = self.discriminator(generated_imgs, y)
        d_output = d_g_z_y.reshape(-1)
        d_zeros = torch.zeros((x.shape[0]), device=self.used_device)
        loss_fake = self.criterion(d_output,
                                   d_zeros)
        self.log("d_loss", loss_fake + loss_real)

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
                              normalize=True,
                              nrow=5)

    def cond_sample(self, y):
        # Conditional generate based on label y
        z = torch.tensor(
            np.random.normal(self.loc_scale[0], self.loc_scale[1], (y.shape[0], self.generator.latent_dim, 1, 1)),
            device=self.used_device, dtype=torch.float)
        # Generate images
        generated_imgs = self(z, y)
        return generated_imgs
