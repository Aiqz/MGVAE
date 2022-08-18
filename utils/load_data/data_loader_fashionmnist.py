from datetime import date
import os
import numpy as np
import torch
from torch import Tensor
from pathlib import Path
from typing import List, Optional, Sequence, Union, Any, Callable
from torchvision.datasets.folder import default_loader
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.datasets import CelebA, MNIST, FashionMNIST
import zipfile
from torch.utils.data.sampler import WeightedRandomSampler

from ..aug_ways import *
# from .smote import smote
# from .etg import ETG
# from .exemplar_aug import exemplar_aug
# from .re_sampling import re_sampling
# from .cvae_aug import cvae_aug


class MyFashionMNIST(FashionMNIST):
    def __init__(self, root: str, train: bool = True, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, download: bool = False,
                 my_target=[0], downsample_size=[0], test_downsample_size=-1, aug=None, exemplar_model=None) -> None:
        super(MyFashionMNIST, self).__init__(root, train,
                                      transform, target_transform, download)
        if len(my_target) == 0:
            raise NotImplementedError("target list is None!")

        if len(my_target) != len(downsample_size):
            raise NotImplementedError("target / down_sample size not same!")
        if train:
            for i, target in enumerate(my_target):
                self.index_of_target_number = (
                    self.targets == target).nonzero(as_tuple=True)[0]
                if downsample_size[i] == 0:
                    cur_data = self.data[self.index_of_target_number]
                    cur_label = torch.zeros(cur_data.shape[0]).float()
                    # cur_label = self.targets[self.index_of_target_number]
                else:
                    cur_data = self.data[self.index_of_target_number][0:downsample_size[i]]
                    cur_label = torch.ones(cur_data.shape[0]).float()
                    # cur_label = self.targets[self.index_of_target_number][0:downsample_size[i]]
                if i == 0:
                    self.target_data = cur_data
                    self.target_label = cur_label
                else:
                    self.target_data = torch.cat(
                        (self.target_data, cur_data), dim=0)
                    self.target_label = torch.cat(
                        (self.target_label, cur_label), dim=0)

            self.data = self.target_data
            self.targets = self.target_label

            if aug == 'smote':
                aug_data, aug_label = smote(self.data, self.targets)
                # Numpy to tensor
                aug_data = torch.from_numpy(aug_data)
                aug_label = torch.from_numpy(aug_label)

                self.data = torch.cat((self.data, aug_data), dim=0)
                self.targets = torch.cat((self.targets, aug_label), dim=0)
                # classes, class_counts = np.unique(self.targets, return_counts=True)
                # print(class_counts)
            elif aug == 'etg':
                if exemplar_model == None:
                    raise NotImplementedError("No exemplar model!")
                aug_data, aug_label = aug_mgvae(
                    self.data, self.targets, model=exemplar_model)

                self.data = torch.cat(
                    (self.data, aug_data.detach().cpu()), dim=0)
                self.targets = torch.cat(
                    (self.targets, aug_label.detach().cpu()), dim=0)
                # classes, class_counts = np.unique(self.targets, return_counts=True)
                # print(class_counts)
            elif aug == 'exemplar':
                if exemplar_model == None:
                    raise NotImplementedError("No exemplar model!")
                aug_data, aug_label = exemplar_aug(
                    self.data, self.targets, model=exemplar_model)
                self.data = torch.cat(
                    (self.data, aug_data.detach().cpu()), dim=0)
                self.targets = torch.cat(
                    (self.targets, aug_label.detach().cpu()), dim=0)
            elif aug == 'rs':
                print("Re-sampling")
                self.train_in_idx = re_sampling(self.data, self.targets)
            elif aug == 'cvae':
                if exemplar_model == None:
                    raise NotImplementedError("No exemplar model!")
                aug_data, aug_label = cvae_aug(
                    self.data, self.targets, model=exemplar_model)
                self.data = torch.cat(
                    (self.data, aug_data.detach().cpu()), dim=0)
                self.targets = torch.cat(
                    (self.targets, aug_label.detach().cpu()), dim=0)

        else:
            if test_downsample_size == -1:
                raise NotImplementedError("Test downsample size is 0!!")
            for i, target in enumerate(my_target):
                self.index_of_target_number = (
                    self.targets == target).nonzero(as_tuple=True)[0]
                if test_downsample_size > len(self.index_of_target_number):
                    raise NotImplementedError(
                        "Test downsample size too large!")
                if test_downsample_size == 0:
                    cur_data = self.data[self.index_of_target_number]
                else:
                    cur_data = self.data[self.index_of_target_number][0:test_downsample_size]
                if downsample_size[i] == 0:
                    cur_label = torch.zeros(cur_data.shape[0]).float()
                    # cur_label = self.targets[self.index_of_target_number]
                    # cur_label = self.targets[self.index_of_target_number][0:test_downsample_size]
                else:
                    cur_label = torch.ones(cur_data.shape[0]).float()
                    # cur_label = self.targets[self.index_of_target_number][0:test_downsample_size]
                if i == 0:
                    self.target_data = cur_data
                    self.target_label = cur_label
                else:
                    self.target_data = torch.cat(
                        (self.target_data, cur_data), dim=0)
                    self.target_label = torch.cat(
                        (self.target_label, cur_label), dim=0)

            # self.data = self.target_data
            # self.targets = self.target_label
            perm = torch.randperm(self.target_data.size(0))
            self.data = self.target_data[perm]
            self.targets = self.target_label[perm]


class FashionMNISTDataset(LightningDataModule):
    def __init__(
        self,
        data_path: str,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        patch_size: Union[int, Sequence[int]] = (256, 256),
        num_workers: int = 0,
        pin_memory: bool = False,
        target_label: List = [0],
        downsample_size=0,
        aug_way=None,
        exemplar_model=None,
        **kwargs,
    ):
        super().__init__()

        self.data_dir = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.target_label = target_label
        self.downsample_size = downsample_size
        self.aug_way = aug_way
        self.exemplar_model = exemplar_model

    def prepare_data(self):
        pass
        # MNIST(self.data_dir, train=True, download=True)
        # MNIST(self.data_dir, train=False, download=True)

        # MyMNIST(self.data_dir, train=True, download=True, downsample_size=self.downsample_size)
        # MyMNIST(self.data_dir, train=False, download=True, downsample_size=self.downsample_size)

    def setup(self, stage: Optional[str] = None) -> None:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                # transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        if stage == "fit" or stage is None:
            # mnist_full = MNIST(self.data_dir, train=True, transform=transform)
            # self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])
            if self.aug_way == 'smote':
                self.fashion_mnist_train = MyFashionMNIST(self.data_dir, train=True, transform=transform,
                                           my_target=self.target_label, downsample_size=self.downsample_size,
                                           aug='smote')
            elif self.aug_way == 'etg':
                if self.exemplar_model == None:
                    raise NotImplementedError("No exemplar model!")
                self.fashion_mnist_train = MyFashionMNIST(self.data_dir, train=True, transform=transform,
                                           my_target=self.target_label, downsample_size=self.downsample_size,
                                           aug='etg',
                                           exemplar_model=self.exemplar_model)
            elif self.aug_way == 'exemplar':
                if self.exemplar_model == None:
                    raise NotImplementedError("No exemplar model!")
                self.fashion_mnist_train = MyFashionMNIST(self.data_dir, train=True, transform=transform,
                                           my_target=self.target_label, downsample_size=self.downsample_size,
                                           aug='exemplar',
                                           exemplar_model=self.exemplar_model)
            elif self.aug_way == 'rs':
                
                self.fashion_mnist_train = MyFashionMNIST(self.data_dir, train=True, transform=transform,
                                           my_target=self.target_label, downsample_size=self.downsample_size,
                                           aug='rs')
            elif self.aug_way == 'cvae':
                self.fashion_mnist_train = MyFashionMNIST(self.data_dir, train=True, transform=transform,
                                           my_target=self.target_label, downsample_size=self.downsample_size,
                                           aug='cvae', exemplar_model=self.exemplar_model)
            else:
                self.fashion_mnist_train = MyFashionMNIST(self.data_dir, train=True, download=True, transform=transform,
                                           my_target=self.target_label, downsample_size=self.downsample_size)

        # Assign test dataset for use in dataloader(s)
        if stage == "validate" or stage is None:
            # self.mnist_val = MNIST(self.data_dir, train=False, transform=transform)
            self.fashion_mnist_val = MyFashionMNIST(self.data_dir, train=False, download=True, transform=transform, 
                                     my_target=self.target_label, downsample_size=self.downsample_size,
                                     test_downsample_size=0)

        if stage == "test" or stage is None:
            # self.mnist_test = MNIST(self.data_dir, train=False, transform=transform)
            self.fashion_mnist_test = MyFashionMNIST(self.data_dir, train=False, transform=transform,
                                      my_target=self.target_label, downsample_size=self.downsample_size,
                                      test_downsample_size=0)

    def train_dataloader(self):
        if self.aug_way == 'rs':
            return DataLoader(self.fashion_mnist_train,
                              batch_size=self.train_batch_size,
                              sampler=WeightedRandomSampler(self.fashion_mnist_train.train_in_idx, len(self.fashion_mnist_train.train_in_idx)),
                              num_workers=self.num_workers,
                            #   shuffle=True,
                              pin_memory=self.pin_memory,)
        else:
            return DataLoader(
                self.fashion_mnist_train,
                batch_size=self.train_batch_size,
                num_workers=self.num_workers,
                shuffle=True,
                pin_memory=self.pin_memory,
            )

    def val_dataloader(self):
        return DataLoader(
            self.fashion_mnist_val,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.fashion_mnist_test,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )
