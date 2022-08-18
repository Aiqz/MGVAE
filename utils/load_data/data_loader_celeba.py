from enum import unique
import torch.utils.data as data
from PIL import Image
import os
import json
from pathlib import Path
from torchvision import transforms, utils, datasets
from typing import List, Optional, Sequence, Union, Any, Callable
import random
import numpy as np
import torch
import torchvision.utils as vutils
from torch import cuda
from torch.utils.data import Dataset, DataLoader, TensorDataset
from pytorch_lightning import LightningDataModule
from torch.utils.data.sampler import WeightedRandomSampler

from ..aug_ways import *

class CelebADataset(LightningDataModule):

    def __init__(
        self,
        data_path: str,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        num_workers: int = 0,
        pin_memory: bool = False,
        image_size:int = 64,
        aug_way:str = 'erm',
        exemplar_model = None,
        **kwargs,
    ):
        super().__init__()
        self.data_dir = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.image_size = image_size
        self.aug_way = aug_way
        self.exemplar_model = exemplar_model

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.CenterCrop(self.image_size),
                # transforms.RandomCrop(self.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]),
            'val': transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
            ]),
            'test': transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
            ])
        }

        if stage == "fit" or stage is None:
            self.dataset_train = datasets.ImageFolder(self.data_dir + '/train', data_transforms['train'])
            print(self.dataset_train.class_to_idx)
            if self.aug_way == 'erm':
                print("No augmentation!")
            elif self.aug_way == 'rw' or self.aug_way == 'cbrw':
                print(self.aug_way)
            else:
                train_data, train_label = self.get_train_data()
                if self.aug_way == 'smote':
                    aug_data, aug_label = smote(train_data, train_label)
                    aug_data = torch.from_numpy(aug_data)
                    aug_label = torch.from_numpy(aug_label)
                    train_data = torch.cat((train_data, aug_data), dim=0)
                    train_label = torch.cat((train_label, aug_label), dim=0)
                    
                elif self.aug_way == 'rs':
                    self.train_in_idx = re_sampling(train_data, train_label)

                elif self.aug_way == 'cvae':
                    aug_data, aug_label = cvae_aug(train_data, train_label, model=self.exemplar_model)
                    train_data = torch.cat((train_data, aug_data.detach().cpu()), dim=0)
                    train_label = torch.cat((train_label, aug_label.detach().cpu()), dim=0)
                    
                elif self.aug_way == 'cdcgan':
                    aug_data, aug_label = cond_gan_aug(train_data, train_label, model=self.exemplar_model)
                    train_data = torch.cat((train_data, aug_data.detach().cpu()), dim=0)
                    train_label = torch.cat((train_label, aug_label.detach().cpu()), dim=0)
                    
                elif self.aug_way == 'etg':
                    if self.exemplar_model is None:
                        raise NotImplementedError("No exemplar model!")
                    else:
                        # aug_data, aug_label = ETG(train_data, train_label, model=self.exemplar_model)
                        aug_data, aug_label = aug_mgvae_long_tail(train_data, train_label, model_list=self.exemplar_model)
                        train_data = torch.cat((train_data, aug_data.detach().cpu()), dim=0)
                        train_label = torch.cat((train_label, aug_label.detach().cpu()), dim=0)

                self.dataset_train = TensorDataset(train_data.float(), train_label.long())

        if stage == "validate" or stage is None:
            self.dataset_val = datasets.ImageFolder(self.data_dir + '/test', data_transforms['val'])
            val_data, val_label = self.get_val_data()
            perm = torch.randperm(val_data.size(0))
            val_data = val_data[perm]
            val_label = val_label[perm]
            self.dataset_val = TensorDataset(val_data.float(), val_label.long())

        if stage == "test" or stage is None:
            self.dataset_test = datasets.ImageFolder(self.data_dir + '/test', data_transforms['test'])

    def get_train_data(self):
        print("Getting training data...")
        num_data = len(self.dataset_train)
        train_data = torch.zeros(num_data, 3, self.image_size, self.image_size)
        train_label = torch.zeros(num_data)
        for i in range(num_data):
            train_data[i], train_label[i] = self.dataset_train.__getitem__(i)
        return train_data, train_label

    def get_val_data(self):
        print("Getting val data...")
        num_data = len(self.dataset_val)
        val_data = torch.zeros(num_data, 3, self.image_size, self.image_size)
        val_label = torch.zeros(num_data)
        for i in range(num_data):
            val_data[i], val_label[i] = self.dataset_val.__getitem__(i)
        return val_data, val_label

    def train_dataloader(self):
        if self.aug_way == 'rs':
            train_loader = DataLoader(
                self.dataset_train,
                batch_size=self.train_batch_size,
                sampler=WeightedRandomSampler(self.train_in_idx, len(self.train_in_idx)),
                num_workers=self.num_workers,
                pin_memory=self.pin_memory)
        else:
            train_loader = DataLoader(
                self.dataset_train, 
                batch_size=self.train_batch_size, 
                shuffle=True, 
                num_workers=self.num_workers, 
                pin_memory=self.pin_memory)

        return train_loader
    
    def val_dataloader(self):
        val_loader = DataLoader(
            self.dataset_val,
            batch_size=self.val_batch_size, 
            shuffle=False, 
            num_workers=self.num_workers, 
            pin_memory=self.pin_memory
        )
        return val_loader

    def test_dataloader(self):
        test_loader = DataLoader(
            self.dataset_test,
            batch_size=self.val_batch_size, 
            shuffle=True, 
            num_workers=self.num_workers, 
            pin_memory=self.pin_memory
        )
        return test_loader

