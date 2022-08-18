import os
import torch
import pandas as pd
import numpy as np

from pytorch_lightning import LightningDataModule
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import Dataset, DataLoader, TensorDataset
from typing import List, Optional, Sequence, Union, Any, Callable
from torch.utils.data.sampler import WeightedRandomSampler
from imblearn.datasets import fetch_datasets
from sklearn.model_selection import train_test_split

from .split_balanced import split_balanced
from ..aug_ways import *

class TabularDataset(LightningDataModule):
    def __init__(
        self,
        data_path: str,
        data_name: str,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        num_workers: int = 0,
        pin_memory: bool = True,
        target_label: int = 0, # 0:all data; 1:majority; 2:minority
        downsample_size=0,
        aug_way=None,
        exemplar_model=None,
        **kwargs,
    ):
        super().__init__()

        self.data_dir = data_path
        self.data_name = data_name
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.target_label = target_label
        self.downsample_size = downsample_size
        self.aug_way = aug_way
        self.exemplar_model = exemplar_model
    
    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:

        if self.data_name == 'musk':
            musk_data_path = os.path.join(self.data_dir, 'MUSK_2', 'musk_csv.csv')
            data = pd.read_csv(musk_data_path)

            # Total 6598 = [5581, 1017]
            # Minmax normalization to [-1, 1]
            scaler = MinMaxScaler(feature_range=(-1, 1))
            
            X = data.iloc[:,3:169]
            X_numpy = scaler.fit_transform(X)
            y_numpy = data.iloc[:, 169].to_numpy()

            if self.target_label == 0:
                X_numpy = X_numpy
                y_numpy = y_numpy
            elif self.target_label == 1:
                X_numpy = X_numpy[1017:, :]
                y_numpy = y_numpy[1017:]
            elif self.target_label == 2:
                X_numpy = X_numpy[0:1017, :]
                y_numpy = y_numpy[0:1017]
            
            # Split train / test
            X_train, X_test, y_train, y_test = split_balanced(X_numpy, y_numpy, test_size=200)


        elif self.data_name == 'water_quality':
            data = pd.read_csv("/home/qingzhong/data/Kaggle/water_quality/waterQuality1.csv")
            data_value = data.values

            x = data_value[:, :-1]
            y = data_value[:, -1]

            f = np.where(x[:,1]=='#NUM!')
            s = f[0][:]
            x = np.delete(x,s,0)
            y = np.delete(y,s,0)

            x[:,1] = x[:,1].astype(np.float64)
            y_numpy = y.astype(np.float64)


            scaler = MinMaxScaler()
            X_numpy = scaler.fit_transform(x)

            if self.target_label == 0:
                X_numpy = X_numpy
                y_numpy = y_numpy
            elif self.target_label == 1:
                index = np.nonzero(y_numpy==0)[0]
                X_numpy = X_numpy[index, :]
                y_numpy = y_numpy[index]
            elif self.target_label == 2:
                index = np.nonzero(y_numpy==1)[0]
                X_numpy = X_numpy[index, :]
                y_numpy = y_numpy[index]
            
            X_train, X_test, y_train, y_test = split_balanced(X_numpy, y_numpy, test_size=300)
            

        elif self.data_name == 'isolet':
            data = fetch_datasets(data_home='/home/qingzhong/data/')['isolet']
            X = data.data
            y = data.target

            X_numpy = X
            y_numpy = (y == 1).astype(np.float64)

            if self.target_label == 0:
                X_numpy = X_numpy
                y_numpy = y_numpy
            elif self.target_label == 1:
                index = np.nonzero(y_numpy==0)[0]
                X_numpy = X_numpy[index, :]
                y_numpy = y_numpy[index]
            elif self.target_label == 2:
                index = np.nonzero(y_numpy==1)[0]
                X_numpy = X_numpy[index, :]
                y_numpy = y_numpy[index]
            
            X_train, X_test, y_train, y_test = split_balanced(X_numpy, y_numpy, test_size=200)



        if stage == "fit" or stage is None:
            if self.aug_way == 'erm':
                print("No augmentation!")

            elif self.aug_way == 'rs':
                self.train_in_idx = re_sampling(X_train, y_train)

            elif self.aug_way == 'smote':
                aug_data, aug_label = smote(X_train, y_train)
                X_train = np.concatenate((X_train, aug_data), axis=0)
                y_train = np.concatenate((y_train, aug_label), axis=0)

            elif self.aug_way == 'etg':
                aug_data, aug_label = aug_mgvae_Tabular(X_train, y_train, model=self.exemplar_model)
                X_train = np.concatenate((X_train, aug_data), axis=0)
                y_train = np.concatenate((y_train, aug_label), axis=0)
            
            elif self.aug_way == 'cvae':
                if self.exemplar_model == None:
                    raise NotImplementedError("No exemplar model!")
                aug_data, aug_label = cvae_aug_tabular(X_train, y_train, model=self.exemplar_model)
                X_train = np.concatenate((X_train, aug_data), axis=0)
                y_train = np.concatenate((y_train, aug_label), axis=0)

            self.dataset_train = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())

        if stage == "validate" or stage is None:
            perm = np.random.permutation(X_test.shape[0])
            X_test = X_test[perm]
            y_test = y_test[perm]
            self.dataset_val = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long())

        if stage == "test" or stage is None:
            self.dataset_test = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long())
    
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
