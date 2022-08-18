from cProfile import label
from email import header


#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
    @File    :   cvae_aug.py
    @Time    :   2022/03/29 09:16:34
    @Author  :   Qingzhong Ai 
    @Contact :   aqz1995@163.com
    @Desc    :   None
'''
import os
import torch
import numpy as np
import torchvision.utils as vutils

def cvae_aug(data, targets, model):
    classes, class_counts = np.unique(targets, return_counts=True)
    n_class = len(classes)
    n_max = max(class_counts)

    model.eval()

    print("#------ Augmenting with OCVAE... ------#")
    for k in range(1, n_class):
        print("Augmenting for class {}".format(k))
        indices = np.where(targets == k)[0]
        class_len = len(indices)

        i = 0
        per_samples = 100
        aug_samples = torch.zeros_like(data)[:n_max-class_len].float().to('cuda')
        while i + per_samples < n_max - class_len:
            index = [j for j in range(i, i+per_samples)]
            label = torch.ones(per_samples) * k
            label = label.type(torch.LongTensor).to('cuda')
            new_samples_save = model.sample(100, current_device='cuda', labels = label)

            # For MNIST and Fashion
            new_samples = new_samples_save.squeeze().float() * 255.0
            # For CelebA
            # new_samples = new_samples_save.detach().squeeze()

            # if i == 0:
            #     vutils.save_image(new_samples_save.data,
            #                 os.path.join('/home/qingzhong/code/Exemplar_transfer_generation/logs/CVAE_fashionmnist', 
            #                             "Generation", 
            #                             f"Gen_{k}.png"),
            #                 normalize=True,
            #                 nrow=10)
            aug_samples[index] = new_samples

            i += per_samples
        
        index = [j for j in range(i, n_max - class_len)]

        label = torch.ones(len(index)) * k
        label = label.type(torch.LongTensor).to('cuda')
        # For MNIST and Fashion
        new_samples = model.sample(len(index), current_device='cuda', labels = label).squeeze().float() * 255.0
        # For CelebA
        # new_samples = model.sample(len(index), current_device='cuda', labels = label)
        aug_samples[index] = new_samples
        aug_label = torch.ones(aug_samples.shape[0]).float() * k

        if k == 1:
            aug_data_total = aug_samples
            aug_label_total = aug_label
        else:
            aug_data_total = torch.cat((aug_data_total, aug_samples), dim=0)
            aug_label_total = torch.cat((aug_label_total, aug_label), dim=0)
        
        print(aug_samples.shape)
        print(aug_label.shape)
    
    return aug_data_total.byte(), aug_label_total

def cvae_aug_tabular(data, targets, model):
    data = torch.from_numpy(data)
    targets = torch.from_numpy(targets)
    classes, class_counts = np.unique(targets, return_counts=True)
    n_class = len(classes)
    n_max = max(class_counts)

    model.eval()

    print("#------ Augmenting with CVAE... ------#")
    for k in range(1, n_class):
        print("Augmenting for class {}".format(k))
        indices = np.where(targets == k)[0]
        class_len = len(indices)

        i = 0
        per_samples = 100
        aug_samples = torch.zeros_like(data)[:n_max-class_len].float().to('cuda')

        while i + per_samples < n_max - class_len:
            index = [j for j in range(i, i+per_samples)]
            label = torch.ones(per_samples) * k
            label = label.type(torch.LongTensor).to('cuda')
            new_samples_save = model.sample(per_samples, current_device='cuda', labels = label)

            new_samples = new_samples_save.squeeze().float()

            aug_samples[index] = new_samples

            i += per_samples
        
        index = [j for j in range(i, n_max - class_len)]

        label = torch.ones(len(index)) * k
        label = label.type(torch.LongTensor).to('cuda')
        new_samples = model.sample(len(index), current_device='cuda', labels = label).squeeze().float()
        aug_samples[index] = new_samples
        aug_label = torch.ones(aug_samples.shape[0]).float() * k

        if k == 1:
            aug_data_total = aug_samples
            aug_label_total = aug_label
        else:
            aug_data_total = torch.cat((aug_data_total, aug_samples), dim=0)
            aug_label_total = torch.cat((aug_label_total, aug_label), dim=0)
        
        print(aug_samples.shape)
        print(aug_label.shape)
    
    return aug_data_total.cpu().detach().numpy(), aug_label_total.cpu().detach().numpy()