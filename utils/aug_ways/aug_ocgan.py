#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
    @File    :   cdcgan_aug.py
    @Time    :   2022/03/31 09:34:34
    @Author  :   Qingzhong Ai 
    @Contact :   aqz1995@163.com
    @Desc    :   None
'''
import os
import torch
import numpy as np
import torchvision.utils as vutils

def cond_gan_aug(data, targets, model):
    classes, class_counts = np.unique(targets, return_counts=True)
    n_class = len(classes)
    n_max = max(class_counts)

    model.eval()

    print("#------ Augmenting with Cond_GAN... ------#")
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

            new_samples_save = model.cond_sample(y=label)
            new_samples = new_samples_save.detach().squeeze()

            # if i == 0:
            #     vutils.save_image(new_samples_save.data,
            #                 os.path.join('/home/qingzhong/code/Exemplar_transfer_generation/logs/CGAN_mnist', 
            #                             "Generation", 
            #                             f"Gen_{k}.png"),
            #                 normalize=False,
            #                 nrow=10)
            aug_samples[index] = new_samples

            i += per_samples
        
        index = [j for j in range(i, n_max - class_len)]

        label = torch.ones(len(index)) * k
        label = label.type(torch.LongTensor).to('cuda')
        new_samples = model.cond_sample(y=label)
        aug_samples[index] = new_samples.detach().squeeze()
        aug_label = torch.ones(aug_samples.shape[0]).float() * k

        if k == 1:
            aug_data_total = aug_samples
            aug_label_total = aug_label
        else:
            aug_data_total = torch.cat((aug_data_total, aug_samples), dim=0)
            aug_label_total = torch.cat((aug_label_total, aug_label), dim=0)
        
        print(aug_samples.shape)
        print(aug_label.shape)
    
    return aug_data_total, aug_label_total
