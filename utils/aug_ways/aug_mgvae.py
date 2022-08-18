from more_itertools import sample
from soupsieve import select
import torch
import os
import numpy as np
import torchvision.utils as vutils

def aug_mgvae(data, targets, model):
    # For two classes
    classes, class_counts = np.unique(targets, return_counts=True)
    n_class = len(classes)
    n_max = max(class_counts)

    model.eval()
    aug_data = []
    aug_label = []

    print("#------ Augmenting with MGVAE... ------#")
    for k in range(1, n_class):
        print("Augmenting for class {}".format(k))
        indices = np.where(targets == k)[0]
        class_data = data[indices]
        class_len = len(indices)
        
        i = 0
        per_samples = 100
        # aug_samples = None
        aug_samples = torch.zeros_like(data)[:n_max-class_len].float().to('cuda')
        selected_images = torch.zeros(32, 1, 28, 28)
        j = 0
        while i + per_samples < n_max - class_len:
            index = [j for j in range(i, i+per_samples)]
            samples_plot = model.sample(100, current_device='cuda')

            new_samples_save = model.index_sample(index, current_device='cuda')
            # For MNIST and Fashion
            new_samples = new_samples_save.squeeze().float() * 255.0
            # For CelebA
            # new_samples = new_samples_save.detach().squeeze()

            aug_samples[index] = new_samples

            i += per_samples
            

        index = [j for j in range(i, n_max - class_len)]
        # new_samples = model.sample((n_max - class_len - i), current_device='cuda').squeeze() * 255.0
        new_samples = model.index_sample(index, current_device='cuda').squeeze().float() * 255.0
        aug_samples[index] = new_samples
        # aug_samples = torch.cat((aug_samples, new_samples), dim=0)
        aug_label = torch.ones(aug_samples.shape[0]).float()

        # for i in range(n_max - class_len):
        #     print(i)
        #     new_samples = model.sample(100, current_device='cuda').squeeze() * 255.0
        #     aug_data.append(new_samples)
        #     aug_label.append(k)
        print(aug_samples.shape)
        print(aug_label.shape)

    return aug_samples.byte(), aug_label


def aug_mgvae_long_tail(data, targets, model_list):
    classes, class_counts = np.unique(targets, return_counts=True)
    n_class = len(classes)
    n_max = max(class_counts)

    aug_data_total = []
    aug_label_total = []

    combined_images = torch.zeros(500, 3, 64, 64).float().to('cuda')
    selected_images = torch.zeros(50, 3, 64, 64).float().to('cuda')
    print("#------ Augmenting with MGVAE... ------#")
    for k in range(1, n_class):
        print("Augmenting for class {}".format(k))
        model = model_list[k-1]
        model.eval()
        indices = np.where(targets == k)[0]
        class_data = data[indices]
        class_len = len(indices)

        i = 0
        per_samples = 100
        # aug_samples = None
        aug_samples = torch.zeros_like(data)[:n_max-class_len].float().to('cuda')
        while i + per_samples < n_max - class_len:
            index = [j for j in range(i, i+per_samples)]
            
            new_samples_save = model.index_sample(index, current_device='cuda')
            # For MNIST and Fashion
            # new_samples = new_samples_save.squeeze().float() * 255.0
            # For CelebA
            new_samples = new_samples_save.detach().squeeze()

            aug_samples[index] = new_samples

            i += per_samples
            

        index = [j for j in range(i, n_max - class_len)]
        # new_samples = model.sample((n_max - class_len - i), current_device='cuda').squeeze() * 255.0
        new_samples = model.index_sample(index, current_device='cuda').detach().squeeze()
        aug_samples[index] = new_samples
        # aug_samples = torch.cat((aug_samples, new_samples), dim=0)
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


def aug_mgvae_Tabular(data, targets, model):
    # For two classes
    data = torch.from_numpy(data)
    targets = torch.from_numpy(targets)

    classes, class_counts = np.unique(targets, return_counts=True)
    n_class = len(classes)
    n_max = max(class_counts)

    model.eval()
    aug_data = []
    aug_label = []

    print("#------ Augmenting with MGVAE... ------#")
    for k in range(1, n_class):
        print("Augmenting for class {}".format(k))
        indices = np.where(targets == k)[0]
        class_data = data[indices]
        class_len = len(indices)
        
        i = 0
        per_samples = 100
        # aug_samples = None
        aug_samples = torch.zeros_like(data)[:n_max-class_len].float().to('cuda')
        
        while i + per_samples < n_max - class_len:
            index = [j for j in range(i, i+per_samples)]
            new_samples_save = model.index_sample(index, current_device='cuda')

            new_samples = new_samples_save.squeeze().float()

            aug_samples[index] = new_samples

            i += per_samples
            

        index = [j for j in range(i, n_max - class_len)]

        new_samples = model.index_sample(index, current_device='cuda').squeeze().float()
        aug_samples[index] = new_samples
        # aug_samples = torch.cat((aug_samples, new_samples), dim=0)
        aug_label = torch.ones(aug_samples.shape[0]).float()

        print(aug_samples.shape)
        print(aug_label.shape)

    return aug_samples.cpu().detach().numpy(), aug_label.cpu().detach().numpy()
