import torch
import os
import numpy as np
import torchvision.utils as vutils

def re_sampling(data, targets):
    print("Re-sampling")
    _, class_counts = np.unique(targets, return_counts=True)
    
    length = data.shape[0]
    num_samples = list(class_counts)

    selected_list = []

    for i in range(length):
        label = targets[i]
        selected_list.append(1.0 / num_samples[int(label)])
    # print(selected_list)
    return selected_list