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

            if i < 1000:
                
                if i == 900:
                    s_index = list(range(0, 16))
                    for ii in s_index:
                        iii = ii * 2
                        selected_images[j:j+2] = samples_plot[iii:iii+2]
                        j = j + 2
                    # selected_images[:] = samples_plot[s_index]
                    vutils.save_image(selected_images.data,
                            os.path.join('/home/qingzhong/code/Exemplar_transfer_generation/logs/ETG_fashion', 
                                        "Generation", 
                                        "Appendix_fashionmnist_600.png"),
                            padding=1,
                            pad_value=255,
                            normalize=True,
                            nrow=2)
                    vutils.save_image(selected_images.data,
                            os.path.join('/home/qingzhong/code/Exemplar_transfer_generation/logs/ETG_fashion', 
                                        "Generation", 
                                        "Appendix_fashionmnist_600.pdf"),
                            padding=1,
                            pad_value=255,
                            normalize=True,
                            nrow=2)

                vutils.save_image(samples_plot.data,
                            os.path.join('/home/qingzhong/code/Exemplar_transfer_generation/logs/ETG_fashion', 
                                        "Generation", 
                                        f"Gen_{i}.png"),
                            normalize=True,
                            nrow=2)

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

            if i == 0:
                reference_images, generated_images = model.index_sample_with_input(index, current_device='cuda')
                vutils.save_image(reference_images[0:5].data,
                            os.path.join('/home/qingzhong/code/Exemplar_transfer_generation/logs/ETG_celeba', 
                                        "Generation", 
                                        "Black.png"),
                            normalize=True,
                            nrow=5)
                if k == 1:
                    index_of_reference = [i  for i in range(0, 500, 5)]
                    index_of_generated = [i+1 for i in index_of_reference]
                    combined_images[index_of_reference] = reference_images
                    combined_images[index_of_generated] = generated_images
                else:
                    index_of_reference = [i  for i in range(0, 500, 5)]
                    index_of_generated = [i+k for i in index_of_reference]
                    combined_images[index_of_generated] = generated_images

                if k == 4:

                    # _index = [14, 20, 34, 41, 44]
                    # _index = [ 54, 50, 57, 62, 61]
                    # _index = [0,2,4,5,8,9,11,12,15,22]
                    _index = [79,84,86,87,88,89,92,94,96,98]
                    j = 0
                    for ii in _index:
                        for iii in range(5):
                            selected_images[j] = combined_images[ii * 5 + iii]
                            j = j + 1
                    vutils.save_image(combined_images.data,
                            os.path.join('/home/qingzhong/code/Exemplar_transfer_generation/logs/ETG_celeba', 
                                        "Generation", 
                                        f"Gen_{i}.png"),
                            normalize=True,
                            nrow=5)
                    vutils.save_image(selected_images.data,
                            os.path.join('/home/qingzhong/code/Exemplar_transfer_generation/logs/ETG_celeba', 
                                        "Generation", 
                                        f"Appendix_celeba_1.png"),
                            normalize=True,
                            nrow=5)
                    vutils.save_image(selected_images.data,
                            os.path.join('/home/qingzhong/code/Exemplar_transfer_generation/logs/ETG_celeba', 
                                        "Generation", 
                                        f"Appendix_celeba_1.pdf"),
                            normalize=True,
                            nrow=5)
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
