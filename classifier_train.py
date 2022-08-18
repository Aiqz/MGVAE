import os
import yaml
import argparse
import numpy as np
from pathlib import Path
import torch.backends.cudnn as cudnn
import torch
import torchmetrics
from torch import nn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin

from models import *
from utils import *

parser = argparse.ArgumentParser(description='Classifier Training')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help='path to the config file',
                    default='configs/erm.yaml')

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)


tb_logger = TensorBoardLogger(save_dir=config['logging_params']['save_dir'],
                              name=config['logging_params']['name'],)

# For reproducibility
seed_everything(config['exp_params']['manual_seed'], True)

# Load dataset
if config['dataset'] == 'mnist':
    if config['data_params_aug']['aug_way'] == 'etg':
        # Load majority data
        data_majority = MNISTDataset(
            **config["data_params_majority"], pin_memory=len(config['trainer_params']['gpus']) != 0)
        data_majority.setup()
        # Creating model
        model = MGVAE_MLP(**config['model_params'], dataset_loader=data_majority.train_dataloader())
        experiment = MGVAE_MLP_Experiment(model, config['exp_params'])
        # Load pretrained model
        if config['model_params']['pretrained'] == True:
            print("Loading pretrained model!")
            experiment = experiment.load_from_checkpoint('/home/qingzhong/code/Exemplar_transfer_generation/logs/Exemplar_Transfer_Generation/NoPF3000/checkpoints/last.ckpt')
            print("Load model success!")
        experiment.to('cuda')
        experiment.eval()
        exemplar_model = experiment.model
        # Load balance dataset augmented by ETG
        data_balanced_aug = MNISTDataset(**config["data_params_aug"], pin_memory=len(config['trainer_params']['gpus']) != 0, exemplar_model=exemplar_model)
    
    elif config['data_params_aug']['aug_way'] == 'cvae':
        model = Condition_VAE_MLP(**config['model_params'], params=config['exp_params'])

        if config['model_params']['pretrained'] == True:
            print("Loading pretrained model!")
            model = model.load_from_checkpoint(
                '/home/qingzhong/code/Exemplar_transfer_generation/logs/Condition_VAE_MLP_MNIST/Pretrain_model_m60/checkpoints/last.ckpt',
                **config['model_params'], 
                params=config['exp_params']
            )
            model.to('cuda')
            model.eval()
            print("Load model success!")
        else:
            raise NotImplementedError("No pretrained model!")
        data_balanced_aug = MNISTDataset(**config["data_params_aug"], exemplar_model=model)
    
    elif config['data_params_aug']['aug_way'] == 'cgan':
        model = CGAN(**config['model_params'])
        if config['model_params']['pretrained'] == True:
            print("Loading pretrained model!")
            model_path = "logs/CGAN/Pretrain_model_m60/checkpoints/epoch=49-step=15449.ckpt"
            print(model_path)
            model = model.load_from_checkpoint(
                model_path,
                **config['model_params']
            )
            model.to('cuda')
            model.eval()
            print("Load model success!")
        else:
            raise NotImplementedError("No pretrained model!")
        data_balanced_aug = MNISTDataset(**config["data_params_aug"], exemplar_model=model)
    else:
        data_balanced_aug = MNISTDataset(**config["data_params_aug"])

elif config['dataset'] == 'fashionmnist':
    if config['data_params_aug']['aug_way'] == 'etg':
        # Load majority data
        data_majority = FashionMNISTDataset(
            **config["data_params_majority"], pin_memory=len(config['trainer_params']['gpus']) != 0)
        data_majority.setup()
        # Create model
        model = MGVAE_MLP(**config['model_params'], dataset_loader=data_majority.train_dataloader())
        experiment = MGVAE_MLP_Experiment(model, config['exp_params'])
        # Load pretrained model
        if config['model_params']['pretrained'] == True:
            print("Loading pretrained model!")
            experiment = experiment.load_from_checkpoint('logs/Exemplar_Transfer_Generation_Fashion/Pretrain_exemplar_0_4_m10_ewc5e4_ns_lr0.005_s100/checkpoints/last.ckpt')
            print("Load model success!")
        experiment.to('cuda')
        experiment.eval()
        exemplar_model = experiment.model
        # Load balance dataset augmented by ETG
        data_balanced_aug = FashionMNISTDataset(**config["data_params_aug"], pin_memory=len(config['trainer_params']['gpus']) != 0, exemplar_model=exemplar_model)
    
    elif config['data_params_aug']['aug_way'] == 'cvae':
        model = Condition_VAE_MLP(**config['model_params'], params=config['exp_params'])

        if config['model_params']['pretrained'] == True:
            print("Loading pretrained model!")
            model = model.load_from_checkpoint(
                'logs/Condition_VAE_MLP_FashionMNIST/Pretrain_model_m10/checkpoints/last.ckpt',
                **config['model_params'],
                params=config['exp_params']
            )
            model.to('cuda')
            model.eval()
            print("Load model success!")
        else:
            raise NotImplementedError("No pretrained model!")
        data_balanced_aug = FashionMNISTDataset(**config["data_params_aug"], exemplar_model=model)

    else:
        data_balanced_aug = FashionMNISTDataset(**config["data_params_aug"])

elif config['dataset'] == 'celeba':
    if config['data_params_aug']['aug_way'] == 'etg':
        # Load majority data for exemplar
        data_majority = CelebADataset(**config["data_params_majority"], pin_memory=len(config['trainer_params']['gpus']) != 0)
        data_majority.setup()
        exemplar_data, _ = data_majority.get_train_data()
        # Load pretrained models
        model = MGVAE_Conv(**config["model_params"], exemplar_data=exemplar_data, params=config['exp_params'])
        # Load pretrained model
        if config['model_params']['pretrained'] == True:
            print("Loading pretrained model!")
            model_bald = model.load_from_checkpoint(
                'logs/ETG_Longtail_Celeba/Pretrain_exemplar_bald_ewc50/checkpoints/epoch=99-step=2499.ckpt',
                **config["model_params"],
                exemplar_data=exemplar_data,
                params=config['exp_params']
            )
            model_bald.to('cuda')
            model_bald.eval()
            model_blonde = model.load_from_checkpoint(
                'logs/ETG_Longtail_Celeba/Pretrain_exemplar_blonde_ewc50/checkpoints/epoch=99-step=9999.ckpt',
                **config["model_params"],
                exemplar_data=exemplar_data,
                params=config['exp_params']
            )
            model_blonde.to('cuda')
            model_blonde.eval()
            model_brown = model.load_from_checkpoint(
                'logs/ETG_Longtail_Celeba/Pretrain_exemplar_brown_ewc50/checkpoints/epoch=99-step=999.ckpt',
                **config["model_params"],
                exemplar_data=exemplar_data,
                params=config['exp_params']
            )
            model_brown.to('cuda')
            model_brown.eval()
            model_gray = model.load_from_checkpoint(
                'logs/ETG_Longtail_Celeba/Pretrain_exemplar_gray_ewc50/checkpoints/epoch=99-step=199.ckpt',
                **config["model_params"],
                exemplar_data=exemplar_data,
                params=config['exp_params']
            )
            model_gray.to('cuda')
            model_gray.eval()
            # model_list = [model_bald]
            model_list = [model_blonde, model_bald, model_brown, model_gray]
            print("Load model success!")
        # Load balance dataset augmented by ETG
        data_balanced_aug = CelebADataset(**config["data_params_aug"], exemplar_model=model_list)

    elif config['data_params_aug']['aug_way'] == 'cvae':
        model = Condition_VAE_Conv(**config['model_params'], params=config['exp_params'])

        if config['model_params']['pretrained'] == True:
            print("Loading pretrained model!")
            model = model.load_from_checkpoint(
                'logs/Condition_VAE/version_0/checkpoints/last.ckpt',
                **config['model_params'], 
                params=config['exp_params']
            )
            model.to('cuda')
            model.eval()
            print("Load model success!")
        else:
            raise NotImplementedError("No pretrained model!")
        data_balanced_aug = CelebADataset(**config["data_params_aug"], exemplar_model=model)

    elif config['data_params_aug']['aug_way'] == 'cdcgan':
        model = CDCGAN(**config['model_params'])
        if config['model_params']['pretrained'] == True:
            print("Loading pretrained model!")
            model = model.load_from_checkpoint(
                'logs/CDCGAN/version_2/checkpoints/epoch=99-step=33699.ckpt',
                **config['model_params']
            )
            model.to('cuda')
            model.eval()
            print("Load model success!")
        else:
            raise NotImplementedError("No pretrained model!")
        data_balanced_aug = CelebADataset(**config["data_params_aug"], exemplar_model=model)

    else:
        data_balanced_aug = CelebADataset(**config["data_params_aug"])

elif config['dataset'] == 'tabular':

    if config['data_params_aug']['aug_way'] == 'etg':
        # Load majority data
        data_majority = TabularDataset(**config["data_params_majority"])
        data_majority.setup()
        # Create model
        model = MGVAE_Tabular(**config['model_params'], dataset_loader=data_majority.train_dataloader())
        experiment = MGVAE_MLP_Experiment(model, config['exp_params'])
        # Load pretrained model
        if config['model_params']['pretrained'] == True:
            print("Loading pretrained model!")
            experiment = experiment.load_from_checkpoint(
                'logs/Exemplar_Transfer_Generation_Huawei_1/version_0/checkpoints/epoch=99-step=99.ckpt'
            )
            print("Load model success!")
        experiment.to('cuda')
        experiment.eval()
        exemplar_model = experiment.model
        # Load balance dataset augmented by ETG
        data_balanced_aug = TabularDataset(**config["data_params_aug"], exemplar_model=exemplar_model)
    
    elif config['data_params_aug']['aug_way'] == 'cvae':
        model = Condition_VAE_Tabular(**config['model_params'], params=config['exp_params'])
        if config['model_params']['pretrained'] == True:
            print("Loading pretrained model!")
            model = model.load_from_checkpoint(
                'logs/Exemplar_Transfer_Generation_Huawei/Pretrain_model_ewc500/checkpoints/last.ckpt',
                **config['model_params'], 
                params=config['exp_params']
            )
            model.to('cuda')
            model.eval()
            print("Load model success!")
        else:
            raise NotImplementedError("No pretrained model!")
        data_balanced_aug = TabularDataset(**config["data_params_aug"], exemplar_model=model)
        
    else:
        data_balanced_aug = TabularDataset(**config["data_params_aug"])
        
# Setup dataset
data_balanced_aug.setup() 

# Classifier model
if config['dataset'] == 'mnist':
    CLASS, N_SAMPLES_PER_CLASS = np.unique(data_balanced_aug.mnist_train.targets, return_counts=True)
elif config['dataset'] == 'fashionmnist':
    CLASS, N_SAMPLES_PER_CLASS = np.unique(data_balanced_aug.fashion_mnist_train.targets, return_counts=True)
elif config['dataset'] == 'celeba':
    if config["data_params_aug"]['aug_way'] == 'smote' or \
        config["data_params_aug"]['aug_way'] == 'etg' or \
            config["data_params_aug"]['aug_way'] == 'rs' or \
                config["data_params_aug"]['aug_way'] == 'cvae' or \
                    config["data_params_aug"]['aug_way'] == 'cdcgan':
        # For datasets belong to "TensorDataset"
        CLASS, N_SAMPLES_PER_CLASS = np.unique(data_balanced_aug.dataset_train.tensors[1], return_counts=True)
    else:
        # For datasets belong to "ImageFolder"
        CLASS, N_SAMPLES_PER_CLASS = np.unique(data_balanced_aug.dataset_train.targets, return_counts=True)
elif config['dataset'] == 'tabular':
    CLASS, N_SAMPLES_PER_CLASS = np.unique(data_balanced_aug.dataset_train.tensors[1], return_counts=True)
else:
    raise NotImplementedError("Wrong dataset name!")

N_CLASS = len(CLASS)
print(CLASS)
print(N_SAMPLES_PER_CLASS)
if config["data_params_aug"]['aug_way'] == 'rw':
    per_cls_weights = 1 / np.array(N_SAMPLES_PER_CLASS)
    per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(N_SAMPLES_PER_CLASS)
    per_cls_weights = torch.FloatTensor(per_cls_weights).to("cuda")
elif config["data_params_aug"]['aug_way'] == 'cbrw':
    beta = 0.9999
    effective_num = 1.0 - np.power(beta, N_SAMPLES_PER_CLASS)
    per_cls_weights = (1.0 - beta) / np.array(effective_num)
    per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(N_SAMPLES_PER_CLASS)
    per_cls_weights = torch.FloatTensor(per_cls_weights).to("cuda")
else:
    per_cls_weights = torch.ones(N_CLASS).to("cuda")

print("Weights for per class:", per_cls_weights)

criterion = nn.CrossEntropyLoss(weight=per_cls_weights, reduction='none')


if config['dataset'] == 'celeba':
    # criterion = nn.CrossEntropyLoss()
    cls_model = pl_resnet(num_classes=N_CLASS, criterion=criterion)
elif config['dataset'] == 'tabular':
    if config["data_params_aug"]['data_name'] == 'musk':
        cls_model = simple_mlp(input_dim = 166, output_dim=2, criterion=criterion)
    elif config["data_params_aug"]['data_name'] == 'water_quality':
        cls_model = simple_mlp(input_dim = 20, output_dim=2, criterion=criterion)
    elif config["data_params_aug"]['data_name'] == 'isolet':
        cls_model = simple_mlp(input_dim = 617, output_dim=2, criterion=criterion)
    else:
        raise NotImplementedError("Wrong tabular dataset name!")
else:
    cls_model = simple_mlp(input_dim=784, output_dim=2, criterion=criterion)

runner = Trainer(logger=tb_logger,
                 callbacks=[
                     LearningRateMonitor(),
                     ModelCheckpoint(save_top_k=2,
                                     dirpath=os.path.join(
                                         tb_logger.log_dir, "checkpoints"),
                                     monitor="valid_loss",
                                     save_last=True),
                 ],
                 strategy=DDPPlugin(find_unused_parameters=False),
                 replace_sampler_ddp=False,
                 **config['trainer_params'])

# print(f"======= Training {config['model_params']['name']} =======")
runner.fit(cls_model, datamodule=data_balanced_aug)