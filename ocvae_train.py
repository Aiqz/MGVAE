import os
import yaml
import argparse
import numpy as np
from pathlib import Path
import torch.backends.cudnn as cudnn
import torch
import torchvision.utils as vutils
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin

from models import *
from utils import *

parser = argparse.ArgumentParser(description='Runner for Condition-VAE')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help='path to the config file',
                    default='configs/condition_vae.yaml')

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

# Dataset loader
if config["dataset"] == "mnist":
    data_imbalanced = MNISTDataset(**config['data_params'])
    data_imbalanced.setup()
elif config["dataset"] == "fashionmnist":
    data_imbalanced = FashionMNISTDataset(**config['data_params'])
    data_imbalanced.setup()
elif config["dataset"] == "celeba":
    data_imbalanced = CelebADataset(**config["data_params"])
    data_imbalanced.setup()
elif config["dataset"] == "tabular":
    data_imbalanced = TabularDataset(**config["data_params"])
    data_imbalanced.setup()

# Load model
if config['model_params']['name'] == "Condition_VAE_MLP":
    model = Condition_VAE_MLP(**config['model_params'], params=config['exp_params'])
elif config['model_params']['name'] == "Condition_VAE_Conv":
    model = Condition_VAE_Conv(**config['model_params'], params=config['exp_params'])
elif config['model_params']['name'] == "Condition_VAE_Tabular":
    model = Condition_VAE_Tabular(**config['model_params'], params=config['exp_params'])

# Load pretrained model
if config['model_params']['pretrained'] == True:
    print("Loading pretrained model!")
    if config['model_params']['name'] == "Condition_VAE":
        model = model.load_from_checkpoint(
            'path of pretrained model!!!'
        )
    print("Load model success!")

runner = Trainer(logger=tb_logger,
                 callbacks=[
                     LearningRateMonitor(),
                     ModelCheckpoint(save_top_k=-1,
                                     every_n_epochs=50,
                                     dirpath=os.path.join(
                                         tb_logger.log_dir, "checkpoints"),
                                     monitor='val_loss',
                                     save_last=True),
                 ],
                 strategy=DDPPlugin(find_unused_parameters=True),
                 **config['trainer_params'])

Path(f"{tb_logger.log_dir}/Samples").mkdir(exist_ok=True, parents=True)
Path(f"{tb_logger.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)

# print(f"======= Training {config['model_params']['name']} =======")
runner.fit(model, datamodule=data_imbalanced)
