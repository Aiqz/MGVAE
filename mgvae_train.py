import os
import yaml
import argparse
import numpy as np
from pathlib import Path
# from models.exemplar_vae_tabular import Exemplar_VAE_Tabular
# from models.experiment import VAEXperiment
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

parser = argparse.ArgumentParser(description='training for MGVAE.')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help='path to the config file',
                    default='configs/configs_generation/mgvae_mlp.yaml')

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
    data_majority = MNISTDataset(**config["data_params_majority"])
    data_majority.setup()
    data_minority = MNISTDataset(**config["data_params_minority"])
    data_minority.setup()
elif config["dataset"] == "fashionmnist":
    data_majority = FashionMNISTDataset(**config["data_params_majority"])
    data_majority.setup()
    data_minority = FashionMNISTDataset(**config["data_params_minority"])
    data_minority.setup()
elif config["dataset"] == "celeba":
    data_majority = CelebADataset(**config["data_params_majority"], pin_memory=len(config['trainer_params']['gpus']) != 0)
    data_majority.setup()
    print("Loading majority data for CelebA....")
    exemplar_data, _ = data_majority.get_train_data()
    data_minority = CelebADataset(**config["data_params_minority"], pin_memory=len(config['trainer_params']['gpus']) != 0)
    data_minority.setup()
elif config["dataset"] == "tabular":
    print("Loading data for", config["data_params_majority"]["data_name"])
    data_majority = TabularDataset(**config["data_params_majority"])
    data_majority.setup()
    data_minority = TabularDataset(**config["data_params_minority"])
    data_minority.setup()

# for standard exemplar-vae inbalanced data
# data_imbalanced = MNISTDataset(**config["data_params"], pin_memory=len(config['trainer_params']['gpus']) != 0)
# data_imbalanced.setup()

# Load model
if config['model_params']['name'] == "MGVAE_MLP":
    model = MGVAE_MLP(**config['model_params'], dataset_loader=data_majority.train_dataloader())
    experiment = MGVAE_MLP_Experiment(model, config['exp_params'])

elif config['model_params']['name'] == "MGVAE_Tabular":
    model = MGVAE_Tabular(**config['model_params'], dataset_loader=data_majority.train_dataloader())
    experiment = MGVAE_MLP_Experiment(model, config['exp_params'])
    
elif config['model_params']['name'] == "MGVAE_Conv":
    model = MGVAE_Conv(**config['model_params'], exemplar_data=exemplar_data, params=config['exp_params'])
    # model.to('cuda')


# Load pretrained model
if config['model_params']['pretrained'] == True:
    print("Loading pretrained model!")
    if config['model_params']['name'] == "MGVAE_MLP":
        experiment = experiment.load_from_checkpoint(
            'logs/Exemplar_Transfer_Generation_Fashion/Pretrain_model_0_4/checkpoints/last.ckpt'
        )

    elif config['model_params']['name'] == "MGVAE_Tabular":
        experiment = experiment.load_from_checkpoint(
            'logs/Exemplar_Transfer_Generation_Huawei_1/Pretrain_exemplar_model/checkpoints/last.ckpt'
        )

    elif config['model_params']['name'] == "MGVAE_Conv":
        model = model.load_from_checkpoint(
            'logs/Exemplar_Transfer_Generation_CelebA/Pretrain_model_1_400_nc2000_mse(1)/checkpoints/last.ckpt', 
            **config['model_params'], 
            exemplar_data=exemplar_data, 
            params=config['exp_params'],
            ewc_params=config['ewc_params']
            )
    print("Load model success!")

if config['model_params']['name'] == "MGVAE_MLP" and config['ewc_params']['ewc'] == True:
    print("With EWC loss!")
    experiment.to('cuda')
    experiment_ewc = MGVAE_MLP_Experiment_EWC(experiment.model, experiment.params, ewc_weight=config['ewc_params']['ewc_weight'])

if config['model_params']['name'] == "MGVAE_Tabular" and config['ewc_params']['ewc'] == True:
    print("With EWC loss, weight:", config['ewc_params']['ewc_weight'])
    experiment.to('cuda')
    experiment_ewc = MGVAE_MLP_Experiment_EWC(experiment.model, experiment.params, ewc_weight=config['ewc_params']['ewc_weight'])

if config['model_params']['name'] == "MGVAE_Conv" and config['ewc_params']['ewc'] == True:
    print("With EWC loss, weight:", config['ewc_params']['ewc_weight'])
    model.to('cuda')
    model.ewc_pare()
    
# Save minority
# print(fashion_data_minority.fashion_mnist_train.data.shape)
# minotity_samples = fashion_data_minority.fashion_mnist_train.data.float().unsqueeze(dim=1)

# vutils.save_image(minotity_samples,
#                           os.path.join('/hdd/aiqingzhong/code22/Exemplar_transfer_generation/logs', 
#                                        "Minority", 
#                                        f"minority_samples.png"),
#                           normalize=True,
#                           nrow=10)

# experiment.to('cuda')
# experiment.eval()
# new_samples = experiment.model.sample(1, current_device='cuda').squeeze()


runner = Trainer(logger=tb_logger,
                 callbacks=[
                     LearningRateMonitor(),
                     ModelCheckpoint(
                                     save_top_k=-1,
                                     every_n_epochs=50,
                                     dirpath=os.path.join(
                                         tb_logger.log_dir, "checkpoints"),
                                     save_last=True),
                 ],
                 strategy=DDPPlugin(find_unused_parameters=True),
                 **config['trainer_params'])

Path(f"{tb_logger.log_dir}/Samples").mkdir(exist_ok=True, parents=True)
Path(f"{tb_logger.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)

# print(f"======= Training {config['model_params']['name']} =======")
if config['model_params']['name'] == "MGVAE_MLP" or config['model_params']['name'] == "MGVAE_Tabular":
    if config['ewc_params']['ewc'] == True:
        runner.fit(experiment_ewc, datamodule=data_minority)
    else:
        runner.fit(experiment, datamodule=data_majority)

elif config['model_params']['name'] == "MGVAE_Conv":
    if config['ewc_params']['ewc'] == True:
        runner.fit(model, datamodule=data_minority)
    else:
        runner.fit(model, datamodule=data_majority)

else:
    raise NotImplementedError("Wrong model name!")
