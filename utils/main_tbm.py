import os
import random
import wandb

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from train_tbm import *
from test_tbm import *
from utils_tbm import *
from tqdm.auto import tqdm

# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
random.seed(hash("setting random seeds") % 2**32 - 1)
np.random.seed(hash("improves reproducibility") % 2**32 - 1)
torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def model_pipeline(cfg:dict) -> None:
    # tell wandb to get started
    with wandb.init(project="pytorch-synthesis", config=cfg):
      # access all HPs through wandb.config, so logging matches execution!
      config = wandb.config

      # make the model, data, and optimization problem
      model, train_loader, test_loader, criterion, optimizer = make(config, '../../preprocessed.csv')

      # and use them to train the model
      train(model, train_loader, criterion, optimizer, config)

      # and test its final performance
      test(model, test_loader)

    return model

if __name__ == "__main__":
    wandb.login()

    config = dict(
        epochs=20,
        batch_size=1,
        learning_rate=5e-3,
        csv_path='../../preprocessed.csv',
        architecture="LSTM Autoencoder",
        subset_size = None)
    model = model_pipeline(config)

