import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from time import time
import wandb
import os
from ViT import DiffViT
import itertools
from train import ModelTrainer, ModelBuilder
# =============================
# Data Module with configurable augmentation
# =============================
class DataModule:
    def __init__(self, config):
        self.config = config

    def setup(self):
        CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
        CIFAR_STD  = (0.2470, 0.2435, 0.2616)

        if self.config.get("data_augmentation", True):
            train_transforms = [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(p=0.5)
            ]
        else:
            train_transforms = []

        train_transforms += [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
        ]
        self.train_transform = transforms.Compose(train_transforms)
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
        ])

        self.train_dataset = datasets.CIFAR10(
            root='./data', train=True, download=True, transform=self.train_transform
        )
        self.test_dataset = datasets.CIFAR10(
            root='./data', train=False, download=True, transform=self.test_transform
        )

    def get_dataloaders(self, batch_size=128, num_workers=4):
        total = len(self.test_dataset)
        val_size = total // 2
        test_size = total - val_size
        val_subset, test_subset = torch.utils.data.random_split(
            self.test_dataset, [val_size, test_size]
        )

        train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_subset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_subset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
        return train_loader, val_loader, test_loader

# =============================
# Runner
# =============================
def run_experiment(config, run_name):
    dm = DataModule(config); dm.setup()
    train_loader, val_loader, test_loader = dm.get_dataloaders(batch_size=config.get('batch_size',128))
    builder = ModelBuilder(config); model = builder.build()
    trainer = ModelTrainer(model, config, train_loader, val_loader, test_loader, run_name)
    trainer.init_wandb(project_name="ViT-Comparison")
    metrics = trainer.train(num_epochs=config['epochs'])
    trainer.plot_metrics()
    if trainer.wandb_run: trainer.wandb_run.finish()
    return metrics

if __name__ == '__main__':
    base_config = {
        'image_size':32,
        'patch_size':4,
        'num_channels':3,
        'qkv_bias':True,
        'attention_probs_dropout_prob':0.1,
        'hidden_dropout_prob':0.1,
        'initializer_range':0.02,
        'num_classes':10,
        'use_differential_attention':True,
        'positional_embedding':'none',
        'epochs':50,
        'batch_size':128
    }

    combos = [(96, 368, 8, 4), (96, 368, 12, 4), (96, 368, 12, 8)]
    aug_opts = [True, False]

    for h,i,l,a in combos:
        for aug in aug_opts:
            cfg = base_config.copy()
            cfg.update({
                'hidden_size':h,
                'intermediate_size':i,
                'num_hidden_layers':l,
                'num_attention_heads':a,
                'data_augmentation':aug
            })
            name = f"DiffViT_h{h}_i{i}_l{l}_a{a}_{'aug' if aug else 'noaug'}"
            print(f"Running {name}")
            run_experiment(cfg, name)
