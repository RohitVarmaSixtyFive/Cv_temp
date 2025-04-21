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
from datetime import datetime
import itertools

# Import ViT model (assuming ViT.py contains your model definition)
from ViT import ViTForClassfication
from train import ModelBuilder, ModelTrainer

# =============================
# Data Module with configurable augmentation
# =============================
class DataModule:
    """Handle dataset loading, preprocessing, and batch creation."""
    def __init__(self, config):
        self.config = config
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    def setup(self):
        # CIFAR-10 statistics
        CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
        CIFAR_STD  = (0.2470, 0.2435, 0.2616)

        # Use data augmentation if specified
        if self.config.get("data_augmentation", True):
            self.train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
            ])
        else:
            self.train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
            ])

        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])

        self.train_dataset = datasets.CIFAR10(
            root='./data', train=True, download=True, transform=self.train_transform
        )
        self.test_dataset = datasets.CIFAR10(
            root='./data', train=False, download=True, transform=self.test_transform
        )

    def get_dataloaders(self, batch_size=128, num_workers=8):
        # Split test dataset into val and test
        total = len(self.test_dataset)
        val_size = total // 2
        test_size = total - val_size
        val_subset, test_subset = torch.utils.data.random_split(
            self.test_dataset, [val_size, test_size]
        )

        train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=batch_size,
            shuffle=True, num_workers=num_workers, pin_memory=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_subset, batch_size=batch_size,
            shuffle=False, num_workers=num_workers, pin_memory=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_subset, batch_size=batch_size,
            shuffle=False, num_workers=num_workers, pin_memory=True
        )
        return train_loader, val_loader, test_loader


# =============================
# Experiment Runner
# =============================
def run_experiment(config, run_name=None, num_epochs=50, batch_size=128, checkpoint_interval=5):
    data_module = DataModule(config)
    data_module.setup()
    train_loader, val_loader, test_loader = data_module.get_dataloaders(batch_size=batch_size)

    builder = ModelBuilder(config)
    model = builder.build()

    # Assume ModelTrainer and its methods are imported or defined above
    trainer = ModelTrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        run_name=run_name
    )
    trainer.init_wandb(project_name="ViT-Comparison")
    metrics = trainer.train(
        num_epochs=config.get('epochs', 50),
        checkpoint_interval=checkpoint_interval
    )
    trainer.plot_metrics()

    return metrics

# =============================
# Main: Define hyperparameters and run experiments
# =============================
def main():
    # Base configuration
    base_config = {
        "image_size": 32,
        "patch_size": 4,
        "num_channels": 3,
        "qkv_bias": True,
        "attention_probs_dropout_prob": 0.1,
        "hidden_dropout_prob": 0.1,
        "initializer_range": 0.02,
        "num_classes": 10,
        "use_faster_attention": True,
        "positional_embedding": "none",
        "epochs": 50,
    }

    # Specific hyperparameter combinations to test
    combos = [
        (96, 368, 8, 4),
        (96, 368, 12, 4),
        (96, 368, 12, 8)
    ]
    aug_options = [True, False]

    for hidden_size, intermediate_size, num_hidden_layers, num_attention_heads in combos:
        for aug in aug_options:
            config = base_config.copy()
            config.update({
                'hidden_size': hidden_size,
                'intermediate_size': intermediate_size,
                'num_hidden_layers': num_hidden_layers,
                'num_attention_heads': num_attention_heads,
                'data_augmentation': aug
            })
            run_name = f"ViT_h{hidden_size}_i{intermediate_size}_l{num_hidden_layers}_a{num_attention_heads}_{'aug' if aug else 'noaug'}"
            print(f"\n===== Running {run_name} =====")
            metrics = run_experiment(
                config=config,
                run_name=run_name
            )
            print(f"Results for {run_name}: {metrics}\n")

if __name__ == "__main__":
    main()
