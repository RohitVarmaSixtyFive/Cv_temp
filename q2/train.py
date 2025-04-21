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
# Import ViT model (assuming ViT.py contains your model definition)
from ViT import DiffViT

class DataModule:
    """Handle dataset loading, preprocessing, and batch creation."""
    
    def __init__(self, config):
        self.config = config
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        
    def setup(self):
        # Define transforms
        CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
        CIFAR_STD  = (0.2470, 0.2435, 0.2616)

        self.train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),   
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
        
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])

        
        self.train_dataset = datasets.CIFAR10(root='./data', train=True, 
                                             download=True, transform=self.train_transform)
        self.test_dataset =  datasets.CIFAR10(root='./data', train=False, 
                                            download=True, transform=self.test_transform)
        
    def get_dataloaders(self, batch_size=128, num_workers=8):
        # Split train_dataset into train and val subsets (90% train, 10% val)

        total_test = len(self.test_dataset)
        val_size = int(0.5 * total_test)
        test_size = total_test - val_size
        val_subset, test_subset = torch.utils.data.random_split(self.test_dataset, [test_size, val_size])
        
        ## save one image to plot
        # img, label = self.train_dataset[10]
        # # img = self.display_transform(img)
        # img = img.permute(1, 2, 0)  # Change to HWC format for display
        # img = (img * 255).numpy().astype(np.uint8)  # Convert to uint8
        # print(label)
        # plt.imshow(img)
        # plt.axis('off')
        # plt.savefig("sample_image.png", dpi=300, bbox_inches='tight')
        # plt.show()

        
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


class ModelBuilder:
    """Build and configure the ViT model."""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def build(self):
        model = DiffViT(self.config).to(self.device)
        return model


class ModelTrainer:
    """Handle model training, evaluation, and logging."""
    
    def __init__(self, model, config, train_loader, val_loader, test_loader, run_name=None):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = next(model.parameters()).device
        
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        self.test_losses = []
        self.test_accuracies = []
        
        self.criterion = nn.CrossEntropyLoss()
        
        
        base_lr = self.config.get("base_lr", 3e-4)
                
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr) 
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config['epochs'])
        
        
        self.run_name = f"DiffViT-Run-{config['patch_size']}-{config['hidden_size']}-{config['num_attention_heads']}-{config['num_hidden_layers']}-{config['intermediate_size']}-{config['positional_embedding']}"
        self.wandb_run = None
        
        save_dir = "models"
        os.makedirs(save_dir, exist_ok=True)
    
    def init_wandb(self, project_name="DiffViT-ViT-Training-Final"):
        """Initialize wandb with the model config."""
        self.wandb_run = wandb.init(project=project_name, name=self.run_name, config=self.config)
        
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        epoch_loss = 0.0
        start_time = time()
        
        for batch_idx, (data, target) in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch} Training")):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            
            output, _ = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
            self.optimizer.step()
            
            running_loss += loss.item()
            epoch_loss += loss.item()
            
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        
        # Calculate epoch metrics
        train_loss = epoch_loss / len(self.train_loader)
        train_accuracy = 100. * correct / total
        
        self.train_losses.append(train_loss)
        self.train_accuracies.append(train_accuracy)
        
        epoch_time = time() - start_time
        print(f'\nEpoch {epoch} Training Summary:')
        print(f'Average Loss: {train_loss:.6f}')
        print(f'Accuracy: {correct}/{total} ({train_accuracy:.2f}%)')
        print(f'Time: {epoch_time:.2f}s')
        
        # Log to wandb
        if self.wandb_run:
            self.wandb_run.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "train_time": epoch_time
            })
        
        return train_loss, train_accuracy
    
    def validate(self, epoch, is_test=False):
        """Validate or test the model based on the flag."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        start_time = time()
        
        # Choose the appropriate dataloader and description
        if is_test:
            loader = self.test_loader
            loader_desc = f"Epoch {epoch} Testing"
        else:
            loader = self.val_loader
            loader_desc = f"Epoch {epoch} Validation"
        
        with torch.no_grad():
            for data, target in tqdm(loader, desc=loader_desc):
                data, target = data.to(self.device), target.to(self.device)
                output, _ = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        avg_loss = total_loss / len(loader)
        accuracy = 100. * correct / total
        epoch_time = time() - start_time
        
        if is_test:
            print(f'\nEpoch {epoch} Test Summary:')
            print(f'Average Loss: {avg_loss:.6f}')
            print(f'Accuracy: {correct}/{total} ({accuracy:.2f}%)')
            print(f'Time: {epoch_time:.2f}s')
            if self.wandb_run:
                self.wandb_run.log({
                    "epoch": epoch,
                    "test_loss": avg_loss,
                    "test_accuracy": accuracy,
                    "test_time": epoch_time
                })
            return avg_loss, accuracy
        else:
            self.val_losses.append(avg_loss)
            self.val_accuracies.append(accuracy)
            print(f'\nEpoch {epoch} Validation Summary:')
            print(f'Average Loss: {avg_loss:.6f}')
            print(f'Accuracy: {correct}/{total} ({accuracy:.2f}%)')
            print(f'Time: {epoch_time:.2f}s')
            if self.wandb_run:
                self.wandb_run.log({
                    "epoch": epoch,
                    "val_loss": avg_loss,
                    "val_accuracy": accuracy,
                    "val_time": epoch_time
                })
            return avg_loss, accuracy
    
    def train(self, num_epochs=50, save_dir="models", checkpoint_interval=5):

        os.makedirs(save_dir, exist_ok=True)
        
        print(f"\n{'='*20} Starting Training: {self.run_name} {'='*20}\n")
        
        best_val_acc = 0.0
        
        for epoch in range(1, num_epochs + 1):
            print(f"\n{'='*15} Epoch {epoch}/{num_epochs} {'='*15}")
            
            # Training phase
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validation phase
            val_loss, val_acc = self.validate(epoch, is_test=False)
            
            # Test phase
            test_loss, test_acc = self.validate(epoch, is_test=True)
            
            # Track best validation accuracy
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                print(f"New best validation accuracy: {best_val_acc:.2f}%")
                
                # Save only the model state dictionary
                best_model_path = os.path.join(save_dir, f"{self.run_name}.pth")
                torch.save(self.model.state_dict(), best_model_path)
                print(f"Saved best accuracy model to '{best_model_path}'")
            
            self.scheduler.step()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Current learning rate: {current_lr}")
            
            if self.wandb_run:
                self.wandb_run.log({"learning_rate": current_lr, "epoch": epoch})
            
            print(f"\nEpoch {epoch} Complete | Train Loss: {train_loss:.6f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.6f} | Val Acc: {val_acc:.2f}% | Test Loss: {test_loss:.6f} | Test Acc: {test_acc:.2f}%")
            
            # Save checkpoint with just the model state dict
            if epoch % checkpoint_interval == 0:
                checkpoint_path = os.path.join(save_dir, f"{self.run_name}.pth")
                print(f"Saving model checkpoint at epoch {epoch}...")
                torch.save(self.model.state_dict(), checkpoint_path)
        
        # Final test on best model
        print("\nPerforming final test on best model...")
        final_test_loss, final_test_acc = self.validate(epoch, is_test=True)
        
        print(f"\nFinal Test Results | Loss: {final_test_loss:.6f} | Accuracy: {final_test_acc:.2f}%")
        
        # Save final model with just the model state dict
        final_path = os.path.join(save_dir, f"{self.run_name}_final.pth")
        torch.save(self.model.state_dict(), final_path)
        print(f"Final model saved to '{final_path}'")
        
        if self.wandb_run:
            self.wandb_run.log({
                "final_test_loss": final_test_loss,
                "final_test_accuracy": final_test_acc,
                "best_val_accuracy": best_val_acc,
                "total_epochs": epoch
            })
        
        return {
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'final_test_loss': final_test_loss,
            'final_test_accuracy': final_test_acc,
            'best_val_accuracy': best_val_acc,
            'total_epochs': epoch
        }
    
    def plot_metrics(self, save_path=None):
        """Plot and optionally save training metrics."""
        num_epochs = len(self.train_losses)
        plt.figure(figsize=(16, 8))

        plt.subplot(2, 2, 1)
        plt.plot(range(1, num_epochs + 1), self.train_losses, 'b-', label='Training Loss')
        plt.plot(range(1, num_epochs + 1), self.val_losses, 'r-', label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 2, 2)
        plt.plot(range(1, num_epochs + 1), self.train_accuracies, 'b-', label='Training Accuracy')
        plt.plot(range(1, num_epochs + 1), self.val_accuracies, 'r-', label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        plt.grid(True)
        
        # Learning rate plot
        if hasattr(self, 'lr_history') and self.lr_history:
            plt.subplot(2, 2, 3)
            plt.plot(range(1, len(self.lr_history) + 1), self.lr_history, 'g-')
            plt.xlabel('Epochs')
            plt.ylabel('Learning Rate')
            plt.title('Learning Rate Schedule')
            plt.grid(True)
            
        # Test accuracy plot if available
        if hasattr(self, 'test_accuracies') and self.test_accuracies:
            plt.subplot(2, 2, 4)
            plt.plot(range(1, len(self.test_accuracies) + 1), self.test_accuracies, 'p-')
            plt.xlabel('Epochs')
            plt.ylabel('Test Accuracy (%)')
            plt.title('Test Accuracy')
            plt.grid(True)

        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved training metrics plot to '{save_path}'")
        
        if self.wandb_run:
            self.wandb_run.log({"training_metrics": wandb.Image(plt)})
        
        plt.show()


def run_experiment(config, run_name=None, num_epochs=50, batch_size=512, checkpoint_interval=5):

    # Setup data
    data_module = DataModule(config)
    data_module.setup()
    train_loader, val_loader, test_loader = data_module.get_dataloaders(batch_size=batch_size)
    
    # Build model
    builder = ModelBuilder(config)
    model = builder.build()
    print(f"Model built and placed on {next(model.parameters()).device}")
    
    # Print model summary
    print(f"\nModel Configuration:")
    print(f"Patch Size: {config['patch_size']}")
    print(f"Hidden Size: {config['hidden_size']}")
    print(f"Attention Heads: {config['num_attention_heads']}")
    print(f"Hidden Layers: {config['num_hidden_layers']}")
    print(f"Intermediate Size: {config['intermediate_size']}")
    print(f"Positional Embedding: {config['positional_embedding']}")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    
    trainer = ModelTrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        run_name=run_name
    )
    
    trainer.init_wandb()
    
    if trainer.wandb_run:
        trainer.wandb_run.log({
            "total_params": total_params,
            "trainable_params": trainable_params,
            "batch_size": batch_size,
            "dataset_size": len(data_module.train_dataset)
        })
    
    metrics = trainer.train(num_epochs=num_epochs, checkpoint_interval=checkpoint_interval)
    
    metrics_path = f"{run_name}_metrics.png" if run_name else "vit_training_metrics.png"
    trainer.plot_metrics(save_path=metrics_path)
    
    if trainer.wandb_run:
        trainer.wandb_run.finish()
    
    return model, metrics


import itertools

def main():

    # if torch.cuda.is_available():
    #     print(f"Device: {torch.cuda.get_device_name(0)}")
    #     print(f"Allocated VRAM: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    #     print(f"Cached VRAM: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
    # else:
    #     print("CUDA not available.")

    
    base_config = {
    "image_size": 32,
    "patch_size": 4,
    "num_channels": 3,
    "hidden_dropout_prob": 0.1,
    "attention_probs_dropout_prob": 0.1,
    "initializer_range": 0.02,
    "qkv_bias": True,
    "num_classes": 10,
    "use_differential_attention": True,  
    "positional_embedding": "none",
    "epochs":50
    }

    hyperparams = {
        "hidden_size":        [128],
        "intermediate_size":  [512],
        "num_hidden_layers":  [8],
        "num_attention_heads":[4]
    }
    
    # config_1d_learned={
    #     "image_size": 32,
    #     "patch_size": 4,
    #     "num_channels": 3,
    #     "hidden_dropout_prob": 0.1,
    #     "attention_probs_dropout_prob": 0.1,
    #     "initializer_range": 0.02,
    #     "qkv_bias": True,
    #     "num_classes": 10,
    #     "use_differential_attention": True,  
    #     "positional_embedding": "1d_learned",
    #     "epochs":50,
    #     "hidden_size": 256,
    #     "intermediate_size": 1024,
    #     "num_hidden_layers":  8,
    #     "num_attention_heads": 4
    # }
    
    # config_none={
    #     "image_size": 32,
    #     "patch_size": 4,
    #     "num_channels": 3,
    #     "hidden_dropout_prob": 0.1,
    #     "attention_probs_dropout_prob": 0.1,
    #     "initializer_range": 0.02,
    #     "qkv_bias": True,
    #     "num_classes": 10,
    #     "use_differential_attention": True,  
    #     "positional_embedding": "none",
    #     "epochs":50,
    #     "hidden_size": 256,
    #     "intermediate_size": 1024,
    #     "num_hidden_layers":  8,
    #     "num_attention_heads": 4
    # }
    
    # config_sinusoidal={
    #     "image_size": 32,
    #     "patch_size": 4,
    #     "num_channels": 3,
    #     "hidden_dropout_prob": 0.1,
    #     "attention_probs_dropout_prob": 0.1,
    #     "initializer_range": 0.02,
    #     "qkv_bias": True,
    #     "num_classes": 10,
    #     "use_differential_attention": True,  
    #     "positional_embedding": "sinusoidal",
    #     "epochs":50,
    #     "hidden_size": 256,
    #     "intermediate_size": 1024,
    #     "num_hidden_layers":  8,
    #     "num_attention_heads": 4
    # }

    keys, values = zip(*hyperparams.items())
    configs_to_run = {}
    for idx, combo in enumerate(itertools.product(*values), start=1):
        cfg = base_config.copy()
        cfg.update(dict(zip(keys, combo)))
        configs_to_run[f"config{idx}"] = cfg

    # configs_to_run[f"config_1d"] = config_1d_learned
    # configs_to_run[f"config_none"] = config_none
    # configs_to_run[f"config_sinusoidal"] = config_sinusoidal

    for name, config in configs_to_run.items():
        print(f"\n\n{'='*30} Running experiment: {name} {'='*30}\n")
        model, metrics = run_experiment(
            config=config,
            run_name=name,
            num_epochs=50,
            batch_size=128,
            checkpoint_interval=5
        )
        print(f"\n{'='*30} Completed experiment: {name} {'='*30}\n")
        print(f"Training Summary for {name}:")
        print(f"  • Best Validation Accuracy: {metrics['best_val_accuracy']:.2f}%")
        print(f"  • Final Test Accuracy:      {metrics['final_test_accuracy']:.2f}%")
        print(f"  • Total Epochs Run:         {metrics['total_epochs']}\n")

if __name__ == "__main__":
    main()
