"""Training loop for the RGB -> heatmap model (used by rgb_to_graph/main.py).

ModelTrainer wraps a torch model with an Adam optimizer and configurable
loss, and provides:
- run_epoch / validate:  standard train & eval passes (tqdm progress,
  per-batch loss, average inference time on validation).
- train:                 the epoch loop — periodically saves input/output
  visualization images, updates the training-curve plot, and checkpoints
  the model state_dict every `checkpoint_interval` epochs.

All paths, hyperparameters and intervals come from the shared config dict.
"""
import os
import time
from typing import Dict, Any, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

class ModelTrainer:
    """Handles model training and evaluation."""
    
    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        self.model = model
        self.config = config
        self.device = config['device']
        
        # Setup optimizer and criterion
        self.optimizer = optim.Adam(
            model.parameters(), 
            lr=config['lr'],
            weight_decay=config.get('weight_decay', 1e-3)
        )
        self.criterion = config.get('criterion', nn.MSELoss())
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        

    def run_epoch(self, train_loader: DataLoader) -> float:
        """Run one training epoch, showing a tqdm progress bar."""
        self.model.train()
        total_loss = 0.0

        # wrap the DataLoader in tqdm
        loader = tqdm(train_loader, desc="Training", unit="batch", leave=False)
        for inputs, targets in loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss    = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            # update the bar with current loss
            loader.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        loader.close()
        return avg_loss

    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Run validation and return loss and average inference time."""
        self.model.eval()
        total_loss = 0.0
        total_time = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                start_time = time.time()
                outputs = self.model(inputs)
                total_time += (time.time() - start_time)
                
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                
        avg_loss = total_loss / len(val_loader)
        avg_time = total_time / len(val_loader)
        
        return avg_loss, avg_time

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Tuple[List[float], List[float]]:
        """Main training loop."""
        print(f"Starting training for {self.config['epochs']} epochs...")
        
        for epoch in range(self.config['epochs']):
            train_loss = self.run_epoch(train_loader)
            val_loss, avg_time = self.validate(val_loader)
            
            
            # Visualize progress
            if epoch % self.config.get('plot_interval', 10) == 0:
                self._visualize_inference(epoch=epoch,
                                            val_loader=val_loader)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.plot_training_curves(self.train_losses, self.val_losses)
            print(f"Epoch [{epoch+1}/{self.config['epochs']}], "
                  f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                  f"Inference Time: {avg_time:.4f}s")
        
            # Save final model
            if (epoch + 1) % self.config.get('checkpoint_interval', 10) == 0:
                self._save_checkpoint(epoch)
        
        return self.train_losses, self.val_losses
    
    def _save_checkpoint(self, epoch: int) -> None:
        """Save model checkpoint."""
        checkpoint_path = os.path.join(
            self.config['checkpoints_dir'], 
            f"model_epoch_{epoch}.pth"
        )
        torch.save(self.model.state_dict(), checkpoint_path)
        print(f"Model saved to {checkpoint_path}")

    def plot_training_curves(self, train_losses: List[float], val_losses: List[float]) -> None:
        """Plot and save training curves."""
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Train Loss', alpha=0.8)
        plt.plot(val_losses, label='Validation Loss', alpha=0.8)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Losses')
        plt.legend()
        plt.grid(True, alpha=0.3)

        save_path = os.path.join(self.config['results_dir'], 'training_curves.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Training curves saved to {save_path}")

    def _visualize_inference(self, epoch: int, val_loader: DataLoader) -> None:
        """Visualize model inference results."""


        inputs, _ = next(iter(val_loader))
        inputs = inputs.to(self.device)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(inputs)[0].detach().cpu().numpy()

        
        # Plotting code here (e.g., using matplotlib)
        unnormalized_inputs = self.model.unNormalize(inputs[0]).detach().cpu().numpy().transpose(1, 2, 0)
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(unnormalized_inputs)
        plt.title("Input Image")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(outputs[0])
        plt.title("Model Output")
        plt.axis('off')

        plt.savefig(os.path.join(self.config['results_dir'], f'epoch_{epoch}.png'))
        plt.close()