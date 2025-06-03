import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.checkpoint import checkpoint
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import wandb
import random
import os
import math

def visualize_prediction(model, test_loader, device, sample_idx=0):
    model.eval()
    # Get a sample from the test set
    for i, (slow_input, target_seq, fast_input) in enumerate(test_loader):  
        if i == sample_idx:
            break
    slow_input = slow_input.to(device)
    fast_input = fast_input.to(device)  
    target_seq = target_seq.to(device)
    with torch.no_grad():
        output = model(slow_input, fast_input, future_seq=10) 
    
    # Plot
    fig, axes = plt.subplots(3, 10, figsize=(20, 6))
    
    # Input sequence
    for t in range(10):
        axes[0, t].imshow(slow_input[0, t, 0].cpu().numpy(), cmap='gray')
        axes[0, t].set_title(f'Input t={t}')
        axes[0, t].axis('off')
    
    # Target sequence
    for t in range(10):
        axes[1, t].imshow(target_seq[0, t, 0].cpu().numpy(), cmap='gray')
        axes[1, t].set_title(f'Target t={t+10}')
        axes[1, t].axis('off')
    
    # Predicted sequence
    for t in range(10):
        axes[2, t].imshow(output[0, t, 0].cpu().numpy(), cmap='gray')
        axes[2, t].set_title(f'Pred t={t+10}')
        axes[2, t].axis('off')
    
    plt.tight_layout()
    plt.savefig('mnist_prediction.png')
    plt.close()
    
    # Log the visualization to WandB
    wandb.log({"Predictions": wandb.Image('mnist_prediction.png')})