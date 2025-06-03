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


def train(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    train_loss = 0
    total_batches = len(train_loader)
    
    # Track total time for the epoch
    start_epoch = torch.cuda.Event(enable_timing=True)
    end_epoch = torch.cuda.Event(enable_timing=True)
    start_epoch.record()
    for batch_idx, (slow_input, target_seq, fast_input) in enumerate(train_loader):
        slow_input = slow_input.to(device)
        fast_input = fast_input.to(device)
        target_seq = target_seq.to(device)

        start_iter = torch.cuda.Event(enable_timing=True)
        end_iter = torch.cuda.Event(enable_timing=True)
        start_iter.record()

        optimizer.zero_grad()
        output = model(slow_input, fast_input, future_seq=10)

        loss = criterion(output, target_seq)
        loss.backward()
        optimizer.step()

        end_iter.record()
        torch.cuda.synchronize()
        iteration_time = start_iter.elapsed_time(end_iter)/1000

        train_loss += loss.item()
        
        # Log time to WandB
        wandb.log({"Batch Loss": loss.item(), "Iteration Time (s)": iteration_time})
        
        if batch_idx % 100 == 0:  # Print every 100 batches
            print(f"Epoch {epoch}, Batch {batch_idx}, Time per Iteration: {iteration_time:.4f}s")

    end_epoch.record()
    torch.cuda.synchronize()

    epoch_time = start_epoch.elapsed_time(end_epoch)/1000
    avg_iteration_time = epoch_time / total_batches
            
    avg_train_loss = train_loss / len(train_loader)
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, avg_train_loss),  )
    
    # Log epoch train loss to WandB
    wandb.log({"Epoch Train Loss": avg_train_loss, "Epoch": epoch, "Avg Iteration Time (s)": avg_iteration_time})
    
    return avg_train_loss


def validate(model, val_loader, criterion, device, epoch):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for (slow_input, target_seq, fast_input) in val_loader:  
            slow_input = slow_input.to(device)
            fast_input = fast_input.to(device)
            target_seq = target_seq.to(device)
            output = model(slow_input, fast_input, future_seq=10)  
            loss = criterion(output, target_seq)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)
    wandb.log({"Epoch Validation Loss": avg_val_loss, "Epoch": epoch})
    return avg_val_loss