#### Import Libraries ####
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import wandb
import random
import os
import math
import yaml


#### Import custom modules ####
from data.data_loader_v3 import data_loaders
from models.ConvLSTMabla import Predictor
from training.train import train
from training.validate import validate
from training.test import test_metrics
from utils.early_stopping import EarlyStopping
from utils.count_parameters import count_parameters
from utils.metrics import UnifiedFocalLoss, DiceLoss, BufferDiceLoss, HybridDiceLoss
from utils.Analysis import diagnose, buffer_zone_vis

    
def main(config=None):

    #### Config ####
    torch.cuda.empty_cache() # Clear GPU memory
    torch.manual_seed(42) # Set random seed for reproducibility
    random.seed(42) # Set random seed for reproducibility
    np.random.seed(42) # Set random seed for reproducibility

    #### Archiotecture Presets ####
    arch_presets = {
        "2 Layers": {
            "num_layers": 2, 
            "hidden_dims": [64, 64], 
            "kernel_sizes": [(5,5), (5,5)]
        },
        "3 Layers": {
            "num_layers": 3,
            "hidden_dims": [64, 128, 128],
            "kernel_sizes": [(5,5), (5,5), (5,5)]
        }
    }

    #### Initialize WandB ####
    wandb.login(key="d42992a374fbc96ee65d1955f037e71d58e30f45")
    wandb.init(config=config, project="THESIS", name="10_fut_biweekly_exp1")
    config = wandb.config
    project_name = wandb.run.name

    #### Load dataset and scenarios ####
    time = "bi-monthly"  # or "weekly", "monthly"
    case = 1
    time_dict = {"weekly": 34, "bi-monthly": 72, "monthly": 144}
    output_freq = 2  # hours
    wave_steps_per_frame = time_dict[time] // output_freq 
    data_path = "/p/11207608-coclico/MSc_students/Daniel/Scripts/outputs/original"
    
    folder_outputs = os.path.join("outputs", project_name)
    os.makedirs(folder_outputs, exist_ok=True)
    
    ## Load data loaders ####
    train_loader, val_loader, test_loader = data_loaders(data_path, batch_size=config.batch_size, time=time, case=case)
    
    #### Call WandB ####
    arch = config.architecture
    preset = arch_presets[arch]

    # Update config with consistent values
    config.num_layers   = preset["num_layers"]
    config.hidden_dims  = preset["hidden_dims"]
    config.kernel_sizes = preset["kernel_sizes"]
    input_dim = config.input_dim
    hidden_dims = config.hidden_dims
    kernel_sizes = config.kernel_sizes
    num_layers = config.num_layers
    lstm_hidden_size = config.lstm_hidden_size
    epochs = config.epochs
    learning_rate = config.learning_rate
    dropout = config.dropout
    future_seq = config.future_seq
    weight = config.UFL_weight
    delta = config.UFL_delta
    gamma = config.UFL_gamma
    alpha = config.HDL_alpha
    loss_function = config.loss_function


    #### Set device to cuda if available ####
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #### Create model ####
    model = Predictor(input_dim=input_dim, hidden_dims=hidden_dims, kernel_sizes=kernel_sizes, num_layers=num_layers, lstm_hidden_size=lstm_hidden_size, dropout=dropout, wave_steps_per_frame=wave_steps_per_frame, future_seq=future_seq, use_wave_context=False).to(device)
    total_params = count_parameters(model)
    wandb.watch(model)

    #if torch.cuda.device_count() > 1:
    #    print(f"Using {torch.cuda.device_count()} GPUs for training.")
    #    model = nn.DataParallel(model)

    #model.to(device)

    #### Define Loss functions
    if config.loss_function == "UnifiedFocalLoss":
        criterion = UnifiedFocalLoss(weight=weight, delta=delta, gamma=gamma)
    elif config.loss_function == "DiceLoss":
        criterion = DiceLoss()
    elif config.loss_function == "BufferDiceLoss":
        criterion = BufferDiceLoss()
    elif config.loss_function == "HybridDiceLoss":
        criterion = HybridDiceLoss(alpha=alpha)
    else:
        raise ValueError("Unsupported loss function")

    

    #### Define Optimizer and Scheduler ####
    if config.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif config.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    else:
        raise ValueError("Unsupported optimizer")
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    #### Print config values ####
    print('Using device:', device)
    print(f"hidden_dims: {hidden_dims}, length: {len(hidden_dims)}")
    print(f"kernel_sizes: {kernel_sizes}, length: {len(kernel_sizes)}")
    print(f"num_layers: {num_layers}")
    #print(f"Model is running with {model.module.convlstm.num_layers} layers.")

    #### Init empty lists to store losses and activate early stopping ####
    train_losses = []
    val_losses = []
    eval_table_data = []
    early_stopper = EarlyStopping(patience=10, min_delta=0.0001, verbose=True)
    best_val_loss = float('inf')

    #### Iterate over epochs to train the model ####
    for epoch in range(1, epochs + 1):

        # Call train and validate functions in each epoch
        train_loss = train(model, train_loader, criterion, optimizer, device, epoch, future_seq)
        avg_recall, avg_iou, avg_precision, avg_dice_buffer, val_loss, per_sample_losses, dice_buffer_s, idx, all_outputs, all_targets, all_inputs, all_wave_context, all_wave_fut, all_buffer_masks = validate(model, val_loader, criterion, device, epoch, eval_table_data, future_seq)

        wandb.log({"avg_recall": avg_recall, 
                   "avg_iou": avg_iou, 
                   "avg_precision": avg_precision, 
                   "avg_dice": avg_dice_buffer,
                   "epoch": epoch,
                   "train_loss": train_loss,
                   "val_loss": val_loss})

        # Scheduler, Help the model to learn by reducing the learning rate (!!!! Include a log to WandB!!!!)
        scheduler.step(val_loss)

        # Append losses to lists
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Save only the best model checking validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = os.path.join(folder_outputs, "best_model.pth")
            torch.save(model.state_dict(), model_path)
            artifact = wandb.Artifact(name="Best-ConvLSTM", type="model")
            artifact.add_file(model_path)
            wandb.log_artifact(artifact)
            print("✅ Saved new best model.")

        # Check for early stopping
        if early_stopper.step(val_loss):
            break  # stop training if early stopping triggered
    
    #### Log the validation per sample losses and metrics ####
    for i in range(len(idx)):
        losses = per_sample_losses[i]
        wandb.log({"Validation Loss per Sample": losses,
                    "Validation Dice Buffer Score per Sample": dice_buffer_s[i],
                    "Sample index Validation": idx[i]}
                    )
        
    #### Number of training steps ####
    steps = len(train_loader) * epochs
    for i in range(steps):
        wandb.log({"Training Steps": i+1})

    # # Test metrics and validation analysis
    test_data = []
    test_metrics(model, test_loader, device, test_data, future_seq)
    diagnose(all_outputs, all_targets, dice_buffer_s, all_inputs, all_wave_context, all_wave_fut, title_prefix="Validation")
    buffer_zone_vis(all_targets, all_outputs, all_buffer_masks, title="Validation", future_seq=future_seq)

    # # Log .py files to WandB
    wandb.run.log_code(".")

    # # Create DataFrame and log it
    table_columns2 = ["Metric"] + [f"t={i+1}" for i in range(future_seq)]
    df2= pd.DataFrame(test_data, columns=table_columns2)
    wandb_table2 = wandb.Table(dataframe=df2)
    wandb.log({"Test Evaluation Table": wandb_table2})
        
    print("Training complete!")
    return model

if __name__ == '__main__':

    with open("config/config.yaml", "r") as file:
        config = yaml.safe_load(file)
    model = main(config)