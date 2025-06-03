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

#### Import custom modules ####
from data_loader_v3 import data_loaders
from metrics import  buffer_zone_metrics, UnifiedFocalLoss, UnifiedFocalLossSample, dice_score, dice_score_shoreline_buffer, buffer_mask
from Analysis import diagnose, buffer_zone_vis



####  Class ConvLSTMCell ####
class ConvLSTMCell(nn.Module):
    """
    Basic ConvLSTM cell.
    """
    def __init__(self, input_dim, hidden_dim, kernel_size, dropout, bias=True): #norm
        """
        Initialize ConvLSTM cell.
        
        Parameters:
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: int
            Size of the convolutional kernel.
        bias: bool
            Whether to add bias or not.
        """
        super(ConvLSTMCell, self).__init__()
        
        #### Initialize parameters ####
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        self.dropout_rate = dropout

        #### Convolutional layer ####
        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim, # input + hidden state 
            out_channels=4 * self.hidden_dim,  # For the four gates
            kernel_size=self.kernel_size, # kernel size
            padding=self.padding, # padding to keep the same size
            bias=self.bias # bias for the convolution
        )

        #### Dropout layer to prevent overfitting ####
        self.dropout = nn.Dropout2d(p=self.dropout_rate) if self.dropout_rate > 0 else None

    #### Forward pass of ConvLSTM cell ####  
    def forward(self, input_tensor, cur_state):
        """
        Forward propagation.
        
        Parameters:
        ----------
        input_tensor: 4D tensor
            Input tensor of shape (batch_size, input_dim, height, width)
        cur_state: tuple
            Current hidden and cell states (h_cur, c_cur)
            
        Returns:
        -------
        h_next, c_next: next hidden and cell states
        """

        #### Get current hidden and cell states ####
        h_cur, c_cur = cur_state
        
        #### concatenate input and hidden state ####
        combined = torch.cat([input_tensor, h_cur], dim=1)
        
        #### Apply convolution to combined input ####
        combined_conv = self.conv(combined)
        
        #### Split the combined output into the 4 gates
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        
        #### Apply gate activations
        i = torch.sigmoid(cc_i)  # input gate
        f = torch.sigmoid(cc_f)  # forget gate
        o = torch.sigmoid(cc_o)  # output gate
        g = torch.tanh(cc_g)     # cell gate
        
        #### Update cell state and hidden state
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        #### Apply dropout if specified ####
        h_next = h_next if self.dropout is None else self.dropout(h_next)
        
        #### Return next hidden and cell states ####
        return h_next, c_next

#### ConvLSTM class ####
class ConvLSTM(nn.Module):
    """
    ConvLSTM module for sequence prediction with multiple layers and varying hidden dimensions.
    """
    def __init__(self, input_dim, hidden_dims, kernel_sizes, num_layers, dropout , batch_first=True, bias=True):
        """
        Initialize ConvLSTM.
        Parameters:
        ----------
        input_dim: int
            Number of channels in input
        hidden_dims: list of ints
            List of hidden dimensions for each layer
        kernel_sizes: list of tuples
            List of kernel sizes for each layer
        num_layers: int
            Number of LSTM layers stacked on each other
        batch_first: bool
            If True, dimension 0 is batch, dimension 1 is time, dimension 2 is channel.
            If False, dimension 0 is time, dimension 1 is batch, dimension 2 is channel.
        bias: bool
            Whether to add bias or not
        """

        #### super to initialize parent class ####
        super(ConvLSTM, self).__init__()

        #### Check input parameters ####
        assert len(hidden_dims) == num_layers, "Length of hidden_dims must match num_layers"
        assert len(kernel_sizes) == num_layers, "Length of kernel_sizes must match num_layers"

        #### Initialize parameters ####
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.kernel_sizes = kernel_sizes
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.dropout = dropout

        #### Stack ConvLSTM cells ####
        cell_list = []
        for i in range(self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dims[i - 1]
            cell_list.append(ConvLSTMCell(
                input_dim=cur_input_dim,
                hidden_dim=self.hidden_dims[i],
                kernel_size=self.kernel_sizes[i],
                dropout=self.dropout,
                bias=self.bias
            ))
        #### Store the cell list as a ModuleList for proper parameter management
        self.cell_list = nn.ModuleList(cell_list)

    #### Initialize hidden state ####
    def _init_hidden(self, batch_size, image_size):
        """
        Initialize hidden state.
        Parameters:
        ----------
        batch_size: int
            Size of the batch
        image_size: tuple
            Height and width of the feature maps
        Returns:
        -------
        init_states: list
            List of tuples (h, c) for each layer
        """
        height, width = image_size
        init_states = []
        for i in range(self.num_layers):
            h = torch.zeros(batch_size, self.hidden_dims[i], height, width, device=self.cell_list[0].conv.weight.device)
            c = torch.zeros(batch_size, self.hidden_dims[i], height, width, device=self.cell_list[0].conv.weight.device)
            init_states.append((h, c))
        return init_states
    
    #### Forward pass through ConvLSTM layers ####
    def forward(self, input_tensor, hidden_state=None):
        """
        Forward pass through ConvLSTM layers.
        Parameters:
        ----------
        input_tensor: 5D tensor
            Input of shape (batch_size, time, channels, height, width) if batch_first
            or (time, batch_size, channels, height, width) otherwise
        hidden_state: list of tuples
            List of tuples (h, c) for each layer
        Returns:
        -------
        layer_output_list: list
            List of outputs from each layer
        last_state_list: list
            List of final states from each layer
        """
        #### Make sure we're working with batch first format
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        #### Get dimensions of input tensor ####
        batch_size, seq_len, _, height, width = input_tensor.size()

        #### Initialize hidden states if none provided
        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size, (height, width))

        #### Initialize lists to store outputs and states ####
        layer_output_list = []
        last_state_list = []

        #### Process each sequence element
        for layer_idx in range(self.num_layers):

            #### Get current hidden state ####
            h, c = hidden_state[layer_idx]
            output_inner = []

            #### Process each timestep in the sequence ####
            for t in range(seq_len):
                # Get input for this timestep
                if layer_idx == 0:
                    # For the first layer, input comes from the original input sequence
                    x = input_tensor[:, t, :, :, :]
                else:
                    # For subsequent layers, input comes from the output of the previous layer
                    x = layer_output_list[layer_idx - 1][:, t, :, :, :]

                # set the hidden state to the current hidden state
                h, c = self.cell_list[layer_idx](x, (h, c))

                # Store output
                output_inner.append(h)

            # Stack outputs along time dimension
            layer_output = torch.stack(output_inner, dim=1)
            layer_output_list.append(layer_output)
            last_state_list.append((h, c))

        # Return outputs and final states
        return layer_output_list[-1], last_state_list

#### ProgressiveWaveContextEncoder class ####
class ProgressiveWaveContextEncoder(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, num_contexts=10):

        #### Initialize parameters ####	
        super(ProgressiveWaveContextEncoder, self).__init__()
        ## LSTM encoder to process wave data
        self.encoder = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        ## Projection layer to reduce dimensionality
        self.projection = nn.Linear(hidden_size, hidden_size)
        ## Number of contexts to generate
        self.num_contexts = num_contexts

    #### Forward pass through the encoder ####
    def forward(self, wave_input, wave_future):
        """
        wave_input: [B, 240, 3]   → past wave data
        wave_future: [B, 240, 3]  → future wave data (up to 10 days)
        Returns:
            wave_contexts: [B, 10, H] — one context per future image
        """

        #### Get batch size and number of channels ####
        B, _, _ = wave_input.shape
        contexts = []

        #### Process each context ####
        for t in range(self.num_contexts):
            # Combine full past + up to t+1 days of future ( This is wrong, 24 should be different depending on time)
            future_slice = wave_future[:, : (t + 1) * 24, :]  # [B, 24*(t+1), 3]
            combined = torch.cat([wave_input, future_slice], dim=1)  # [B, 240 + t*24, 3]

            _, (h_n, _) = self.encoder(combined)  # [1, B, H]
            ctx = self.projection(h_n[-1])        # [B, H]
            contexts.append(ctx)

        return torch.stack(contexts, dim=1)  # [B, 10, H]

#### Predictor class ####
class Predictor(nn.Module):
    def __init__(self, input_dim, hidden_dims, kernel_sizes, num_layers, lstm_hidden_size, dropout):
        ##### Initialize parameters ####
        super(Predictor, self).__init__()

        #### Initialize ConvLSTM ####
        self.convlstm = ConvLSTM(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            kernel_sizes=kernel_sizes,
            num_layers=num_layers,
            dropout=dropout,

        )
        #### Initialize fusion conv layer to combine ConvLSTM output and wave context ####
        self.fusion_conv = nn.Conv2d(
            in_channels=hidden_dims[-1] + lstm_hidden_size,
            out_channels=input_dim,
            kernel_size=1
        )

        #### Initialize wave context encoder ####
        self.wave_encoder = ProgressiveWaveContextEncoder(input_size=3, hidden_size=lstm_hidden_size)

        #### Activation function ####
        self.activation = nn.Sigmoid()

        #### Initialize Conv2D layer to generate final output ####
        self.conv_output = nn.Conv2d(hidden_dims[-1], input_dim, kernel_size=1)

    #### Forward pass through the model ####
    def forward(self, slow_input, fast_input, fast_fut_input, future_seq=1):

        #### Get dimensions of input tensor of binary masks ####
        batch_size, seq_len, _, h, w = slow_input.shape
        
        #### Process binary mask 
        _, lstm_states = self.convlstm(slow_input)
        hidden_state = lstm_states  # (B, T, C, H, W)
        
        #### Process wave stream + future wave data ####
        wave_context = self.wave_encoder(fast_input, fast_fut_input)  # (B, 10, H)

        #### Get the last frame of the slow input ####
        current_input = slow_input[:, -1]  # Keep time_steps dimension intact (B, 1, C, H, W)
        
        #### Iterative prediction ####
        predictions = []

        #### Get the last frame of the slow input ####
        last_frame = slow_input[:, -1:]  # Keep time_steps dimension intact (B, 1, C, H, W)

        #### Initialize hidden state for ConvLSTM ####
        lstm_h, lstm_c = None, None
        
        #### Iterate through future sequence length ####
        for t in range(future_seq):
            #### Get corresponding wave context 
            wave_idx = t % 10
            current_wave = wave_context[:, wave_idx].unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1, 1)
            current_wave = current_wave.unsqueeze(1).expand(-1, 1, -1, h, w) 
            
            #### Process current input through ConvLSTM ####
            current_input = current_input.unsqueeze(1)  
            lstm_output, hidden_state = self.convlstm(current_input, hidden_state)
            current_input = self.conv_output(lstm_output[:, 0])

            #### Concatenate ConvLSTM output and wave context ####
            fused = torch.cat([lstm_output, current_wave], dim=2)
            fused = fused.squeeze(1)

            #### Apply fusion conv layer to combine outputs ####
            pred = self.fusion_conv(fused).unsqueeze(1)  # (B, 1, C, H, W)

            #### activation to get final prediction with sigmoid ####
            pred = self.activation(pred)

            #### Append prediction to the list ####
            predictions.append(pred)

            #### concat the prediction to the last frame to keep the time_steps dimension ####
            last_frame = torch.cat([last_frame, pred], dim=1)  # Append prediction while keeping time_steps dimension
        
        return torch.cat(predictions, dim=1)

#### Train definition ####
def train(model, train_loader, criterion, optimizer, device, epoch):

    #### Set model to training mode ####
    model.train()
    train_loss = 0
    total_batches = len(train_loader)
    
    #### Track total time for the epoch
    start_epoch = torch.cuda.Event(enable_timing=True)
    end_epoch = torch.cuda.Event(enable_timing=True)
    start_epoch.record()

    #### Loop through training data per batch ####
    for batch_idx, (slow_input, target_seq, fast_input, fast_fut_input) in enumerate(train_loader):

        #### Move data to device ####
        slow_input = slow_input.to(device)
        fast_input = fast_input.to(device)
        target_seq = target_seq.to(device)
        fast_fut_input = fast_fut_input.to(device)

        #### Log the time taken for each iteration ####
        start_iter = torch.cuda.Event(enable_timing=True)
        end_iter = torch.cuda.Event(enable_timing=True)
        start_iter.record()

        #### zero gradients to avoid accumulation ####
        optimizer.zero_grad()

        #### Output from the model ####
        output = model(slow_input, fast_input, fast_fut_input, future_seq=1)

        #### Compute loss ####
        loss = criterion(output, target_seq[:, 0])

        #### Compute gradients backward to learn ####
        loss.backward()

        #### Clip gradients to avoid exploding gradients ####
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        #### Update weights ####
        optimizer.step()

        #### Log the time taken for each iteration ####
        end_iter.record()
        torch.cuda.synchronize()
        iteration_time = start_iter.elapsed_time(end_iter)/1000

        #### Store loss for the epoch ####
        train_loss += loss.item()
        
        # Log time to WandB
        wandb.log({"Batch Loss": loss.item(), "Iteration Time (s)": iteration_time})

        #### Print progress every 100 batches ####
        if batch_idx % 100 == 0:  # Print every 100 batches
            print(f"Epoch {epoch}, Batch {batch_idx}, Time per Iteration: {iteration_time:.4f}s")

    #### time taken for the epoch ####
    end_epoch.record()
    torch.cuda.synchronize()
    epoch_time = start_epoch.elapsed_time(end_epoch)/1000
    avg_iteration_time = epoch_time / total_batches
    
    #### Average loss for the epoch ####
    avg_train_loss = train_loss / len(train_loader)
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, avg_train_loss),  )
    
    # Log epoch train loss to WandB
    wandb.log({"Epoch Train Loss": avg_train_loss, "Epoch": epoch, "Avg Iteration Time (s)": avg_iteration_time}, commit=False)
    
    return avg_train_loss


def validate(model, val_loader, criterion2, device, epoch, eval_table_data):
    model.eval()

    #### Define variables to store #####
    per_sample_losses = []
    all_inputs = []
    all_targets = []
    all_outputs = []
    all_wave_contexts = []
    all_wave_contexts_future = []
    all_buffer_masks = []
    dice_scores = []
    dice_buffer_s = []
    iou_scores = []
    precision_scores = []
    recall_scores = []
    idx = []

    
    #### initialize metrics ####
    val_loss = 0
    total_samples = 0
    dice_s = 0
    dice_b = 0
    iou_s = 0
    precision_s = 0
    recall_s = 0


    #### torch no_grad() to avoid gradient calculation ####
    with torch.no_grad():

        #### loop through validation data per batch 
        for batch_idx, (slow_input, target_seq, fast_input, fast_fut_input) in enumerate(val_loader):  

            #### move data to device ####
            slow_input = slow_input.to(device)
            fast_input = fast_input.to(device)
            target_seq = target_seq.to(device)
            fast_fut_input = fast_fut_input.to(device)

            #### forward pass ####
            output = model(slow_input, fast_input, fast_fut_input, future_seq=1)

            #### Compute loss ####
            loss = criterion2(output, target_seq[:, 0])
            val_loss += loss.sum().item()

            #### Compute metrics ####
            B = output.shape[0]
            total_samples += B
            dice = dice_score(output, target_seq[:, 0], delta=0.5)
            dice_s += dice.sum().item()

            #### Metrics with buffer zone ####
            buffer_masks = buffer_mask(target_seq[:, 0])
            dice_buffer = dice_score_shoreline_buffer(output, target_seq[:, 0], buffer_masks, delta=0.5)
            dice_b += dice_buffer.sum().item()
            ious, recalls, precisions = buffer_zone_metrics(output, target_seq[:, 0], buffer_masks)
            iou_s += ious.sum().item()
            precision_s += precisions.sum().item()
            recall_s += recalls.sum().item()
            
            #### Cut off future wave data ####
            future_seq = 1
            cutoff = int(fast_fut_input.size(1) * future_seq * 0.1)
            fast_fut = fast_fut_input[:, :cutoff]

            #### Store variables per sample for analysis ####
            for i in range(B):  # B = batch size
                sample_idx = batch_idx * val_loader.batch_size + i
                idx.append(sample_idx)
                
                #### Store per sample losses and metrics ####
                per_sample_losses.append(loss[i].item())
                dice_scores.append(dice[i].item())
                dice_buffer_s.append(dice_buffer[i].item())
                iou_scores.append(ious[i].item())
                precision_scores.append(precisions[i].item())
                recall_scores.append(recalls[i].item())

                
                ### Store variables per sample for analysis ###
                all_inputs.append(slow_input[i].cpu())            # shape: [T, 1, H, W]
                all_targets.append(target_seq[i, 0].cpu())        # shape: [1, H, W]
                all_outputs.append(output[i].cpu())               # shape: [1, H, W]
                all_wave_contexts.append(fast_input[i].cpu())     # shape: [W_input_len, 3]
                all_wave_contexts_future.append(fast_fut[i].cpu())  # shape: [W_future_len, 3]
                all_buffer_masks.append(buffer_masks[i].cpu())   # shape: [H, W]

    #### AVG Loss and Metrics per Epoch ####
    avg_val_loss = val_loss / total_samples
    avg_dice = dice_s / total_samples
    avg_dice_buffer = dice_b / total_samples
    avg_iou = iou_s / total_samples
    avg_precision = precision_s / total_samples
    avg_recall = recall_s / total_samples

    print("====> Validation Total samples:", total_samples)
    print('====> Validation set loss: {:.4f}'.format(avg_val_loss))
    print('====> Validation set Dice: {:.4f}'.format(avg_dice))
    print('====> Validation set Dice Buffer: {:.4f}'.format(avg_dice_buffer))
    print('====> Validation set IOU: {:.4f}'.format(avg_iou))
    print('====> Validation set Precision: {:.4f}'.format(avg_precision))
    print('====> Validation set Recall: {:.4f}'.format(avg_recall))

    wandb.log({
    "Epoch Validation Loss": avg_val_loss,
    "Epoch Validation Dice": avg_dice,
    "Epoch Validation Dice Buffer": avg_dice_buffer,
    "Epoch Validation IOU": avg_iou,
    "Epoch Validation Precision": avg_precision,
    "Epoch Validation Recall": avg_recall,
    "Epoch": epoch
    }, commit=False)

    return avg_val_loss, per_sample_losses, dice_buffer_s, idx, all_outputs, all_targets, all_inputs, all_wave_contexts, all_wave_contexts_future, all_buffer_masks


def test_metrics(model, test_loader, device, test_data, future_seq=1):
    model.eval()

    #### Define variables to store #####
    all_inputs = []
    all_targets = []
    all_outputs = []
    all_wave_contexts = []
    all_wave_contexts_future = []
    all_buffer_masks = []
    dice_scores = []
    dice_buffer_s = []
    iou_scores = []
    precision_scores = []
    recall_scores = []
    idx = []

    
    #### initialize metrics ####
    total_samples = 0
    dice_s = 0
    dice_b = 0
    iou_s = 0
    precision_s = 0
    recall_s = 0
    

    #### torch no_grad() to avoid gradient calculation ####
    with torch.no_grad():

        #### loop through test data per batch  
        for batch_idx, (slow_input, target_seq, fast_input, fast_fut_input) in enumerate(test_loader):  

            #### move data to device ####
            slow_input = slow_input.to(device)
            fast_input = fast_input.to(device)
            target_seq = target_seq.to(device)
            fast_fut_input = fast_fut_input.to(device)

            #### forward pass ####
            output = model(slow_input, fast_input, fast_fut_input, future_seq)
            
            #### Compute metrics ####
            B = output.shape[0]
            total_samples += B
            dice = dice_score(output, target_seq[:, 0], delta=0.5)
            dice_s += dice.sum().item()

            #### Metrics with buffer zone ####
            buffer_masks = buffer_mask(target_seq[:, 0])
            dice_buffer = dice_score_shoreline_buffer(output, target_seq[:, 0], buffer_masks, delta=0.5)
            dice_b += dice_buffer.sum().item()
            ious, recalls, precisions = buffer_zone_metrics(output, target_seq[:, 0], buffer_masks)
            iou_s += ious.sum().item()
            precision_s += precisions.sum().item()
            recall_s += recalls.sum().item()

            #### Cut off future wave data ####
            #future_seq = 1
            cutoff = int(fast_fut_input.size(1) * future_seq * 0.1)
            fast_fut = fast_fut_input[:, :cutoff]

            #### Store variables per sample for analysis ####
            for i in range(B):  # B = batch size
                sample_idx = batch_idx * test_loader.batch_size + i
                idx.append(sample_idx)

                #### Store per sample metrics ####
                dice_scores.append(dice[i].item())
                dice_buffer_s.append(dice_buffer[i].item())
                iou_scores.append(ious[i].item())
                precision_scores.append(precisions[i].item())
                recall_scores.append(recalls[i].item())
                
                #### Store variables per sample for analysis ###
                all_inputs.append(slow_input[i].cpu())            # shape: [T, 1, H, W]
                all_targets.append(target_seq[i, 0].cpu())        # shape: [1, H, W]
                all_outputs.append(output[i].cpu())               # shape: [1, H, W]
                all_wave_contexts.append(fast_input[i].cpu())     # shape: [W_input_len, 3]
                all_wave_contexts_future.append(fast_fut[i].cpu())  # shape: [W_future_len, 3]
                all_buffer_masks.append(buffer_masks[i].cpu())   # shape: [H, W]

    
    #### Analysis per sample ####
    diagnose(all_outputs, all_targets, dice_buffer_s, all_inputs, all_wave_contexts, all_wave_contexts_future, title_prefix="Testing")
    buffer_zone_vis(all_targets, all_outputs, all_buffer_masks, title="Testing")

    #### AVG Metrics ####
    avg_dice = dice_s / total_samples
    avg_dice_buffer = dice_b / total_samples
    avg_iou = iou_s / total_samples
    avg_precision = precision_s / total_samples
    avg_recall = recall_s / total_samples
    print("====> Testing Total samples:", total_samples)
    print('====> Testing set Dice: {:.4f}'.format(avg_dice))
    print('====> Testing set Dice Buffer: {:.4f}'.format(avg_dice_buffer))
    print('====> Testing set IOU: {:.4f}'.format(avg_iou))
    print('====> Testing set Precision: {:.4f}'.format(avg_precision))
    print('====> Testing set Recall: {:.4f}'.format(avg_recall))

    #### Log metrics to WandB per sample ####
    for i in range(len(idx)):
        score = dice_scores[i]
        score_buffer = dice_buffer_s[i]
        wandb.log({"Dice_Score Testing": score,
                    "Dice_Score Buffer Testing": score_buffer,
                    "IOU_Score Testing": iou_scores[i],
                    "Precision_Score Testing": precision_scores[i],
                    "Recall_Score Testing": recall_scores[i],
                    "Sample index Testing": idx[i]+1}
                 )

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params:,}")  # Format with commas
    
    # 🔹 Log to WandB
    wandb.log({"Total Parameters": total_params}, commit=False)
    
    return total_params

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0001, verbose=True):
        """
        Args:
            patience (int): How many epochs to wait after last improvement.
            min_delta (float): Minimum change to qualify as improvement.
            verbose (bool): If True, prints a message when stopping early.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def step(self, val_loss):
        score = -val_loss  # minimize loss → maximize (-loss)

        if self.best_score is None:
            self.best_score = score
            return False  # continue training

        if score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"⏳ EarlyStopping patience {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                if self.verbose:
                    print("🛑 Early stopping triggered.")
                self.early_stop = True
                return True
        else:
            self.best_score = score
            self.counter = 0  # reset counter
        return False

    
def main():

    #### Config ####
    torch.cuda.empty_cache() # Clear GPU memory
    torch.manual_seed(42) # Set random seed for reproducibility
    random.seed(42) # Set random seed for reproducibility
    np.random.seed(42) # Set random seed for reproducibility

    #{wandb.util.generate_id()}
    #### Initialize WandB ####
    wandb.login(key="d42992a374fbc96ee65d1955f037e71d58e30f45")
    wandb.init(project="THESIS",
        name=f"hybrid_v11_30Epochs_case1_try",
        config={
        "input_dim": 1,
        "hidden_dims": [ 64, 64],
        "kernel_sizes": [(5, 5), (5, 5)],
        "num_layers": 2,
        "lstm_hidden_size": 64,
        "batch_size": 8,
        "epochs": 30,
        "learning_rate": 0.001,
        "optimizer": "Adam",
        "loss_function": "MSELoss",
        "dropout": 0.2,
        "future_seq": 1,
    })

    #### Load dataset and scenarios ####
    time = "bi-monthly"
    case = 1
    data_path = "/p/11207608-coclico/MSc_students/Daniel/Scripts/outputs/"
    train_loader, val_loader, test_loader = data_loaders(data_path, batch_size=wandb.config.batch_size, time=time, case=case)
    
    #### Call WandB ####
    input_dim = wandb.config.input_dim
    hidden_dims = wandb.config.hidden_dims
    kernel_sizes = wandb.config.kernel_sizes
    num_layers = wandb.config.num_layers
    batch_size = wandb.config.batch_size
    lstm_hidden_size = wandb.config.lstm_hidden_size
    epochs = wandb.config.epochs
    learning_rate = wandb.config.learning_rate
    dropout = wandb.config.dropout


    #### Set device to cuda if available ####
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #### Create model ####
    model = Predictor(input_dim=input_dim, hidden_dims=hidden_dims, kernel_sizes=kernel_sizes, num_layers=num_layers, lstm_hidden_size=lstm_hidden_size, dropout=dropout).to(device)
    total_params = count_parameters(model)
    wandb.watch(model)

    #### Define Loss functions
    criterion = UnifiedFocalLoss(weight=0.5, delta=0.7, gamma=0.75) # for training
    criterion2 = UnifiedFocalLossSample(weight=0.5, delta=0.7, gamma=0.75) # for validation

    #### Define Optimizer and Scheduler ####
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    #### Print config values ####
    print('Using device:', device)
    print(f"hidden_dims: {hidden_dims}, length: {len(hidden_dims)}")
    print(f"kernel_sizes: {kernel_sizes}, length: {len(kernel_sizes)}")
    print(f"num_layers: {num_layers}")
    print(f"Model is running with {model.convlstm.num_layers} layers.")

    #### Init empty lists to store losses and activate early stopping ####
    train_losses = []
    val_losses = []
    eval_table_data = []
    early_stopper = EarlyStopping(patience=10, min_delta=0.0001, verbose=True)
    best_val_loss = float('inf')

    #### Iterate over epochs to train the model ####
    for epoch in range(1, epochs + 1):

        # Call train and validate functions in each epoch
        train_loss = train(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, per_sample_losses, dice_buffer_s, idx, all_outputs, all_targets, all_inputs, all_wave_context, all_wave_fut, all_buffer_masks = validate(model, val_loader, criterion2, device, epoch, eval_table_data)
        
        # Scheduler, Help the model to learn by reducing the learning rate (!!!! Include a log to WandB!!!!)
        scheduler.step(val_loss)

        # Append losses to lists
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Save only the best model checking validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = "best_model.pth"
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

    
    #model.load_state_dict(torch.load(model_path))
    test_data = []
    test_metrics(model, test_loader, device, test_data)
    diagnose(all_outputs, all_targets, dice_buffer_s, all_inputs, all_wave_context, all_wave_fut, title_prefix="Validation")
    buffer_zone_vis(all_targets, all_outputs, all_buffer_masks, title="Validation")

    wandb.run.log_code(".")
    
    #table_columns = ["Metric"] + [f"t={i+1}" for i in range(1)] + ["Epoch"]
    #table_columns2 = ["Metric"] + [f"t={i+1}" for i in range(1)]

    

    # Create DataFrame and log it
    #df = pd.DataFrame(eval_table_data, columns=table_columns)
    #df2= pd.DataFrame(test_data, columns=table_columns2)
    #wandb_table = wandb.Table(dataframe=df)
    #wandb_table2 = wandb.Table(dataframe=df2)
    #wandb.log({"Per-timestep Evaluation Table": wandb_table})
    #wandb.log({"Test Evaluation Table": wandb_table2})

    # Visualize predictions
    #visualize_prediction(model, test_loader, device)
    
    # Plot loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig('loss_curves.png')
    plt.close()
    
    # Log loss curves to WandB
    wandb.log({"Loss Curves": wandb.Image('loss_curves.png')}, commit=False)
    
    print("Training complete!")
    return model

if __name__ == '__main__':
    model = main()