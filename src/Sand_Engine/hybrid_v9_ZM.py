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

from data_loader_v2 import get_dataloaders



# Initialize WandB
wandb.login(key="d42992a374fbc96ee65d1955f037e71d58e30f45")
wandb.init(project="THESIS",
    name=f"hybrid_v5_zm-{wandb.util.generate_id()}",
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
    #"norm": "batch",
})

#Load dataset

data_path = "/p/11207608-coclico/MSc_students/Daniel/Scripts/outputs/v3_256/daily/"
train_loader, val_loader, test_loader = get_dataloaders(data_path, batch_size=wandb.config.batch_size)

# Verify shapes
for inputs, targets, wave, wave_fut in train_loader:
    print("Train Inputs shape:", inputs.shape)  # Should be (B, T-10, 1, H, W)
    print("Train Targets shape:", targets.shape)  # Should be (B, 10, 1, H, W)
    print("Wave shape:", wave.shape)  # Should be (B, 480, 1)
    print("Wave shape:", wave_fut.shape)  # Should be (B, 240, 3)
    break

for inputs, targets, wave, wave_fut in val_loader:
    print("Validation Inputs shape:", inputs.shape)  # Should be (B, T-10, 1, H, W)
    print("Validation Targets shape:", targets.shape)  # Should be (B, 10, 1, H, W)
    print("Wave shape:", wave.shape)  # Should be (B, 480, 1)
    print("Wave shape:", wave_fut.shape)  # Should be (B, 240, 3)
    break

for inputs, targets, wave, wave_fut in test_loader:
    print("Test Inputs shape:", inputs.shape)  # Should be (B, T-10, 1, H, W)
    print("Test Targets shape:", targets.shape)  # Should be (B, 10, 1, H, W)
    print("Wave shape:", wave.shape)  # Should be (B, 480, 1)
    print("Wave shape:", wave_fut.shape)  # Should be (B, 240, 3)
    break



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
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        self.dropout_rate = dropout
        #self.norm_type = norm.lower()
        
        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,  # For the four gates
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias
        )

        # if self.norm_type == 'batch':
        #     self.norm = nn.BatchNorm2d(self.hidden_dim)
        # elif self.norm_type == 'layer':
        #     self.norm = nn.GroupNorm(1, self.hidden_dim)
        # else:
        #     self.norm = None

        self.dropout = nn.Dropout2d(p=self.dropout_rate) if self.dropout_rate > 0 else None
        
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
        h_cur, c_cur = cur_state
        
        # Concatenate along channel axis
        combined = torch.cat([input_tensor, h_cur], dim=1)
        
        #print("Combined shape:", combined.shape)  # Debugging line
        # Convolutional operation
        combined_conv = self.conv(combined)
        
        # Split the combined output into the 4 gates
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        
        # Apply gate activations
        i = torch.sigmoid(cc_i)  # input gate
        f = torch.sigmoid(cc_f)  # forget gate
        o = torch.sigmoid(cc_o)  # output gate
        g = torch.tanh(cc_g)     # cell gate
        
        # Update cell state and hidden state
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        # if self.norm is not None:
        #     h_next = self.norm(h_next)
        
        h_next = h_next if self.dropout is None else self.dropout(h_next)
        
        return h_next, c_next

class ConvLSTM(nn.Module):
    """
    ConvLSTM module for sequence prediction with multiple layers and varying hidden dimensions.
    """
    def __init__(self, input_dim, hidden_dims, kernel_sizes, num_layers, dropout , batch_first=True, bias=True): #,norm
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
        super(ConvLSTM, self).__init__()
        assert len(hidden_dims) == num_layers, "Length of hidden_dims must match num_layers"
        assert len(kernel_sizes) == num_layers, "Length of kernel_sizes must match num_layers"

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.kernel_sizes = kernel_sizes
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.dropout = dropout
        #self.norm = norm

        # Create a list of ConvLSTM cells
        cell_list = []
        for i in range(self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dims[i - 1]
            cell_list.append(ConvLSTMCell(
                input_dim=cur_input_dim,
                hidden_dim=self.hidden_dims[i],
                kernel_size=self.kernel_sizes[i],
                dropout=self.dropout,
                #norm=self.norm,
                bias=self.bias
            ))
        self.cell_list = nn.ModuleList(cell_list)

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
        # Make sure we're working with batch first format
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        # Get dimensions
        batch_size, seq_len, _, height, width = input_tensor.size()

        # Initialize hidden states if none provided
        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size, (height, width))

        layer_output_list = []
        last_state_list = []

        # Process each sequence element
        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                # Get input for this timestep
                if layer_idx == 0:
                    # For the first layer, input comes from the original input sequence
                    x = input_tensor[:, t, :, :, :]
                else:
                    # For subsequent layers, input comes from the output of the previous layer
                    x = layer_output_list[layer_idx - 1][:, t, :, :, :]

                # Process through the ConvLSTM cell
                h, c = self.cell_list[layer_idx](x, (h, c))

                # Store output
                output_inner.append(h)

            # Stack outputs along time dimension
            layer_output = torch.stack(output_inner, dim=1)
            layer_output_list.append(layer_output)
            last_state_list.append((h, c))

        # Return outputs as needed
        return layer_output_list[-1], last_state_list
    
class ProgressiveWaveContextEncoder(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, num_contexts=10):
        super(ProgressiveWaveContextEncoder, self).__init__()
        self.encoder = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.projection = nn.Linear(hidden_size, hidden_size)
        self.num_contexts = num_contexts

    def forward(self, wave_input, wave_future):
        """
        wave_input: [B, 240, 3]   → past wave data
        wave_future: [B, 240, 3]  → future wave data (up to 10 days)
        Returns:
            wave_contexts: [B, 10, H] — one context per future image
        """
        B, _, _ = wave_input.shape
        contexts = []

        for t in range(self.num_contexts):
            # Combine full past + up to t+1 days of future
            future_slice = wave_future[:, : (t + 1) * 24, :]  # [B, 24*(t+1), 3]
            combined = torch.cat([wave_input, future_slice], dim=1)  # [B, 240 + t*24, 3]

            _, (h_n, _) = self.encoder(combined)  # [1, B, H]
            ctx = self.projection(h_n[-1])        # [B, H]
            contexts.append(ctx)

        return torch.stack(contexts, dim=1)  # [B, 10, H]


class Predictor(nn.Module):
    def __init__(self, input_dim, hidden_dims, kernel_sizes, num_layers, lstm_hidden_size, dropout): #norm
        super(Predictor, self).__init__()
        self.convlstm = ConvLSTM(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            kernel_sizes=kernel_sizes,
            num_layers=num_layers,
            dropout=dropout,
            #norm=norm
        )

        self.fusion_conv = nn.Conv2d(
            in_channels=hidden_dims[-1] + lstm_hidden_size,
            out_channels=input_dim,
            kernel_size=1
        )
        self.wave_encoder = ProgressiveWaveContextEncoder(input_size=3, hidden_size=lstm_hidden_size)

        self.activation = nn.Sigmoid()

        self.conv_output = nn.Conv2d(hidden_dims[-1], input_dim, kernel_size=1)

    def forward(self, slow_input, fast_input, fast_fut_input, future_seq=10):
        batch_size, seq_len, _, h, w = slow_input.shape
        
        # Process satellite stream
        _, lstm_states = self.convlstm(slow_input)
        hidden_state = lstm_states  # (B, T, C, H, W)
        
        # Process wave stream (10 segments of 360 hours)

        wave_context = self.wave_encoder(fast_input, fast_fut_input)  # (B, 10, H)

        current_input = slow_input[:, -1]  # Keep time_steps dimension intact (B, 1, C, H, W)
        
        # Iterative prediction
        predictions = []
        last_frame = slow_input[:, -1:]  # Keep time_steps dimension intact (B, 1, C, H, W)
        lstm_h, lstm_c = None, None
        
        for t in range(future_seq):
            # Get corresponding wave context (cycle if needed)
            wave_idx = t % 10
            current_wave = wave_context[:, wave_idx].unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1, 1)
            
            current_input = current_input.unsqueeze(1)  # Add time dimension
            # Forward pass through ConvLSTM
            lstm_output, hidden_state = self.convlstm(current_input, hidden_state)
            # Generate prediction


            # Fuse with wave context
            current_wave = current_wave.unsqueeze(1).expand(-1, 1, -1, h, w) 
            #print("Current wave shape after expand:", current_wave.shape)  # Debugging line
            #print("LSTM output shape:", lstm_output.shape)  # Debugging line
            #print("Current wave shape:", current_wave.shape)  # Debugging line
            current_input = self.conv_output(lstm_output[:, 0])

            
            fused = torch.cat([lstm_output, current_wave], dim=2)
            #print("Fused shape:", fused.shape)  # Debugging line
            fused = fused.squeeze(1)
            pred = self.fusion_conv(fused).unsqueeze(1)  # (B, 1, C, H, W)
            pred = self.activation(pred)
            predictions.append(pred)
            last_frame = torch.cat([last_frame, pred], dim=1)  # Append prediction while keeping time_steps dimension
        
        return torch.cat(predictions, dim=1)
        
def train(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    train_loss = 0
    total_batches = len(train_loader)
    
    # Track total time for the epoch
    start_epoch = torch.cuda.Event(enable_timing=True)
    end_epoch = torch.cuda.Event(enable_timing=True)
    start_epoch.record()
    for batch_idx, (slow_input, target_seq, fast_input, fast_fut_input) in enumerate(train_loader):
        slow_input = slow_input.to(device)
        fast_input = fast_input.to(device)
        target_seq = target_seq.to(device)
        fast_fut_input = fast_fut_input.to(device)

        start_iter = torch.cuda.Event(enable_timing=True)
        end_iter = torch.cuda.Event(enable_timing=True)
        start_iter.record()

        optimizer.zero_grad()
        output = model(slow_input, fast_input, fast_fut_input, future_seq=10)

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
        for (slow_input, target_seq, fast_input, fast_fut_input) in val_loader:  
            slow_input = slow_input.to(device)
            fast_input = fast_input.to(device)
            target_seq = target_seq.to(device)
            fast_fut_input = fast_fut_input.to(device)

            output = model(slow_input, fast_input, fast_fut_input, future_seq=10)  
            loss = criterion(output, target_seq)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)
    wandb.log({"Epoch Validation Loss": avg_val_loss, "Epoch": epoch})
    return avg_val_loss

def visualize_prediction(model, test_loader, device, sample_idx=0):
    model.eval()
    # Get a sample from the test set
    for i, (slow_input, target_seq, fast_input, fast_fut_input) in enumerate(test_loader):  
        if i == sample_idx:
            break
    slow_input = slow_input.to(device)
    fast_input = fast_input.to(device)  
    target_seq = target_seq.to(device)
    fast_fut_input = fast_fut_input.to(device)
    with torch.no_grad():
        output = model(slow_input, fast_input, fast_fut_input, future_seq=10) 
    
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
def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params:,}")  # Format with commas
    
    # 🔹 Log to WandB
    wandb.log({"Total Parameters": total_params})
    
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
    torch.cuda.empty_cache()
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    
    # Hyperparameters (already logged in WandB init)
    input_dim = wandb.config.input_dim
    hidden_dims = wandb.config.hidden_dims
    kernel_sizes = wandb.config.kernel_sizes
    num_layers = wandb.config.num_layers
    batch_size = wandb.config.batch_size
    lstm_hidden_size = wandb.config.lstm_hidden_size
    epochs = wandb.config.epochs
    learning_rate = wandb.config.learning_rate
    dropout = wandb.config.dropout
    #norm = wandb.config.norm

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    print(f"hidden_dims: {hidden_dims}, length: {len(hidden_dims)}")
    print(f"kernel_sizes: {kernel_sizes}, length: {len(kernel_sizes)}")
    print(f"num_layers: {num_layers}")
    # Create model
    model = Predictor(input_dim=input_dim, hidden_dims=hidden_dims, kernel_sizes=kernel_sizes, num_layers=num_layers, lstm_hidden_size=lstm_hidden_size, dropout=dropout).to(device) #, norm=norm
    print(f"Model is running with {model.convlstm.num_layers} layers.")

    # Count trainable parameters
    total_params = count_parameters(model)
    
    # Log model architecture to WandB
    wandb.watch(model)
    
    # Loss function
    criterion = nn.MSELoss()
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train_losses = []
    val_losses = []

    early_stopper = EarlyStopping(patience=5, min_delta=0.0001, verbose=True)
    best_val_loss = float('inf')
    
    # Train model
    for epoch in range(1, epochs + 1):
        train_loss = train(model, train_loader, criterion, optimizer, device, epoch)
        val_loss = validate(model, test_loader, criterion, device, epoch)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Save only the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = "best_model.pth"
            torch.save(model.state_dict(), model_path)

            artifact = wandb.Artifact(name="Best-ConvLSTM", type="model")
            artifact.add_file(model_path)
            wandb.log_artifact(artifact)
            print("✅ Saved new best model.")

        if early_stopper.step(val_loss):
            break  # stop training if early stopping triggered

    wandb.run.log_code(".")
    
    # Visualize predictions
    visualize_prediction(model, test_loader, device)
    
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
    wandb.log({"Loss Curves": wandb.Image('loss_curves.png')})
    
    print("Training complete!")
    return model

if __name__ == '__main__':
    model = main()