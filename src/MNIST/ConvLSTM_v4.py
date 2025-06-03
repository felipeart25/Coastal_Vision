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

# Initialize WandB
wandb.login(key="d42992a374fbc96ee65d1955f037e71d58e30f45")
wandb.init(project="convlstm-mnist",
    name=f"run-{wandb.util.generate_id()}",
    config={
    "input_dim": 1,
    "hidden_dims": [128, 64, 64],
    "kernel_sizes": [(5, 5),(5, 5),(5, 5)],
    "num_layers": 3,
    "batch_size": 8,
    "epochs": 20,
    "learning_rate": 0.001,
    "optimizer": "Adam",
    "loss_function": "MSELoss"
})

# Load dataset (example: MNIST-like sequences)
os.system('wget "https://github.com/felipeart25/Coastal_Vision/raw/main/data/Data/mnist_test_seq.npy" -O mnist_test_seq.npy')
data = np.load("mnist_test_seq.npy")  # Shape: (num_sequences, time_steps, channels, height, width)
data = torch.tensor(data, dtype=torch.float32) / 255.0  # Normalize to [0, 1]
data = data.unsqueeze(2)
data = data.permute(1, 0, 2, 3, 4)  # Swap axes 

# Print shape
print("Original data shape:", data.shape)  # Should be (num_sequences, time_steps, 1, height, width)

# Split into train (70%), validation (15%), and test (15%)
train_size = int(0.8 * len(data))  # 70% for training
val_size = int(0.1 * len(data))   # 15% for validation
test_size = len(data) - train_size - val_size  # Remaining 15% for testing

train_data = data[:train_size]
val_data = data[train_size:train_size + val_size]
test_data = data[train_size + val_size:]

print("Train data shape:", train_data.shape)
print("Validation data shape:", val_data.shape)
print("Test data shape:", test_data.shape)

# Prepare datasets
# Input: first T-10 frames, Target: next 10 frames
T = 20  # Number of input frames (T-10 for input, 10 for target)
train_dataset = TensorDataset(train_data[:, :T-10], train_data[:, -10:])  # Input: T-10, Target: 10
val_dataset = TensorDataset(val_data[:, :T-10], val_data[:, -10:])
test_dataset = TensorDataset(test_data[:, :T-10], test_data[:, -10:])

# Create DataLoaders
batch_size = wandb.config.batch_size
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Verify shapes
for inputs, targets in train_loader:
    print("Train Inputs shape:", inputs.shape)  # Should be (B, T-10, 1, H, W)
    print("Train Targets shape:", targets.shape)  # Should be (B, 10, 1, H, W)
    break

for inputs, targets in val_loader:
    print("Validation Inputs shape:", inputs.shape)  # Should be (B, T-10, 1, H, W)
    print("Validation Targets shape:", targets.shape)  # Should be (B, 10, 1, H, W)
    break

for inputs, targets in test_loader:
    print("Test Inputs shape:", inputs.shape)  # Should be (B, T-10, 1, H, W)
    print("Test Targets shape:", targets.shape)  # Should be (B, 10, 1, H, W)
    break



class ConvLSTMCell(nn.Module):
    """
    Basic ConvLSTM cell.
    """
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
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
        
        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,  # For the four gates
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias
        )
        
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
        
        return h_next, c_next

class ConvLSTM(nn.Module):
    """
    ConvLSTM module for sequence prediction with multiple layers and varying hidden dimensions.
    """
    def __init__(self, input_dim, hidden_dims, kernel_sizes, num_layers, batch_first=True, bias=True):
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

        # Create a list of ConvLSTM cells
        cell_list = []
        for i in range(self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dims[i - 1]
            cell_list.append(ConvLSTMCell(cur_input_dim, self.hidden_dims[i], self.kernel_sizes[i], self.bias))
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

class Predictor(nn.Module):
    def __init__(self, input_dim, hidden_dims, kernel_sizes, num_layers):
        super(Predictor, self).__init__()
        self.convlstm = ConvLSTM(input_dim=input_dim,
                                 hidden_dims=hidden_dims,
                                 kernel_sizes=kernel_sizes,
                                 num_layers=num_layers)
        self.conv_output = nn.Conv2d(hidden_dims[-1], input_dim, kernel_size=1)

    def forward(self, x, future_seq=10):
        # Process input sequence
        _, lstm_states = self.convlstm(x)
        # Generate future predictions
        current_input = x[:, -1]  # Last input frame
        outputs = []
        hidden_state = lstm_states
        for _ in range(future_seq):
            # Reshape for input to ConvLSTM cell
            current_input = current_input.unsqueeze(1)  # Add time dimension
            # Forward pass through ConvLSTM
            lstm_output, hidden_state = self.convlstm(current_input, hidden_state)
            # Generate prediction
            current_input = self.conv_output(lstm_output[:, 0])
            # Store prediction
            outputs.append(current_input.unsqueeze(1))
        # Concatenate all predictions
        outputs = torch.cat(outputs, dim=1)
        return outputs
    
def train(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    train_loss = 0
    total_batches = len(train_loader)
    
    # Track total time for the epoch
    start_epoch = torch.cuda.Event(enable_timing=True)
    end_epoch = torch.cuda.Event(enable_timing=True)
    start_epoch.record()
    for batch_idx, (input_seq, future_seq) in enumerate(train_loader):
        input_seq, future_seq = input_seq.to(device), future_seq.to(device)
        start_iter = torch.cuda.Event(enable_timing=True)
        end_iter = torch.cuda.Event(enable_timing=True)
        start_iter.record()

        optimizer.zero_grad()
        output = model(input_seq)

        loss = criterion(output, future_seq)
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
        for input_seq, target_seq in val_loader:
            input_seq, target_seq = input_seq.to(device), target_seq.to(device)
            
            output = model(input_seq)
            loss = criterion(output, target_seq)
            
            val_loss += loss.item()
    
    avg_val_loss = val_loss / len(val_loader)
    
    # Log epoch validation loss to WandB
    wandb.log({"Epoch Validation Loss": avg_val_loss, "Epoch": epoch})
    
    return avg_val_loss

def visualize_prediction(model, test_loader, device, sample_idx=0):
    model.eval()
    
    # Get a sample from the test set
    for i, (input_seq, target_seq) in enumerate(test_loader):
        if i == sample_idx:
            break
    
    input_seq = input_seq.to(device)
    target_seq = target_seq.to(device)
    
    with torch.no_grad():
        output = model(input_seq)
    
    # Plot
    fig, axes = plt.subplots(3, 10, figsize=(20, 6))
    
    # Input sequence
    for t in range(10):
        axes[0, t].imshow(input_seq[0, t, 0].cpu().numpy(), cmap='gray')
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
    epochs = wandb.config.epochs
    learning_rate = wandb.config.learning_rate

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    
    # Create model
    model = Predictor(input_dim=input_dim, hidden_dims=hidden_dims, kernel_sizes=kernel_sizes, num_layers=num_layers).to(device)
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
    
    # Train model
    for epoch in range(1, epochs + 1):
        train_loss = train(model, train_loader, criterion, optimizer, device, epoch)
        val_loss = validate(model, test_loader, criterion, device, epoch)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

    # Save model
    model_path = "convlstm_mnist.pth"
    torch.save(model.state_dict(), model_path)
    artifact = wandb.Artifact(name="Conv-LSTM", type="model")
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)  # Log model checkpoint to WandB

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