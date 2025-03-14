import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.checkpoint import checkpoint
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import wandb
import random
import os

wandb.login(key="d42992a374fbc96ee65d1955f037e71d58e30f45")

# Initialize WandB
wandb.init(project="convlstm-mnist", 
           name=f"run-{wandb.util.generate_id()}",
           config={
    "input_dim": 1,
    "hidden_dim": 64,
    "kernel_size": (3, 3),
    "num_layers": 2,
    "batch_size": 8,
    "epochs": 10,
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


import torch
import torch.nn as nn
import torch.nn.functional as F

class SAM(nn.Module):
    """Self-Attention Memory Module"""
    def __init__(self, in_channels):
        super().__init__()
        reduction_factor = 16  # Changed from 8 to 16
        self.query = nn.Conv2d(in_channels, in_channels//reduction_factor, 1)
        self.key = nn.Conv2d(in_channels, in_channels//reduction_factor, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x, prev_memory):
        batch_size, channels, height, width = x.size()
        
        # Skip memory integration when unnecessary
        if prev_memory is None or torch.allclose(prev_memory, torch.zeros_like(prev_memory)):
            # Simplified attention when no memory is available
            q = self.query(x).view(batch_size, -1, height*width).permute(0, 2, 1)
            k = self.key(x).view(batch_size, -1, height*width)
            v = self.value(x).view(batch_size, -1, height*width)
            
            # Scale attention to prevent gradient issues
            attention = torch.bmm(q, k) / (channels ** 0.5)
            attention = F.softmax(attention, dim=-1)
            
            out = torch.bmm(v, attention.permute(0, 2, 1))
            out = out.view(batch_size, -1, height, width)
            
            new_memory = self.gamma * out + x
            return out + x, new_memory
            
        # Memory is available, use original implementation
        q = self.query(x).view(batch_size, -1, height*width).permute(0, 2, 1)
        k = self.key(x).view(batch_size, -1, height*width)
        v = self.value(x).view(batch_size, -1, height*width)
        
        k_m = self.key(prev_memory).view(batch_size, -1, height*width)
        v_m = self.value(prev_memory).view(batch_size, -1, height*width)
        
        # Attention scaling for numerical stability
        k = torch.cat([k, k_m], dim=2)
        v = torch.cat([v, v_m], dim=2)
        
        attention = torch.bmm(q, k) / (channels ** 0.5)  # Scale by sqrt(channels)
        attention = F.softmax(attention, dim=-1)
        
        out = torch.bmm(v, attention.permute(0, 2, 1))
        out = out.view(batch_size, -1, height, width)
        
        new_memory = self.gamma * out + x
        return out + x, new_memory

class SAConvLSTMCell(nn.Module):
    """Self-Attention ConvLSTM Cell"""
    def __init__(self, input_dim, hidden_dim, kernel_size):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        padding = kernel_size[0] // 2, kernel_size[1] // 2
        
        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding
        )
        self.sam = SAM(hidden_dim)
        
    def forward(self, x, prev_state):
        h_prev, c_prev, m_prev = prev_state
        
        # ConvLSTM operations
        combined = torch.cat([x, h_prev], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        
        c_cur = f * c_prev + i * g
        h_cur = o * torch.tanh(c_cur)
        
        # Self-attention memory
        h_attn, m_cur = self.sam(h_cur, m_prev)
        
        # Store h_attn in the state instead of h_cur
        return h_attn, (h_attn, c_cur, m_cur)  # Changed h_cur to h_attn

class SimpleSAConvLSTM(nn.Module):
    """Simplified SA-ConvLSTM without separate encoder and decoder"""
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=4, kernel_size=3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Create stack of SAConvLSTM cells
        self.cells = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                cell_input_dim = input_dim
            else:
                cell_input_dim = hidden_dim
                
            self.cells.append(SAConvLSTMCell(cell_input_dim, hidden_dim, kernel_size))
        
        # Output layer
        self.conv_out = nn.Conv2d(hidden_dim, input_dim, kernel_size=1)
        
    def init_hidden(self, batch_size, height, width, device):
        """Initialize hidden states for all layers"""
        hidden = []
        for _ in range(self.num_layers):
            h = torch.zeros(batch_size, self.hidden_dim, height, width, device=device)
            c = torch.zeros(batch_size, self.hidden_dim, height, width, device=device)
            m = torch.zeros(batch_size, self.hidden_dim, height, width, device=device)
            hidden.append((h, c, m))
        return hidden
    
    def forward(self, x, future_seq=10):
        """
        Optimized forward pass with reduced sequential operations
        """
        # Get tensor dimensions
        batch_size, seq_len, _, height, width = x.size()
        device = x.device
        
        # Initialize hidden states for all layers
        hidden = self.init_hidden(batch_size, height, width, device)
        
        # Process input sequence - try to parallelize when possible
        input_frames = [x[:, t] for t in range(seq_len)]
        
        # Process each timestep
        for t in range(seq_len):
            h = input_frames[t]
            # Process through layers sequentially
            for layer_idx, cell in enumerate(self.cells):
                h, hidden[layer_idx] = cell(h, hidden[layer_idx])
        
        # Generate future predictions more efficiently
        outputs = []
        h = hidden[-1][0]  # Start with the final hidden state
        pred = self.conv_out(h)
        outputs.append(pred.unsqueeze(1))
        
        # Generate remaining predictions
        for _ in range(1, future_seq):
            h = outputs[-1][:, 0]  # Last prediction
            
            # Reuse the for loop to avoid code duplication
            for layer_idx, cell in enumerate(self.cells):
                h, hidden[layer_idx] = cell(h, hidden[layer_idx])
                
            pred = self.conv_out(h)
            outputs.append(pred.unsqueeze(1))
        
        # Concatenate all predictions at once (more efficient)
        return torch.cat(outputs, dim=1)


class Predictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers):
        super(Predictor, self).__init__()
        # Use the new SAConvLSTM implementation
        self.saconvlstm = SimpleSAConvLSTM(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            kernel_size=kernel_size
        )
    
    def forward(self, x, future_seq=10):
        # Directly return predictions from SAConvLSTM
        return self.saconvlstm(x, future_seq=future_seq)
    
def train(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    train_loss = 0
    for batch_idx, (input_seq, future_seq) in enumerate(train_loader):
        input_seq, future_seq = input_seq.to(device), future_seq.to(device)

        optimizer.zero_grad()
        output = model(input_seq)
        #print(future_seq.shape, output.shape )
        loss = criterion(output, future_seq)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        
        # Log batch loss to WandB
        wandb.log({"Batch Loss": loss.item()})
            
    avg_train_loss = train_loss / len(train_loader)
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, avg_train_loss))
    
    # Log epoch train loss to WandB
    wandb.log({"Epoch Train Loss": avg_train_loss, "Epoch": epoch})
    
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
    
def main():
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    
    # Hyperparameters (already logged in WandB init)
    input_dim = wandb.config.input_dim
    hidden_dim = wandb.config.hidden_dim
    kernel_size = wandb.config.kernel_size
    num_layers = wandb.config.num_layers
    batch_size = wandb.config.batch_size
    epochs = wandb.config.epochs
    learning_rate = wandb.config.learning_rate

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    
    # Create model
    model = Predictor(input_dim=input_dim, hidden_dim=hidden_dim, kernel_size=kernel_size, num_layers=num_layers).to(device)
    
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
    model_path = "SAconvlstm_mnist.pth"
    torch.save(model.state_dict(), model_path)
    artifact = wandb.Artifact(name="Conv-LSTM", type="model")
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)
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


