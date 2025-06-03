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

wandb.login(key="d42992a374fbc96ee65d1955f037e71d58e30f45")

# Initialize WandB
wandb.init(project="convlstm-mnist", 
           name=f"run-{wandb.util.generate_id()}",
           config={
    "input_dim": 1,
    "hidden_dim": 64,
    "kernel_size": (3, 3),
    "num_layers": 2,
    "batch_size": 4,
    "epochs": 5,
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
input_frames = 10
future_frames = 10
train_dataset = TensorDataset(train_data[:, :input_frames], train_data[:, input_frames:input_frames+future_frames])
val_dataset = TensorDataset(val_data[:, :input_frames], val_data[:, input_frames:input_frames+future_frames])
test_dataset = TensorDataset(test_data[:, :input_frames], test_data[:, input_frames:input_frames+future_frames])

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
import numpy as np


# New SAAttnMem module from second architecture
class SAAttnMem(nn.Module):
    def __init__(self, input_dim, d_model, kernel_size):
        super().__init__()
        pad = kernel_size[0] // 2, kernel_size[1] // 2
        self.d_model = d_model
        self.input_dim = input_dim
        
        self.conv_h = nn.Conv2d(input_dim, d_model*3, kernel_size=1)
        self.conv_m = nn.Conv2d(input_dim, d_model*2, kernel_size=1)
        self.conv_z = nn.Conv2d(d_model*2, d_model, kernel_size=1)
        self.conv_output = nn.Conv2d(input_dim + d_model, 
                                     input_dim*3, 
                                     kernel_size=kernel_size, 
                                     padding=pad)

    def forward(self, h, m):
        hq, hk, hv = torch.split(self.conv_h(h), self.d_model, dim=1)
        mk, mv = torch.split(self.conv_m(m), self.d_model, dim=1)
        
        N, C, H, W = hq.shape
        S = H * W
        
        # Reshape for attention
        hq = hq.view(N, C, S)
        hk = hk.view(N, C, S)
        hv = hv.view(N, C, S)
        mk = mk.view(N, C, S)
        mv = mv.view(N, C, S)
        
        # Current branch attention
        scores_h = torch.bmm(hq.transpose(1,2), hk) / math.sqrt(C)
        attn_h = F.softmax(scores_h, dim=-1)
        Zh = torch.bmm(attn_h, hv.transpose(1,2)).transpose(1,2)
        Zh = Zh.view(N, C, H, W)
        
        # Memory branch attention
        scores_m = torch.bmm(hq.transpose(1,2), mk) / math.sqrt(C)
        attn_m = F.softmax(scores_m, dim=-1)
        Zm = torch.bmm(attn_m, mv.transpose(1,2)).transpose(1,2)
        Zm = Zm.view(N, C, H, W)
        
        # Fusion and gating
        Z = self.conv_z(torch.cat([Zh, Zm], dim=1))
        i, g, o = torch.split(
            self.conv_output(torch.cat([Z, h], dim=1)), 
            self.input_dim, 
            dim=1
        )
        
        i = torch.sigmoid(i)
        g = torch.tanh(g)
        m_next = i * g + (1 - i) * m
        h_next = torch.sigmoid(o) * m_next
        
        return h_next, m_next

class SAConvLSTMCell(nn.Module):
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
        
        self.sa = SAAttnMem(
            input_dim=hidden_dim,
            d_model=hidden_dim,
            kernel_size=kernel_size
        )

    def forward(self, x, prev_state):
        h_prev, c_prev, m_prev = prev_state
        
        combined = torch.cat([x, h_prev], dim=1)
        combined_conv = self.conv(combined)
        
        cc_i, cc_f, cc_o, cc_g = torch.split(
            combined_conv, 
            self.hidden_dim, 
            dim=1
        )
        
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        
        c_cur = f * c_prev + i * g
        h_cur = o * torch.tanh(c_cur)
        
        # Apply self-attention memory module
        h_new, m_new = self.sa(h_cur, m_prev)
        
        return h_new, (h_new, c_cur, m_new)

class SimpleSAConvLSTM(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=1, kernel_size=(3,3)):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.cells = nn.ModuleList()
        for i in range(num_layers):
            cell_input_dim = input_dim if i == 0 else hidden_dim
            self.cells.append(
                SAConvLSTMCell(
                    cell_input_dim, 
                    hidden_dim, 
                    kernel_size
                )
            )
        
        self.conv_out = nn.Conv2d(
            hidden_dim, 
            input_dim, 
            kernel_size=1
        )

    def init_hidden(self, batch_size, height, width, device):
        hidden = []
        for _ in range(self.num_layers):
            h = torch.zeros(batch_size, self.hidden_dim, height, width, device=device)
            c = torch.zeros(batch_size, self.hidden_dim, height, width, device=device)
            m = torch.zeros(batch_size, self.hidden_dim, height, width, device=device)
            hidden.append((h, c, m))
        return hidden


    def forward(self, x, future_frames=10):
        # Process input sequence
        batch_size, seq_len, _, height, width = x.size()
        hidden = self.init_hidden(batch_size, height, width, x.device)
        
        for t in range(seq_len):
            input_tensor = x[:, t]
            for layer_idx, cell in enumerate(self.cells):
                if layer_idx == 0:
                    h, hidden[layer_idx] = cell(input_tensor, hidden[layer_idx])
                else:
                    h, hidden[layer_idx] = cell(h, hidden[layer_idx])
        
        # Generate exactly `future_frames` predictions
        outputs = []
        for _ in range(future_frames):
            for layer_idx, cell in enumerate(self.cells):
                if layer_idx == 0:
                    pred_input = self.conv_out(hidden[-1][0]) if len(outputs) == 0 else outputs[-1]
                    h, hidden[layer_idx] = cell(pred_input, hidden[layer_idx])
                else:
                    h, hidden[layer_idx] = cell(h, hidden[layer_idx])
            pred = torch.sigmoid(self.conv_out(h))  # Ensure output is in [0,1]
            outputs.append(pred.unsqueeze(1))
        
        return torch.cat(outputs, dim=1)

class Predictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers):
        super().__init__()
        self.saconvlstm = SimpleSAConvLSTM(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            kernel_size=kernel_size
        )

    def forward(self, x, future_frames=10, **kwargs):
        return self.saconvlstm(x, future_frames=future_frames, **kwargs)

# Training loop with scheduled sampling
def train(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    train_loss = 0
    teacher_forcing_prob = max(0.8 * math.exp(-epoch / 10), 0.1)
    
    for batch_idx, (input_seq, target_seq) in enumerate(train_loader):
        input_seq, target_seq = input_seq.to(device), target_seq.to(device)
        optimizer.zero_grad()
        
        # Use teacher forcing probabilistically
        if torch.rand(1) < teacher_forcing_prob:
            # Concatenate input and target for teacher forcing
            full_seq = torch.cat([input_seq, target_seq], dim=1)
            output = model(full_seq, future_frames=target_seq.size(1))
        else:
            # Autoregressive prediction
            output = model(input_seq, future_frames=target_seq.size(1))
        
        loss = criterion(output, target_seq)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        wandb.log({"Batch Loss": loss.item()})
    
    avg_loss = train_loss / len(train_loader)
    wandb.log({"Epoch Train Loss": avg_loss, "Epoch": epoch})
    return avg_loss

# Validation loop
def validate(model, val_loader, criterion, device, epoch):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for input_seq, target_seq in val_loader:
            input_seq, target_seq = input_seq.to(device), target_seq.to(device)
            output = model(input_seq, future_frames=target_seq.size(1))
            loss = criterion(output, target_seq)
            val_loss += loss.item()
    avg_loss = val_loss / len(val_loader)
    wandb.log({"Validation Loss": avg_loss})
    return avg_loss


def visualize_prediction(model, test_loader, device, sample_idx=0):
    model.eval()
    input_seq, target_seq = next(iter(test_loader))  # Get first batch
    input_seq, target_seq = input_seq.to(device), target_seq.to(device)
    
    with torch.no_grad():
        output = model(input_seq, future_frames=target_seq.size(1))
    
    # Plot input, target, and prediction
    fig, axes = plt.subplots(3, 10, figsize=(20, 6))
    for t in range(10):
        axes[0, t].imshow(input_seq[0, t, 0].cpu().numpy(), cmap='gray')
        axes[1, t].imshow(target_seq[0, t, 0].cpu().numpy(), cmap='gray')
        axes[2, t].imshow(output[0, t, 0].cpu().numpy(), cmap='gray')
    plt.tight_layout()
    wandb.log({"Predictions": wandb.Image(fig)})
    plt.close()

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
    print(f"Model is running with {model.saconvlstm.num_layers} layers.")

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


