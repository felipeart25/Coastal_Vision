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
    "num_layers": 4,
    "batch_size": 4,
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



# New SAM module that replicates the article’s self-attention memory module
class SAM(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # For current input features
        self.conv_query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.conv_key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.conv_value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        # For memory branch (mapping previous memory M_{t-1})
        self.conv_mem_key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.conv_mem_value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        # Fusion: fuse [Zh; Zm] -> aggregated feature Z via a 1x1 conv
        self.conv_z = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
        
        # Memory update gating with depth-wise separable convolutions.
        # (In the paper, the gating is computed as: 
        #   i'_t = sigmoid(W_mzi*Z + W_mhi*Ht + b_i),
        #   g'_t = tanh(W_mzg*Z + W_mhg*Ht + b_g),
        #   M_t = (1 - i'_t) ∘ M_{t-1} + i'_t ∘ g'_t,
        #   o'_t = sigmoid(W_mzo*Z + W_mho*Ht + b_o),
        #   Ĥ_t = o'_t ∘ M_t.)
        #
        # We approximate this using depth-wise (grouped) 3x3 convs.
        self.conv_i = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.conv_hi = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        
        self.conv_g = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.conv_hg = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        
        self.conv_o = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.conv_ho = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        
    def forward(self, x, prev_memory):
        # x: current hidden state Ht (B, C, H, W)
        B, C, H, W = x.size()
        N = H * W

        # ---- Current branch (compute Zh) ----
        query = self.conv_query(x).view(B, C // 8, N)       # (B, C//8, N)
        key_current = self.conv_key(x).view(B, C // 8, N)       # (B, C//8, N)
        value_current = self.conv_value(x).view(B, C, N)        # (B, C, N)
        
        # Compute pairwise similarity (current branch)
        e_current = torch.bmm(query.transpose(1, 2), key_current)  # (B, N, N)
        attn_current = F.softmax(e_current, dim=-1)
        Zh = torch.bmm(value_current, attn_current.transpose(1, 2)).view(B, C, H, W)
        
        # ---- Memory branch (compute Zm) ----
        if prev_memory is None:
            # If no previous memory, set Zm to zeros (or you could use x)
            Zm = torch.zeros_like(x)
        else:
            key_mem = self.conv_mem_key(prev_memory).view(B, C // 8, N)
            value_mem = self.conv_mem_value(prev_memory).view(B, C, N)
            e_mem = torch.bmm(query.transpose(1, 2), key_mem)  # (B, N, N)
            attn_mem = F.softmax(e_mem, dim=-1)
            Zm = torch.bmm(value_mem, attn_mem.transpose(1, 2)).view(B, C, H, W)
        
        # ---- Fuse aggregated features ----
        Z_cat = torch.cat([Zh, Zm], dim=1)  # (B, 2C, H, W)
        Z = self.conv_z(Z_cat)             # (B, C, H, W)
        
        # ---- Memory update ----
        # Input gate (i'_t)
        i_t = torch.sigmoid(self.conv_i(Z) + self.conv_hi(x))
        # Candidate update (g'_t)
        g_t = torch.tanh(self.conv_g(Z) + self.conv_hg(x))
        if prev_memory is None:
            prev_memory = torch.zeros_like(x)
        M_new = (1 - i_t) * prev_memory + i_t * g_t
        
        # Output gate (o'_t)
        o_t = torch.sigmoid(self.conv_o(Z) + self.conv_ho(x))
        H_hat = o_t * M_new  # Final output
        
        return H_hat, M_new

# SA-ConvLSTM Cell that embeds the new SAM module
class SAConvLSTMCell(nn.Module):
    """Self-Attention ConvLSTM Cell with updated SAM module"""
    def __init__(self, input_dim, hidden_dim, kernel_size):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        padding = (kernel_size // 2, kernel_size // 2) if isinstance(kernel_size, int) else (kernel_size[0] // 2, kernel_size[1] // 2)

        
        # Standard ConvLSTM operations: compute gates from input and previous hidden state
        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding
        )
        # Use the revised SAM for self-attention memory update
        self.sam = SAM(hidden_dim)
        
    def forward(self, x, prev_state):
        h_prev, c_prev, m_prev = prev_state
        
        # Concatenate input and previous hidden state and apply convolution
        combined = torch.cat([x, h_prev], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        
        # Update cell state
        c_cur = f * c_prev + i * g
        h_cur = o * torch.tanh(c_cur)
        
        # Apply the SAM to h_cur (and previous memory) to get self-attended hidden state and new memory
        h_attn, m_cur = self.sam(h_cur, m_prev)
        
        return h_attn, (h_attn, c_cur, m_cur)

# A simple SA-ConvLSTM module that stacks multiple cells
class SimpleSAConvLSTM(nn.Module):
    """Simplified SA-ConvLSTM with stacked SAConvLSTMCells"""
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=4, kernel_size=3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Create a stack of SAConvLSTM cells
        self.cells = nn.ModuleList()
        for i in range(num_layers):
            cell_input_dim = input_dim if i == 0 else hidden_dim
            self.cells.append(SAConvLSTMCell(cell_input_dim, hidden_dim, kernel_size))

        
        # Output layer: map hidden state back to input dimension
        self.conv_out = nn.Conv2d(hidden_dim, input_dim, kernel_size=1)
        
    def init_hidden(self, batch_size, height, width, device):
        """Initialize hidden, cell and memory states for all layers"""
        hidden = []
        for _ in range(self.num_layers):
            h = torch.zeros(batch_size, self.hidden_dim, height, width, device=device)
            c = torch.zeros(batch_size, self.hidden_dim, height, width, device=device)
            m = torch.zeros(batch_size, self.hidden_dim, height, width, device=device)
            hidden.append((h, c, m))
        return hidden
    
    def forward(self, x, future_seq=10):
        """
        x: Input tensor with shape (batch_size, seq_len, channels, height, width)
        future_seq: Number of future frames to predict
        """
        batch_size, seq_len, _, height, width = x.size()
        device = x.device
        
        # Initialize hidden states for each layer
        hidden = self.init_hidden(batch_size, height, width, device)
        
        # Process input sequence
        for t in range(seq_len):
            input_tensor = x[:, t]
            for layer_idx, cell in enumerate(self.cells):
                if layer_idx == 0:
                    h, hidden[layer_idx] = cell(input_tensor, hidden[layer_idx])
                else:
                    h, hidden[layer_idx] = cell(h, hidden[layer_idx])
        
        # Generate future predictions autoregressively
        outputs = []
        for _ in range(future_seq):
            for layer_idx, cell in enumerate(self.cells):
                if layer_idx == 0:
                    # For the first layer, use the last prediction (or the output from the input sequence)
                    if len(outputs) == 0:
                        pred_input = self.conv_out(hidden[-1][0])
                    else:
                        pred_input = outputs[-1][:, 0]
                    h, hidden[layer_idx] = cell(pred_input, hidden[layer_idx])
                else:
                    h, hidden[layer_idx] = cell(h, hidden[layer_idx])
            pred = self.conv_out(hidden[-1][0])
            outputs.append(pred.unsqueeze(1))
        
        outputs = torch.cat(outputs, dim=1)
        return outputs

# Predictor model that wraps the SA-ConvLSTM
class Predictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers):
        super().__init__()
        self.saconvlstm = SimpleSAConvLSTM(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            kernel_size=kernel_size
        )
    
    def forward(self, x, future_seq=10):
        return self.saconvlstm(x, future_seq=future_seq)
    
def train(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    train_loss = 0
    total_batches = len(train_loader)

    start_epoch = torch.cuda.Event(enable_timing=True)
    end_epoch = torch.cuda.Event(enable_timing=True)
    start_epoch.record()

    if epoch < 40:
        teacher_forcing_prob = 1.0  # Pure teacher forcing early on
    else:
        teacher_forcing_prob = max(0.9 * math.exp(-((epoch - 40) / 40)), 0.2)  # Slower decay 
    
    for batch_idx, (input_seq, future_seq) in enumerate(train_loader):
        input_seq, future_seq = input_seq.to(device), future_seq.to(device)
        
        start_iter = torch.cuda.Event(enable_timing=True)
        end_iter = torch.cuda.Event(enable_timing=True)
        start_iter.record()

        optimizer.zero_grad()

        # Get dimensions
        batch_size, seq_len, _, height, width = future_seq.size()

        # Process input sequence to initialize hidden state
        _, hidden_state = model.convlstm(input_seq)
        
        # Initialize the first input with the last frame of the input sequence
        current_input = input_seq[:, -1, :, :, :]
        outputs = []

        # Generate each frame step-by-step
        for t in range(seq_len):
            # Forward step with current input and hidden state
            lstm_output, hidden_state = model.convlstm(
                current_input.unsqueeze(1),  # Add time dimension
                hidden_state
            )
            next_frame = model.conv_output(lstm_output[:, 0])
            outputs.append(next_frame.unsqueeze(1))  # Add time dimension
            
            # Decide to use teacher forcing or not
            use_teacher_forcing = torch.rand(1).item() < teacher_forcing_prob
            if use_teacher_forcing:
                current_input = future_seq[:, t]
            else:
                current_input = next_frame.detach()  # Detach to avoid backprop through time

        # Combine all predicted frames
        output = torch.cat(outputs, dim=1)
        
        # Compute loss
        loss = criterion(output, future_seq)
        loss.backward()
        optimizer.step()

        end_iter.record()
        torch.cuda.synchronize()
        iteration_time = start_iter.elapsed_time(end_iter) / 1000

        train_loss += loss.item()
        
        wandb.log({
            "Batch Loss": loss.item(),
            "Iteration Time (s)": iteration_time,
            "Teacher Forcing Prob": teacher_forcing_prob
        })
        
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}, Time: {iteration_time:.4f}s")

    end_epoch.record()
    torch.cuda.synchronize()
    epoch_time = start_epoch.elapsed_time(end_epoch) / 1000
    avg_iteration_time = epoch_time / total_batches
            
    avg_train_loss = train_loss / len(train_loader)
    print(f'Epoch {epoch} Average Train Loss: {avg_train_loss:.4f}')
    
    wandb.log({
        "Epoch Train Loss": avg_train_loss,
        "Epoch": epoch,
        "Avg Iteration Time (s)": avg_iteration_time
    })
    
    return avg_train_loss


def validate(model, val_loader, criterion, device, epoch):
    model.eval()
    val_loss = 0
    
    with torch.no_grad():
        for input_seq, target_seq in val_loader:
            input_seq, target_seq = input_seq.to(device), target_seq.to(device)
            
            # Initialize hidden state with input sequence
            _, hidden_state = model.convlstm(input_seq)
            
            # Start with last frame of input sequence
            current_input = input_seq[:, -1]
            batch_size, seq_len, _, height, width = target_seq.size()
            outputs = []
            
            # Generate predictions autoregressively
            for t in range(seq_len):
                # Forward step
                lstm_output, hidden_state = model.convlstm(
                    current_input.unsqueeze(1),  # Add time dimension
                    hidden_state
                )
                next_frame = model.conv_output(lstm_output[:, 0])
                outputs.append(next_frame.unsqueeze(1))
                
                # Use prediction as next input
                current_input = next_frame
            
            # Combine predictions and compute loss
            output = torch.cat(outputs, dim=1)
            loss = criterion(output, target_seq)
            val_loss += loss.item()
    
    avg_val_loss = val_loss / len(val_loader)
    wandb.log({"Epoch Validation Loss": avg_val_loss, "Epoch": epoch})
    
    return avg_val_loss

def visualize_prediction(model, test_loader, device, sample_idx=0, num_timesteps=10):
    model.eval()
    
    # Get a sample from the test set
    for i, (input_seq, target_seq) in enumerate(test_loader):
        if i == sample_idx:
            break
    
    input_seq = input_seq.to(device)
    target_seq = target_seq.to(device)
    
    with torch.no_grad():
        # Initialize hidden state with input sequence
        _, hidden_state = model.convlstm(input_seq)
        
        # Autoregressively generate predictions (like validation)
        current_input = input_seq[:, -1]  # Last frame of input
        predictions = []
        
        for t in range(num_timesteps):
            # Predict next frame
            lstm_output, hidden_state = model.convlstm(
                current_input.unsqueeze(1),  # Add time dimension
                hidden_state
            )
            next_frame = model.conv_output(lstm_output[:, 0])
            predictions.append(next_frame.unsqueeze(1))
            
            # Feed prediction back as input
            current_input = next_frame
        
        predictions = torch.cat(predictions, dim=1)
    
    # Calculate per-timestep loss
    timestep_loss = [
        F.mse_loss(predictions[:, t], target_seq[:, t]).item()
        for t in range(num_timesteps)
    ]
    
    # Plotting
    fig, axes = plt.subplots(4, num_timesteps, figsize=(25, 8))
    
    # Input sequence (first T-10 frames)
    for t in range(input_seq.shape[1]):
        axes[0, t].imshow(input_seq[0, t, 0].cpu().numpy(), cmap='gray')
        axes[0, t].set_title(f'Input t={t}', fontsize=8)
        axes[0, t].axis('off')
    
    # Target frames (next 10 frames)
    for t in range(num_timesteps):
        axes[1, t].imshow(target_seq[0, t, 0].cpu().numpy(), cmap='gray')
        axes[1, t].set_title(f'Target t={t}', fontsize=8)
        axes[1, t].axis('off')
    
    # Predicted frames
    for t in range(num_timesteps):
        axes[2, t].imshow(predictions[0, t, 0].cpu().numpy(), cmap='gray')
        axes[2, t].set_title(f'Pred t={t}\nLoss: {timestep_loss[t]:.4f}', fontsize=8)
        axes[2, t].axis('off')
    
    # Error heatmap (|Target - Prediction|)
    for t in range(num_timesteps):
        error = torch.abs(target_seq[0, t, 0] - predictions[0, t, 0]).cpu().numpy()
        axes[3, t].imshow(error, cmap='hot', vmin=0, vmax=1)
        axes[3, t].set_title(f'Error t={t}', fontsize=8)
        axes[3, t].axis('off')
    
    plt.tight_layout()
    plt.savefig('autoregressive_predictions.png')
    plt.close()
    
    # Log to WandB
    wandb.log({
        "Autoregressive Predictions": wandb.Image('autoregressive_predictions.png'),
        "Per-Timestep Loss": wandb.plot.line(
            x=range(num_timesteps),
            y=timestep_loss,
            title="Per-Timestep Prediction Loss",
            xname="Timestep",
            yname="MSE Loss"
        )
    })
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


