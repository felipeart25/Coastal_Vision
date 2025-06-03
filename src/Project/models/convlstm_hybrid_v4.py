import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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
    def __init__(self, input_dim, hidden_dims, kernel_sizes, num_layers, lstm_hidden_size):
        super(Predictor, self).__init__()
        self.convlstm = ConvLSTM(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            kernel_sizes=kernel_sizes,
            num_layers=num_layers
        )
        self.wave_lstm = nn.LSTM(
            input_size=1,
            hidden_size=lstm_hidden_size,
            num_layers=1,
            batch_first=True
        )
        self.fusion_conv = nn.Conv2d(
            in_channels=hidden_dims[-1] + lstm_hidden_size,
            out_channels=input_dim,
            kernel_size=1
        )

        self.conv_output = nn.Conv2d(hidden_dims[-1], input_dim, kernel_size=1)

    def forward(self, slow_input, fast_input, future_seq=10):
        batch_size, seq_len, _, h, w = slow_input.shape
        
        # Process satellite stream
        _, lstm_states = self.convlstm(slow_input)
        hidden_state = lstm_states  # (B, T, C, H, W)
        
        # Process wave stream (10 segments of 360 hours)

        wave_output, _ = self.wave_lstm(fast_input) # wave_output shape: [B, 3600, lstm_hidden_size]
        #print("Wave output shape:", wave_output.shape)  # Debugging line
        B, T, H = wave_output.shape  # T should be 3600, H is lstm_hidden_size
        segment_length = T // 10      # This should be 360

        # Reshape to [B, 10, segment_length, H]
        wave_output_reshaped = wave_output.view(B, 10, segment_length, H)

        # Compute the average over the segment_length dimension to get a context per segment
        wave_context = wave_output_reshaped.mean(dim=2)  # Now shape: [B, 10, lstm_hidden_size]

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
            #print("LSTM output shape:", lstm_output.shape)  # Debugging line
            #print("Current wave shape:", current_wave.shape)  # Debugging line
            current_input = self.conv_output(lstm_output[:, 0])

            
            fused = torch.cat([lstm_output, current_wave], dim=2)
            fused = fused.squeeze(1)
            pred = self.fusion_conv(fused).unsqueeze(1)  # (B, 1, C, H, W)
            predictions.append(pred)
            last_frame = torch.cat([last_frame, pred], dim=1)  # Append prediction while keeping time_steps dimension
        
        return torch.cat(predictions, dim=1)