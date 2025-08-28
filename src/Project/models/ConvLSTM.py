import torch
import torch.nn as nn


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
        #ssert len(hidden_dims) == num_layers, "Length of hidden_dims must match num_layers"
        #ssert len(kernel_sizes) == num_layers, "Length of kernel_sizes must match num_layers"

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
    def __init__(self, input_size=3, hidden_size=64, future_seq=1, wave_steps_per_frame=34):

        #### Initialize parameters ####	
        super(ProgressiveWaveContextEncoder, self).__init__()
        ## LSTM encoder to process wave data
        self.encoder = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        ## Projection layer to reduce dimensionality
        self.projection = nn.Linear(hidden_size, hidden_size)
        ## Number of contexts to generate
        self.future_seq = future_seq
        ## brinf time fro dictionary
        self.wave_steps_per_frame = wave_steps_per_frame

    #### Forward pass through the encoder ####
    def forward(self, wave_input, wave_future):
        """
        Args:
            wave_input: [B, T_past, 3] — past wave data (e.g., 360)
            wave_future: [B, T_future, 3] — future wave data (e.g., 1800)
        Returns:
            wave_contexts: [B, future_seq, H] — 1 vector per future image frame
        """
        B, _, _ = wave_input.shape
        contexts = []

        for t in range(self.future_seq):
            steps = (t + 1) * self.wave_steps_per_frame
            if wave_future.shape[1] < steps:
                raise ValueError(f"Wave future data too short! Need {(t+1)*self.wave_steps_per_frame} steps, but got {wave_future.shape[1]}.")

            future_slice = wave_future[:, :steps, :]  # [B, steps, 3]
            combined = torch.cat([wave_input, future_slice], dim=1)  # [B, total_steps, 3]

            _, (h_n, _) = self.encoder(combined)  # h_n: [1, B, H]
            ctx = self.projection(h_n[-1])        # ctx: [B, H]
            contexts.append(ctx)

        return torch.stack(contexts, dim=1)  # [B, future_seq, H]

#### Predictor class ####
class Predictor(nn.Module):
    def __init__(self, input_dim, hidden_dims, kernel_sizes, num_layers, lstm_hidden_size, dropout, wave_steps_per_frame, future_seq=1):
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
        self.wave_encoder = ProgressiveWaveContextEncoder(input_size=3, hidden_size=lstm_hidden_size, future_seq=future_seq, wave_steps_per_frame=wave_steps_per_frame)

        #### Activation function ####
        self.activation = nn.Sigmoid()

        #### Initialize Conv2D layer to generate final output ####
        self.conv_output = nn.Conv2d(hidden_dims[-1], input_dim, kernel_size=1)

    #### Forward pass through the model ####
    def forward(self, slow_input, fast_input, fast_fut_input, future_seq=1, inference=False):

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
            wave_idx = t % wave_context.size(1)

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

            # Update current_input for next iteration
            # Only during inference, apply binarization
             # Remove the channel dimension
            if inference:
                current_input = (pred > 0.5).float().squeeze(1)
            else:
                current_input = pred.detach().squeeze(1)  # Use soft predictions during training

            #### concat the prediction to the last frame to keep the time_steps dimension ####
            last_frame = torch.cat([last_frame, pred], dim=1)  # Append prediction while keeping time_steps dimension
        
        return torch.cat(predictions, dim=1)