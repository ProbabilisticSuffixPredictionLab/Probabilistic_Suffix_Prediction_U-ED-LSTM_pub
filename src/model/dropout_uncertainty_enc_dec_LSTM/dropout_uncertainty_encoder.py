"""
Encoder consisting of two an two-layerd LSTM with LSTM cells using dropout as a Bayesian approximation.
"""

from .dropout_uncertainty_LSTM_cell import DropoutUncertaintyLSTMCell

import torch
from torch import nn, Tensor
from typing import Optional, Tuple, List, Union

class DropoutUncertaintyLSTMEncoder(nn.Module):
    def __init__(self,
                 hidden_size: int,
                 num_layers: int,
                 # dynamic attributes
                 embeddings,
                 data_indices_enc: list,
                 input_size: int,
                 # static attributes
                 static_embeddings: Optional[nn.ModuleList] = None,
                 static_data_indices: Optional[List[List[int]]] = None,
                 dropout: Optional[float] = None):
        """
        Encoder part of the Encoder-Decoder LSTM
        
        ARGS:
        - input_size: Size of input features
        - hidden_size: Size of hidden layer
        - embeddings: Embedding modules for categorical data encoder
        - data_indices_enc: Indicies of tensors used as input of the encoder model
        - num_layers: Number hidden layers in the LSTM
        - dropout: Dropout probability
        """
        super(DropoutUncertaintyLSTMEncoder, self).__init__()
        
        # Encoder learnable embeddings
        self.embeddings = embeddings
        
        # List of two lists (categorical, numerical) each containing the indices of tensors required for encoder
        self.data_indices_enc = data_indices_enc

        # Static feature configuration
        if static_data_indices is None:
            self.static_data_indices = [[], []]
        else:
            self.static_data_indices = [list(static_data_indices[0]), list(static_data_indices[1])]
        self.static_cat_indices = self.static_data_indices[0]
        self.static_num_indices = self.static_data_indices[1]

        if static_embeddings is None:
            self.static_embeddings = nn.ModuleList()
        else:
            self.static_embeddings = static_embeddings
        
        cat_static_dim = sum(embedding.embedding_dim for embedding in self.static_embeddings)
        num_static_dim = len(self.static_num_indices)
        computed_static_dim = cat_static_dim + num_static_dim
        self.static_input_size = computed_static_dim
        
        self.static_fc = None
        self.static_merger = None
        
        if self.static_input_size > 0:
            self.static_fc = nn.Sequential(
                nn.Linear(self.static_input_size, hidden_size),
                nn.ReLU()
            )
            self.static_merger = nn.Linear(hidden_size * 2, hidden_size)
        
         # Linear for dynamic before inserted into lstm layer
        self.input_proj = nn.Linear(input_size, hidden_size)
        
        # Create a first cell:
        self.first_layer = DropoutUncertaintyLSTMCell(input_size=hidden_size, hidden_size=hidden_size, dropout=dropout)
        
        # Create multiple LSTM cells based on num_layer
        self.hidden_layers = nn.ModuleList([DropoutUncertaintyLSTMCell(input_size=hidden_size, hidden_size=hidden_size, dropout=dropout) for i in range(num_layers-1)])

    def regularizer(self) -> Tuple[float, float]:
        """
        L2 regularization of Encoder weights, biases and dropout.
        
        OUTPUTS:
        - total_weight_reg: L2 weight regularization term
        - total_bias_reg: L2 bias regularization term
        """
        total_weight_reg, total_bias_reg = self.first_layer.regularizer()
        
        for l in self.hidden_layers:
            weight, bias = l.regularizer()
            total_weight_reg += weight
            total_bias_reg += bias

        # Projection layer (weaker prior)
        proj_weight_reg = 0.1 * torch.sum(self.input_proj.weight ** 2)
        proj_bias_reg = 0.1 * torch.sum(self.input_proj.bias ** 2)
        total_weight_reg += proj_weight_reg
        total_bias_reg += proj_bias_reg

        # regularizer for static
        if self.static_fc is not None:
            for layer in self.static_fc:
                if isinstance(layer, nn.Linear):
                    total_weight_reg += 0.1 * torch.sum(layer.weight ** 2)
                    total_bias_reg += 0.1 * torch.sum(layer.bias ** 2)

        if self.static_merger is not None:
            total_weight_reg += 0.1 * torch.sum(self.static_merger.weight ** 2)
            total_bias_reg += 0.1 * torch.sum(self.static_merger.bias ** 2)
            
        return total_weight_reg, total_bias_reg
        
    def forward(self,
                input: List,
                static_inputs: Optional[Union[Tensor, List, Tuple, dict]] = None) -> Tuple[Tensor, Tensor]:
        """
        Forward pass through the encoder to get the final (hidden) state vector as input for the decoder.
        
        INPUTS:
        - input: Prefixes, Tensor: seq_len, batch_size, input_size
        
        OUTPUTS:
        - h,c: Last hidden and cell states of the last layer.
        """
        
        if static_inputs is not None and self.static_fc is None:
            raise ValueError("Static inputs provided but encoder was initialized without static feature size.")

        # Transform the input into 
        prefixes = self.__data_enc_for_model(data=input) # dim: Tensor: seq_len x batch_size x input feature (cat as embedding) 

        # Project input features to hidden size
        input_proj = self.input_proj(prefixes)
        
        # Outputs: All hidden states of all cells in the layer, h,c: last hidden state and cell state in the layer
        outputs, (h, c), _ = self.first_layer(input=input_proj, hx=None, z=None)
        
        # Pass through the remaining LSTM cell: Layer gets for: input: h_n Tensor, hx: (h, c)
        for _, layer in enumerate(self.hidden_layers):
            outputs, (h, c), _ = layer(input=outputs, hx=(h, c), z=None)

        if self.static_fc is not None:
            static_tensor = self.__prepare_static_tensor(static_inputs, h.device)
            if static_tensor is not None:
                static_latent = self.static_fc(static_tensor)
                merged_hidden = torch.cat((h, static_latent), dim=-1)
                h = self.static_merger(merged_hidden)

        return (h, c)
    
    def __data_enc_for_model(self, data):
        """
        Transform the dataloader input (prefix or suffix input) into a tensor structure for the encoder.
        
        INPUTS:
        - data: dataloader input
        
        OUTPUTs:
        - prefixes: Returns model input: Tensor seq_len x batch_size x input features (also embedded)
        """       
        cats = [data[0][i] for i in self.data_indices_enc[0]] # dims: list (n categorical values): Each with Tensor: batch_size x (window_size - suffix size)
        nums = [data[1][i] for i in self.data_indices_enc[1]] # dims: list (n numerical values): Each with Tensor: batch_size x (window_size - suffix size)
        
        assert len(cats) == len(self.data_indices_enc[0]) and len(nums) == len(self.data_indices_enc[1]), \
            f"Encoder: Number of input tensor is unequal the number of indices"
                
        # Embedd categorical tensors
        embedded_cats = []
        for i, embedd in enumerate(self.embeddings):        
            embedded_cats.append(embedd(cats[i]))

        # Merged categroical data
        merged_cats = torch.cat([cat for cat in embedded_cats], dim=-1)
    
        if len(nums):
            # Merged numerical inputs
            merged_nums = torch.cat([num.unsqueeze(2) for num in nums], dim=-1)
        else:
            merged_nums = torch.tensor([], device=merged_cats.device)
        prefixes = torch.cat((merged_cats, merged_nums), dim=-1).permute(1,0,2) # dim: seq_len x batch_size x input_features
        return prefixes

    def __prepare_static_tensor(self,
                                static_inputs: Optional[Union[Tensor, List, Tuple, dict]],
                                device: torch.device) -> Optional[Tensor]:
        """
        Build a dense tensor from optional static categorical and numerical inputs.
        """
        
        if static_inputs is None or self.static_input_size == 0:
            return None

        # Backwards compatibility: allow pre-projected tensors
        if isinstance(static_inputs, Tensor):
            tensor = static_inputs.to(device)
            if tensor.dim() == 1:
                tensor = tensor.unsqueeze(1)
            return tensor.float()

        cat_tensor = None
        num_tensor = None

        if isinstance(static_inputs, dict):
            cat_tensor = static_inputs.get('categorical')
            num_tensor = static_inputs.get('numerical')
        elif isinstance(static_inputs, (list, tuple)):
            if len(static_inputs) > 0:
                cat_tensor = static_inputs[0]
            if len(static_inputs) > 1:
                num_tensor = static_inputs[1]
        else:
            raise TypeError("static_inputs must be tensors, dicts or iterables of tensors")

        static_parts: List[Tensor] = []

        if self.static_cat_indices and len(self.static_embeddings) > 0:
            if cat_tensor is None or cat_tensor.numel() == 0:
                raise ValueError("Static categorical features specified, but tensor is missing.")
            cat_tensor = cat_tensor.to(device)
            if cat_tensor.dim() == 1:
                cat_tensor = cat_tensor.unsqueeze(1)
            embedded_static = []
            for idx, embedding in zip(self.static_cat_indices, self.static_embeddings):
                if idx >= cat_tensor.size(1):
                    raise IndexError("Static categorical index out of range")
                feature_values = cat_tensor[:, idx].long()
                embedded_static.append(embedding(feature_values))
            if embedded_static:
                static_parts.append(torch.cat(embedded_static, dim=-1))

        if self.static_num_indices:
            if num_tensor is None or num_tensor.numel() == 0:
                raise ValueError("Static numerical features specified, but tensor is missing.")
            num_tensor = num_tensor.to(device).float()
            if num_tensor.dim() == 1:
                num_tensor = num_tensor.unsqueeze(1)
            selected_nums = []
            for idx in self.static_num_indices:
                if idx >= num_tensor.size(1):
                    raise IndexError("Static numerical index out of range")
                selected_nums.append(num_tensor[:, idx].unsqueeze(1))
            if selected_nums:
                static_parts.append(torch.cat(selected_nums, dim=-1))

        if not static_parts:
            return None

        return torch.cat(static_parts, dim=-1)