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
                 static_input_size: int = 0,
                 dropout: Optional[float] = None):
        """
        Encoder part of the Encoder-Decoder LSTM
        
        ARGS:
        - input_size: Size of input features
        - hidden_size: Size of hidden layer
        - embeddings: Embedding modules for categorical data encoder
        - data_indices_enc: Indicies of tensors used as input of the encoder model
        - num_layers: Number hidden layers in the LSTM
        - static_input_size: Flattened feature size of optional static inputs
        - dropout: Dropout probability
        """
        super(DropoutUncertaintyLSTMEncoder, self).__init__()
        
        # Encoder learnable embeddings
        self.embeddings = embeddings
        
        self.static_embeddings = static_embeddings if static_embeddings is not None else nn.ModuleList()
        
        # List of two lists (categorical, numerical) each containing the indices of tensors required for encoder
        self.data_indices_enc = data_indices_enc

        if static_data_indices is None:
            self.static_data_indices = [[], []]
        else:
            self.static_data_indices = static_data_indices
        self.static_cat_indices = self.static_data_indices[0]
        self.static_num_indices = self.static_data_indices[1]
        
        self.static_input_size = static_input_size
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
        
        # Transform the input into 
        prefixes = self.__data_enc_for_model(data=input) # dim: Tensor: seq_len x batch_size x input feature (cat as embedding) 

        # Project input features to hidden size
        input_proj = self.input_proj(prefixes)
        
        # Outputs: All hidden states of all cells in the layer, h,c: last hidden state and cell state in the layer
        outputs, (h, c), _ = self.first_layer(input=input_proj, hx=None, z=None)
        
        # Pass through the remaining LSTM cell: Layer gets for: input: h_n Tensor, hx: (h, c)
        for _, layer in enumerate(self.hidden_layers):
            outputs, (h, c), _ = layer(input=outputs, hx=(h, c), z=None)

        if self.static_fc is not None and static_inputs is not None:
            static_tensor = self.__static_data_enc_model(static_inputs)
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

    def __static_data_enc_model(self,
                                static_inputs: Optional[Union[Tensor, List, Tuple]]) -> Optional[Tensor]:

        if static_inputs is None or self.static_input_size == 0:
            return None

        if not isinstance(static_inputs, (list, tuple)):
            raise TypeError("static_inputs must be a tuple (static_cats, static_nums)")

        static_cats, static_nums = static_inputs

        merged_static_cats = None
        if static_cats is not None and len(self.static_embeddings) > 0:
            # Ensure correct dtype
            static_cats = static_cats.long()

            embedded = []
            for i, emb in enumerate(self.static_embeddings):
                embedded.append(emb(static_cats[:, i]))

            merged_static_cats = torch.cat(embedded, dim=-1)

        merged_static_nums = None
        if static_nums is not None:
            if isinstance(static_nums, Tensor):
                merged_static_nums = static_nums
            else:
                merged_static_nums = torch.cat(
                    [num.unsqueeze(1) for num in static_nums], dim=-1
                )

        if merged_static_cats is not None and merged_static_nums is not None:
            return torch.cat((merged_static_cats, merged_static_nums), dim=-1)
        elif merged_static_cats is not None:
            return merged_static_cats
        elif merged_static_nums is not None:
            return merged_static_nums
        else:
            return None
