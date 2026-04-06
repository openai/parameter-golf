# imports
import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    """
    The feedforward transition function for the encoder
    layer. Simple feedforward network with a single
    hidden layer.
    """
    def __init__(self, d_input, d_model, d_output, model_config='ll', padding='left', dropout=0.0):
        """
        Args:
            d_input: dimension of the input
            d_model: dimension of the hidden layer
            d_output: dimension of the output
            model_config: ll -> fully connected neural net
            lc -> convolutional network
            padding: 
            dropout: model dropout to stabilize training
        """
        super.__init__()

        layers = []

        # Idea here is first linear project d_input -> d_model
        # Then for hidden layers project d_model -> d_model
        # For final output projection d_model -> d_output
        sizes = ([(d_input, d_model)] + 
                 [(d_model, d_model)]*(len(model_config)-2) + 
                 [(d_model, d_output)])

        for layer, size in zip(list(model_config), sizes):
            if layer == 'l':
                layers.append(nn.Linear(*size))
            elif layer == 'c':
                # Still need to write the convolution class, but for now feedforward should make due
                layers.append(Conv(*size, kernel_size=3, pad_type=padding))
            else:
                raise ValueError("Unknown layer type {}".format(layer))

        self.layers = nn.ModuleList(layers)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        for num, layer in enumerate(self.layers):
            x = layer(x)
            # Apply non-linearity and dropout for hidden layers
            if num < len(self.layers):
                x = self.relu(x)
                x = self.dropout(x)

        # return
        return x