import torch
import torch.nn as nn



class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) model for binary decision tasks in graph construction.
    """

    def __init__(self, model_args):
        super(MLP, self).__init__()
        input_size = model_args.get('input_size')  # Should be 2E (previous actions + positional encoding)
        hidden_sizes = model_args.get('hidden_sizes', [128, 64])  # List of hidden layer sizes
        output_size = 1  # Single logit output
        dropout = model_args.get('dropout', 0.0)  # Dropout rate
        batch_norm = model_args.get('batch_norm', False)  # Use batch normalization if True
        activation = model_args.get('activation', 'ReLU')  # Activation function
        init_method = model_args.get('init_method', None)  # Initialization method

        
        # Define activation function mapping
        activation_map = {
            'ReLU': nn.ReLU,
            'LeakyReLU': nn.LeakyReLU,
            'Tanh': nn.Tanh,
            'ELU': nn.ELU,
            'GELU': nn.GELU,
            'swish': nn.SiLU,
        }
        activation_fn = activation_map.get(activation, nn.ReLU)

        # Build the network layers
        layers = []
        in_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_size, hidden_size))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(activation_fn())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_size = hidden_size

        # Output layer
        layers.append(nn.Linear(in_size, output_size))
        layers.append(nn.Sigmoid())
        self.network = nn.Sequential(*layers)

        # Apply weight initialization
        if init_method:
            self.apply(self.initialize_weights(init_method))

        
    def initialize_weights(self, method):
        def init_fn(m):
            if isinstance(m, nn.Linear):
                if method == 'xavier':
                    nn.init.xavier_uniform_(m.weight)
                elif method == 'he':
                    nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                elif method == 'normal':
                    nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        return init_fn

    def forward(self, x):
        """
        Forward pass of the MLP model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 2 * num_edges).
            input_normalizer (BinaryInputNormalizer): Normalizer for input preprocessing.

        Returns:
            torch.Tensor: Output logits of shape (batch_size,).
        """
        logits = self.network(x)
        return logits