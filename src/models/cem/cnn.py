import torch.nn as nn



import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeRNN(nn.Module):
    def __init__(self, num_edges, embed_dim=64, hidden_dim=128, num_layers=1):
        super(EdgeRNN, self).__init__()
        self.num_edges = num_edges

        # Edge feature embedding
        self.edge_feature_embed = nn.Linear(3, embed_dim)

        # RNN/LSTM Layer
        self.rnn = nn.LSTM(input_size=embed_dim, hidden_size=hidden_dim,
                           num_layers=num_layers, batch_first=True)

        # Output Layer
        self.output_layer = nn.Linear(hidden_dim, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        """
        x: Tensor of shape (batch_size, 2 * num_edges)
        """
        batch_size = x.size(0)
        num_edges = self.num_edges

        # Split edge decisions and current edge indicators
        edge_decisions = x[:, :num_edges]  # Shape: (batch_size, num_edges)
        current_edge_indicator = x[:, num_edges:]  # Shape: (batch_size, num_edges)

        # Prepare edge features (same as before)
        included = (edge_decisions == 1).float().unsqueeze(-1)
        excluded = (edge_decisions == 0).float().unsqueeze(-1)
        edge_status = torch.cat([included, excluded], dim=-1)
        current_edge_flag = current_edge_indicator.unsqueeze(-1)
        edge_features = torch.cat([edge_status, current_edge_flag], dim=-1)  # Shape: (batch_size, num_edges, 4)

        # Embed edge features
        edge_embeddings = self.edge_feature_embed(edge_features)  # Shape: (batch_size, num_edges, embed_dim)

        # RNN/LSTM Forward Pass
        outputs, (hn, cn) = self.rnn(edge_embeddings)  # outputs: (batch_size, num_edges, hidden_dim)

        # Get the outputs corresponding to the current edge
        current_edge_indices = torch.argmax(current_edge_indicator, dim=1)  # Shape: (batch_size,)
        batch_indices = torch.arange(batch_size)
        current_edge_outputs = outputs[batch_indices, current_edge_indices, :]  # Shape: (batch_size, hidden_dim)

        # Output Layer
        logits = self.output_layer(current_edge_outputs)#.squeeze(-1)  # Shape: (batch_size,)
        logits = self.sigmoid(logits)

        return logits  # Logits for binary decision


class EdgeTransformer(nn.Module):
    def __init__(self,num_edges, embed_dim=64, num_heads=4, num_layers=2,
                 use_positional_encoding=True, max_len=2000):
        super(EdgeTransformer, self).__init__()
        self.num_edges = num_edges
        self.embed_dim = embed_dim
        self.use_positional_encoding = use_positional_encoding

        # Edge feature embedding
        self.edge_feature_embed = nn.Linear(3, embed_dim)  # 4 features per edge

        # Positional Encoding
        if self.use_positional_encoding:
            self.positional_encoding = PositionalEncoding(embed_dim, max_len=max_len)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output Layer
        self.output_layer = nn.Linear(embed_dim, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        """
        x: Tensor of shape (batch_size, 2 * num_edges)
        """
        batch_size = x.size(0)
        num_edges = self.num_edges

        # Split edge decisions and current edge indicators
        edge_decisions = x[:, :num_edges]  # Shape: (batch_size, num_edges)
        current_edge_indicator = x[:, num_edges:]  # Shape: (batch_size, num_edges)

        # Prepare edge features
        # Edge inclusion status as one-hot encoding: [included, excluded, undecided]
        included = (edge_decisions == 1).float().unsqueeze(-1)
        excluded = (edge_decisions == 0).float().unsqueeze(-1)
        edge_status = torch.cat([included, excluded], dim=-1)  # Shape: (batch_size, num_edges, 3)

        # Current edge flag
        current_edge_flag = current_edge_indicator.unsqueeze(-1)  # Shape: (batch_size, num_edges, 1)

        # Combine features
        edge_features = torch.cat([edge_status, current_edge_flag], dim=-1)  # Shape: (batch_size, num_edges, 4)

        # Embed edge features
        edge_embeddings = self.edge_feature_embed(edge_features)  # Shape: (batch_size, num_edges, embed_dim)

        # Optional Positional Encoding
        if self.use_positional_encoding:
            edge_embeddings = self.positional_encoding(edge_embeddings)

        # Prepare for Transformer (requires shape: (sequence_length, batch_size, embed_dim))
        edge_embeddings = edge_embeddings.transpose(0, 1)  # Shape: (num_edges, batch_size, embed_dim)

        # Transformer Encoder
        transformer_output = self.transformer_encoder(edge_embeddings)  # Shape: (num_edges, batch_size, embed_dim)

        # Transpose back to (batch_size, num_edges, embed_dim)
        transformer_output = transformer_output.transpose(0, 1)  # Shape: (batch_size, num_edges, embed_dim)

        # Extract the embedding corresponding to the current edge
        # Current edge index for each sample in the batch
        current_edge_indices = torch.argmax(current_edge_indicator, dim=1)  # Shape: (batch_size,)
        batch_indices = torch.arange(batch_size)

        # Gather embeddings of the current edges
        current_edge_embeddings = transformer_output[batch_indices, current_edge_indices, :]  # Shape: (batch_size, embed_dim)

        # Output Layer
        logits = self.output_layer(current_edge_embeddings)#.squeeze(-1)  # Shape: (batch_size,)
        logits = self.sigmoid(logits)
        return logits  # Logits for binary decision

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        position = torch.arange(0, max_len).unsqueeze(1)  # Shape: (max_len, 1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-torch.log(torch.tensor(10000.0)) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, embed_dim)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: Tensor of shape (batch_size, sequence_length, embed_dim)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
