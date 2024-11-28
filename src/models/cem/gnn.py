import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool

class GNN(nn.Module):
    def __init__(self, num_nodes, hidden_dim=64, num_layers=3, output_type='edge'):
        super(GNN, self).__init__()
        self.num_nodes = num_nodes
        self.output_type = output_type  # 'edge' or 'graph'

        # Define GNN layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(3, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        # Edge MLP for edge-level prediction
        if self.output_type == 'edge':
            self.edge_mlp = nn.Sequential(
                nn.Linear(2 * hidden_dim + 3, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
        else:
            # Graph-level MLP for graph-level prediction
            self.graph_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )

        # Precompute edge indices for a complete undirected graph
        edge_indices = []
        for i in range(num_nodes):
            for j in range(i+1, num_nodes):
                edge_indices.append([i, j])
                edge_indices.append([j, i])  # Since the graph is undirected
        self.register_buffer('edge_index', torch.tensor(edge_indices, dtype=torch.long).t())

        # Map from edge tuple to index and vice versa
        self.edge_tuples = [(i, j) for i in range(num_nodes) for j in range(i+1, num_nodes)]
        self.edge_to_index = {edge: idx for idx, edge in enumerate(self.edge_tuples)}
        self.index_to_edge = {idx: edge for idx, edge in enumerate(self.edge_tuples)}
        self.num_edges = len(self.edge_tuples)

    def forward(self, x):
        """
        x: Tensor of shape (batch_size, 2 * num_edges)
        """
        batch_size = x.size(0)
        data_list = []
        for i in range(batch_size):
            x_i = x[i]
            # Split edge decisions and current edge indicator
            edge_decisions = x_i[:self.num_edges]
            current_edge_indicator = x_i[self.num_edges:]

            # Edge features
            edge_features = torch.zeros((self.num_edges, 3), dtype=torch.float)
            # Binary flag if the edge currently exists (1 if included, 0 otherwise)
            edge_features[:, 0] = (edge_decisions == 1).float()
            # Binary flag if we have already made a decision on this edge
            edge_features[:, 1] = (edge_decisions != 0.5).float()
            # Binary flag if this is the edge being decided on right now
            edge_features[:, 2] = current_edge_indicator

            # Reconstruct adjacency matrix
            adjacency_matrix = torch.zeros((self.num_nodes, self.num_nodes))
            for idx, decision in enumerate(edge_decisions):
                if decision == 1:
                    i, j = self.edge_tuples[idx]
                    adjacency_matrix[i, j] = 1
                    adjacency_matrix[j, i] = 1  # Undirected graph

            # Node features
            degrees = adjacency_matrix.sum(dim=1)
            normalized_degree = degrees / (self.num_nodes - 1)
            degree_zero = (degrees == 0).float()
            # Binary flag if node is attached to the current edge
            current_edge_idx = torch.argmax(current_edge_indicator).item()
            i_node, j_node = self.edge_tuples[current_edge_idx]
            node_attached = torch.zeros(self.num_nodes)
            node_attached[i_node] = 1
            node_attached[j_node] = 1
            node_features = torch.stack([normalized_degree, degree_zero, node_attached], dim=1).float()

            # Create graph data object
            data = Data(x=node_features, edge_index=self.edge_index, edge_attr=edge_features)
            data_list.append(data)

        # Batch the graphs
        batch = Batch.from_data_list(data_list)
        batch = batch.to(torch.device('mps'))

        # GNN forward pass
        x = batch.x
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)

        if self.output_type == 'edge':
            # Edge-level prediction
            row, col = edge_index
            edge_representation = torch.cat([x[row], x[col], edge_attr], dim=1)
            edge_logits = self.edge_mlp(edge_representation).squeeze(-1)
            # Mask to get the current edges
            current_edge_mask = (edge_attr[:, 2] == 1)
            outputs = edge_logits[current_edge_mask]
            outputs = torch.sigmoid(outputs)
            return outputs  # Shape: [batch_size]
        else:
            # Graph-level prediction
            x = global_mean_pool(x, batch.batch)
            graph_logits = self.graph_mlp(x)#.squeeze(-1)
            outputs = torch.sigmoid(graph_logits)
            return outputs  # Shape: [batch_size]
