import numpy as np
import torch 
import random

import networkx as nx 
import matplotlib.pyplot as plt 
import datetime
import json
import os

# Imports
from src.models.cem.mlp import MLP
from src.models.cem.gnn import GNN
from src.models.cem.cnn import EdgeRNN,EdgeTransformer
from src.models.cem.lr import LR

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def state_to_graph1(args, state, directed=False):
    """
    Efficiently constructs the graph from a given state and returns the NetworkX graph,
    adjacency matrix, edge list, and degree sequence.

    Args:
        args: An object with attributes like `n` (number of nodes).
        state: A 1D array representing the presence (1) or absence (0) of edges in the graph.
        directed: Whether the graph should be directed.

    Returns:
        G (networkx.Graph): The constructed NetworkX graph.
        adjMatG (numpy.ndarray): The adjacency matrix of the graph.
        edgeListG (list of lists): The neighbor list representation of the graph.
        Gdeg (numpy.ndarray): The degree sequence of the graph.
    """
    n = args.n
    state = np.asarray(state, dtype=np.int8)

    # Generate the upper triangle indices of the adjacency matrix
    triu_indices = np.triu_indices(n, k=1)
    
    # Fill the adjacency matrix
    adjMatG = np.zeros((n, n), dtype=np.int8)
    adjMatG[triu_indices] = state
    adjMatG += adjMatG.T  # Symmetrize for undirected graph

    # Generate degree sequence directly from the adjacency matrix
    Gdeg = adjMatG.sum(axis=1)

    # Create neighbor list
    edgeListG = [np.flatnonzero(adjMatG[i]).tolist() for i in range(n)]

    # Create a NetworkX graph
    if directed:
        G = nx.from_numpy_array(adjMatG, create_using=nx.DiGraph)
    else:
        G = nx.from_numpy_array(adjMatG, create_using=nx.Graph)

    return G, adjMatG, edgeListG, Gdeg



import networkx as nx
import numpy as np

def reconstruct_matrix(non_diagonal_entries, n):
    """
    Reconstruct a square matrix with zeros on the diagonal 
    and given non-diagonal entries.

    Parameters:
        non_diagonal_entries (list or array): The values to fill in the non-diagonal positions.
        n (int): The size of the square matrix (n x n).

    Returns:
        np.ndarray: The reconstructed square matrix.
    """
    # Initialize a zero matrix
    matrix = np.zeros((n, n), dtype=int)
    
    # Create a mask for non-diagonal entries
    non_diagonal_mask = ~np.eye(n, dtype=bool)
    
    # Fill the non-diagonal entries
    matrix[non_diagonal_mask] = non_diagonal_entries
    
    return matrix


def state_to_graph(args, state, directed=False):
    """
    Constructs the graph G from the given state and returns the NetworkX graph,
    adjacency matrix, edge list, and degree sequence.
    
    Args:
        args: An object with attributes like `n` (number of nodes).
        state: A list or array representing the edge states (1 for edge, 0 for no edge).
        directed: Whether the graph should be directed.

    Returns:
        G (networkx.Graph): The constructed NetworkX graph.
        adjMatG (numpy.ndarray): The adjacency matrix of the graph.
        edgeListG (numpy.ndarray): The neighbor list representation of the graph.
        Gdeg (numpy.ndarray): The degree sequence of the graph.
    """
    # Initialize adjacency matrix, edge list, and degree sequence
    adjMatG = np.zeros((args.n, args.n), dtype=np.int8)  # Adjacency matrix
    edgeListG = np.zeros((args.n, args.n), dtype=np.int8)  # Neighbor list
    Gdeg = np.zeros(args.n, dtype=np.int8)  # Degree sequence

    if directed:
        adjMatG = reconstruct_matrix(state[:int(len(state)/2)], args.n)
        G = nx.DiGraph(adjMatG)
        return G, adjMatG, None, None
    else:


        # Populate adjacency matrix and edge list
        count = 0
        for i in range(args.n):
            for j in range(i + 1, args.n):
                if state[count] == 1:
                    adjMatG[i][j] = 1
                    adjMatG[j][i] = 1
                    edgeListG[i][Gdeg[i]] = j
                    edgeListG[j][Gdeg[j]] = i
                    Gdeg[i] += 1
                    Gdeg[j] += 1
                count += 1
        G = nx.Graph(adjMatG)
        return G, adjMatG, edgeListG, Gdeg

  


def initialize_model(args, MYN, device, seed):
    # Step 0 : seed everything
    seed_everything(seed)

    # Step 1: Choose Model
    if args.model == "MLP":
        model_args = {
            'input_size': 2 * MYN,
            'hidden_sizes': args.hidden_sizes,
            'dropout': args.dropout,
            'batch_norm': args.batch_norm,
            'activation': args.activation,
            'init_method': args.init_method
            }
        model = MLP(model_args).to(device)
    elif args.model == "LR": 
        model = LR({ 'input_size': 2 * MYN, }).to(device)
    elif args.model == "GNN":
        model_args = { }
        #model = GNN(args.n, hidden_dim=32, num_layers=3, output_type='edge1').to(device)
        #model = EdgeRNN(args.MYN).to(device)
        model = EdgeTransformer(args.MYN).to(device)
     
    # Step 2: Select optimizer
    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    elif args.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    elif args.optimizer == "RMSprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")
    
    return model, optimizer



def display_graph(adjMatG):
    print("Best adjacency matrix in current step:")
    print(adjMatG)

    G = nx.convert_matrix.from_numpy_array(adjMatG)

    plt.clf()
    nx.draw_circular(G)

    plt.axis('equal')
    plt.draw()
    plt.pause(0.001)
    plt.show()



def create_output_folder(args):
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    if args.reward_function == 9999:
        problem_id = 'No_Iso_Triangle'
        folder_name = f"{args.base_folder}/conj_{problem_id}__n_{args.n}_{args.model}/cem_run__seed_{args.seed}_{timestamp}_id_{args.current_idx}"
    elif args.reward_function == 9998:
        problem_id = 'No_3_In_Line'
        folder_name = f"{args.base_folder}/conj_{problem_id}__n_{args.n}_{args.model}/cem_run__seed_{args.seed}_{timestamp}_id_{args.current_idx}"
    elif args.reward_function == 9997:
        problem_id = 'No_Right_Triangle'
        folder_name = f"{args.base_folder}/conj_{problem_id}__n_{args.n}_{args.model}/cem_run__seed_{args.seed}_{timestamp}_id_{args.current_idx}"
    elif args.reward_function == 9996:
        problem_id = 'No_Golomb_Ruler'
        folder_name = f"{args.base_folder}/conj_{problem_id}__n_{args.n}_{args.model}/cem_run__seed_{args.seed}_{timestamp}_id_{args.current_idx}"
    elif args.reward_function == 995:
        problem_id = 'No_Parallelograms'
        folder_name = f"{args.base_folder}/conj_{problem_id}__n_{args.n}_{args.model}/cem_run__seed_{args.seed}_{timestamp}_id_{args.current_idx}"
    else:
        directed= "directed" if args.directed else "undirected"
        folder_name = f"{args.base_folder}/graph_conj_{args.reward_function}__{directed}_n_{args.n}_{args.model}/cem_run__seed_{args.seed}_graph_init_{args.init_graph}_{timestamp}_id_{args.current_idx}"
   
    output_folder = os.path.join("results", folder_name)
    os.makedirs(output_folder, exist_ok=True)
    return output_folder