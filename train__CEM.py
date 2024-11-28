import argparse
import torch
import os
import json
import numpy as np


from src.rl_algo.cem.train import train_CEM_graph_agent
from src.rl_algo.cem.utils import create_output_folder
from src.rewards.score import reward_dict
from src.rewards.utils import load_pattern_graphs


# Combinations

ez_combinations = [

# conjecture 1 --> test out 18 and 19 nodes
    [19 , 1, False,False],     # [ num_nodes, reward_function, directed,isoceles_triangle]
    [18 , 1, False,False],     # [ num_nodes, reward_function, directed,isoceles_triangle]
    # conjecture 5 --> test out  10 nodes
    [10 , 5, False,False],     # [ num_nodes, reward_function, directed,isoceles_triangle]
    # conjecture 8 --> test out  15 nodes
    [15 , 8, False,False],     # [ num_nodes, reward_function, directed,isoceles_triangle]
    # conjecture 9 --> test out  18 nodes
    [19 , 9, False,False],
    # conjecture 10 --> test out  11 nodes
    [11 , 10, False,False],


    # Isoceles triangle
    [4 , 9999, False,True],
    [5 , 9999, False,True],
    [6 , 9999, False,True],
    [7 , 9999, False,True],
    [8 , 9999, False,True],


    # Tournament
    [3 , 999, True,False],
    [5 , 999, True,False],
    [8 , 999, True,False],


    # Graffiti

    # conjecture 93 --> test out  16+ nodes
    [15 , 93, False,False],
    [16 , 93, False,False],
    [17 , 93, False,False],
    [18 , 93, False,False],


    # conjecture 96 --> test out  20+ nodes
    [17 , 96, False,False],
    [18 , 96, False,False],
    [19 , 96, False,False],
    [20 , 96, False,False],

    # conjecture 97 --> test out  14+ nodes
    [11 , 97, False,False],
    [12 , 97, False,False],
    [13 , 97, False,False],
    [14 , 97, False,False],

    # conjecture 98 --> test out  15+ nodes
    [12 , 98, False,False],
    [13 , 98, False,False],
    [14 , 98, False,False],
    [15 , 98, False,False],

    # conjecture 99 --> test out  7+ nodes
    [5 , 99, False,False],
    [6 , 99, False,False],
    [7 , 99, False,False],
    [8 , 99, False,False],



]


combinations = [

    
    # conjecture 3 --> test out  13 and 14 nodes
    [13 , 3, False,False],     # [ num_nodes, reward_function, directed,isoceles_triangle]
    [14 , 3, False,False],     # [ num_nodes, reward_function, directed,isoceles_triangle]

    # conjecture 4 --> test out  36 nodes
    [36 , 4, False,False],     # [ num_nodes, reward_function, directed,isoceles_triangle]
    

    # Isoceles triangle
    [9 , 9999, False,True],
    [10 , 9999, False,True],
    [16 , 9999, False,True],

    # Tournament
    [10 , 999, True,False],
    [14 , 999, True,False],

    
    # conjecture 94 --> test out  50+ nodes
    [50 , 94, False,False],
    [51 , 94, False,False],


    # conjecture 95 --> test out  67+ nodes
    [67 , 95, False,False],
    [68 , 95, False,False],


    # Directed: seymour, cacette, woodall
    [10 , 998, True,False],
    [11 , 998, True,False],
    [12 , 998, True,False],
    [13 , 998, True,False],
    [14 , 998, True,False],
    [15 , 998, True,False],
    [16 , 998, True,False],
    [17 , 998, True,False],
    [18 , 998, True,False],
    [19 , 998, True,False],
    [20 , 998, True,False],
    [21 , 998, True,False],
    [22 , 998, True,False],
    [23 , 998, True,False],
    [24 , 998, True,False],
    [25 , 998, True,False],
    [26 , 998, True,False],
    [27 , 998, True,False],
    [28 , 998, True,False],
    [29 , 998, True,False],
    [30 , 998, True,False],
    [31 , 998, True,False],


    [10 , 997, True,False],
    [11 , 997, True,False],
    [12 , 997, True,False],
    [13 , 997, True,False],
    [14 , 997, True,False],
    [15 , 997, True,False],
    [16 , 997, True,False],
    [17 , 997, True,False],
    [18 , 997, True,False],
    [19 , 997, True,False],
    [20 , 997, True,False],
    [21 , 997, True,False],
    [22 , 997, True,False],
    [23 , 997, True,False],
    [24 , 997, True,False],
    [25 , 997, True,False],
    [26 , 997, True,False],
    [27 , 997, True,False],
    [28 , 997, True,False],
    [29 , 997, True,False],
    [30 , 997, True,False],
    [31 , 997, True,False],


    [10 , 996, True,False],
    [11 , 996, True,False],
    [12 , 996, True,False],
    [13 , 996, True,False],
    [14 , 996, True,False],
    [15 , 996, True,False],
    [16 , 996, True,False],
    [17 , 996, True,False],
    [18 , 996, True,False],
    [19 , 996, True,False],
    [20 , 996, True,False],
    [21 , 996, True,False],
    [22 , 996, True,False],
    [23 , 996, True,False],
    [24 , 996, True,False],
    [25 , 996, True,False],
    [26 , 996, True,False],
    [27 , 996, True,False],
    [28 , 996, True,False],
    [29 , 996, True,False],
    [30 , 996, True,False],
    [31 , 996, True,False],

    

    

]

def parse_args():
    parser = argparse.ArgumentParser(description="Train a reinforcement learning model using cross entropy method.")
    # >>>>>> Environment arguments <<<<<<
    parser.add_argument('--n', type=int, default=7, help='Number of vertices in the graph.')
    parser.add_argument('--directed', type=bool, default=False, help='Whether the graph is directed.')
    parser.add_argument('--reward_function', type=int, default=9999, help='Reward function to use.')
    parser.add_argument('--seed', type=int, default=29092000, help='Seed for random number generators.')
    parser.add_argument('--init_graph', type=str, default='empty', help='Initial graph type.', choices=['empty', 'complete',"random_tree"])

    parser.add_argument('--isoceles_triangle',type=bool, default=True,help = 'are we overriding graph problem w/ the isoceles triangle problem')

    # >>>>>> Reward shaping arguments <<<<<<
    parser.add_argument('--reward_weight_is_tree_pos', type=float, default=0.5, help='Reward weight for tree reward.')
    parser.add_argument('--reward_weight_is_tree_neg', type=float, default=0.5, help='Reward weight for tree reward.')
    parser.add_argument('--reward__is_tree', type=bool, default=False, help='Whether to use tree reward.')
    parser.add_argument('--reward_weight_cyclomatic', type=float, default=0.01, help='Reward weight for connectedness reward.')
    parser.add_argument('--reward__use_cylomatic', type=bool, default=False, help='Whether to use connectedness reward.')
    parser.add_argument('--INF', type=float, default=10000, help='Large negative value for bad actions such as disconnected graphs.')

    # >>>>>> Cross Entropy Method arguments <<<<<<
    parser.add_argument('--iterations', type=int, default=1000000, help='Number of iterations for training.')
    parser.add_argument('--n_sessions', type=int, default=200, help='Number of new sessions per iteration.')
    parser.add_argument('--percentile', type=int, default=91, help='Percentile for selecting elite sessions.')
    parser.add_argument('--super_percentile', type=int, default=92, help='Percentile for selecting super sessions.')

    # >>>>>> Machine Learning Models <<<<<<
    parser.add_argument('--model', type=str, default='MLP', help='What ML model to use.')
    parser.add_argument('--hidden_sizes', nargs='+', type=int, default=[128, 64, 4], help='Hidden layer sizes for the MLP model.')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate for the MLP model.')
    parser.add_argument('--batch_norm', type=bool, default=False, help='Whether to use batch normalization in the MLP model.')
    parser.add_argument('--activation', type=str, default='swish', help='Activation function for the MLP model.',choices = ['ReLU', 'LeakyReLU', 'Sigmoid', 'Tanh'])
    parser.add_argument('--init_method', type=str, default='xavier', help='Initialization method for the MLP model.')

    # >>>>>> Training arguments <<<<<<
    parser.add_argument('--learning_rate', type=float, default=0.00001, help='Learning rate for the optimizer.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training.')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer to use (SGD, Adam, RMSprop).')
    
    # >>>>>> Logging arguments <<<<<<
    parser.add_argument('--logger', type=str, default='tensorboard', choices=['wandb', 'tensorboard'], help='Logger to use.')
    parser.add_argument('--wandb_entity', type=str, default='dezaarna', help='Wandb entity.')
    parser.add_argument('--wandb_project', type=str, default='CEM_runs', help='Wandb project name.')
    parser.add_argument('--save_every_k_iters', type=int, default=100, help='Save best graphs every k iterations.')
    parser.add_argument('--print_every_k', type=int, default=5, help='Print progress every k iterations.')
    parser.add_argument('--print_on_improvement_of_reward', type=bool, default=True, help='Print progress when reward improves.')

    parser.add_argument('--current_idx', type=int, default=69, help='')
    parser.add_argument('--base_folder', type=str, default='cem_thurs_nov28', help='Output folder for saving results.')
    parser.add_argument('--mckay_dir', type=str,
                         default='/Users/adeza3/Desktop/PhD_year1/Courses/AI_4_Math/simple_graphs_mckay',
                         help='')
    # >>>>>> Cache arguments <<<<<<
    parser.add_argument('--reward_tolerance', type=float, default=0.01, help='Reward tolerance for caching solutions.')

    # >>>>>> Stagnation detection arguments <<<<<<
    parser.add_argument('--stagnation_epochs', type=int, default=100, help='Number of epochs to detect stagnation.')
    

    # >>>>>> Tolerance arguments <<<<<<
    parser.add_argument('--tol_for_valid_counter_example', type=float, default=0.000001, help='Normalize input.')

    # Plotting arguments
    parser.add_argument('--fps', type=float, default=1, help='fps.')
    

    # Normalize input
    parser.add_argument('--normalize_input', type=bool, default=False, help='Normalize input.')
    args = parser.parse_args()
    return args



if __name__ == "__main__":


    args = parse_args()

    # Print arguments
    print("\n\n\t\t Arguments:")
    for arg in vars(args):
        print(f"\t\t\t {arg}: {getattr(args, arg)}")

    # Set device (handle CUDA, MPS, or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_built() else "cpu")

    # Create output folder based on arguments
    output_folder = create_output_folder(args)

    args.output_folder = output_folder

    # Save all arguments to a JSON file
    args_path = os.path.join(output_folder, "args.json")
    with open(args_path, "w") as f:
        json.dump(vars(args), f, indent=4)
    print(f"Arguments saved to {args_path}")

    args.device = device
    print("\n\n\t\t Starting training on device: ", device)
    print("\n\n\t\t Output folder: ", output_folder)


    args.calc_score = reward_dict[args.reward_function]



    # Colors for plotting
    args.colors = {   'node_color': 'skyblue',
            'highlight_edge_color': 'green',
            'existing_edge_color': 'black',
            'undecided_color': 'grey',
            'decision_1_color': 'salmon',
            'decision_0_color': 'grey', 
            'possible_edge_color': 'lightcoral'}

    args.state_vector_colors = {'upper_triangle_color': 'coolwarm', 'positional_encoding_color': 'coolwarm'}


    if args.reward_function == 999:
        args.pattern_graphs,args.tournament_max_score = load_pattern_graphs(args.mckay_dir,num_nodes = args.n  )


    # We are actually solving the no isoceles triangle  problem on n by n grid
    if args.isoceles_triangle:
        print("\n\n\n\n \t\t\t >>> Training CEM Agent on No Isoceles problem")


        # Precompute grid positions
        args.positions = [(i, j) for i in range(args.n) for j in range(args.n)]
        
        # Precompute squared distances between all grid points
        distance_matrix = np.zeros((args.n**2, args.n**2), dtype=int)
        for idx1, (i1, j1) in enumerate(args.positions):
            for idx2, (i2, j2) in enumerate(args.positions):
                dx = i1 - i2
                dy = j1 - j2
                dist2 = dx * dx + dy * dy
                distance_matrix[idx1, idx2] = dist2
        args.distance_matrix = distance_matrix
        from src.rewards.terminal_scores import *
        if args.reward_function == 9999:
            args.terminal_reward = terminal_rewards___no_isosceles_triangle[args.n]
        elif args.reward_function == 9998:
            args.terminal_reward = terminal_rewards___no_three_in_line[args.n]
        elif args.reward_function == 9997:
            args.terminal_reward = terminal_rewards___no_right_triangles[args.n]
        elif args.reward_function == 9996:
            args.terminal_reward = terminal_rewards___golomb_ruler[args.n]
        elif args.reward_function == 9995:
            args.terminal_reward = terminal_rewards___no_parallelograms[args.n]
        else:
            print(" \n\n\t\t Playing grid game but reward conjecture is wrong")
            exit()


    train_CEM_graph_agent(args)












