import networkx as nx
import random
import time
import logging
from copy import deepcopy
import matplotlib.pyplot as plt
import os
import numpy as np
from datetime import datetime
import argparse
from nmcts import NMCS_trees, NMCS_connected_graphs
from scores import *

def remove_randleaf(G):
    '''Removes a random leaf from G'''
    # Find all leaf nodes (degree == 1)
    leaves = [v for v in G.nodes() if G.degree(v) == 1]
    if not leaves:
        return None
    # Select a random leaf to remove
    leaf = random.choice(leaves)
    G.remove_node(leaf)
    return leaf

def remove_subdiv(G):
    '''Removes a random subdivision node (degree == 2) from G'''
    # Find all nodes with degree 2
    deg_2 = [v for v in G.nodes() if G.degree(v) == 2]
    if not deg_2:
        # If none, try removing a random leaf instead
        return remove_randleaf(G)
    # Select a random degree-2 node
    random_vertex = random.choice(deg_2)
    neighbors = list(G.neighbors(random_vertex))
    if len(neighbors) == 2:
        # Reconnect the neighbors directly
        G.add_edge(neighbors[0], neighbors[1])
    # Remove the selected node
    G.remove_node(random_vertex)
    return random_vertex

def AMCS(score_function, initial_graph=None, max_depth=5, max_level=5, trees_only=True, output_dir="output"):
    '''Adaptive Multi-level Monte Carlo Search (AMCS) algorithm'''
    if trees_only:
        # Use the NMCS algorithm for trees
        NMCS = NMCS_trees
    else:
        # Use the NMCS algorithm for connected graphs
        NMCS = NMCS_connected_graphs

    if initial_graph is None:
        # Generate a random tree with 5 nodes as the initial graph
        initial_graph = nx.random_tree(10)

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Set up logging to record progress
    logging.basicConfig(
        filename=os.path.join(output_dir, 'amcs.log'),
        level=logging.INFO,
        format='%(asctime)s - %(message)s'
    )
    logging.info("Starting AMCS algorithm")
    logging.info(f"Initial graph nodes: {initial_graph.number_of_nodes()}, edges: {initial_graph.number_of_edges()}")
    logging.info(f"Best score (initial): {float(score_function(initial_graph))}")
    
    # Initialize search parameters
    depth = 0
    level = 1
    min_order = initial_graph.number_of_nodes()
    current_graph = deepcopy(initial_graph)
    iteration = 0
    iteration_times = []
    
    # Main loop: continue until a satisfactory graph is found or max levels are reached
    while score_function(current_graph) <= 0 and level <= max_level:
        start_iteration_time = time.time()
        next_graph = deepcopy(current_graph)
        
        # Simplify the graph by removing nodes while it's larger than the minimum order
        while next_graph.number_of_nodes() > min_order:
            if random.random() < depth / (depth + 1):
                if random.random() < 0.5:
                    # Remove a random leaf
                    vertex = remove_randleaf(next_graph)
                else:
                    # Remove a random subdivision node
                    vertex = remove_subdiv(next_graph)
                if vertex is not None:
                    # Relabel nodes to maintain continuity
                    next_graph = nx.relabel_nodes(
                        next_graph,
                        {i: i-1 if i > vertex else i for i in next_graph.nodes()}
                    )
                else:
                    break
            else:
                break
        
        # Apply the NMCS algorithm to explore new graphs
        next_graph = NMCS(next_graph, depth, level, score_function)
        best_score = max(score_function(next_graph), score_function(current_graph))
        logging.info(f"Best score (lvl {level}, dpt {depth}): {float(best_score)}")
        
        # Save the adjacency matrix for progress tracking
        adj_matrix = nx.adjacency_matrix(next_graph).todense()
        adj_matrix_path = os.path.join(output_dir, f'progress_adjacency_matrix_iter{iteration}.npy')
        np.save(adj_matrix_path, adj_matrix)
        logging.info(f"Progress adjacency matrix saved to {adj_matrix_path}")
        iteration += 1

        # Track the time taken for this iteration
        iteration_time = time.time() - start_iteration_time
        iteration_times.append(iteration_time)
        logging.info(f"Iteration {iteration} took {iteration_time:.2f} seconds")
        
        if score_function(next_graph) > score_function(current_graph):
            # If a better graph is found, update current graph and reset depth and level
            current_graph = deepcopy(next_graph)
            depth = 0
            level = 1
        elif depth < max_depth:
            # If not, increase the depth to explore deeper
            depth += 1
        else:
            # If max depth reached, increase the level
            depth = 0
            level += 1
    score = score_function(current_graph)
    if score > 0:#0.00001:
        # If a satisfactory graph is found
        logging.info("Counterexample found GIVEN score is: {}".format(score))
        print("\n\t\t >> Counterexample found GIVEN score is: {}".format(score))
        print("\t\t >> Countereample found with {} nodes and {} edges".format(current_graph.number_of_nodes(), current_graph.number_of_edges()))
        print("\n")
        # Plot and save the graph
        plt.figure(figsize=(10, 7))
        nx.draw(current_graph, with_labels=True)
        plot_path = os.path.join(output_dir, 'counterexample_plot.png')
        plt.savefig(plot_path)
        logging.info(f"Counterexample plot saved to {plot_path}")
        
        # Save the adjacency matrix of the counterexample
        adj_matrix = nx.adjacency_matrix(current_graph).todense()
        adj_matrix_path = os.path.join(output_dir, 'counterexample_adjacency_matrix.npy')
        np.save(adj_matrix_path, adj_matrix)
        logging.info(f"Counterexample adjacency matrix saved to {adj_matrix_path}")

        # Create a progress plot of all graphs (function not defined in provided code)
        create_progress_plot(output_dir, iteration)

        return current_graph,current_graph.number_of_nodes()
    else:
        # If no satisfactory graph is found
        logging.info("No counterexample found")
        print("No counterexample found")
    
    # Create a summary log file with the results
    summary_path = os.path.join(output_dir, 'summary.txt')
    with open(summary_path, 'w') as summary_file:
        summary_file.write("AMCS Algorithm Summary\n")
        summary_file.write(f"Total iterations: {iteration}\n")
        summary_file.write(f"Total time: {sum(iteration_times):.2f} seconds\n")
        summary_file.write(f"Average time per iteration: {np.mean(iteration_times):.2f} seconds\n")
        if score_function(current_graph) > 0:
            summary_file.write("Counterexample found\n")
        else:
            summary_file.write("No counterexample found\n")
    logging.info(f"Summary log saved to {summary_path}")
    
    return current_graph,current_graph.number_of_nodes()

def create_progress_plot(output_dir, num_iterations):
    '''Create a plot with multiple subplots showing the progress of the algorithm.'''
    plt.figure(figsize=(25, 25))
    cols = 5
    rows = (num_iterations // cols) + 1
    
    for i in range(num_iterations):
        adj_matrix_path = os.path.join(output_dir, f'progress_adjacency_matrix_iter{i}.npy')
        adj_matrix = np.load(adj_matrix_path)
        G = nx.from_numpy_array(adj_matrix)
        
        plt.subplot(rows, cols, i + 1)
        pos = nx.spring_layout(G, seed=42)  # Use spring layout for better visual clarity
        nx.draw(G, pos, with_labels=True, node_size=100, font_size=8, node_color='skyblue', edge_color='gray', linewidths=0.5, font_weight='bold')
        plt.title(f'Iteration {i}', fontsize=10)
    
    plt.tight_layout()
    progress_plot_path = os.path.join(output_dir, 'progress_plot.png')
    plt.savefig(progress_plot_path, bbox_inches='tight')
    logging.info(f"Progress plot saved to {progress_plot_path}")
    #print(f"Progress plot saved to {progress_plot_path}")






def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conjecture", type=int,   default=66, help="")
    parser.add_argument("--max_depth",  type=int,   default=5, help="")
    parser.add_argument("--max_level",  type=int,   default=3, help="")
    parser.add_argument("--trees_only",  type=bool,   default=False, help="")
    parser.add_argument("--base_dir",  type=str,   default='/Users/adeza3/Desktop/PhD_year1/Courses/ISYE6740/adaptive_MCTS/undirected/results', help="")

    args = parser.parse_args()

    # Create output directory based on arguments and current date-time
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{args.base_dir}/conj_{args.conjecture}_depth_{args.max_depth}_level_{args.max_level}_treesonly_{args.trees_only}_{timestamp}"
    from score_lapace import auto_lapla_29
    conjecture_dict = {
        0: bouwer,
        1: Conj1_score,
        2: Conj2_score,
        3: Conj3_score,
        4: Conj4_score,
        5: Conj5_score,
        6: Conj6_score,
        7: Conj7_score,
        8: Conj8_score,
        9: Conj9_score,
        10: Conj10_score,
        11:reward_function,
        12:check_conjecture_already_solved,
        13:scoring_function_color,
        14:scoring_function_Melnikov,

        15:conj_graffiti_29,

        66:auto_lapla_29

    }

    seeds = list(range(3))
    
    #seeds = [42, 43, 44, 45, 46,69,6969,696969,120,121,122,234,45135,54322,245,413]
    ns = []
    for seed in seeds:

        # seed everything
        random.seed(seed)
        np.random.seed(seed)
        print("\n\n\n\n \t \t >>>>>>>> RUNNING SEED: ", seed)
        print("\n\n")


        # Run the AMCS algorithm

        start_time = time.time()
        graph,n = AMCS(conjecture_dict[args.conjecture], max_depth=args.max_depth, max_level=args.max_level, trees_only=args.trees_only, output_dir=output_dir)
        
        
        # visualize the graph
        plt.figure(figsize=(10, 7))
        nx.draw(graph, with_labels=True)
        plt.show()

        logging.info("Search time: %s seconds" % (time.time() - start_time))
        print("Search time: %s seconds" % (time.time() - start_time))
        ns.append(n)



    print("\n\n\n\n \t \t >>>>>>>> FINAL RESULTS: ")
    print("\n\n")
    print("Results: ", ns)
    print(" Length: ", len(ns))

    
if __name__ == "__main__":
    main()