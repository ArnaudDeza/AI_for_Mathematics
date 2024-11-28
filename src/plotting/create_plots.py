import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

















def plot_graph_layouts_for_counter_example(adj_matrix, reward, output_folder, is_directed, node_size=500, font_size=10, edge_thickness=1.0, output_file_name=None, plot_title=None):
    """
    Plots the given graph in multiple layouts and saves it as a single figure.
    
    Parameters:
    adj_matrix (numpy.ndarray): Adjacency matrix of the graph
    reward (float): Scalar reward value for the graph
    output_folder (str): Path to the output folder to save the figure
    is_directed (bool): Whether the graph is directed or undirected
    node_size (int): Size of the nodes
    font_size (int): Font size for node labels
    edge_thickness (float): Thickness of the edges
    output_file_name (str): Name of the output file (optional)
    plot_title (str): Title of the plot (optional)
    """
    # Create the graph from adjacency matrix
    if is_directed:
        G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph())
    else:
        G = nx.from_numpy_array(adj_matrix)

    # Define the different layout functions to use
    layouts = {
        'Spring Layout': nx.spring_layout,
        'Circular Layout': nx.circular_layout,
        'Spectral Layout': nx.spectral_layout,
        'Shell Layout': nx.shell_layout,
        'Kamada-Kawai Layout': nx.kamada_kawai_layout,
        
    }

    # Prepare the output folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Create the figure with multiple subplots
    num_layouts = len(layouts)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    if plot_title is None:
        plot_title = f"Graph with Reward: {reward}"
    fig.suptitle(plot_title, fontsize=16)

    # Flatten axes for easy iteration
    axes = axes.flatten()

    # Plot the graph with each layout
    for ax, (layout_name, layout_func) in zip(axes, layouts.items()):
        pos = layout_func(G)
        ax.set_title(layout_name)
        if is_directed:
            nx.draw(G, pos, ax=ax, with_labels=True, node_color='lightblue', edge_color='gray', node_size=node_size, font_size=font_size, width=edge_thickness, arrows=True)
        else:
            nx.draw(G, pos, ax=ax, with_labels=True, node_color='lightblue', edge_color='gray', node_size=node_size, font_size=font_size, width=edge_thickness)

    # Hide any unused subplots
    for i in range(len(layouts), len(axes)):
        fig.delaxes(axes[i])

    # Save the figure to the specified output folder
    if output_file_name is None:
        output_file_name = f'graph_layouts_reward_{reward}.png'
    output_file = os.path.join(output_folder, output_file_name)
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()