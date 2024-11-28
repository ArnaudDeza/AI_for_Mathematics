import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
import imageio
import os
from matplotlib import gridspec

def create_rl_graph_gif(n=5, fps=1, output_folder='output', output_filename='graph_conjecture_rl.gif',
                        node_layout='spring', layout_seed=42, highlight_existing_edges=True, colors=None,
                        state_vector_colors=None, adjacency_matrix=None, reward=None, gif_title='RL Graph Visualization',
                        extra_frames=25):
    """
    Create a GIF visualizing an RL environment where a graph is built edge by edge.

    Parameters:
    - n (int): Number of nodes in the graph.
    - fps (int): Frames per second for the GIF speed.
    - output_folder (str): Folder to save the output GIF.
    - output_filename (str): Name of the output GIF file.
    - node_layout (str): Type of node layout during edge addition ('spring', 'circular', 'random', 'fixed').
    - layout_seed (int): Seed for layout reproducibility.
    - highlight_existing_edges (bool): Whether to plot all possible edges with a light shade and highlight existing edges.
    - colors (dict): Dictionary specifying colors for nodes, existing edges, and possible edges.
    - state_vector_colors (dict): Dictionary specifying colors for state vector components.
    - adjacency_matrix (np.ndarray): User-provided adjacency matrix.
    - reward (float): Reward associated with the provided adjacency matrix.
    - gif_title (str): Title of the whole GIF.
    - extra_frames (int): Number of additional frames to display after the environment steps.
    """
    if adjacency_matrix is not None:
        n = adjacency_matrix.shape[0]
    else:
        pass  # n is as given

    iterations = int(n * (n - 1) / 2)  # Total number of edges to consider

    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    output_path = os.path.join(output_folder, output_filename)

    # Default colors if not provided
    if colors is None:
        colors = {
            'node_color': 'skyblue',
            'existing_edge_color': 'black',
            'possible_edge_color': 'gray',
            'highlight_edge_color': 'green',
            'undecided_color': 'lightblue',
            'decision_1_color': 'salmon',
            'decision_0_color': 'lightcoral'
        }

    # Default state vector colors if not provided
    if state_vector_colors is None:
        state_vector_colors = {
            'upper_triangle_color': 'Blues',
            'positional_encoding_color': 'Oranges'
        }

    # Creating the graph and initializing the adjacency matrix for plotting
    G = nx.Graph()
    G.add_nodes_from(range(n))
    adjacency_matrix_plot = np.full((n, n), np.nan)  # Initialize with NaN to differentiate undecided edges

    # Possible edges (loop over all possible edges in an undirected graph)
    possible_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_edges = len(possible_edges)

    # Actions list based on adjacency matrix or random actions
    if adjacency_matrix is not None:
        upper_triangle_indices = np.triu_indices(n, k=1)
        actions_list = adjacency_matrix[upper_triangle_indices].astype(int).flatten()
    else:
        actions_list = np.random.choice([0, 1], size=num_edges)

    # Create a gif to visualize the RL playthrough
    gif_images = []

    # Define fixed layout if specified
    if node_layout == 'fixed':
        pos_fixed = {i: (np.cos(2 * np.pi * i / n), np.sin(2 * np.pi * i / n)) for i in range(n)}

    # Iterate over all possible edges to simulate the RL agent's decisions
    for iteration, (edge, action) in enumerate(zip(possible_edges, actions_list)):
        if action == 1:
            G.add_edge(*edge)
            adjacency_matrix_plot[edge[0], edge[1]] = 1
            adjacency_matrix_plot[edge[1], edge[0]] = 1
        else:
            adjacency_matrix_plot[edge[0], edge[1]] = 0
            adjacency_matrix_plot[edge[1], edge[0]] = 0

        # Create the state vector
        # First half: Flattened upper triangle of the current adjacency matrix
        upper_triangle_indices = np.triu_indices(n, k=1)
        state_vector_upper_triangle = adjacency_matrix_plot[upper_triangle_indices].flatten()
        # Second half: Positional encoding for the current edge being considered
        state_vector_positional = np.zeros(num_edges)
        edge_index = iteration  # Since we're looping over possible_edges in order
        state_vector_positional[edge_index] = 1

        # Plotting the current state
        fig = plt.figure(figsize=(15, 12))
        spec = gridspec.GridSpec(ncols=2, nrows=3, figure=fig, height_ratios=[2, 0.5, 0.5], width_ratios=[1, 1], hspace=0.5)

        # Plot the current graph
        ax0 = fig.add_subplot(spec[0, 0])
        if node_layout == 'spring':
            pos = nx.spring_layout(G, seed=layout_seed)
        elif node_layout == 'circular':
            pos = nx.circular_layout(G)
        elif node_layout == 'random':
            pos = nx.random_layout(G, seed=layout_seed)
        elif node_layout == 'fixed':
            pos = pos_fixed
        else:
            raise ValueError("Unsupported node layout. Choose from 'spring', 'circular', 'random', or 'fixed'.")

        # Draw edges and nodes
        if highlight_existing_edges:
            # Draw all possible edges with light color
            nx.draw_networkx_edges(G, pos, edgelist=possible_edges, alpha=0.1, edge_color=colors['possible_edge_color'], ax=ax0)
            # Draw existing edges with deep color
            nx.draw_networkx_edges(G, pos, edgelist=G.edges(), edge_color=colors['existing_edge_color'], ax=ax0)
        else:
            nx.draw(G, pos, with_labels=True, node_color=colors['node_color'], edge_color=colors['existing_edge_color'], node_size=500, ax=ax0)

        # Highlight the edge being considered
        nx.draw_networkx_edges(G, pos, edgelist=[edge], width=2.5, edge_color=colors['highlight_edge_color'], ax=ax0)
        nx.draw_networkx_nodes(G, pos, node_color=colors['node_color'], node_size=500, ax=ax0)
        nx.draw_networkx_labels(G, pos, ax=ax0, font_size=14)
        ax0.set_title(f'Graph at Step {iteration + 1}', fontsize=16)

        # Plot the upper triangle of the adjacency matrix
        ax1 = fig.add_subplot(spec[0, 1])
        mask = np.triu(np.ones_like(adjacency_matrix_plot, dtype=bool))
        annotated_matrix = np.where(adjacency_matrix_plot == 1, 1, np.where(adjacency_matrix_plot == 0, 0, np.nan))
        cmap = sns.color_palette([colors['undecided_color'], colors['decision_0_color'], colors['decision_1_color']])
        sns.heatmap(annotated_matrix, mask=~mask, annot=True, cmap=cmap, cbar=False, ax=ax1, vmin=-0.5, vmax=1.5,
                    linewidths=1.5, linecolor='white', annot_kws={'size': 20})

        # Highlight the current edge in the adjacency matrix
        highlight_mask = np.zeros_like(adjacency_matrix_plot, dtype=bool)
        highlight_mask[edge[0], edge[1]] = True
        sns.heatmap(adjacency_matrix_plot, mask=~highlight_mask, annot=True, cmap=[colors['highlight_edge_color']],
                    cbar=False, ax=ax1, linewidths=1.5, linecolor='white', annot_kws={'size': 20})

        ax1.set_title('Adjacency Matrix', fontsize=16)
        ax1.tick_params(axis='both', which='major', labelsize=14)

        # Plot the state vector (upper triangle)
        ax2 = fig.add_subplot(spec[1, :])
        sns.heatmap(state_vector_upper_triangle.reshape(1, -1), cmap=state_vector_colors['upper_triangle_color'],
                    cbar=False, ax=ax2, linewidths=1.5, linecolor='white', annot=True, annot_kws={'size': 18})
        ax2.set_title('State Vector - Upper Triangle', fontsize=16)
        ax2.set_yticks([])
        ax2.set_xticks(np.arange(num_edges))
        ax2.set_xticklabels(['E' + str(i + 1) for i in range(num_edges)], rotation=90, fontsize=10)
        # Highlight the current edge in the state vector (upper triangle)
        ax2.add_patch(plt.Rectangle((edge_index, 0), 1, 1, fill=False, edgecolor=colors['highlight_edge_color'], lw=3))

        # Plot the state vector (positional encoding)
        ax3 = fig.add_subplot(spec[2, :])
        sns.heatmap(state_vector_positional.reshape(1, -1), cmap=state_vector_colors['positional_encoding_color'],
                    cbar=False, ax=ax3, linewidths=1.5, linecolor='white', annot=True, annot_kws={'size': 18})
        ax3.set_title('State Vector - Positional Encoding', fontsize=16)
        ax3.set_yticks([])
        ax3.set_xticks(np.arange(num_edges))
        ax3.set_xticklabels(['P' + str(i + 1) for i in range(num_edges)], rotation=90, fontsize=10)
        # Highlight the current position in the positional encoding vector
        ax3.add_patch(plt.Rectangle((edge_index, 0), 1, 1, fill=False, edgecolor=colors['highlight_edge_color'], lw=3))

        # Add reward and iteration information
        current_reward = reward if reward is not None and iteration == num_edges - 1 else 0
        plt.suptitle(f'{gif_title}\nStep: {iteration + 1} | Reward: {current_reward}', fontsize=18)

        # Save frame to gif
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        gif_images.append(image)

        plt.close()


    if extra_frames > 0:

            # Prepare the final frame without state vectors, but with 3 different layouts
            # We will create the figure once and reuse it for extra_frames times
            fig = plt.figure(figsize=(15, 12))
            spec = gridspec.GridSpec(ncols=2, nrows=2, figure=fig, width_ratios=[1, 1], height_ratios=[2, 1])
            # First row: Graph and Adjacency Matrix
            ax0 = fig.add_subplot(spec[0, 0])
            if node_layout == 'spring':
                pos_main = nx.spring_layout(G, seed=layout_seed)
            elif node_layout == 'circular':
                pos_main = nx.circular_layout(G)
            elif node_layout == 'random':
                pos_main = nx.random_layout(G, seed=layout_seed)
            elif node_layout == 'fixed':
                pos_main = pos_fixed
            else:
                pos_main = nx.spring_layout(G, seed=layout_seed)

            nx.draw(G, pos_main, with_labels=True, node_color=colors['node_color'], edge_color=colors['existing_edge_color'], node_size=500, ax=ax0)
            ax0.set_title('Final Graph', fontsize=16)

            ax1 = fig.add_subplot(spec[0, 1])
            mask = np.triu(np.ones_like(adjacency_matrix_plot, dtype=bool))
            annotated_matrix = np.where(adjacency_matrix_plot == 1, 1, np.where(adjacency_matrix_plot == 0, 0, np.nan))
            sns.heatmap(annotated_matrix, mask=~mask, annot=True, cmap=cmap, cbar=False, ax=ax1, vmin=-0.5, vmax=1.5,
                        linewidths=1.5, linecolor='white', annot_kws={'size': 20})
            ax1.set_title('Adjacency Matrix', fontsize=16)
            ax1.tick_params(axis='both', which='major', labelsize=14)

            # Second row: 3 small subplots with different layouts
            layouts = [ 'kamada_kawai', 'spectral']
            ax_positions = [fig.add_subplot(spec[1, i]) for i in range(2)]  # Adjusted to 3 subplots
            #ax_positions.append(fig.add_subplot(spec[1, 1], projection='rectilinear'))  # Add third subplot

            for ax, layout_name in zip(ax_positions, layouts):
                if layout_name == 'spring':
                    pos = nx.spring_layout(G, seed=layout_seed)
                elif layout_name == 'kamada_kawai':
                    pos = nx.kamada_kawai_layout(G)
                elif layout_name == 'spectral':
                    pos = nx.spectral_layout(G)
                else:
                    pos = nx.spring_layout(G, seed=layout_seed)
                nx.draw(G, pos, with_labels=False, node_color=colors['node_color'], edge_color=colors['existing_edge_color'],
                        node_size=300, ax=ax)
                ax.set_title(f'Layout: {layout_name}', fontsize=14)

    plt.suptitle(f'{gif_title}\nFinal Graph | Reward: {reward if reward is not None else 0}', fontsize=18)

    # Draw the canvas once
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # Append the same frame multiple times
    for _ in range(extra_frames):
        gif_images.append(image)

    plt.close()

    # Save gif
    imageio.mimsave(output_path, gif_images, fps=fps)

    print(f'GIF saved as {output_path}')












