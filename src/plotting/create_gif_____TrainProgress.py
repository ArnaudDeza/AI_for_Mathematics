




import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

# Function to plot adjacency matrix in different ways
def plot_adjacency_matrix(ax, adj_matrix, directed=False, plot_type='matshow'):
    """
    Plot the adjacency matrix in different formats.

    Parameters:
    - ax: Axis object for plotting.
    - adj_matrix: The adjacency matrix to plot.
    - directed: Boolean indicating if the graph is directed.
    - plot_type: Type of plot ('matshow', 'numbers', 'mixed').
    """
    if plot_type == 'matshow':
        # Plot the adjacency matrix using matshow.
        if directed:
            ax.matshow(adj_matrix, cmap='Blues')
        else:
            ax.matshow(np.triu(adj_matrix), cmap='Blues')
    elif plot_type == 'numbers':
        # Plot the adjacency matrix using numbers in each cell.
        ax.clear()
        ax.set_title('Adjacency Matrix', fontsize=20)
        ax.set_xlabel('Nodes', fontsize=16)
        ax.set_ylabel('Nodes', fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=12)
        n = len(adj_matrix)
        # Loop through rows and columns to display each number.
        for i in range(n):
            for j in range(i, n) if not directed else range(n):
                ax.text(j, i, f'{adj_matrix[i][j]}', ha='center', va='center', color='black')
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(n - 0.5, -0.5)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
    elif plot_type == 'mixed':
        # Plot both matrix values and color map for visual representation.
        ax.matshow(np.triu(adj_matrix) if not directed else adj_matrix, cmap='Blues')
        n = len(adj_matrix)
        for i in range(n):
            for j in range(i, n) if not directed else range(n):
                ax.text(j, i, f'{adj_matrix[i][j]}', ha='center', va='center', color='black')
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(n - 0.5, -0.5)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))

# Function to plot a graph with a specified layout
def plot_graph_with_layout(ax, G, layout_func, i, score,directed = False, highlight=False):
    """
    Plot the graph using a given layout function.

    Parameters:
    - ax: Axis object for plotting.
    - G: Graph to plot.
    - layout_func: Function for determining the layout of nodes.
    - i: Iteration number to display in the title.
    - score: Reward or score for the graph.
    - highlight: Whether to highlight nodes and edges.
    """
    pos = layout_func(G)  # Get positions of nodes based on layout function
    # Set node and edge colors based on whether to highlight or not
    node_colors = ['green' if highlight else 'salmon'] * len(G.nodes)
    edge_colors = ['green' if highlight else 'salmon'] * len(G.edges)

    if directed:
        # Draw nodes and labels
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=700)
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=15)
        # Draw edges with arrows for directed graphs
        nx.draw_networkx_edges(
            G, pos, ax=ax, edge_color=edge_colors,
            arrowstyle='-|>', arrowsize=20 if isinstance(G, nx.DiGraph) else 0,
            connectionstyle='arc3,rad=0.1'  # Optional for curved edges
        )

    else:
        # Draw the graph with specified properties
        nx.draw(G, pos, ax=ax, with_labels=True, node_color=node_colors, edge_color=edge_colors, node_size=700, font_size=15)
    # Set title based on score
    if score <= 0.0:
        ax.set_title(f'Graph @ iter = {i} with score = {round(score, 5)}', fontsize=23)
    else:
        ax.set_title(f'Found Counter-example @ iteration = {i} !! \n score is {round(score, 8)} !!', fontsize=26, color='green')
    ax.axis('off')

# Function to create a GIF showing the progression of reward graphs
def create_reward_graph_gif(graph_data,
                            output_file='reward_graph_progression.gif',
                            layout_func=nx.spring_layout,
                            fps=2,
                            interval=500,
                            directed=False,
                            adj_plot_type='matshow',
                            bottom_right_plot_data=None,
                            hold_frames=10):
    """
    Create a GIF showing the progression of reward graphs over iterations.

    Parameters:
    - graph_data: List of tuples containing (adjacency matrix, reward, iteration).
    - output_file: Output file name for the GIF.
    - layout_func: Function to use for graph layout (default is spring layout).
    - fps: Frames per second for the GIF.
    - interval: Interval between frames in milliseconds.
    - directed: Boolean indicating if the graph is directed or not.
    - adj_plot_type: Type of plot for adjacency matrix ('matshow', 'numbers', 'mixed').
    - bottom_right_plot_data: Dictionary containing data to compare rewards.
    - hold_frames: Number of frames to hold the last frame with zooming in effect.
    """
    # Set up figure and axes for subplots
    fig, axes = plt.subplots(2, 2, figsize=(24, 16))
    graph_ax, reward_ax, adj_matrix_ax, reward_comparison_ax = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    # Add a title for the entire figure with LaTeX math equation
    fig.suptitle(r"Graph Analysis and Reward Plotting: Exploring Conjecture $\lambda_1 + \lambda_2 \leq n$",
                 fontsize=32, y=0.98, ha='center')

    # Extract rewards and iterations from graph_data
    rewards = [data[1] for data in graph_data]
    iterations = [data[2] for data in graph_data]

    total_frames = len(graph_data) + hold_frames

    # Function to update each frame in the animation
    def update(frame):
        # Clear all axes for the next frame
        graph_ax.clear()
        reward_ax.clear()
        adj_matrix_ax.clear()
        reward_comparison_ax.clear()

        if frame < len(graph_data):
            # Extract adjacency matrix, reward, and iteration for the current frame
            adj_matrix, reward, iteration = graph_data[frame]
            G = create_graph_from_adj_matrix(adj_matrix, directed=directed)
            highlight = reward > 0  # Highlight nodes/edges if reward indicates a valid counter-example

            # Plot the graph on the top left
            plot_graph_with_layout(graph_ax, G, layout_func, iteration, reward,directed=directed, highlight=highlight)

            # Plot reward vs iteration on the top right
            reward_ax.plot(iterations[:frame + 1], rewards[:frame + 1], color='b', marker='o', linestyle='-', linewidth=2, label='Reward')
            reward_ax.set_xlim(0, max(iterations) + 3)
            reward_ax.set_ylim(min(rewards) - 0.5, max(rewards) + 0.3)
        else:
            # Hold the last frame and zoom in on the y-axis
            adj_matrix, reward, iteration = graph_data[-1]
            G = create_graph_from_adj_matrix(adj_matrix, directed=directed)
            highlight = reward > 0

            # Plot the graph on the top left
            plot_graph_with_layout(graph_ax, G, layout_func, iteration, reward,directed=directed, highlight=highlight)

            # Plot reward vs iteration on the top right
            reward_ax.plot(iterations, rewards, color='b', marker='o', linestyle='-', linewidth=2, label='Reward')
            reward_ax.set_xlim(0, max(iterations) + 3)
            # Adjust the y-axis limits to zoom in over time
            zoom_factor = (frame - len(graph_data) + 1) / hold_frames
            new_ymin = min(rewards) + zoom_factor * (max(rewards) - min(rewards)) * 0.5
            reward_ax.set_ylim(new_ymin, max(rewards) + 0.3)

        # Set labels, title, and grid for reward vs iteration plot
        reward_ax.set_xlabel('Iteration', fontsize=19)
        reward_ax.set_ylabel('Reward of Best Construction', fontsize=19)
        reward_ax.set_title('Reward of Best Construction vs Iteration', fontsize=20)
        reward_ax.tick_params(axis='both', which='major', labelsize=14)
        reward_ax.axhline(0, color='red', linestyle='--', linewidth=3, label='Counter-example found')
        reward_ax.legend(loc='lower right', fontsize=18, shadow=True, fancybox=True)
        reward_ax.grid(True)

        # Plot the adjacency matrix on the bottom left
        plot_adjacency_matrix(adj_matrix_ax, adj_matrix, directed=directed, plot_type=adj_plot_type)
        #plot_heatmap_of_upper_triangle(adj_matrix_ax,[data[0] for data in graph_data] )
        adj_matrix_ax.set_title('Adjacency Matrix', fontsize=20)
        adj_matrix_ax.set_xlabel('Nodes', fontsize=16)
        adj_matrix_ax.set_ylabel('Nodes', fontsize=16)
        adj_matrix_ax.tick_params(axis='both', which='major', labelsize=14)

        # Plot reward comparison on the lower right
        if bottom_right_plot_data:
            colors = ['g', 'orange', 'purple', 'brown']
            markers = ['o', 's', '^', 'D']
            for idx, (key, value) in enumerate(bottom_right_plot_data.items()):
                reward_comparison_ax.plot(iterations[:min(frame + 1, len(graph_data))], value[:min(frame + 1, len(graph_data))], color=colors[idx], marker=markers[idx], linestyle='-', linewidth=2, label=key)
            reward_comparison_ax.set_xlabel('Iteration', fontsize=16)
            reward_comparison_ax.set_ylabel('Reward Value', fontsize=16)
            reward_comparison_ax.set_title('Comparison of Rewards', fontsize=20)
            reward_comparison_ax.axhline(0, color='red', linestyle='--', linewidth=3, label='Counter-example found')
            reward_comparison_ax.set_xlim(0, max(iterations) + 5)
            # Set y-axis limits based on reward data
            y_lim = (min(np.min(value) for value in bottom_right_plot_data.values()) - 1,
                     max(np.max(value) for value in bottom_right_plot_data.values()) + 1)
            reward_comparison_ax.set_ylim(y_lim)
            reward_comparison_ax.legend(loc='lower right', fontsize=18, shadow=True, fancybox=True)
            reward_comparison_ax.grid(True)
            reward_comparison_ax.tick_params(axis='both', which='major', labelsize=12)

    # Create and save the animation as a GIF
    ani = FuncAnimation(fig, update, frames=total_frames, repeat=True, interval=interval)
    ani.save(output_file, writer=PillowWriter(fps=fps))
    plt.close()



# Function to create NetworkX graph from an adjacency matrix
def create_graph_from_adj_matrix(adj_matrix, directed=False):
    G = nx.DiGraph() if directed else nx.Graph()
    n = len(adj_matrix)
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n) if not directed else range(n):
            if adj_matrix[i][j] == 1:
                G.add_edge(i, j)
    return G