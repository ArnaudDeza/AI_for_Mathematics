import gymnasium
from gymnasium import spaces
import numpy as np
from typing import Optional

class LinearEnvironment(gymnasium.Env):
    """
    LinearEnvironment is a Gymnasium environment where an agent iteratively decides
    whether to include or exclude edges in a graph, one edge at a time, in a predefined order.

    The environment supports both undirected and directed graphs and allows customization
    of various parameters such as the initial graph, reward function, etc.
    """


    def __init__(self, config: Optional[dict] = None):
        config = config or {}
        # Initialize parameters
        self.number_of_nodes = config.get("number_of_nodes", 19)
        self.value_fun = config.get("value_fun", None)
        self.check_every = config.get("check_at_every_step", False)
        self.dense_reward = config.get("dense_reward", False)
        self.init = config.get("init_graph", None)
        self.start_with_complete_graph = config.get("start_with_complete_graph", True)
        self.self_loops = config.get("self_loops", False)
        self.verbose = config.get("verbose", True)
        self.directed = config.get("directed", False)
        self.time_horizon = config.get("time_horizon", None)

        # Determine the number of edges based on graph type
        if self.directed:
            if self.self_loops:
                self.number_of_edges = self.number_of_nodes * self.number_of_nodes
            else:
                self.number_of_edges = self.number_of_nodes * (self.number_of_nodes - 1)
        else:
            if self.self_loops:
                self.number_of_edges = self.number_of_nodes * (self.number_of_nodes + 1) // 2
            else:
                self.number_of_edges = self.number_of_nodes * (self.number_of_nodes - 1) // 2

        # Action space:
        # - Undirected Graphs: Discrete(2) -> 0: no edge, 1: edge exists
        # - Directed Graphs: Discrete(3) -> 0: no edge, 1: edge from i to j, 2: edge from j to i
        if self.directed:
            self.action_space = spaces.Discrete(3)
        else:
            self.action_space = spaces.Discrete(2)

        # Observation space: Flattened adjacency matrix
        # For undirected graphs without self-loops: number_of_edges
        # For directed graphs: number_of_edges
        self.observation_space = spaces.MultiBinary(self.number_of_edges*2)

        self.best_score_ever = -np.inf
        self.current_edge_index = 0  # Index of the current edge in the predefined order

        # Precompute the list of edges in the predefined order
        self.edges = self._generate_edge_list()

        # Reset the environment to the initial state
        self.reset()

    def _generate_edge_list(self):
        """
        Generate the list of edges in a predefined order based on the graph type.

        Returns:
        - edges (list of tuples): List of edge indices (i, j).
        """
        edges = []
        if self.directed:
            # For directed graphs, include all ordered pairs (i, j)
            for i in range(self.number_of_nodes):
                for j in range(self.number_of_nodes):
                    if not self.self_loops and i == j:
                        continue
                    edges.append((i, j))
        else:
            # For undirected graphs, include upper triangle (including or excluding diagonal)
            for i in range(self.number_of_nodes):
                if self.self_loops:
                    start_j = i
                else:
                    start_j = i + 1
                for j in range(start_j, self.number_of_nodes):
                    edges.append((i, j))
        return edges

    def state_to_observation(self):
        """
        Convert the current state to an observation.

        Returns:
        - observation (np.ndarray): Flattened adjacency matrix.
        """
        # Extract edge values in the order of self.edges
        edge_values = np.array([self.graph[i, j] for (i, j) in self.edges], dtype=np.int8)

        return edge_values
    
    def get_full_observation(self):
        """
        flatten the adjacency matrix and append a position one hot encoding of currente edge index
        Returns:
        - observation (np.ndarray): Flattened adjacency matrix with position one hot encoding.
        """
        flattened_graph = self.state_to_observation()
        position_one_hot = np.zeros((self.number_of_edges,), dtype=np.int8)
        if self.done:
            position_one_hot[self.number_of_edges - 1] = 1
        else:
            position_one_hot[self.current_edge_index] = 1
        return np.concatenate((flattened_graph, position_one_hot))


    def reset(self, *, seed=None, options=None):
        """
        Reset the environment to the initial state.

        Parameters:
        - seed (int): Random seed.
        - options (dict): Additional options.

        Returns:
        - observation (np.ndarray): The initial observation.
        - info (dict): Additional information (empty dictionary).
        """
        super().reset(seed=seed, options=options)
        
        self.done = False

        # Initialize adjacency matrix
        shape = (self.number_of_nodes, self.number_of_nodes)
        if self.init is not None:
            # Use the provided initial graph
            self.graph = np.copy(self.init)
            # Ensure the graph respects the directedness and self-loop settings
            if not self.directed:
                self.graph = np.triu(self.graph)
                self.graph += self.graph.T - np.diag(self.graph.diagonal())
            if not self.self_loops:
                np.fill_diagonal(self.graph, 0)
        else:
            if self.start_with_complete_graph:
                # Start with a complete graph
                self.graph = np.ones(shape, dtype=np.int8)
                if not self.self_loops:
                    np.fill_diagonal(self.graph, 0)  # Remove self-loops
                if not self.directed:
                    # For undirected graphs, ensure symmetry
                    self.graph = np.triu(self.graph)
                    self.graph += self.graph.T - np.diag(self.graph.diagonal())
            else:
                # Start with an empty graph
                self.graph = np.zeros(shape, dtype=np.int8)

        # Set the current edge index to 0
        self.current_edge_index = 0

        # Evaluate the initial graph
        self.old_value = self.value_fun(self.graph)

        self.best_score_in_episode = -np.inf

        # Return the initial observation and info
        observation = self.get_full_observation()
        info = {}
        return observation,info

    def step(self, action):
        """
        Take an action in the environment.

        Parameters:
        - action (int): The action to take

        Returns:
        - observation (np.ndarray): The new observation.
        - reward (float): The reward received.
        - done (bool): Whether the episode has ended.
        - truncated (bool): Whether the episode was truncated (always False here).
        - info (dict): Additional information (empty dictionary).
        """
        info = {}

        # Check if the episode is already done
        if self.done:
            # If the episode has ended, return the current state
            observation = self.get_full_observation()
            return observation, 0.0, self.done, False, info

        # Get the current edge
        if self.current_edge_index >= self.number_of_edges:
            # If all edges have been processed, the episode should be done
            self.done = True
            observation = self.get_full_observation()
            new_value = self.value_fun(self.graph)
            # Compute the final reward
            if self.dense_reward:
                self.last_reward = new_value - self.old_value
            else:
                self.last_reward = new_value
            self.old_value = new_value
            return observation, self.last_reward, self.done, False, info

        i, j = self.edges[self.current_edge_index]

        # Apply the action based on graph type
        if self.directed:
            if action == 0:
                # No edge
                self.graph[i, j] = 0
            elif action == 1:
                # Edge from i to j
                self.graph[i, j] = 1
            elif action == 2:
                # Edge from j to i
                self.graph[j, i] = 1
            else:
                raise ValueError(f"Invalid action {action} for directed graph.")
        else:
            if action == 0:
                # No edge
                self.graph[i, j] = 0
                self.graph[j, i] = 0  # Ensure symmetry
            elif action == 1:
                # Edge exists
                self.graph[i, j] = 1
                self.graph[j, i] = 1  # Ensure symmetry
            else:
                raise ValueError(f"Invalid action {action} for undirected graph.")

        # Evaluate the new graph
        new_value = self.value_fun(self.graph)

        # Compute the reward
        if self.dense_reward:
            # Use the difference in value as reward
            self.last_reward = new_value - self.old_value
        else:
            self.last_reward = 0.0  # No reward until the episode ends

        self.old_value = new_value

        # Update best scores
        if new_value > self.best_score_ever:
            self.best_score_ever = new_value
        if new_value > self.best_score_in_episode:
            self.best_score_in_episode = new_value

        # Optionally check for termination at every step
        if self.check_every:
            # Assuming that a "counterexample" is identified when value exceeds a threshold
            if new_value > 1e-12:
                self.done = True

        # Move to the next edge
        self.current_edge_index += 1
        if self.current_edge_index >= self.number_of_edges:
            self.done = True
            # Compute the final reward if not using dense rewards
            if not self.dense_reward:
                self.last_reward = new_value  # Final reward at the end of episode

        # Get the new observation
        observation = self.get_full_observation()

        # Optionally print verbose information
        if self.verbose and self.done:
            print(f"best_score_ever={self.best_score_ever}, "
                  f"best_score_in_episode={self.best_score_in_episode}, "
                  f"final_score={new_value}")

        info['reward'] = self.last_reward
        info['terminated'] = self.done
        
        return observation, self.last_reward, self.done, False, info

    def render(self, mode='human'):
        """
        Render the environment (optional).

        Parameters:
        - mode (str): The mode in which to render the environment.
        """
        if mode == 'human':
            print("Current Graph Adjacency Matrix:")
            print(self.graph)
        else:
            super().render(mode=mode)

    def close(self):
        """
        Perform any necessary cleanup (optional).
        """
        pass

