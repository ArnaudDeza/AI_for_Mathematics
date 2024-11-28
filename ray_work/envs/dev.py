import numpy as np
import gymnasium
from gymnasium import spaces
from typing import Optional, Dict, Any, Tuple, List

import networkx as nx
import matplotlib.pyplot as plt


class FlipEnvironment(gymnasium.Env):
    """
    FlipEnvironment is a Gymnasium environment where an agent flips edges in a graph
    to find counterexamples for graph theory conjectures.

    At every step, the agent selects an edge to flip its status (add/remove).
    The observation includes the flattened adjacency matrix and the sequence of actions taken so far.

    The environment supports both undirected and directed graphs and allows customization
    of various parameters such as the initial graph, reward function, etc.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}

        # Initialize parameters
        self.number_of_nodes: int = config.get("number_of_nodes", 19)  # Number of nodes in the graph
        self.value_fun = config.get("value_fun", None)  # Function to calculate the value/reward of the graph
        self.check_every: bool = config.get("check_at_every_step", False)  # Check for counterexamples at every step
        self.dense_reward: bool = config.get("dense_reward", False)  # Use the difference in value as reward at each step
        self.init: Optional[np.ndarray] = config.get("init_graph", None)  # Initial graph
        self.start_with_complete_graph: bool = config.get("start_with_complete_graph", True)  # Start with complete or empty graph
        self.self_loops: bool = config.get("self_loops", False)  # Allow self-loops
        self.verbose: bool = config.get("verbose", True)  # Verbose output
        self.directed: bool = config.get("directed", False)  # Directed or undirected graph
        self.time_horizon: Optional[int] = config.get("time_horizon", None)  # Episode length

        # Ensure that value_fun is provided
        if self.value_fun is None:
            raise ValueError("A 'value_fun' must be provided in the config.")

        # Determine the number of possible edges based on graph type
        if self.directed:
            # Directed graph
            if self.self_loops:
                self.number_of_edges = self.number_of_nodes ** 2
                self.action_to_edge: List[Tuple[int, int]] = [
                    (i, j) for i in range(self.number_of_nodes) for j in range(self.number_of_nodes)
                ]
            else:
                self.number_of_edges = self.number_of_nodes * (self.number_of_nodes - 1)
                self.action_to_edge = [
                    (i, j)
                    for i in range(self.number_of_nodes)
                    for j in range(self.number_of_nodes)
                    if i != j
                ]
        else:
            # Undirected graph
            if self.self_loops:
                self.number_of_edges = self.number_of_nodes * (self.number_of_nodes + 1) // 2
                self.action_to_edge = [
                    (i, j)
                    for i in range(self.number_of_nodes)
                    for j in range(i, self.number_of_nodes)
                ]
            else:
                self.number_of_edges = self.number_of_nodes * (self.number_of_nodes - 1) // 2
                self.action_to_edge = [
                    (i, j)
                    for i in range(self.number_of_nodes)
                    for j in range(i + 1, self.number_of_nodes)
                ]

        # Action space is the number of edges in the graph
        self.action_space = spaces.Discrete(self.number_of_edges)

        # Observation space includes adjacency matrix and action sequence
        # The action sequence is fixed to 'stop' length, padded with 'number_of_edges' to indicate no action
        self.stop: int = self.time_horizon if self.time_horizon is not None else self.number_of_edges
        self.observation_space = spaces.Dict({
            'adjacency': spaces.MultiBinary(self.number_of_nodes * self.number_of_nodes),
            'action_sequence': spaces.Box(
                low=0,
                high=self.number_of_edges,
                shape=(self.stop,),
                dtype=np.int32
            )
        })

        # Initialize other variables
        self.last_reward: float = 0.0
        self.current: Tuple[int, int] = (0, 0)
        self.best_score_ever: float = -np.inf
        self.best_score_in_episode: float = -np.inf
        self.action_sequence: List[int] = []

        # Reset the environment to the initial state
        self.reset()

    def state_to_observation(self) -> Dict[str, Any]:
        """
        Convert the current state to an observation dictionary.

        Returns:
            Dict[str, Any]: Dictionary containing the flattened adjacency matrix and action sequence.
        """
        adjacency_flat = self.graph.flatten()
        # Create a fixed-length action sequence with padding
        action_seq = np.full(self.stop, self.number_of_edges, dtype=np.int32)  # Padding value is self.number_of_edges
        for idx, action in enumerate(self.action_sequence):
            if idx < self.stop:
                action_seq[idx] = action
            else:
                break  # Only keep up to 'stop' actions
        return {
            'adjacency': adjacency_flat,
            'action_sequence': action_seq
        }

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Reset the environment to the initial state.

        Parameters:
            seed (Optional[int]): Random seed.
            options (Optional[Dict[str, Any]]): Additional options.

        Returns:
            Tuple[Dict[str, Any], Dict[str, Any]]: The initial observation and info.
        """
        super().reset(seed=seed)

        # Initialize the adjacency matrix
        if self.init is not None:
            if not isinstance(self.init, np.ndarray):
                raise ValueError("'init_graph' must be a numpy ndarray.")
            if self.init.shape != (self.number_of_nodes, self.number_of_nodes):
                raise ValueError(f"'init_graph' must have shape ({self.number_of_nodes}, {self.number_of_nodes}).")
            self.graph = self.init.astype(np.int8).copy()
            if not self.directed:
                # Ensure symmetry for undirected graphs
                self.graph = np.triu(self.graph)
                self.graph += self.graph.T - np.diag(self.graph.diagonal())
            if not self.self_loops:
                np.fill_diagonal(self.graph, 0)  # Remove self-loops if not allowed
        else:
            if self.start_with_complete_graph:
                # Start with a complete graph
                self.graph = np.ones((self.number_of_nodes, self.number_of_nodes), dtype=np.int8)
                if not self.self_loops:
                    np.fill_diagonal(self.graph, 0)  # Remove self-loops
                if not self.directed:
                    # For undirected graphs, ensure symmetry
                    self.graph = np.triu(self.graph)
                    self.graph += self.graph.T - np.diag(self.graph.diagonal())
            else:
                # Start with an empty graph
                self.graph = np.zeros((self.number_of_nodes, self.number_of_nodes), dtype=np.int8)

        # Initialize counters and rewards
        self.timestep_it: int = 0
        self.done: bool = False
        self.old_value: float = self.value_fun(self.graph)
        self.best_score_ever: float = self.old_value
        self.best_score_in_episode: float = self.old_value
        self.action_sequence: List[int] = []

        # Return the initial observation and info
        observation = self.state_to_observation()
        info: Dict[str, Any] = {}
        return observation, info

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """
        Take an action in the environment.

        Parameters:
            action (int): The action to take (edge to flip).

        Returns:
            Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]: Observation, reward, done, truncated, info.
        """
        info: Dict[str, Any] = {}

        # Validate action
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}. Action must be in [0, {self.number_of_edges - 1}].")

        # Get the edge to flip
        i, j = self.action_to_edge[action]
        self.current = (i, j)
        self.action_sequence.append(action)

        # Log the action with direction if verbose
        if self.verbose:
            direction = "self-loop" if i == j else f"{i}→{j}"
            print(f"Action taken: Flip edge {direction} (action index: {action})")

        # Check for self-loops if they are not allowed
        if not self.self_loops and i == j:
            if self.verbose:
                print(f"Invalid move: trying to flip a self-loop at node {i}. Action {action} is ignored.")
            # Do not change the graph or timestep
            observation = self.state_to_observation()
            reward = self.last_reward
            return observation, reward, self.done, False, info

        # Flip the edge
        self.graph[i, j] = 1 - self.graph[i, j]
        if not self.directed:
            # Ensure symmetry for undirected graphs
            self.graph[j, i] = self.graph[i, j]

        # Increment the timestep
        self.timestep_it += 1

        # Check if the episode should end
        if self.timestep_it >= self.stop:
            self.done = True

        # Get the new observation
        observation = self.state_to_observation()

        # Evaluate the new graph
        new_value = self.value_fun(self.graph)

        # Optionally check for counterexamples at every step
        if self.check_every and self.timestep_it < self.stop:
            if new_value > 1e-12:
                self.done = True
                if self.verbose:
                    print(f"Counterexample found at step {self.timestep_it} with value {new_value}.")

        # Calculate the reward
        if self.dense_reward:
            # Use the difference in value as reward
            reward = new_value - self.old_value
        elif self.done:
            # Provide the final reward at the end of the episode
            reward = new_value
        else:
            # No reward until the episode ends
            reward = 0.0

        # Update the old value
        self.old_value = new_value

        # Update the best scores
        if new_value > self.best_score_ever:
            self.best_score_ever = new_value
        if new_value > self.best_score_in_episode:
            self.best_score_in_episode = new_value

        # Optionally print verbose information
        if self.verbose and self.done:
            print(f"Episode done. best_score_ever={self.best_score_ever}, "
                  f"best_score_in_episode={self.best_score_in_episode}, final_score={new_value}")

        self.last_reward = reward

        return observation, reward, self.done, False, info

    def get_edge_from_action(self, action: int) -> Tuple[int, int]:
        """
        Retrieve the (i, j) edge corresponding to a given action.

        Parameters:
            action (int): The action index.

        Returns:
            Tuple[int, int]: The (i, j) edge indices.
        """
        if not self.action_space.contains(action):
            raise ValueError(f"Action {action} is out of bounds.")
        return self.action_to_edge[action]








def visualize_graph(graph: np.ndarray, directed: bool):
    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    
    G.add_nodes_from(range(len(graph)))
    
    for i in range(len(graph)):
        for j in range(len(graph)):
            if graph[i, j]:
                G.add_edge(i, j)
    
    pos = nx.spring_layout(G)
    plt.figure(figsize=(8, 6))
    if directed:
        nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=20)
    else:
        nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_nodes(G, pos, node_size=700)
    nx.draw_networkx_labels(G, pos)
    plt.title("Directed Graph" if directed else "Undirected Graph")
    plt.axis('off')
    plt.show()







def example_value_fun(graph: np.ndarray) -> float:
    # Example value function: count the number of edges
    return np.sum(graph)




if __name__ == "__main__":
    config = {
        "number_of_nodes": 5,
        "value_fun": example_value_fun,
        "directed": False,
        "self_loops": False,
        "start_with_complete_graph": False,
        "dense_reward": True,
        "verbose": True,
        "time_horizon": 10
    }

    env = FlipEnvironment(config)

    observation, info = env.reset()
    print("Initial Observation:", observation)

    done = False
    total_reward = 0.0
    while not done:
        action = env.action_space.sample()  # Random action for demonstration
        observation, reward, done, truncated, info = env.step(action)
        total_reward += reward
        print(f"Action taken: {action}, Reward: {reward}, Done: {done}")
        print("Current Adjacency Matrix:\n", env.graph)

    print(f"Total Reward: {total_reward}")
    # Example usage after taking some actions
    visualize_graph(env.graph, env.directed)
