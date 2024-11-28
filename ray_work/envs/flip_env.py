import numpy as np
import gymnasium

from typing import Optional


class FlipEnvironment(gymnasium.Env):
    """
    FlipEnvironment is a Gymnasium environment where an agent flips edges in a graph
    to find counterexamples for graph theory conjectures. 
    
    In Flip, each action requires the agent to select an edge and compulsorily flip it.

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

        # Determine the number of possible edges based on graph type
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

         # Define action and observation spaces
        self.action_space = gymnasium.spaces.Discrete(self.number_of_edges)
        self.observation_space = gymnasium.spaces.MultiBinary(self.number_of_nodes * self.number_of_nodes)
        
        # Set the time horizon (episode length)
        if self.time_horizon is None:
            self.stop = self.number_of_edges
        else: 
            self.stop = self.time_horizon

        # Initialize other variables
        self.last_reward = 0
        self.current = [0, 0]
        self.best_score_ever = -np.inf

        # Reset the environment to the initial state
        self.reset()

    def state_to_observation(self):
        """
        Convert the adjacency matrix to a flattened observation.

        Returns:
        - observation (np.ndarray): Flattened adjacency matrix.
        """
        return self.graph.flatten()
    
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

        # Initialize the adjacency matrix
        shape = (self.number_of_nodes, self.number_of_nodes)
        if self.init is not None:
            # Use the provided initial graph
            self.graph = np.copy(self.init)
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

        # Initialize counters and rewards
        self.timestep_it = 0
        self.done = False
        self.old_value = self.value_fun(self.graph)
        self.best_score_in_episode = -np.inf

        # Return the initial observation and info
        observation = self.state_to_observation()
        info = {}
        return observation, info
    
    def step(self, action):
        """
        Take an action in the environment.

        Parameters:
        - action (int): The action to take (edge to flip).

        Returns:
        - observation (np.ndarray): The new observation.
        - reward (float): The reward received.
        - done (bool): Whether the episode has ended.
        - truncated (bool): Whether the episode was truncated (always False here).
        - info (dict): Additional information (empty dictionary).
        """
        info = {}

        # Map the action to edge indices
        if self.directed:
            # For directed graphs, action maps to (i, j)
            i = action // self.number_of_nodes
            j = action % self.number_of_nodes
        else:
            # For undirected graphs, action maps to upper triangle indices
            if self.self_loops:
                idx = np.triu_indices(self.number_of_nodes)
            else:
                idx = np.triu_indices(self.number_of_nodes, k=1)
            if action >= len(idx[0]):
                raise ValueError("Invalid action for undirected graph.")
            i = idx[0][action]
            j = idx[1][action]

        self.current = [i, j]

        # Check for self-loops if they are not allowed
        if not self.self_loops and i == j:
            if self.verbose:
                print("Invalid move: trying to add a self-loop")
            # Return current state without changing anything
            observation = self.state_to_observation()
            return observation, self.last_reward, self.done, False, info
        else:
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

            # Calculate the reward
            if self.dense_reward:
                # Use the difference in value as reward
                self.last_reward = new_value - self.old_value
            elif self.done:
                # Provide the final reward at the end of the episode
                self.last_reward = new_value
            else:
                # No reward until the episode ends
                self.last_reward = 0

            # Update the old value
            self.old_value = new_value

            # Update the best scores
            if new_value > self.best_score_ever:
                self.best_score_ever = new_value
            if new_value > self.best_score_in_episode:
                self.best_score_in_episode = new_value

            # Optionally print verbose information
            if self.verbose and self.done:
                print(f"best_score_ever={self.best_score_ever}, "
                      f"best_score_in_episode={self.best_score_in_episode}, "
                      f"final_score={new_value}")
            
            return observation, self.last_reward, self.done, False, info
    
        