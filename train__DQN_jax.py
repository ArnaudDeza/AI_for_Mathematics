import random
import time

import flax
import flax.linen as nn
import gymnasium as gym
import torch
import jax
import jax.numpy as jnp
import numpy as np
import wandb
import optax
from flax.training.train_state import TrainState
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
import argparse
from utils import make_env


from src.models.dqn.mlp import *

def parse_args():
    """
    Parses command-line arguments for configuring the experiment.
    """
    parser = argparse.ArgumentParser(description="Experiment Arguments")
    # Environment specific arguments
    parser.add_argument("--env_type", type=str, default="linear", help="the type of the environment", choices=["flip", "global", "linear", "local"])
    parser.add_argument("--number_of_nodes", type=int, default=19, help="the number of nodes in the environment")
    parser.add_argument("--directed", type=bool, default=False, help="Graph is directed or not")
    parser.add_argument("--conjecture", type=str, default="conj_2_1", help="which conjecture to test", choices=["conj_2_1","conj_2_3"])
    parser.add_argument("--start_with_complete_graph", type=bool, default=False, help="Start with a complete graph if True, else empty graph.")
    parser.add_argument("--time_horizon", type=int, default=-100, help="Maximum number of steps (episode length)")
    parser.add_argument("--dense_reward", type=bool, default=False, help="Whether to use dense rewards (difference in value at each step)")
    parser.add_argument("--check_at_every_step", type=bool, default=False, help="Whether to check for counterexamples at every step")
    parser.add_argument("--verbose", type=bool, default=False, help="Verbosity flag")
    parser.add_argument("--self_loops", type=bool, default=False, help="Whether self-loops are allowed")

    # Q-Network specific arguments
    parser.add_argument("--q_network_type", type=str, default="fc", help="Type of Q-Network to use", choices=["fc", "dueling", "cnn"])
    parser.add_argument("--hidden_layers", type=int, nargs='+', default=[128, 64, 16], help="Hidden layer sizes")
    parser.add_argument("--activation", type=str, default="relu", help="Activation function", choices=["relu", "leaky_relu", "tanh"])
    parser.add_argument("--cnn_num_channels", type=int, nargs='+', default=[32, 64], help="Number of channels for CNN layers")
    parser.add_argument("--cnn_kernel_sizes", type=int, nargs='+', default=[8, 4], help="Kernel sizes for CNN layers")
    parser.add_argument("--cnn_strides", type=int, nargs='+', default=[4, 2], help="Strides for CNN layers")
    parser.add_argument("--cnn_hidden_layers", type=int, nargs='+', default=[512], help="Hidden layer sizes after CNN")
    
    # General arguments
    parser.add_argument("--seed", type=int, default=23,     help="Seed of the experiment")
    parser.add_argument("--track", action="store_true",   help="If toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb_project_name", type=str, default="RL_Graph_Theory", help="The wandb's project name")
    parser.add_argument("--wandb_entity", type=str, default="dezaarna",  help="The entity (team) of wandb's project")
    parser.add_argument("--save_model", action="store_true",   help="Whether to save model into the `runs/{run_name}` folder")
    
    # Algorithm-specific arguments
    parser.add_argument("--total_timesteps", type=int, default=500000,  help="Total timesteps of the experiments")
    parser.add_argument("--learning_rate", type=float, default=1e-4,  help="The learning rate of the optimizer")
    parser.add_argument("--num_envs", type=int, default=1,  help="The number of parallel game environments")
    parser.add_argument("--buffer_size", type=int, default=300000,  help="The replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,  help="The discount factor gamma")
    parser.add_argument("--tau", type=float, default=0.4, help="The target network update rate")
    parser.add_argument("--target_network_frequency", type=int, default=10000,   help="The timesteps it takes to update the target network")
    parser.add_argument("--batch_size", type=int, default=512,  help="The batch size of sample from the replay memory")
    parser.add_argument("--start_e", type=float, default=1,  help="The starting epsilon for exploration")
    parser.add_argument("--end_e", type=float, default=0.05,  help="The ending epsilon for exploration")
    parser.add_argument("--exploration_fraction", type=float, default=0.7,  help="The fraction of `total-timesteps` it takes from start-e to go end-e")
    parser.add_argument("--learning_starts", type=int, default=100000,  help="Timestep to start learning")
    parser.add_argument("--train_frequency", type=int, default=10, help="The frequency of training")
    
    return parser.parse_args()

class QNetwork(nn.Module):
    """
    Defines the Q-network architecture using Flax's Linen API.
    """
    action_dim: int  # Number of possible actions
    hidden_layers: list = (128, 64, 16)  # Sizes of hidden layers
    activation: callable = nn.relu       # Activation function
    
    @nn.compact
    def __call__(self, x: jnp.ndarray):
        """ 
        Forward pass for the Q-network.
        """
        # Iterate through hidden layers, applying Dense layer and activation
        for units in self.hidden_layers:
            x = nn.Dense(units)(x)
            x = self.activation(x)

        # Output layer with size equal to number of actions
        x = nn.Dense(self.action_dim)(x)
        return x

class TrainState(TrainState):
    """
    Extends Flax's TrainState to include target network parameters.
    """
    target_params: flax.core.FrozenDict

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    """
    Linearly decays epsilon from start_e to end_e over the specified duration.
    """
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)
def get_activation(name):
    """
    Returns the activation function based on the name.
    """
    if name == "relu":
        return nn.relu
    elif name == "leaky_relu":
        return nn.leaky_relu
    elif name == "tanh":
        return nn.tanh
    else:
        raise ValueError(f"Unsupported activation function: {name}")
    
if __name__ == "__main__":
    args = parse_args()  # Parse command-line arguments
    import stable_baselines3 as sb3

    # Check for compatible version of stable_baselines3
    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:

poetry run pip install "stable_baselines3==2.0.0a1"
"""
        )
    
    # Determine the effective time horizon
    args.time_horizon = None if args.time_horizon < 0 else args.time_horizon
    
    # Calculate the number of edges based on whether the graph is directed and if self-loops are allowed
    if args.directed:
        num_edges = args.number_of_nodes * args.number_of_nodes if args.self_loops else args.number_of_nodes * (args.number_of_nodes - 1)
    else:
        num_edges = args.number_of_nodes * (args.number_of_nodes + 1) // 2 if args.self_loops else args.number_of_nodes * (args.number_of_nodes - 1) // 2
    
    # Ensure that only one environment is used (vectorized envs not supported)
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"

    # Create a unique run name based on experiment parameters
    run_name = f"DQN_JAX__{args.env_type}__N{args.number_of_nodes}__{args.conjecture}__{args.q_network_type}__{args.seed}"
    
    # Initialize Weights and Biases for experiment tracking
    wandb.init(
        project=args.wandb_project_name,
        entity=args.wandb_entity,
        sync_tensorboard=True,
        config=vars(args),
        name=run_name,
    )

    # Initialize TensorBoard writer for logging
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, q_key = jax.random.split(key, 2)

    # Select the appropriate device (MPS, CUDA, or CPU)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS backend")
    elif torch.cuda.is_available() and args.cuda:
        device = torch.device("cuda")
        print("Using CUDA backend")
    else:
        device = torch.device("cpu")
        print("Using CPU backend")

    # Create the environment(s) using the make_env utility
    envs = gym.vector.SyncVectorEnv([make_env(args) for _ in range(args.num_envs)])

    # Reset the environment and obtain initial observations
    obs, _ = envs.reset(seed=args.seed)

    # Initialize the Q-network with the appropriate number of actions and activation function
    # Select the Q-Network based on the argument
    activation_fn = get_activation(args.activation)
    
    if args.q_network_type == "fc":
        q_network = QNetworkFC(
            action_dim=envs.single_action_space.n,
            hidden_layers=args.hidden_layers,
            activation=activation_fn
        )
    elif args.q_network_type == "dueling":
        q_network = DuelingQNetwork(
            action_dim=envs.single_action_space.n,
            hidden_layers=args.hidden_layers,
            activation=activation_fn
        )
    elif args.q_network_type == "cnn":
        q_network = QNetworkCNN(
            action_dim=envs.single_action_space.n,
            num_channels=args.cnn_num_channels,
            kernel_sizes=args.cnn_kernel_sizes,
            strides=args.cnn_strides,
            activation=activation_fn,
            hidden_layers=args.cnn_hidden_layers
        )
    else:
        raise ValueError(f"Unsupported Q-Network type: {args.q_network_type}")
    


    
    # Initialize the training state, including parameters and optimizer
    q_state = TrainState.create(
        apply_fn=q_network.apply,  # Function to apply the network
        params=q_network.init(q_key, obs),  # Initialize network parameters
        target_params=q_network.init(q_key, obs),  # Initialize target network parameters
        tx=optax.adam(learning_rate=args.learning_rate),  # Optimizer (Adam)
    )

    # JIT compile the apply function for faster execution
    q_network.apply = jax.jit(q_network.apply)
    
    # Initialize target network parameters (not strictly necessary here)
    q_state = q_state.replace(target_params=optax.incremental_update(q_state.params, q_state.target_params, 1))

    # Initialize the replay buffer for experience replay
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        "cpu",
        handle_timeout_termination=False,
    )

    @jax.jit
    def update(q_state, observations, actions, next_observations, rewards, dones):
        """
        Performs a single update step for the Q-network using a batch of experiences.
        """
        # Compute target Q-values using the target network
        q_next_target = q_network.apply(q_state.target_params, next_observations)  # Shape: (batch_size, num_actions)
        q_next_target = jnp.max(q_next_target, axis=-1)  # Shape: (batch_size,)
        # Compute the target for the Bellman update
        next_q_value = rewards + (1 - dones) * args.gamma * q_next_target

        def mse_loss(params):
            """
            Computes the Mean Squared Error loss between predicted Q-values and target Q-values.
            """
            # Predict Q-values for current observations
            q_pred = q_network.apply(params, observations)  # Shape: (batch_size, num_actions)
            # Select the Q-values corresponding to the taken actions
            q_pred = q_pred[jnp.arange(q_pred.shape[0]), actions.squeeze()]  # Shape: (batch_size,)
            # Compute MSE loss
            return ((q_pred - next_q_value) ** 2).mean(), q_pred

        # Compute loss and gradients
        (loss_value, q_pred), grads = jax.value_and_grad(mse_loss, has_aux=True)(q_state.params)
        # Apply gradients to update the network parameters
        q_state = q_state.apply_gradients(grads=grads)
        return loss_value, q_pred, q_state

    start_time = time.time()  # Record the start time for SPS (steps per second) calculation

    # Reset the environment to start the training loop
    obs, _ = envs.reset(seed=args.seed)
    
    for global_step in range(args.total_timesteps):
        # Compute the current epsilon for epsilon-greedy exploration
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        
        # Decide whether to take a random action (exploration) or use the policy (exploitation)
        if random.random() < epsilon:
            # Take a random action
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            # Predict Q-values using the current policy
            q_values = q_network.apply(q_state.params, obs)
            # Select the action with the highest Q-value
            actions = q_values.argmax(axis=-1)
            # Convert actions from JAX device arrays to NumPy arrays
            actions = jax.device_get(actions)

        # Execute the selected actions in the environment
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # Log episodic returns and lengths for TensorBoard
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # Handle final observations for truncated episodes
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        
        # Add the experience to the replay buffer
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # Update the current observation
        obs = next_obs

        # Training logic: perform updates after a certain number of steps
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                # Sample a batch of experiences from the replay buffer
                data = rb.sample(args.batch_size)
                # Perform a gradient descent step to update the Q-network
                loss, old_val, q_state = update(
                    q_state,
                    data.observations.numpy(),
                    data.actions.numpy(),
                    data.next_observations.numpy(),
                    data.rewards.flatten().numpy(),
                    data.dones.flatten().numpy(),
                )

                # Log training metrics every 100 steps
                if global_step % 100 == 0:
                    writer.add_scalar("losses/td_loss", jax.device_get(loss), global_step)
                    writer.add_scalar("losses/q_values", jax.device_get(old_val).mean(), global_step)
                    # Calculate steps per second (SPS)
                    print("SPS:", int(global_step / (time.time() - start_time)))
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

            # Update the target network at specified frequency
            if global_step % args.target_network_frequency == 0:
                q_state = q_state.replace(
                    target_params=optax.incremental_update(q_state.params, q_state.target_params, args.tau)
                )

    # Save the trained model if the save_model flag is set
    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        with open(model_path, "wb") as f:
            f.write(flax.serialization.to_bytes(q_state.params))
        print(f"model saved to {model_path}")

    # Clean up: close the environment and TensorBoard writer
    envs.close()
    writer.close()
