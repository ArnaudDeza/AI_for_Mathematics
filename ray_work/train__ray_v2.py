import argparse
import ray
from ray import tune
from ray.tune.logger import TBXLoggerCallback
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
import yaml
import os
import random
import numpy as np
torch.manual_seed
import torch
from ray_envs.dev import FlipEnvironment
from rewards.conjecture_2_1 import calcScore as calcScore_2_1
from rewards.conjecture_2_2 import calcScore as calcScore_2_2  # Example additional reward function
from ray_models.ppo_agent import FlipEnvPPOModel

# Environment creator
def env_creator(env_config):
    """
    Create the environment instance.
    This function takes an environment configuration and returns an instance of the FlipEnvironment.
    """
    return FlipEnvironment(env_config)


def parse_arguments():
    """
    Parse command line arguments.
    This function uses argparse to define and parse command line arguments required for the script.
    It provides flexibility to the user to specify experiment parameters directly from the command line.
    """
    parser = argparse.ArgumentParser(description="Train RL model using Ray RLlib")

    # General parameters
    parser.add_argument("--experiment_name", type=str, default="default_experiment", help="Name of the experiment")
    parser.add_argument("--random_seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--config_file", type=str, default=None, help="Path to the configuration file")

    # Training parameters
    parser.add_argument("--algorithm", type=str, default="PPO", choices=["PPO", "DQN", "A3C"], help="RL algorithm to use (e.g., 'PPO', 'DQN', 'A3C')")
    parser.add_argument("--num_timesteps", type=int, default=1000000, help="Number of total timesteps for training")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of hyperparameter samples")
    parser.add_argument("--max_concurrent_trials", type=int, default=2, help="Maximum number of concurrent trials")
    parser.add_argument("--checkpoint_freq", type=int, default=10, help="Checkpoint frequency")
    parser.add_argument("--storage_path", type=str, default="./ray_results", help="Directory for results and logs")
    parser.add_argument("--resume_training", action="store_true", help="Resume training from the last checkpoint")

    # Environment parameters
    parser.add_argument("--num_nodes", type=int, default=19, help="Number of nodes in the environment")
    parser.add_argument("--init_graph", type=str, default=None, help="Initial graph configuration (default: None)")
    parser.add_argument("--time_horizon", type=int, default=None, help="Time horizon for the environment (default: None)")
    parser.add_argument("--dense_reward", action='store_true', help="Use dense reward (default: False)")
    parser.add_argument("--check_at_every_step", action='store_true', help="Check environment at every step (default: False)")
    parser.add_argument("--start_with_complete_graph", action='store_true', help="Start with a complete graph (default: False)")
    parser.add_argument("--verbose", action='store_true', help="Verbose mode (default: False)")
    parser.add_argument("--self_loops", action='store_true', help="Allow self-loops in the graph (default: False)")
    parser.add_argument("--directed", action='store_true', help="Use directed graph (default: False)")
    parser.add_argument("--reward_function", type=str, default="calcScore_2_1", help="Reward function to use (e.g., 'calcScore_2_1' or 'calcScore_2_2')")

    # Resource allocation parameters
    parser.add_argument("--cpus_per_trial", type=int, default=4, help="Number of CPUs to allocate per trial")
    parser.add_argument("--gpus_per_trial", type=float, default=1, help="Number of GPUs to allocate per trial")
    parser.add_argument("--gpu_device", type=str, default="cuda", help="Specify GPU device, e.g., 'cuda' or 'mps'")

    # Model configuration parameters
    parser.add_argument("--adj_hidden_sizes", type=int, nargs='+', default=[128, 128], help="List of sizes for adjacency hidden layers")
    parser.add_argument("--action_embedding_dim", type=int, default=64, help="Dimension of action embedding")
    parser.add_argument("--action_seq_hidden_size", type=int, default=128, help="Hidden size for action sequence")
    parser.add_argument("--action_seq_num_layers", type=int, default=1, help="Number of layers for action sequence")
    parser.add_argument("--activation_fn", type=str, default="ReLU", help="Activation function to use")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate")

    return parser.parse_args()


def load_config(config_path):
    """
    Load configuration from a YAML file.
    This function loads experiment configuration parameters from a specified YAML file.
    It allows users to define complex configurations without specifying every parameter via command line.
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def set_random_seed(seed):
    """
    Set the random seed for reproducibility.
    This function sets the random seed for Python, NumPy, and PyTorch to ensure that experiments are reproducible.
    It ensures that random processes within the training are consistent across runs when the same seed is used.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)


def get_search_space():
    """
    Define the hyperparameter search space.
    This function defines a search space for hyperparameters like learning rate, batch size, etc.
    It is used by Ray Tune to perform hyperparameter tuning to find the best possible model configuration.
    """
    return {
        "lr": tune.grid_search([0.001, 0.0001]),  # Example learning rate grid search
        "sgd_minibatch_size": tune.choice([32, 64]),  # Example choice of batch size
        "train_batch_size": tune.choice([500, 1000]),  # Example train batch size
    }


def main():
    """
    Main training script.
    This is the main function that handles argument parsing, environment setup, hyperparameter tuning, and training.
    It orchestrates the entire process of defining the experiment configuration, setting up the training environment, and running the model training.
    """
    args = parse_arguments()

    # Load configuration file if provided
    if args.config_file:
        config_from_file = load_config(args.config_file)
        args = argparse.Namespace(**vars(args), **config_from_file)

    # Set random seed for reproducibility
    set_random_seed(args.random_seed)

    # Select reward function based on argument
    if args.reward_function == "calcScore_2_1":
        value_fun = calcScore_2_1
    elif args.reward_function == "calcScore_2_2":
        value_fun = calcScore_2_2
    else:
        raise ValueError(f"Unknown reward function: {args.reward_function}")

    # Register the environment with Ray
    register_env("flip_env", env_creator)
    ModelCatalog.register_custom_model("custom_torch_ppo_model", FlipEnvPPOModel)

    # Initialize Ray for distributed training
    ray.init(ignore_reinit_error=True, num_gpus=args.gpus_per_trial)

    # Define hyperparameter search space
    search_space = get_search_space()

    # Experiment configuration
    config = {
        "env": "flip_env",  # Name of the environment to be used
        "env_config": {
            "number_of_nodes": args.num_nodes,  # Number of nodes in the environment
            "value_fun": value_fun,  # Reward function used for the environment
            "init_graph": args.init_graph,  # Initial graph configuration
            "time_horizon": args.time_horizon,  # Time horizon for environment
            "dense_reward": args.dense_reward,  # Whether to use dense rewards
            "check_at_every_step": args.check_at_every_step,  # Whether to check environment state at every step
            "start_with_complete_graph": args.start_with_complete_graph,  # Start with a complete graph
            "verbose": args.verbose,  # Verbose output
            "self_loops": args.self_loops,  # Allow self-loops in the graph
            "directed": args.directed  # Use directed graph
        },
        "lr": search_space["lr"],  # Learning rate
        "sgd_minibatch_size": search_space["sgd_minibatch_size"],  # SGD minibatch size
        "train_batch_size": search_space["train_batch_size"],  # Training batch size
        "model": {
            "custom_model": "custom_torch_ppo_model",  # Custom model to use
            "custom_model_config": {
                "adj_hidden_sizes": args.adj_hidden_sizes,  # Sizes for adjacency hidden layers
                "action_embedding_dim": args.action_embedding_dim,  # Dimension of action embedding
                "action_seq_hidden_size": args.action_seq_hidden_size,  # Hidden size for action sequence
                "action_seq_num_layers": args.action_seq_num_layers,  # Number of layers for action sequence
                "activation_fn": args.activation_fn,  # Activation function
                "dropout": args.dropout,  # Dropout rate
                "num_nodes": args.num_nodes,  # Number of nodes
            },
        },
        "framework": "torch",  # Framework to use (PyTorch)
        "num_workers": args.cpus_per_trial - 1,  # Number of workers (subtract 1 for the driver process)
        "num_envs_per_worker": 4,  # Number of environments per worker
        "device": args.gpu_device  # GPU device to use
    }

    # Run the experiment with hyperparameter tuning, logging, and checkpointing
    tune.run(
        args.algorithm,  # RL algorithm to use (e.g., PPO, DQN, A3C)
        name=args.experiment_name,  # Name of the experiment
        stop={"timesteps_total": args.num_timesteps},  # Stop condition for the training
        config=config,  # Configuration dictionary for the experiment
        num_samples=args.num_samples,  # Number of hyperparameter samples for tuning
        max_concurrent_trials=args.max_concurrent_trials,  # Maximum number of concurrent trials
        checkpoint_freq=args.checkpoint_freq,  # Frequency for saving checkpoints
        checkpoint_at_end=True,  # Save checkpoint at the end of training
        storage_path=args.storage_path,  # Directory to store results and logs
        resume=args.resume_training,  # Whether to resume training from last checkpoint
        callbacks=[TBXLoggerCallback()]  # Callback for TensorBoard logging
    )

    # Shutdown Ray to release resources
    ray.shutdown()


if __name__ == "__main__":
    main()