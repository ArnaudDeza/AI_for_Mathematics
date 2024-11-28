import argparse
import ray
from ray import tune
from ray.tune.logger import TBXLoggerCallback
from ray.tune.registry import register_env

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import (
    FullyConnectedNetwork as TorchFullyConnectedNetwork,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
# Register the custom model
from ray.rllib.models import ModelCatalog




from ray_envs.dev import FlipEnvironment
from rewards.conjecture_2_1 import calcScore as calcScore_2_1
from ray_models.ppo_agent import FlipEnvPPOModel



# Environment creator
def env_creator(env_config):
    #return FlipEnvironment(env_config)  # return an env instance
    return FlipEnvironment(env_config)

if __name__ == "__main__":
    # Argument parser for environment and training parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_timesteps", type=int, default=1000000, help="Number of total timesteps for training")
    parser.add_argument("--num_nodes", type=int, default=19, help="Number of nodes in the environment")
    parser.add_argument("--cpus_per_trial", type=int, default=4, help="Number of CPUs to allocate per trial")
    parser.add_argument("--gpus_per_trial", type=float, default=1, help="Number of GPUs to allocate per trial")
    parser.add_argument("--gpu_device", type=str, default="cuda", help="Specify GPU device, e.g., 'cuda' or 'mps'")
    args = parser.parse_args()

    # Register the environment
    register_env("flip_env", env_creator)

    ModelCatalog.register_custom_model("custom_torch_ppo_model", FlipEnvPPOModel)



    # Initialize Ray
    ray.init(ignore_reinit_error=True)

    # Define hyperparameter search space
    search_space = {
        "lr": tune.grid_search([0.001, 0.0001]),  # Example learning rate grid search
        "sgd_minibatch_size": tune.choice([32, 64]),  # Example choice of batch size
        "train_batch_size": tune.choice([500, 1000]),  # Example train batch size
    }

    # Run the experiment with hyperparameter tuning, logging, and checkpointing
    tune.run(
        "PPO",
        
        stop={
            "timesteps_total": args.num_timesteps,
        },
        config={
            "env": "flip_env",
            "env_config": {
                "number_of_nodes": args.num_nodes,
                "value_fun": calcScore_2_1,
                "init_graph": None,
                "time_horizon": None,
                "dense_reward": False,
                "check_at_every_step": False,
                "start_with_complete_graph": True,
                "verbose": False,
                "self_loops": False,
                "directed": False
            },
            "lr": search_space["lr"],
            "sgd_minibatch_size": search_space["sgd_minibatch_size"],
            "train_batch_size": search_space["train_batch_size"],
            "model": {
                "custom_model": "custom_torch_ppo_model",

                "custom_model_config": {
                    "adj_hidden_sizes": [128, 128],
                    "action_embedding_dim": 64,
                    "action_seq_hidden_size": 128,
                    "action_seq_num_layers": 1,
                    "activation_fn": "ReLU",
                    "dropout": 0.0,
                    "num_nodes": args.num_nodes,
                },
            },
            "framework": "torch",
            "num_workers": args.cpus_per_trial - 1,  # Subtract 1 for the driver
            "num_envs_per_worker": 4,  # Each worker runs multiple environments in parallel
            #"num_gpus": args.gpus_per_trial,
            "device": args.gpu_device
        },
        num_samples=10,  # Limit total number of trials
        max_concurrent_trials=2,  # Limit how many trials run in parallel

        checkpoint_freq=10,  # Save model every 10 training iterations
        checkpoint_at_end=True,  # Save model at the end of training
        storage_path="/Users/adeza3/Desktop/PhD_year1/Courses/ISYE6740/ray_results",  # Specify the directory for results and logs
    )

    # Shutdown Ray
    ray.shutdown()


