# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqnpy
import os
import random
import time
from dataclasses import dataclass

import gymnasium
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
import wandb

from utils import make_env



from src.models.dqn.mlp import QNetwork



import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Experiment Arguments")

    # Logging & Saving
    parser.add_argument("--wandb_project_name", type=str, default="RL_Graph_Theory", help="the wandb's project name")
    parser.add_argument("--wandb_entity", type=str, default="dezaarna", help="the wandb's entity name")

    # Environment specific arguments
    parser.add_argument("--env_type", type=str, default="linear", help="the type of the environment", choices=["flip", "global", "linear", "local"])
    parser.add_argument("--number_of_nodes", type=int, default=19, help="the number of nodes in the environment")
    parser.add_argument("--directed", type=bool, default=False, help="Graph is directed or not")
    parser.add_argument("--conjecture", type=str, default="conj_2_1", help="which conjecture to test", choices=["conj_2_1","conj_2_3"])
    parser.add_argument("--start_with_complete_graph", type=bool, default=False, help="Start with a complete graph if True, else empty graph.")
    parser.add_argument("--time_horizon", type=int, default=-1, help="Maximum number of steps (episode length)")
    parser.add_argument("--dense_reward", type=bool, default=False, help="Whether to use dense rewards (difference in value at each step)")
    parser.add_argument("--check_at_every_step", type=bool, default=False, help="Whether to check for counterexamples at every step")
    parser.add_argument("--verbose", type=bool, default=False, help="Verbosity flag")
    parser.add_argument("--self_loops", type=bool, default=False, help="Whether self-loops are allowed")

    # ML model specific arguments
    parser.add_argument("--model_type", type=str, default="fc", help="the type of the environment", choices=["fc","gnn"])


    # General arguments
    parser.add_argument("--exp_name", type=str, default=os.path.basename(__file__)[: -len(".py")],
                        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=111,
                        help="seed of the experiment")
    parser.add_argument("--torch_deterministic", type=bool, default=True,
                        help="if toggled, torch.backends.cudnn.deterministic=False")
    parser.add_argument("--cuda", type=bool, default=True,
                        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=bool, default=False,
                        help="if toggled, this experiment will be tracked with Weights and Biases")
    
    # Algorithm specific arguments
    parser.add_argument("--total_timesteps", type=int, default=500000,
                        help="total timesteps of the experiments")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="the learning rate of the optimizer")
    parser.add_argument("--num_envs", type=int, default=1,
                        help="the number of parallel game environments")
    parser.add_argument("--buffer_size", type=int, default=10000,
                        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=1.0,
                        help="the target network update rate")
    parser.add_argument("--target_network_frequency", type=int, default=500,
                        help="the timesteps it takes to update the target network")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="the batch size of sample from the replay memory")
    parser.add_argument("--start_e", type=float, default=1,
                        help="the starting epsilon for exploration")
    parser.add_argument("--end_e", type=float, default=0.05,
                        help="the ending epsilon for exploration")
    parser.add_argument("--exploration_fraction", type=float, default=0.5,
                        help="the fraction of total-timesteps it takes from start-e to go end-e")
    parser.add_argument("--learning_starts", type=int, default=10000,
                        help="timestep to start learning")
    parser.add_argument("--train_frequency", type=int, default=10,
                        help="the frequency of training")

    return parser.parse_args()







def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


if __name__ == "__main__":
    args = parse_args()
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:

poetry run pip install "stable_baselines3==2.0.0a1"
"""
        )

    # looping over all edges possible (i.e all edges in a complete graph)
    if args.time_horizon < 0:
        args.time_horizon = None


    # Determine the number of edges based on graph type
    if args.directed:
        if args.self_loops:
            num_edges = args.number_of_nodes * args.number_of_nodes
        else:
            num_edges = args.number_of_nodes * (args.number_of_nodes - 1)
    else:
        if args.self_loops:
            num_edges = args.number_of_nodes * (args.number_of_nodes + 1) // 2
        else:
            num_edges = args.number_of_nodes * (args.number_of_nodes - 1) // 2
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"
    



    run_name = f"DQN__{args.env_type}__N{args.number_of_nodes}__{args.conjecture}__{args.model_type}__{args.seed}"
    
    
    wandb.init(
        project=args.wandb_project_name,
        entity=args.wandb_entity,
        sync_tensorboard=True,
        config=vars(args),
        name=run_name,
        #monitor_gym=True,
        #save_code=True,

    )







    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # Check for MPS, CUDA, and fallback to CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS backend")
    elif torch.cuda.is_available() and args.cuda:
        device = torch.device("cuda")
        print("Using CUDA backend")
    else:
        device = torch.device("cpu")
        print("Using CPU backend")

    # Environment setup
    envs = gymnasium.vector.SyncVectorEnv([make_env(args) for _ in range(args.num_envs)])

    q_network = QNetwork(envs).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = QNetwork(envs).to(device)
    target_network.load_state_dict(q_network.state_dict())

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = q_network(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)
                with torch.no_grad():
                    target_max, _ = target_network(data.next_observations).max(dim=1)
                    td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
                old_val = q_network(data.observations).gather(1, data.actions).squeeze()
                loss = F.mse_loss(td_target, old_val)

                if global_step % 100 == 0:
                    writer.add_scalar("losses/td_loss", loss, global_step)
                    writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                    print("SPS:", int(global_step / (time.time() - start_time)))
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # update target network
            if global_step % args.target_network_frequency == 0:
                for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                    target_network_param.data.copy_(
                        args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
                    )

    

       
    envs.close()
    writer.close()