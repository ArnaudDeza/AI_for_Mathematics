# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_atari_envpoolpy
import os
import argparse
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

import gymnasium
import wandb
import datetime


from src.models.ppo.mlp import FC_PPO_Agent
from src.utils import make_env



class RecordEpisodeStatistics(gymnasium.Wrapper):
    def __init__(self, env, deque_size=100):
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.episode_returns = None
        self.episode_lengths = None

    def reset(self, **kwargs):
        observations = super().reset(**kwargs)
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.lives = np.zeros(self.num_envs, dtype=np.int32)
        self.returned_episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.returned_episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        return observations

    def step(self, action):
        observations, rewards, dones, _,infos = super().step(action)
        self.episode_returns += infos["reward"]
        self.episode_lengths += 1
        self.returned_episode_returns[:] = self.episode_returns
        self.returned_episode_lengths[:] = self.episode_lengths
        self.episode_returns *= 1 - infos["terminated"]
        self.episode_lengths *= 1 - infos["terminated"]
        infos["r"] = self.returned_episode_returns
        infos["l"] = self.returned_episode_lengths
        return (
            observations,
            rewards,
            dones,
            infos,
        )

def parse_args():
    parser = argparse.ArgumentParser(description="PPO Agent Training Arguments")

    # Logging & Saving
    parser.add_argument("--wandb_project_name", type=str, default="Wed_nov20_morn", help="the wandb's project name")
    parser.add_argument("--wandb_entity", type=str, default="dezaarna", help="the wandb's entity name")

    # Experiment setup
    parser.add_argument("--exp_name", type=str, default=os.path.basename(__file__)[: -len(".py")], help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=31232, help="seed of the experiment")
    parser.add_argument("--torch_deterministic", type=bool, default=True, help="if toggled, torch.backends.cudnn.deterministic=False")
    parser.add_argument("--cuda", type=bool, default=False, help="if toggled, cuda will be enabled by default")

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



    # Fully connected network specific arguments
    parser.add_argument("--actor_hidden_layers", type=int, nargs="+", default=[ 128,64, 32], help="the hidden layers of the actor network")
    parser.add_argument("--critic_hidden_layers", type=int, nargs="+", default=[ 128,64, 32], help="the hidden layers of the critic network")
    parser.add_argument("--activation_fn", type=str, default="relu", help="the activation function of the network", choices=["relu", "tanh", "sigmoid"])
    parser.add_argument("--normalization", type=str, default="layer", help="the normalization layer of the network", choices=["layer", "batch", "none"])
    parser.add_argument("--dropout", type=float, default=0.0, help="the dropout rate of the network")
    parser.add_argument("--use_attention", type=bool, default=False, help="toggle attention mechanism")
    parser.add_argument("--attention_embed_dim", type=int, default=16, help="the embedding dimension of the attention mechanism")
    parser.add_argument("--attention_num_heads", type=int, default=4, help="the number of heads in the attention mechanism")
    parser.add_argument("--weight_init", type=str, default="xavier", help="the weight initialization method", choices=["orthogonal", "xavier"])


    # Graph neural network specific arguments




    # PPO Algorithm specific arguments
    parser.add_argument("--total_timesteps", type=int, default=10000000, help="total timesteps of the experiment")
    parser.add_argument("--learning_rate", type=float, default=2.5e-5, help="the learning rate of the optimizer")
    parser.add_argument("--num_envs", type=int, default=4, help="the number of parallel game environments")
    parser.add_argument("--num_steps", type=int, default=171, help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal_lr", type=bool, default=True, help="toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.98, help="the discount factor gamma")
    parser.add_argument("--gae_lambda", type=float, default=0.97, help="the lambda for general advantage estimation")
    parser.add_argument("--num_minibatches", type=int, default=4, help="the number of mini-batches")
    parser.add_argument("--update_epochs", type=int, default=6, help="the K epochs to update the policy")
    parser.add_argument("--norm_adv", type=bool, default=True, help="toggle advantages normalization")
    parser.add_argument("--clip_coef", type=float, default=0.1, help="the surrogate clipping coefficient")
    parser.add_argument("--clip_vloss", type=bool, default=True, help="toggle whether or not to use a clipped loss for the value function")
    parser.add_argument("--ent_coef", type=float, default=0.01, help="coefficient of the entropy")
    parser.add_argument("--vf_coef", type=float, default=0.5, help="coefficient of the value function")
    parser.add_argument("--max_grad_norm", type=float, default=0.5, help="the maximum norm for the gradient clipping")
    parser.add_argument("--target_kl", type=float, default=None, help="the target KL divergence threshold")
    
    # Burn-in iterations for speed measure
    parser.add_argument("--measure_burnin", type=int, default=3, help="number of burn-in iterations for speed measure")

    # Parsed arguments
    return parser.parse_args()



if __name__ == "__main__":
    args = parse_args()

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

    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    

    # Get current datetime in the format YYYYMMDD_HHMMSS
    current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Modify the run_name to include the datetime
    run_name = f"PPO__{current_datetime}__{args.env_type}__N{args.number_of_nodes}__{args.conjecture}__{args.model_type}__{args.seed}"

        
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

    if args.model_type == "fc":
        agent = FC_PPO_Agent(num_edges=num_edges,
                                actor_hidden_layers=args.actor_hidden_layers,
                                critic_hidden_layers=args.critic_hidden_layers,
                                activation_fn=args.activation_fn,
                                normalization=args.normalization,
                                dropout=args.dropout,
                                use_attention=args.use_attention,
                                attention_embed_dim=args.attention_embed_dim,
                                attention_num_heads=args.attention_num_heads,
                                weight_init=args.weight_init
                            ).to(device)



    elif args.model_type == "gnn":
        agent = GNN_PPO_Agent(num_nodes=args.number_of_nodes, num_edges=num_edges,device = device).to(device)



    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward,dtype=torch.float32).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()












