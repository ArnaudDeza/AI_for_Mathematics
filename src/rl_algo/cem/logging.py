
import os
import datetime
import numpy as np
import json


def create_output_folder(args):
    if args.directed == True:
        directed  = "directed"
    else:
        directed = "undirected"
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    folder_name = f"{args.base_folder}/{args.reward_function}__{directed}_n_{args.n}_{args.model}/cem_run__seed_{args.seed}_{timestamp}_id_{args.current_idx}"
    output_folder = os.path.join("results", folder_name)
    os.makedirs(output_folder, exist_ok=True)
    return output_folder




def log_results(iteration, super_rewards, mean_all_reward, sessgen_time, randomcomp_time, 
                select1_time, select2_time, select3_time, fit_time, score_time, output_folder):
    # Path to the log file
    log_path = os.path.join(output_folder, f"training_log.json")

    # Data to log
    log_data = {
        "iteration": iteration,

        # Stats about super sessions rewards - min,max, std, mean, median
        "super_rewards_min": np.min(super_rewards),
        "super_rewards_max": np.max(super_rewards),
        "super_rewards_std": np.std(super_rewards),
        "super_rewards_mean": np.mean(super_rewards),
        "super_rewards_median": np.median(super_rewards),
        # "super_rewards": super_rewards.tolist(),

        "mean_all_reward": mean_all_reward,

        "sessgen_time": sessgen_time,
        "randomcomp_time": randomcomp_time,
        "select1_time": select1_time,
        "select2_time": select2_time,
        "select3_time": select3_time,
        "fit_time": fit_time,
        "score_time": score_time,
    }

    # Append to the JSON log file
    if os.path.exists(log_path):
        with open(log_path, "r") as file:
            logs = json.load(file)
    else:
        logs = []

    logs.append(log_data)
    with open(log_path, "w") as file:
        json.dump(logs, file, indent=4)
