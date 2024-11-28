import numpy as np
import time
import torch

from src.rewards.score import score_state_graph




def play_game(args, actions, state_next, states, prob, step, total_score):
    """
    Simulates one step of the game for all sessions.
    
    Parameters:
        args: Namespace containing arguments like n_sessions, MYN, etc.
        actions: Array to store actions taken at each step.
        state_next: Array representing the next state of each session.
        states: Array representing the entire history of states for each session.
        prob: Probabilities of taking action 1 for each session.
        step: Current step in the game.
        total_score: Array to store the total score for each session.

    Returns:
        Updated actions, state_next, states, total_score, terminal flag, and session infos.
    """
    infos = {}
    for i in range(args.n_sessions):
        # Decide action based on probability
        action = 1 if np.random.rand() < prob[i] else 0
        actions[i][step - 1] = action  # Record the action

        # Update the next state for the current session
        state_next[i] = states[i, :, step - 1]
        state_next[i][step - 1] = action  # Update edge decision
        state_next[i][args.MYN + step - 1] = 0  # Reset the next step flag
        
        # Set the next step flag if the current step is not the last
        if step < args.MYN:
            state_next[i][args.MYN + step] = 1
        
        # Check if the game has reached a terminal state
        terminal = step == args.MYN
        if terminal:
            total_score[i], info = score_state_graph(args, state_next[i])
            infos[f"session_{i}"] = info
        else:
            # Record the updated state for the next step
            states[i, :, step] = state_next[i]
    
    return actions, state_next, states, total_score, terminal, infos


def generate_session(args, agent):
    """
    Generates n_sessions of game play using the agent neural network.

    Parameters:
        args: Namespace containing arguments like n_sessions, len_game, etc.
        agent: Neural network that predicts probabilities for actions.

    Returns:
        Tuple containing (states, actions, total_score) and session infos.
    """
    # Initialize actions array to store the actions for each session and step
    actions = np.zeros([args.n_sessions, args.len_game], dtype=int)
    
    # Initialize the state and next state arrays based on the starting graph type
    if args.init_graph == 'empty':
        # Starting from an empty graph
        states = np.zeros([args.n_sessions, args.observation_space, args.len_game], dtype=int)
        state_next = np.zeros([args.n_sessions, args.observation_space], dtype=int)
    elif args.init_graph == 'complete':
        # Starting from a complete graph
        states = np.zeros([args.n_sessions, args.observation_space, args.len_game], dtype=int)
        states[:, :args.MYN, :] = 1  # Set the first half of the graph to 1
        state_next = np.ones([args.n_sessions, args.observation_space], dtype=int)

    # Set the initial step flag
    states[:, args.MYN, 0] = 1

    # Initialize other variables
    total_score = np.zeros(args.n_sessions)
    step, pred_time, play_time = 0, 0, 0

    # Simulate the game
    while True:
        step += 1

        # Predict probabilities using the agent
        tic = time.time()
        state_tensor = torch.from_numpy(states[:, :, step - 1]).float().to(args.device)
        prob = agent(state_tensor).detach().cpu().numpy()  # Detach and move predictions to CPU
        pred_time += time.time() - tic

        # Play one step of the game
        tic = time.time()
        actions, state_next, states, total_score, terminal, infos = play_game(
            args, actions, state_next, states, prob, step, total_score
        )
        play_time += time.time() - tic

        # Break if the game has ended
        if terminal:
            break

    return (states, actions, total_score), infos

