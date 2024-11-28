from torch import nn
import numpy as np
import time
import wandb
import torch
import random
import os
import networkx as nx
from torch.utils.tensorboard import SummaryWriter


from src.rl_algo.cem.game import generate_session
from src.rl_algo.cem.utils import seed_everything, initialize_model
from src.rl_algo.cem.select import select_elites, select_super_sessions

from src.plotting.create_plots import plot_graph_layouts_for_counter_example
from src.plotting.create_gif_____EnvDemo import create_rl_graph_gif

def train_network(args, model, optimizer, train_loader, num_epochs=1):
    ''' Updates the model parameters (in place) using the given optimizer object.  Returns `None`. '''
    criterion,  pbar = nn.BCELoss(),  range(num_epochs)
    for i in pbar:
        for k, batch_data in enumerate(train_loader):
            batch_data = batch_data.to(args.device)
            batch_x = batch_data[:, :-1]
            batch_y = batch_data[:, -1]
            model.zero_grad()
            y_pred = model(batch_x)
            loss = criterion(y_pred, batch_y.unsqueeze(1))
            loss.backward() 
            optimizer.step()


def action_to_adj_matrix(args,action):
    if args.directed:
        adjMatG = np.zeros((args.n,args.n), dtype=int)
        non_diagonal_mask = ~np.eye(args.n, dtype=bool)
        adjMatG[non_diagonal_mask] = action
    else:
        adjMatG = np.zeros((args.n,args.n),dtype=np.int8) #adjacency matrix determined by the state
        count = 0
        for i in range(args.n):
            for j in range(i+1,args.n):
                if action[count] == 1:
                    adjMatG[i][j] = 1
                    adjMatG[j][i] = 1
                count += 1
    return adjMatG

def action_to_grid(args,action):
    included_indices = [idx for idx, val in enumerate(action) if val == 1]
    included_points = [args.positions[idx] for idx in included_indices]
    grid = np.zeros((args.n, args.n), dtype=int)
    for i, j in included_points:
        grid[i, j] = 1
    return grid


def train_CEM_graph_agent(args):

    # Step 0: Set random seed for reproducibility
    seed_everything(args.seed)

    # Step 1: Initialize graph parameters
    N = args.n

    if args.isoceles_triangle:
        MYN = N * N  # Total number of grid points
    else:
        MYN = int(N * (N - 1) / 2) if args.directed == False else N * (N - 1)
    observation_space,len_game = 2 * MYN, MYN
    state_dim = (observation_space,)
    args.state_dim, args.observation_space = state_dim, observation_space
    args.len_game, args.MYN = len_game, MYN


    # Step 2: Initialize model and optimizer
    model, optimizer = initialize_model(args, MYN, args.device,args.seed)
    model = model.to(args.device)

    # Step 3: Initialize logger
    if args.logger == 'wandb':
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args))
        wandb.watch(model)
    elif args.logger == 'tensorboard':
        writer = SummaryWriter(log_dir=args.output_folder)

    # Step 4: Initialize variables
    # Generate initial buffers for super states, actions, and rewards
    super_states =  np.empty((0,len_game,observation_space), dtype = int)
    super_actions = np.array([], dtype = int)
    super_rewards = np.array([])
    # Timing variables
    sessgen_time, fit_time, score_time = 0, 0, 0
    # Initialize variables for stagnation detection and best solutions cache
    best_reward_list_every_iter,best_reward_list_overall= [],[]
    
    best_adj_matrix_list = []
    best_reward = -np.inf
    best_action_list = []
    iter_since_valid_counter_example = 0
    best_action = None

    sessgen_times,randomcomp_times,select1_times,select2_times,select3_times,fit_times,score_times = [],[],[],[],[],[],[]


    # Across all sessions
    mean_reward___all__list = []
    mean_reward___top_100__list = []
    mean_reward___top_50__list = []
    mean_reward___top_20__list = []
    mean_reward___top_25__list = []
    mean_reward___top_10__list = []
    mean_reward___top_2__list = []

    # Across super sessions
    mean_reward_super_top_10__list = []
    best_reward_super__list = []
    mean_reward_super_all__list = []

    # Across elite session
    mean_elite_rewards__list = []


    percentage_below___INF__list = []
    printed_before_found_counter_example = False



    FINISHED = False
    try:
        # Optimized training loop
        for i in range(args.iterations):


            # 1. Generate new sessions (Parallelizing this can be beneficial)
            tic = time.time()
            sessions,session_stats = generate_session(args, model)  # Set verbose=1 for debugging timing
            sessgen_time = time.time() - tic

            # 2. Extract state, action, and reward batches
            tic = time.time()
            states_batch = np.array(sessions[0], dtype=int)
            actions_batch = np.array(sessions[1], dtype=int)
            rewards_batch = np.array(sessions[2])
            states_batch = np.transpose(states_batch, axes=[0, 2, 1])
            states_batch = np.append(states_batch,super_states,axis=0)
            if i>0:
                actions_batch = np.append(actions_batch,np.array(super_actions),axis=0)    
            rewards_batch = np.append(rewards_batch,super_rewards)
            randomcomp_time = time.time()-tic 
            
            # 3. Select elite sessions based on percentile
            tic = time.time()
            elite_states, elite_actions, elite_rewards = select_elites(args,states_batch, actions_batch, rewards_batch, percentile=args.percentile) #pick the sessions to learn from
            select1_time = time.time()-tic

            mean_elite_reward = np.mean(elite_rewards)  # Mean reward of the elite sessions


            # 4. Select super sessions to survive, using a diverse selection strategy
            tic = time.time()
            super_sessions = select_super_sessions(args, states_batch, actions_batch, rewards_batch, percentile=90)
            select2_time = time.time() - tic


            # Sort super sessions by rewards in descending order
            tic = time.time()
            super_sessions = sorted(zip(super_sessions[0], super_sessions[1], super_sessions[2]), key=lambda x: x[2], reverse=True)
            select3_time = time.time() - tic


            # 5. Train the model on elite sessions
            tic = time.time()
            train_data = torch.from_numpy(np.column_stack((elite_states, elite_actions))).float()
            train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=args.batch_size)
            train_network(args,model, optimizer, train_loader)
            fit_time = time.time() - tic



            # 6. Update super sessions
            tic = time.time()
            super_states = np.array([s[0] for s in super_sessions])
            super_actions = np.array([s[1] for s in super_sessions])
            super_rewards = np.array([s[2] for s in super_sessions])

            # Mean rewards across rewards_batch --  all, best 100, best 50, best 25, 10, 5 , 2
            mean_reward___all = np.mean(rewards_batch)
            mean_reward___top_100 = np.mean(rewards_batch[-100:]) if len(rewards_batch)>= 100 else np.mean(rewards_batch)
            mean_reward___top_50 = np.mean(rewards_batch[-50:]) if len(rewards_batch)>= 50 else np.mean(rewards_batch)
            mean_reward___top_25 = np.mean(rewards_batch[-25:]) if len(rewards_batch)>= 25 else np.mean(rewards_batch)
            mean_reward___top_20 = np.mean(rewards_batch[-20:]) if len(rewards_batch)>= 20 else np.mean(rewards_batch)
            mean_reward___top_10 = np.mean(rewards_batch[-10:]) if len(rewards_batch)>= 10 else np.mean(rewards_batch)
            mean_reward___top_2 = np.mean(rewards_batch[-2:]) if len(rewards_batch)>= 2 else np.mean(rewards_batch)
            percentage_below___INF = np.mean(rewards_batch < args.INF + 10) *100




            # Mean across super_rewards --> all, top 10 and overall best
            mean_reward_super_all = np.mean(super_rewards)  # Mean reward of the surviving sessions
            mean_reward_super_top_10 = np.mean(super_rewards[:10]) if len(super_rewards) >= 10 else np.mean(super_rewards)
            best_reward_super = np.max(super_rewards)
            score_time = time.time() - tic


    

            # Update the best known candidate counter-example
            best_reward_this_iter = np.max(super_rewards)
            if best_reward_this_iter > best_reward:
                if printed_before_found_counter_example == False:
                    print("\t\t >> Found new best reward: {} improved from {} -- {} improvement".format(best_reward_this_iter, best_reward, best_reward_this_iter-best_reward))
                    printed_before_found_counter_example = True


                best_reward = best_reward_this_iter
                best_action = super_actions[0]

                if best_reward_this_iter > args.tol_for_valid_counter_example:
                    
                    if not os.path.exists('{}/valid_counter_example_score_{:.5f}.npz'.format(args.output_folder,best_reward_this_iter)):
                        if args.isoceles_triangle:
                            adj_mat__ = action_to_grid(args,best_action)
                        else:
                            adj_mat__ = action_to_adj_matrix(args,best_action)
                        np.savez('{}/valid_counter_example_score_{:.5f}.npz'.format(args.output_folder,best_reward_this_iter),
                                    adj_mat =  adj_mat__,
                                    action =  best_action,
                                    iter = np.array([i]) ,
                                    reward = np.array([best_reward_this_iter]))
                               


            # 7. Logging -- Append values to list for plotting later
            best_reward_list_every_iter.append(best_reward_this_iter)
            best_reward_list_overall.append(best_reward)

            percentage_below___INF__list.append(percentage_below___INF)

            # across all
            mean_reward___all__list.append(mean_reward___all)
            mean_reward___top_100__list.append(mean_reward___top_100)
            mean_reward___top_50__list.append(mean_reward___top_50)
            mean_reward___top_20__list.append(mean_reward___top_25)
            mean_reward___top_25__list.append(mean_reward___top_20)
            mean_reward___top_10__list.append(mean_reward___top_10)
            mean_reward___top_2__list.append(mean_reward___top_2)

            # across super
            mean_reward_super_top_10__list.append(mean_reward_super_top_10)
            best_reward_super__list.append(best_reward_super)
            mean_reward_super_all__list.append(mean_reward_super_all)

            # across elites
            mean_elite_rewards__list.append(mean_elite_reward)

            sessgen_times.append(sessgen_time)                          # Time taken to generate sessions
            randomcomp_times.append(randomcomp_time)                    # Time taken to extract states, actions, and rewards
            select1_times.append(select1_time)                          # Time taken to select elite sessions
            select2_times.append(select2_time)                          # Time taken to select super sessions
            select3_times.append(select3_time)                          # Time taken to sort super sessions
            fit_times.append(fit_time)                                  # Time taken to train the model
            score_times.append(score_time)                              # Time taken to update super sessions
            best_adj_matrix_list.append(super_actions[0]) # use action_to_adj_matrix(args,super_actions[0]) to get adj matrix
            best_action_list.append(best_action)

            # 9. Periodically save results
            if i % args.save_every_k_iters == 0:

                # Save list of scalars and adj matrices
                np.savez('{}/data_lists.npz'.format(args.output_folder),
                     best_reward_list_every_iter=np.array(best_reward_list_every_iter),
                    best_reward_list_overall=np.array(best_reward_list_overall),
                    percentage_below___INF__list = np.array(percentage_below___INF__list),

                    mean_reward___all__list = np.array(mean_reward___all__list),
                    mean_reward___top_100__list = np.array(mean_reward___top_100__list),
                    mean_reward___top_50__list = np.array(mean_reward___top_50__list),
                    mean_reward___top_20__list = np.array(mean_reward___top_20__list),
                    mean_reward___top_25__list = np.array(mean_reward___top_25__list),
                    mean_reward___top_10__list = np.array(mean_reward___top_10__list),
                    mean_reward___top_2__list = np.array(mean_reward___top_2__list),

                    mean_reward_super_top_10__list = np.array(mean_reward_super_top_10__list),
                    best_reward_super__list = np.array(best_reward_super__list),
                    mean_reward_super_all__list = np.array(mean_reward_super_all__list),

                    mean_elite_rewards__list = np.array(mean_elite_rewards__list),

                    sessgen_times=np.array(sessgen_times), randomcomp_times=np.array(randomcomp_times),
                    select1_times=np.array(select1_times), select2_times=np.array(select2_times), select3_times=np.array(select3_times),
                    fit_times=np.array(fit_times), score_times=np.array(score_times),
                    best_adj_matrix_list=np.array(best_adj_matrix_list),
                    best_action_list=np.array(best_action_list)
                                                  )
                


            # 10. Check for termination conditions --> best reward > 0 or i == args.iterations - 1
            if i == args.iterations - 1:                # Hit max iterations
                FINISHED = True
            elif best_reward > args.tol_for_valid_counter_example and iter_since_valid_counter_example > 60:   # Found best counter-example but going for 60 more iterations
                iter_since_valid_counter_example += 1
                FINISHED = True
                
            elif best_reward > args.tol_for_valid_counter_example: # Found a counter-example
                # If first time finding a vlid counter-example
                if iter_since_valid_counter_example == 0:
                    # Loop over all unique positive super_rewards
                    for reward in np.unique(super_rewards):
                        if reward > args.tol_for_valid_counter_example:

                            cur_action_ = super_actions[np.where(super_rewards == reward)[0][0]]
                            if args.isoceles_triangle:
                                adjmat_counter_example = action_to_grid(args,cur_action_)
                            else:
                                adjmat_counter_example = action_to_adj_matrix(args,cur_action_)
                            # Save the best counter-example
                            # check if this file already exists
                            if not os.path.exists('{}/valid_counter_example_score_{:.5f}.npz'.format(args.output_folder,reward)):
                                np.savez('{}/valid_counter_example_score_{:.5f}.npz'.format(args.output_folder,reward),
                                    adj_mat =  adjmat_counter_example,
                                    action = cur_action_,
                                      iter = np.array([i]) ,
                                      reward = np.array([reward]))
                iter_since_valid_counter_example += 1

            # 11. Priniting
            print("\t >>> Iter {}:  Best reward {} \t Mean all / super / elite rewards: {} / {} / {} ".format(i,  round(best_reward_this_iter, 5),round(mean_reward___top_100, 5), round(mean_reward_super_all, 5), round(mean_elite_reward, 5)))
            
            '''if abs(best_reward_this_iter) <= 0.000001:
                print("\n score is zero, exiting")
                print(best_reward_this_iter)
                return'''
            
            #print("\t >>> Iter {}: . Best individuals: {}".format(i, str(np.flip(np.sort(super_rewards)))))
            #print(    "Mean reward: " + str(mean_all_reward) + "\nSessgen: " + str(sessgen_time) + ", other: " + str(randomcomp_time) + ", select1: " + str(select1_time) + ", select2: " + str(select2_time) + ", select3: " + str(select3_time) +  ", fit: " + str(fit_time) + ", score: " + str(score_time)) 
    
            if FINISHED:

                # Save list of scalars and adj matrices
                np.savez('{}/data_lists.npz'.format(args.output_folder),
                     best_reward_list_every_iter=np.array(best_reward_list_every_iter),
                    best_reward_list_overall=np.array(best_reward_list_overall),
                    percentage_below___INF__list = np.array(percentage_below___INF__list),

                    mean_reward___all__list = np.array(mean_reward___all__list),
                    mean_reward___top_100__list = np.array(mean_reward___top_100__list),
                    mean_reward___top_50__list = np.array(mean_reward___top_50__list),
                    mean_reward___top_20__list = np.array(mean_reward___top_20__list),
                    mean_reward___top_25__list = np.array(mean_reward___top_25__list),
                    mean_reward___top_10__list = np.array(mean_reward___top_10__list),
                    mean_reward___top_2__list = np.array(mean_reward___top_2__list),

                    mean_reward_super_top_10__list = np.array(mean_reward_super_top_10__list),
                    best_reward_super__list = np.array(best_reward_super__list),
                    mean_reward_super_all__list = np.array(mean_reward_super_all__list),

                    mean_elite_rewards__list = np.array(mean_elite_rewards__list),

                    sessgen_times=np.array(sessgen_times), randomcomp_times=np.array(randomcomp_times),
                    select1_times=np.array(select1_times), select2_times=np.array(select2_times), select3_times=np.array(select3_times),
                    fit_times=np.array(fit_times), score_times=np.array(score_times),
                    best_adj_matrix_list=np.array(best_adj_matrix_list),
                    best_action_list=np.array(best_action_list)
                                                  )
                

                # if finished and found a counter_example --> then create plots already

                
                npz_files_of_counter_examples = [os.path.join(args.output_folder, f) for f in os.listdir(args.output_folder) if f.endswith('.npz')]
                npz_files_of_counter_examples = [k for k in npz_files_of_counter_examples if 'valid_counter_example' in k]
                for npz_files_of_counter_example in npz_files_of_counter_examples:
                    counter_example_data = np.load(npz_files_of_counter_example)
                    adj_mat_, reward_,iteration_ = counter_example_data['adj_mat'],counter_example_data['reward'][0],counter_example_data['iter'][0]
                    reward_ = round(reward_,6)

                    if args.isoceles_triangle:
                        pass

                    else:
                        # Plot # 1: create figures for each counter-examples with various layouts in it.
                        plot_graph_layouts_for_counter_example(  adj_matrix = adj_mat_,   reward = reward_,
                            output_folder = args.output_folder,  is_directed = args.directed,
                            node_size=600,  font_size=15,  edge_thickness=1.7, output_file_name='counter_example_graph_{}.png'.format(reward_),
                            plot_title='Various Layout of a Counter Example w/ Score = {} @ Iter = {} on {} nodes'.format(reward_,iteration_,args.n ))
                        
                        # Plot #2: Create Env demo given a conjecture
                        #create_rl_graph_gif(  adjacency_matrix=adj_mat_,   reward=reward_,   fps=args.fps,  output_folder=args.output_folder,
                        #    output_filename='env_demo_of_counter_example_graph_{}.gif'.format(reward_), node_layout= 'fixed', layout_seed = 69,  highlight_existing_edges=True,
                        #    colors = args.colors,   state_vector_colors = args.state_vector_colors, gif_title = 'Demo of Valid Counter Example - # nodes = {} -- Score = {}'.format(args.n,reward_),
                        #    extra_frames = -10 )  


                # Plot rewards

                # Plot times


                      
                return
                    
                     
    
    except Exception as e:
        print(f"\n\n\n \t\t > > > An error occurred: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Ensure resources are cleaned up
        if args.logger == 'wandb':
            wandb.finish()
        elif args.logger == 'tensorboard':
            writer.close()




