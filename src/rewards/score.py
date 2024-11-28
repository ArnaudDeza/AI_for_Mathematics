import numpy as np
import networkx as nx


# Imports
from src.rl_algo.cem.utils import state_to_graph
from src.rewards.conjectures_graph_undirected import *

from src.rewards.conj_directed import *
from src. rewards.conj_no_isoceles_triangle import *
# Set the reward function based on the argument
reward_dict = {

    # No isoceles problem
    9999:   conj_no_isoleces,
    9998:   conj_no_three_in_line,
    9997:   count_right_triangles,
    9996:   conj_golomb_ruler,
    9995:   conj_no_parallelograms,





    ########################################################################################
    ########################################################################################
    ###>>>>>>>>>>>>>>>>>>>>>>       Directed Graphs

    #  give the order of a smallest(induced) k-universal graph
    999: conj_smallest_k_universal_graph,
    #  give the order of a smallest(induced) k-universal tournament
    #1000: conj_tournament,
    998: conj_seymour,
    997: conj_cacetta,
    996: conj_woodall,
    ########################################################################################
    ########################################################################################



    ########################################################################################
    ########################################################################################
    ###>>>>>>>>>>>>>>>>>>>>>>       Undirected Graphs

    #                                                                        Miscellaneuous
    0: conj_brouwer,       # Brouwer -- IDK if counter-example even exists to this

    #                                                          10 conj from Chemistry MCTS
    1 : conj_1_score,        # Conjecture 1 -- known counterexample with 9 nodes
    2 : conj_2_score,        # Conjecture 2 -- known counterexample with 8 nodes
    3 : conj_3_score,        # Conjecture 3 -- known counterexample with XXX nodes
    4 : conj_4_score,        # Conjecture 4 -- known counterexample with XXX nodes
    5 : conj_5_score,        # Conjecture 5 -- known counterexample with XXX nodes
    6 : conj_6_score,        # Conjecture 6 -- known counterexample with XXX nodes
    7 : conj_7_score,        # Conjecture 7 -- known counterexample with 7 nodes
    8 : conj_8_score,        # Conjecture 8 -- known counterexample with 6 nodes
    9 : conj_9_score,        # Conjecture 9 -- known counterexample with XXX nodes
    10 : conj_10_score,      # Conjecture 10 -- known counterexample with XXX nodes

    #                                                          32 conj from RL java paper
    11 : conj_11_score,      # Conjecture 11 -- known counterexample with XXX nodes
    12 : conj_12_score,      # Conjecture 12 -- known counterexample with XXX nodes
    13 : conj_13_score,      # Conjecture 13 -- known counterexample with XXX nodes
    14 : conj_14_score,      # Conjecture 14 -- known counterexample with XXX nodes
    15 : conj_15_score,      # Conjecture 15 -- known counterexample with XXX nodes
    16 : conj_16_score,      # Conjecture 16 -- known counterexample with XXX nodes
    17 : conj_17_score,      # Conjecture 17 -- known counterexample with XXX nodes
    18 : conj_18_score,      # Conjecture 18 -- known counterexample with XXX nodes
    19 : conj_19_score,      # Conjecture 19 -- known counterexample with XXX nodes
    20 : conj_20_score,      # Conjecture 20 -- known counterexample with XXX nodes
    21 : conj_21_score,      # Conjecture 21 -- known counterexample with XXX nodes
    22 : conj_22_score,      # Conjecture 22 -- known counterexample with XXX nodes
    23 : conj_23_score,      # Conjecture 23 -- known counterexample with XXX nodes
    24 : conj_24_score,      # Conjecture 24 -- known counterexample with XXX nodes
    25 : conj_25_score,      # Conjecture 25 -- known counterexample with XXX nodes
    26 : conj_26_score,      # Conjecture 26 -- known counterexample with XXX nodes
    27 : conj_27_score,      # Conjecture 27 -- known counterexample with XXX nodes
    28 : conj_28_score,      # Conjecture 28 -- known counterexample with XXX nodes
    29 : conj_29_score,      # Conjecture 29 -- known counterexample with XXX nodes
    30 : conj_30_score,      # Conjecture 30 -- known counterexample with XXX nodes
    31 : conj_31_score,      # Conjecture 31 -- known counterexample with XXX nodes
    32 : conj_32_score,      # Conjecture 32 -- known counterexample with XXX nodes
    33 : conj_33_score,      # Conjecture 33 -- known counterexample with XXX nodes
    34 : conj_34_score,      # Conjecture 34 -- known counterexample with XXX nodes
    35 : conj_35_score,      # Conjecture 35 -- known counterexample with XXX nodes
    36 : conj_36_score,      # Conjecture 36 -- known counterexample with XXX nodes
    37 : conj_37_score,      # Conjecture 37 -- known counterexample with XXX nodes
    38 : conj_38_score,      # Conjecture 38 -- known counterexample with XXX nodes
    39 : conj_39_score,      # Conjecture 39 -- known counterexample with XXX nodes
    40 : conj_40_score,      # Conjecture 40 -- known counterexample with XXX nodes
    41 : conj_41_score,      # Conjecture 41 -- known counterexample with XXX nodes
    42 : conj_42_score,      # Conjecture 42 -- known counterexample with XXX nodes


    #                                                          conj from AutoGraphix
    93: conj_graffiti_715,    # Graffiti 715 -- known counterexample with 16+ nodes
    94: conj_graffiti_139,    # Graffiti 139 -- known counterexample with 50+ nodes
    95: conj_graffiti_137,    # Graffiti 137 -- known counterexample with 67+ nodes
    96: conj_graffiti_289,    # Graffiti 289 -- known counterexample with 20 nodes
    97: conj_graffiti_301,    # Graffiti 301 -- known counterexample with 14 nodes
    98: conj_graffiti_30,    # Graffiti 30 -- known counterexample with 15 nodes
    99: conj_graffiti_29,    # Graffiti 29 -- known counterexample with 7 nodes

    
    }







def score_state_graph(args,state):
    if args.isoceles_triangle:
        score = args.calc_score(args,state)
        information = {}
    else:
        # step 0: convert state to adj_matrix and G i.e networkx graph
        G, adjMatG, edgeListG, Gdeg = state_to_graph(args, state, directed = args.directed)
        
        # step 1: compute score
        # If not tournament conjecture related
        if args.reward_function !=999:
            score, information = args.calc_score(G, adjMatG, args.INF)
        else:
            score, information = args.calc_score(G, adjMatG, args.INF,pattern_graphs = args.pattern_graphs,tournament_max_score = args.tournament_max_score)


        # step 2: optional reward shaping:  If the graph is a tree, add a reward
        if args.reward__is_tree:
            if nx.is_tree(G):
                score += args.reward_weight_is_tree_pos
                information['is_tree'] = True
                information['cyclomatic_number'] = 0 

            else:
                score -= args.reward_weight_is_tree_neg
                information['is_tree'] = False
                # Step 2a: If we know that we do want a tree, look at tree properties that we may want:
                if args.reward__use_cylomatic:
                    # Compute the cyclomatic number (number of cycles)
                    num_components = nx.number_connected_components(G)
                    cyclomatic_number = G.number_of_edges() - args.n + num_components
                    information['cyclomatic_number'] = cyclomatic_number
                    # you want to minimize this as tree's have a cyclomatic number of 0
                    score -= args.reward_weight_cyclomatic * cyclomatic_number
                else:
                    information['cyclomatic_number'] = -1*args.INF

    # Update the information dictionary with final score used
    information['final_reward'] = score
    return score, information