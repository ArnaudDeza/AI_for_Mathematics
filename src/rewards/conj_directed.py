import math
import networkx as nx
import numpy as np


tournament_dict = {
    1:1,        # k is 1
    3:2,        # k is 2            # Got score of +1 lets goo
    5:4,        # k is 3            # Got score of +1 lets goo
    8:11,       # k is 4            # Got score of +1 lets goo
    10:34,      # k is 5           
    14:156,     # k is 6
    16:1044,    # k is 7    
}



def has_self_loops(adj_matrix):
    return np.any(np.diag(adj_matrix) != 0)
def has_parallel_edges(adj_matrix):
    return np.any(adj_matrix > 1)
def has_two_edge_cycles(adj_matrix):
    return np.any((adj_matrix * adj_matrix.T) > 0)



def conj_cacetta(G, adjMatG, INF = 10000):
    info = {}
    n = G.number_of_nodes()

    # Compute the outdegree of each node
    out_degrees = [G.out_degree(node) for node in G.nodes()]
    r = min(out_degrees)

    # Check for multiple edges (no multiple edges allowed in simple graphs)
    # Since G is a DiGraph, unless specified otherwise, it should not have multiple edges
    # However, we can check adj_mat for entries greater than 1
    if np.any(adjMatG > 1):
        # Multiple edges detected, not a simple graph
        return -INF,info

    if r == 0:
        # Minimum outdegree less than required (should be at least 1)
        return -INF,info

    # Find all simple cycles in G
    cycles = list(nx.simple_cycles(G))

    if not cycles:
        # No cycles in G, minimal cycle length is infinite
        return -INF,info
    else:
        cycle_lengths = [len(cycle) for cycle in cycles]
        min_cycle_length = min(cycle_lengths)

        score = min_cycle_length - math.ceil(n / r)
        info["score"] = score
        return score,info




def conj_seymour(G, adjMatG, INF = 10000):
    info = {}
    n = G.number_of_nodes()
    """
    Compute the reward for a candidate counter-example to Seymour's conjecture.
    
    Args:
        adj_matrix (np.ndarray): The adjacency matrix of the input undirected graph.
                                 It's assumed that this will be an undirected graph
                                 (i.e., symmetric adjacency matrix).
                                 
    Returns:
        float: The score/reward for the candidate graph. A positive score indicates
               a valid counter-example, while a non-positive score indicates otherwise.
    """
    n = adjMatG.shape[0]  # Number of vertices in the graph
    max_violation = -np.inf  # Track the maximum violation across all vertices

    if has_parallel_edges(adjMatG) or has_two_edge_cycles(adjMatG):
        # Graph has parallel edges or two-edge cycles, not a simple graph
        return -INF,info

    for v in range(n):
        # Get the first neighborhood N1(v)
        N1 = np.where(adjMatG[v] == 1)[0]
        
        # Get the second neighborhood N2(v)
        N2 = set()
        for u in N1:
            neighbors_of_u = np.where(adjMatG[u] == 1)[0]
            N2.update(neighbors_of_u)
        N2.discard(v)  # Exclude the vertex itself
        N2 -= set(N1)  # Exclude the first neighborhood
        
        # Compute sizes of N1(v) and N2(v)
        size_N1 = len(N1)
        size_N2 = len(N2)
        
        # Compute the violation for this vertex
        violation = size_N1 - size_N2
        
        # Track the maximum violation
        max_violation = max(max_violation, violation)

    # If max_violation > 0, the graph is a valid counter-example
    # Otherwise, return the negative of the maximum violation
    score = max_violation
    info["score"] = score
    return score,info
    

def conj_smallest_k_universal_graph(G, adjMatG, INF = 10000,pattern_graphs = None,tournament_max_score = None):
    info = {}
    n = G.number_of_nodes()
    num_induced = 0
    use_nx_ = True
    if use_nx_:
        for pattern_graph in pattern_graphs:
            GM = nx.isomorphism.GraphMatcher(G,pattern_graph)
            if GM.subgraph_is_isomorphic():
                num_induced += 1
    else:
        G_ = ig.Graph.from_networkx(G)
        for pattern_graph in pattern_graphs:
            g1 = ig.Graph.from_networkx(pattern_graph)
            if G_.subisomorphic_lad(g1):
                num_induced += 1

    score = 1 if num_induced == tournament_max_score else -1*(tournament_max_score - num_induced)
    info["score"] = score
    return score,info


def conj_woodall(G,adjMatG, INF = 10000):
    info = {}
    n = G.number_of_nodes()
    # Compute the size of the minimum directed cut (k)
    try:
        # Use built-in function to find the minimum edge cut
        min_cut_value = nx.algorithms.connectivity.minimum_st_node_cut(G)
        k = len(min_cut_value)
    except nx.NetworkXError:
        # If the graph is not connected
        k = 0
    # Approximate the number of disjoint dijoins
    # Here, we use the fact that the number of disjoint dijoins is at most the minimum out-degree
    min_out_degree = min(dict(G.out_degree()).values())
    num_dijoins = min_out_degree
    # Compute the score
    score = k - num_dijoins
    info["score"] = score
    return score,info
