

import networkx as nx
import numpy as np
from numpy.linalg import eigvals
from numpy import sqrt, cos, pi, argmax, abs
from scipy.linalg import eig
from math import floor
import math
import scipy




# from networkx documentation
def clique_number(g):
    """Returns the clique number of the graph.

    The clique number of a graph is the size of the largest clique in the graph."""
    return max(len(c) for c in nx.find_cliques(g))


# from grinpy source code
def _topological_index(G, func):
    """Return the topological index of ``G`` determined by ``func``"""

    return math.fsum(func(*edge) for edge in G.edges())


# from grinpy source code
def randic_index(G):
    r"""Returns the Randić Index of the graph ``G``.

    The *Randić index* of a graph *G* with edge set *E* is defined as the
    following sum:

    .. math::
        \sum_{vw \in E} \frac{1}{\sqrt{d_G(v) \times d_G(w)}}

    Parameters
    ----------
    G : NetworkX graph
        An undirected graph.

    Returns
    -------
    float
        The Randić Index of a ``G``.

    References
    ----------

    Ivan Gutman, Degree-Based Topological Indices, Croat. Chem. Acta 86 (4)
    (2013) 351-361. http://dx.doi.org/10.5562/cca2294
    """
    _degree = functools.partial(nx.degree, G)
    return _topological_index(
        G, func=lambda x, y: 1 / math.sqrt(_degree(x) * _degree(y))
    )

def compute_mu(G):
    # Compute mu the spectral radius of the graph
    evals = np.linalg.eigvalsh(nx.adjacency_matrix(G).todense())
    lambda1 = max(np.abs(evals))
    return lambda1


def compute_d(G):
    # Compute the d of the graph
    d = [d for n, d in G.degree()]
    return d

def compute_avg_degree(G):
    # For every node compute the average degree of its neighbors
    avg_degree = 0
    for node in G.nodes():
        avg_degree += G.degree(node)
    avg_degree /= G.number_of_nodes()
    return avg_degree


  
def proximity(G):
    """
    Compute the proximity of a graph G.
    
    Parameters:
    - G: networkx.Graph - A NetworkX graph object
    
    Returns:
    - float: The proximity of the graph
    """
    n = len(G)
    shortest_paths = nx.floyd_warshall_numpy(G)  # Compute shortest paths as a dense matrix
    avg_distances = shortest_paths.sum(axis=1) / (n - 1)  # Average distance per node
    return avg_distances.min()  # Return the minimum average distance


def dist_eigenvalue(G, n):
    '''Returns the n-th largest eigenvalue of the distance matrix of G'''
    dist_matrix = nx.floyd_warshall_numpy(G)
    dist_spectrum = eigvals(dist_matrix)
    dist_spectrum.sort()
    return dist_spectrum[-n]

def mod_zagreb_2(G):
    '''Returns the modified second Zagreb index of G'''
    return sum(1 / (G.degree[u] * G.degree[v]) for u, v in G.edges)


def p_A(G):
    #Returns the peak location of the non-zero coefficients of the characteristic polynomial
    char_poly = np.poly(nx.adjacency_matrix(G).todense())
    coefs = np.abs(char_poly)
    nonzero_coefs = coefs[coefs != 0]
    return argmax(nonzero_coefs) + 1


def p_D(G):
    '''Returns the peak location of the normalized coefficients of the distance matrix'''
    dist_matrix = nx.floyd_warshall_numpy(G)
    char_poly = np.poly(dist_matrix)
    abs_coefs = np.abs(char_poly)
    n = G.number_of_nodes()
    norm_coefs = abs_coefs * [2**(k+2-n) for k in range(n + 1)]
    return argmax(norm_coefs)

def m(G):
    '''Returns the number of non-zero coefficients of CPA(G)'''
    char_poly = np.poly(nx.adjacency_matrix(G).todense())
    coefs = np.abs(char_poly)
    num_nonzero_coefs = np.sum(coefs != 0)
    return num_nonzero_coefs


def randic_index(graph):
    """ 
    Compute the Randic index of a graph efficiently.
    Parameters: graph (networkx.Graph): The input graph.
    Returns: float: The Randic index of the graph.
    """
    return sum(1 / math.sqrt(graph.degree[u] * graph.degree[v]) for u, v in graph.edges)


def harmonic_index(G):
    '''Returns the harmonic index of G'''
    return sum([2/(G.degree(u) + G.degree(v)) for u, v in G.edges()])


def connectivity(G):
    '''Returns the algebraic connectivity of G'''
    laplacian = nx.laplacian_matrix(G).todense()
    eigenvalues = np.sort(eigvals(laplacian))
    return eigenvalues[1]



#pattern_grap_dict = {
#    3:pattern_graphs_2,        # k is 2
#    5:pattern_graphs_3,        # k is 3
#    8:pattern_graphs_4,       # k is 4
#    10:pattern_graphs_5,      # k is 5
#}

def load_pattern_graphs(base_dir,num_nodes):
    # num nodes to k and num nodes to max attainable score i.e induced sub graphs
    if num_nodes == 3:      k,tournament_max_score = 2,2
    elif num_nodes == 5:    k,tournament_max_score = 3,4
    elif num_nodes == 8:    k,tournament_max_score = 4,11
    elif num_nodes == 10:   k,tournament_max_score = 5,34
    elif num_nodes == 14:   k,tournament_max_score = 6,156
    elif num_nodes == 16:   k,tournament_max_score = 7,1044
    pattern_graphs = nx.read_graph6(base_dir+'/graph{}.g6'.format(k))
    return pattern_graphs,tournament_max_score
    