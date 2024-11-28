from src.rewards.utils import *



from typing import List

import networkx as nx
import numpy as np
import scipy as sp








def conj_1_score(G, adjMatG, INF = 10000):
    '''
    Score function for conjecture 1
    Conjecture: 
    
    '''
    info = {}
    if not nx.is_connected(G):
        return -INF,info
    n = G.number_of_nodes()
    max_spectrum = max(nx.adjacency_spectrum(G).real)
    matching_size = len(nx.maximal_matching(G))

    info['max_spectrum'] = max_spectrum
    info['matching_size'] = matching_size

    score = sqrt(n - 1) + 1 - max_spectrum - matching_size
    info['score'] = score
    return score, info
    
def conj_2_score(G, adjMatG, INF = 10000):
    '''
    Score function for conjecture 2
    Conjecture: 
    
    '''
    info = {}
    if not nx.is_connected(G):
        return -INF,info
    proximity_val = proximity(G)
    diameter_val = nx.diameter(G)
    floor_val = floor(2 * diameter_val / 3)
    eigenval = dist_eigenvalue(G, floor_val)

    info['proximity'] = proximity_val
    info['diameter'] = diameter_val
    info['eigenval'] = eigenval

    score = -proximity_val - eigenval
    info['score'] = score
    return score, info




def conj_3_score(G, adjMatG, INF = 10000):
    '''
    Score function for conjecture 3
    Conjecture: 
    
    '''
    info = {}
    if not nx.is_connected(G):
        return -INF,info
    n = G.number_of_nodes()

    p_A_val, m_val, p_D_val = p_A(G), m(G),  p_D(G)
    info['p_A'] = p_A_val
    info['p_D'] = p_D_val
    info['m'] = m_val
    if m_val == 0 or n == 0:
        score = 0.0
    else:
        score = abs(p_A_val / m_val - (1 - p_D_val / n)) - 0.28
    info['score'] = score
    return score, info


def conj_4_score(G, adjMatG, INF = 10000):
    '''
    Score function for conjecture 4
    Conjecture: 
    
    '''
    info = {}
    if not nx.is_connected(G):
        return -INF,info
    
    spectrum = sorted(nx.adjacency_spectrum(G).real, reverse=True)
    if len(spectrum) < 2:
        second_largest = 0.0
    else:
        second_largest = spectrum[1]

    harmonic_val = harmonic_index(G)
    
    info['harmonic_val'] = harmonic_val
    info['second_largest'] = second_largest
    score = second_largest - harmonic_val
    info['score'] = score
    return score, info


def conj_5_score(G, adjMatG, INF = 10000):
    '''
    Score function for conjecture 5
    Conjecture: 
    
    '''
    info = {}
    if not nx.is_connected(G):
        return -INF,info
    n = G.number_of_nodes()

    mod_zagreb_val = mod_zagreb_2(G)
    info['mod_zagreb_val'] = mod_zagreb_val
    score = mod_zagreb_val - (n + 1) / 4
    info['score'] = score
    return score, info

def conj_6_score(G, adjMatG, INF = 10000):
    '''
    Score function for conjecture 6
    Conjecture: 
    
    '''
    info = {}
    if not nx.is_connected(G):
        return -INF,info
    n = G.number_of_nodes()

    gamma = len(nx.dominating_set(G))
    info['gamma'] = gamma
    if (2 * n - 2 * gamma) == 0:
        score = 0.0
    else:
        score = (1 - gamma) / (2 * n - 2 * gamma) + (gamma + 1) / 2 - mod_zagreb_2(G)
    info['score'] = score
    return score, info



def conj_6_score(G, adjMatG, INF = 10000):
    '''
    Score function for conjecture 6
    Conjecture: 
    
    '''
    info = {}
    if not nx.is_connected(G):
        return -INF,info
    n = G.number_of_nodes()

    gamma = len(nx.dominating_set(G))
    info['gamma'] = gamma
    if (2 * n - 2 * gamma) == 0:
        score = 0.0
    else:
        score = (1 - gamma) / (2 * n - 2 * gamma) + (gamma + 1) / 2 - mod_zagreb_2(G)
    info['score'] = score
    return score, info




 

def conj_7_score(G, adjMatG, INF = 10000):
    '''
    Score function for conjecture 7
    Conjecture: 
    
    '''
    info = {}
    if not nx.is_connected(G):
        return -INF,info
    n = G.number_of_nodes()
    
    proximity_val = proximity(G) 
    info['proximity'] = proximity_val 
    max_spectrum = max(nx.adjacency_spectrum(G).real)
    info['max_spectrum'] = max_spectrum

    score = max_spectrum * proximity_val - n + 1
    info['score'] = score
    return score, info



def conj_8_score(G, adjMatG, INF = 10000):
    '''
    Score function for conjecture 8
    Conjecture: 
    
    '''
    info = {}
    if not nx.is_connected(G):
        return -INF,info
    n = G.number_of_nodes()
    
    proximity_val = proximity(G)
    connectivity_val = connectivity(G)
    info['proximity'] = proximity_val
    info['connectivity'] = connectivity_val
    if n == 0: score = 0.0
    elif n % 2 == 0: score = 0.5 * (n ** 2 / (n - 1)) * (1 - cos(pi / n))
    else: score = 0.5 * (n + 1) * (1 - cos(pi / n))

    score -= connectivity_val * proximity_val
    info['score'] = score
    return score, info



def conj_9_score(G, adjMatG, INF = 10000):
    '''
    Score function for conjecture 9
    Conjecture: 
    
    '''
    info = {}
    if not nx.is_connected(G):
        return -INF,info
    n = G.number_of_nodes()
    max_spectrum = max(nx.adjacency_spectrum(G).real)
    alpha = len(nx.maximal_independent_set(G))
    info['max_spectrum'] = max_spectrum
    info['alpha'] = alpha
    score = sqrt(n - 1) - n + 1 - max_spectrum + alpha
    info['score'] = score
    return score, info


def conj_10_score(G, adjMatG, INF = 10000):
    '''
    Score function for conjecture 10
    Conjecture: (Aouchiche, 2006). Let \( G \) be a connected graph on \( n \geq 3 \) vertices. Then
    R(G) + \alpha(G) \leq n - 1 + \sqrt{n - 1}.
    '''
    info = {}
    if not nx.is_connected(G):
        return -INF,info
    n = G.number_of_nodes()
    r = randic_index(G)
    alpha = len(nx.maximal_independent_set(G))
    info['randic_index'] = r
    info['alpha'] = alpha
    score = r + alpha - n + 1 - sqrt(n - 1)
    info['score'] = score
    return score, info





def conj_graffiti_29(G, adjMatG, INF = 10000):
    '''
    Score function for conjecture of graffiti 29
    R(G) <= # of distinct negative eigenvalues of G
    '''
    info = {}
    if not nx.is_connected(G):
        return -INF,info
    r = randic_index(G)
    info['randic_index'] = r
    neg_eigenvals = [eig for eig in nx.adjacency_spectrum(G).real if eig < 0]
    score = r - len(set(neg_eigenvals))
    info['score'] = score
    return score, info



def conj_graffiti_30(G, adjMatG, INF = 10000):
    '''
    Score function for conjecture of graffiti 30
    # of distinct positive eigenvalues of G <= sum of temperature of every vertex where temperature of a vertex v  is d(v) / (n-d(v)) where d(v) is the degree of v
    '''
    info = {}
    if not nx.is_connected(G):
        return -INF,info
    n = G.number_of_nodes()
    pos_eigenvals = [eig for eig in nx.adjacency_spectrum(G).real if eig > 0]
    sum_temp = sum([G.degree(v) / (n - G.degree(v)) for v in G.nodes()])
    score = len(set(pos_eigenvals)) - sum_temp
    info['score'] = score
    return score, info

def conj_graffiti_715(G, adjMatG, INF = 10000):
    '''
    Score function for conjecture of graffiti 715
    Scope(eigenvalue vector of adj matrix)<= average degree of vertices who have vertices higher or equal to the mean degree acrioss all vertices
    '''
    info = {}
    if not nx.is_connected(G):
        return -INF,info
    n = G.number_of_nodes()
    spectrum = nx.adjacency_spectrum(G).real
    scope = max(spectrum) - min(spectrum)
    mean_degree = sum([G.degree(v) for v in G.nodes()]) / n
    high_degree_vertices = [v for v in G.nodes() if G.degree(v) >= mean_degree]
    avg_degree_high_degree = sum([G.degree(v) for v in high_degree_vertices]) / len(high_degree_vertices)
    info['scope'] = scope
    info['mean_degree'] = mean_degree
    info['avg_degree_high_degree'] = avg_degree_high_degree
    score = scope - avg_degree_high_degree
    info['score'] = score
    return score, info




def conj_graffiti_301(G, adjMatG, INF = 10000):
    '''
    Score function for conjecture of graffiti 301
    Scope(eigenvalue vector of adj matrix)<= harmonic index of G 
    '''
    info = {}
    if not nx.is_connected(G):
        return -INF,info
    n = G.number_of_nodes()
    spectrum = nx.adjacency_spectrum(G).real
    scope = max(spectrum) - min(spectrum)
    harmonic_val = harmonic_index(G)
    info['scope'] = scope
    info['harmonic_val'] = harmonic_val
    score = scope - harmonic_val
    info['score'] = score
    return score, info

def conj_graffiti_289(G, adjMatG, INF = 10000):
    '''
    Score function for conjecture of graffiti 289
    2nd largest eigenvalye <= average dual degree across all vertices
    '''
    info = {}
    if not nx.is_connected(G):
        return -INF,info
    n = G.number_of_nodes()
    spectrum = sorted(nx.adjacency_spectrum(G).real, reverse=True)
    if len(spectrum) < 2:
        second_largest = 0.0
    else:
        second_largest = spectrum[1]
    # compute dual degree of each vertex i.e the avg degree of its neigbours
    dual_degrees = [sum([G.degree(n) for n in G.neighbors(v)]) / G.degree(v) for v in G.nodes()]
    avg_dual_degree = sum(dual_degrees) / n
    info['second_largest'] = second_largest
    info['avg_dual_degree'] = avg_dual_degree
    score = second_largest - avg_dual_degree
    info['score'] = score
    return score, info






def conj_graffiti_137(G, adjMatG, INF = 10000):
    '''
    Score function for conjecture of graffiti 137
    2nd largest eigenvalue <= harmonic index of G
    '''
    info = {}
    if not nx.is_connected(G):
        return -INF,info
    n = G.number_of_nodes()
    spectrum = sorted(nx.adjacency_spectrum(G).real, reverse=True)
    if len(spectrum) < 2:
        second_largest = 0.0
    else:
        second_largest = spectrum[1]
    harmonic_val = harmonic_index(G)
    info['second_largest'] = second_largest
    info['harmonic_val'] = harmonic_val
    score = second_largest - harmonic_val
    info['score'] = score
    return score, info

def conj_graffiti_139(G, adjMatG, INF = 10000):
    '''
    Score function for conjecture of graffiti 139
    negative of 2nd smallest eigenvalue <= harmonic index of G
    '''
    info = {}
    if not nx.is_connected(G):
        return -INF,info
    n = G.number_of_nodes()
    spectrum = sorted(nx.adjacency_spectrum(G).real, reverse=True)
    if len(spectrum) < 2:
        second_smallest = 0.0
    else:
        second_smallest = spectrum[-2]
    harmonic_val = harmonic_index(G)
    info['second_smallest'] = second_smallest
    info['harmonic_val'] = harmonic_val
    score = -second_smallest - harmonic_val
    info['score'] = score
    return score, info

    




















def calculate_matching_number(graph: nx.Graph) -> int: 
    """
    This function calculates all matchings for a given graph and
    it returns the matching number (i.e. the length of the maximum
    matching).
    """
    max_matching = nx.max_weight_matching(graph)
    return len(max_matching)


def calculate_max_abs_val_eigenvalue(graph: nx.Graph) -> float:
    """
    This function computes the eigenvalues of the adjacency matrix
    that corresponds to a specific graph. It returns the largest
    eigenvalue in absolute value.
    """
    adjacency_matrix = nx.adjacency_matrix(graph).todense() 
    eigenvals = np.linalg.eigvalsh(adjacency_matrix) 
    eigenvals_abs = abs(eigenvals)
    return max(eigenvals_abs)


def signless_laplacian_matrix(G, nodelist=None, weight='weight'):
    """
    Returns the signless Laplacian matrix of G, L = D + A, where
    A is the adjacency matrix and D is the diagonal matrix of node degrees.
    """
    if nodelist is None:
        nodelist = list(G)
    adj_mat = nx.to_scipy_sparse_matrix(
        G, nodelist=nodelist, weight=weight, format='csr'
    )
    n, m = adj_mat.shape
    diags = adj_mat.sum(axis=1)
    deg_mat = sp.sparse.spdiags(diags.flatten(), [0], m, n, format='csr')
    return deg_mat + adj_mat


def calculate_laplacian_eigenvalues(
    graph: nx.Graph, signless_laplacian: bool,
) -> List[float]:
    """
    This function computes the eigenvalues of the laplacian
    matrix that corresponds to a specific graph.
    """
    laplacian_matrix = nx.laplacian_matrix(graph).todense()
    if signless_laplacian: 
        laplacian_matrix = signless_laplacian_matrix(graph).todense()
    eigenvals = np.linalg.eigvalsh(laplacian_matrix)
    return eigenvals




    

def conj_brouwer(G, adjMatG, INF = 10000):
    info = {}
    if not nx.is_connected(G):
        return -INF,info
    if nx.is_regular(G):
        return -INF,info
    if nx.is_tree(G):
        return -INF,info
    

    n = G.number_of_nodes()
    m = G.number_of_edges()
    lamb = np.flip(nx.laplacian_spectrum(G))
    sums = np.cumsum(lamb)
    binomials = np.array([i*(i+1)/2 for i in range(1,n+1)])
    diff = sums - (binomials + m)
    max_reward = max(diff[2:n-2]).real


    '''eigenvals = calculate_laplacian_eigenvalues(
        graph=G, signless_laplacian=False,
    )
    n_eigenvals = len(eigenvals)
    n_edges = G.number_of_edges()
    eigenvals_list =list(eigenvals)
    

    for t in range(1, n_eigenvals+1):
        t_eigenvals = eigenvals_list[:t]
        sum_eigenvals = sum(t_eigenvals)
        reward_t = sum_eigenvals - float(n_edges) - scipy.special.comb(t+1, 2)   
        if t==1: 
            max_reward = reward_t
        else: 
            # Total reward will be the maximum reward_t
            if reward_t > max_reward:
                max_reward = reward_t   '''
    info['score'] = max_reward
    return max_reward, info



############################################################################################################
############################################################################################################
############################################################################################################
# Spectral Graph Theory Conjectures --- 68 of them from a paper



def spectral_68_conjectures(score_index,G, adjMatG, INF = 10000):
    '''
    Score function for conjecture 1 from 68 laplacian spectral conjectures
    Conjecture: 
    
    '''
    info = {}
    if not nx.is_connected(G):
        return -INF,info
    mu = compute_mu(G)
    d = np.array([G.degree(v) for v in G.nodes()], dtype=np.float64)
    neighbor_degree_sums = np.array([sum(G.degree(n) for n in G.neighbors(v)) for v in G.nodes()], dtype=np.float64)
    m = neighbor_degree_sums / np.array([len(list(G.neighbors(v))) for v in G.nodes()], dtype=np.float64)
    if score_index == 1:
        scores = np.sqrt((4 * d**3) / m)
    elif score_index == 2:
        scores = 2 * (m**2) / d
    elif score_index == 3:
        scores = (m**2) / d + m
    elif score_index == 4:
        scores = 2 * (d**2) / m
    elif score_index == 5:
        scores = (d**2) / m + m
    elif score_index == 6:
        scores = np.sqrt(m**2 + 3 * d**2)
    elif score_index == 7:
        scores = (d**2) / m + d
    elif score_index == 8:
        scores = np.sqrt(d * (m + 3 * d))
    elif score_index == 9:
        scores = 0.5 * (m + 3 * d)
    elif score_index == 10:
        scores = np.sqrt(d * (d + 3 * d))
    elif score_index == 11:
        scores = 2 * m**3 / d**2
    elif score_index == 12:
        scores = np.sqrt(2 * d**2 + 2 * m**2)
    elif score_index == 13:
        scores = 2 * m**4 / d**3
    elif score_index == 14:
        scores = 2 * d**3 / m**2
    elif score_index == 15:
        scores = np.sqrt((4 * m**3) / d)
    elif score_index == 16:
        scores = 2 * d**4 / m**3
    elif score_index == 19:
        scores = (4 * d**4 + 12 * d * m**3)**0.25
    elif score_index == 20:
        scores = np.sqrt(7 * d**2 + 9 * m**2) / 2
    elif score_index == 21:
        scores = np.sqrt((d**3 / m) + 3 * m**2)
    elif score_index == 22:
        scores = (2 * d**4 + 14 * d**2 * m**2)**0.25
    elif score_index == 23:
        scores = np.sqrt(d**2 + 3 * d * m)
    elif score_index == 24:
        scores = (6 * d**4 + 10 * m**4)**0.25
    elif score_index == 25:
        scores = (3 * d**4 + 13 * d**2 * m**2)**0.25
    elif score_index == 26:
        scores = np.sqrt(5 * d**2 + 11 * d * m) / 2
    elif score_index == 27:
        scores = np.sqrt(3 * d**2 + 5 * d * m) / 2
    elif score_index == 28:
        scores = np.sqrt(2 * m**4 / d**2 + 2 * d * m)
    elif score_index == 29:
        scores = (m**2 / d) + (3 * m**3 / d)
    elif score_index == 30:
        scores = (m**3 / d**2) + (d**2 / m)
    elif score_index == 31:
        scores = 4 * m**2 / (m + d)
    elif score_index == 32:
        scores = (m**3 * (m + 3 * d)) / d
    info['mu'] = mu
    score = mu - max(scores)
    info['score'] = score
    return score, info


def conj_11_score(G, adjMatG, INF = 10000): return spectral_68_conjectures(1,G, adjMatG, INF)
def conj_12_score(G, adjMatG, INF = 10000): return spectral_68_conjectures(2,G, adjMatG, INF)
def conj_13_score(G, adjMatG, INF = 10000): return spectral_68_conjectures(3,G, adjMatG, INF)
def conj_14_score(G, adjMatG, INF = 10000): return spectral_68_conjectures(4,G, adjMatG, INF)
def conj_15_score(G, adjMatG, INF = 10000): return spectral_68_conjectures(5,G, adjMatG, INF)
def conj_16_score(G, adjMatG, INF = 10000): return spectral_68_conjectures(6,G, adjMatG, INF)
def conj_17_score(G, adjMatG, INF = 10000): return spectral_68_conjectures(7,G, adjMatG, INF)
def conj_18_score(G, adjMatG, INF = 10000): return spectral_68_conjectures(8,G, adjMatG, INF)
def conj_19_score(G, adjMatG, INF = 10000): return spectral_68_conjectures(9,G, adjMatG, INF)
def conj_20_score(G, adjMatG, INF = 10000): return spectral_68_conjectures(10,G, adjMatG, INF)
def conj_21_score(G, adjMatG, INF = 10000): return spectral_68_conjectures(11,G, adjMatG, INF)
def conj_22_score(G, adjMatG, INF = 10000): return spectral_68_conjectures(12,G, adjMatG, INF)
def conj_23_score(G, adjMatG, INF = 10000): return spectral_68_conjectures(13,G, adjMatG, INF)
def conj_24_score(G, adjMatG, INF = 10000): return spectral_68_conjectures(14,G, adjMatG, INF)
def conj_25_score(G, adjMatG, INF = 10000): return spectral_68_conjectures(15,G, adjMatG, INF)
def conj_26_score(G, adjMatG, INF = 10000): return spectral_68_conjectures(16,G, adjMatG, INF)
def conj_27_score(G, adjMatG, INF = 10000): return spectral_68_conjectures(17,G, adjMatG, INF)
def conj_28_score(G, adjMatG, INF = 10000): return spectral_68_conjectures(18,G, adjMatG, INF)
def conj_29_score(G, adjMatG, INF = 10000): return spectral_68_conjectures(19,G, adjMatG, INF)
def conj_30_score(G, adjMatG, INF = 10000): return spectral_68_conjectures(20,G, adjMatG, INF)
def conj_31_score(G, adjMatG, INF = 10000): return spectral_68_conjectures(21,G, adjMatG, INF)
def conj_32_score(G, adjMatG, INF = 10000): return spectral_68_conjectures(22,G, adjMatG, INF)
def conj_33_score(G, adjMatG, INF = 10000): return spectral_68_conjectures(23,G, adjMatG, INF)
def conj_34_score(G, adjMatG, INF = 10000): return spectral_68_conjectures(24,G, adjMatG, INF)
def conj_35_score(G, adjMatG, INF = 10000): return spectral_68_conjectures(25,G, adjMatG, INF)
def conj_36_score(G, adjMatG, INF = 10000): return spectral_68_conjectures(26,G, adjMatG, INF)
def conj_37_score(G, adjMatG, INF = 10000): return spectral_68_conjectures(27,G, adjMatG, INF)
def conj_38_score(G, adjMatG, INF = 10000): return spectral_68_conjectures(28,G, adjMatG, INF)
def conj_39_score(G, adjMatG, INF = 10000): return spectral_68_conjectures(29,G, adjMatG, INF)
def conj_40_score(G, adjMatG, INF = 10000): return spectral_68_conjectures(30,G, adjMatG, INF)
def conj_41_score(G, adjMatG, INF = 10000): return spectral_68_conjectures(31,G, adjMatG, INF)
def conj_42_score(G, adjMatG, INF = 10000): return spectral_68_conjectures(32,G, adjMatG, INF)



############################################################################################################
############################################################################################################
############################################################################################################



