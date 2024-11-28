import networkx as nx
import random
from copy import deepcopy

def add_randleaf(G):
    '''Adds a random leaf to graph G'''
    n = G.number_of_nodes()
    # Select a random existing node
    random_vertex = random.choice(list(G.nodes()))
    # Add a new node connected to the random_vertex
    G.add_edge(random_vertex, n)

def add_leaf(G, v):
    '''Adds a leaf adjacent to a specific vertex v in G'''
    n = G.number_of_nodes()
    # Add a new node connected to vertex v
    G.add_edge(v, n)

def add_randsubdiv(G):
    '''Subdivides a random edge in G by adding a new node between two connected nodes'''
    # Select a random edge to subdivide
    random_edge = random.choice(list(G.edges()))
    # Remove the selected edge
    G.remove_edge(*random_edge)
    # Add a new node
    new_node = G.number_of_nodes()
    # Connect the new node between the two nodes of the removed edge
    G.add_edge(random_edge[0], new_node)
    G.add_edge(new_node, random_edge[1])

def NMCS_trees(current_graph, depth, level, score_function, is_parent=True):
    '''Nested Monte Carlo Search (NMCS) algorithm tailored for trees'''
    # Initialize the best graph and score
    best_graph = deepcopy(current_graph)
    best_score = score_function(current_graph)
    
    if level == 0:
        # At the lowest level, perform random moves
        next_graph = deepcopy(current_graph)
        for i in range(depth):
            if random.random() < 0.5:
                # With 50% chance, add a random leaf
                add_randleaf(next_graph)
            else:
                # Otherwise, subdivide a random edge
                add_randsubdiv(next_graph)
        # Update the best graph if the new score is better
        if score_function(next_graph) > best_score:
            best_graph = deepcopy(next_graph)
    else:
        # At higher levels, explore possible moves
        for x in list(current_graph.nodes()) + list(current_graph.edges()):
            next_graph = deepcopy(current_graph)
            if isinstance(x, tuple):
                # If x is an edge, subdivide it
                add_randsubdiv(next_graph)
            else:
                # If x is a node, add a leaf to it
                add_leaf(next_graph, x)
            # Recursively call NMCS at a lower level
            next_graph = NMCS_trees(next_graph, depth, level-1, score_function, False)
            # Update the best graph if the new score is better
            if score_function(next_graph) > best_score:
                best_graph = deepcopy(next_graph)
                best_score = score_function(next_graph)
                # Early exit condition for large graphs
                if current_graph.number_of_nodes() > 20 and is_parent:
                    break
    return best_graph

def NMCS_connected_graphs(current_graph, depth, level, score_function, is_parent=True):
    '''NMCS algorithm for connected graphs, not limited to trees'''
    best_graph = deepcopy(current_graph)
    best_score = score_function(current_graph)
    
    if level == 0:
        # At the lowest level, perform random moves
        next_graph = deepcopy(current_graph)
        for i in range(depth):
            random_number = random.random()
            if random_number < 0.5 and len(list(nx.complement(next_graph).edges())) != 0:
                # With 50% chance, add a random edge from the complement graph
                random_edge = random.choice(list(nx.complement(next_graph).edges()))
                next_graph.add_edge(*random_edge)
            elif random_number < 0.8:
                # With 30% chance, add a random leaf
                add_randleaf(next_graph)
            else:
                # With 20% chance, subdivide a random edge
                add_randsubdiv(next_graph)
        # Update the best graph if the new score is better
        if score_function(next_graph) > best_score:
            best_graph = deepcopy(next_graph)
    else:
        # At higher levels, explore possible moves
        possible_actions = (
            list(current_graph.nodes()) +
            list(current_graph.edges()) +
            list(nx.complement(current_graph).edges())
        )
        for x in possible_actions:
            next_graph = deepcopy(current_graph)
            if x in current_graph.nodes():
                # If x is a node, add a leaf to it
                add_leaf(next_graph, x)
            elif x in current_graph.edges():
                # If x is an edge, subdivide it
                add_randsubdiv(next_graph)
            else:
                # If x is a non-edge (from the complement), add it
                next_graph.add_edge(*x)
            # Recursively call NMCS at a lower level
            next_graph = NMCS_connected_graphs(next_graph, depth, level-1, score_function, False)
            # Update the best graph if the new score is better
            if score_function(next_graph) > best_score:
                best_graph = deepcopy(next_graph)
                best_score = score_function(next_graph)
                # Early exit condition for large graphs
                if current_graph.number_of_nodes() > 20 and is_parent:
                    break
    return best_graph