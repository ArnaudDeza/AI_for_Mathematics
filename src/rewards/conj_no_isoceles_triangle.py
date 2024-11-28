







def count_isosceles_triangles(included_indices,distance_matrix):
    """
    Counts the number of isosceles triangles formed by the included points.
    """
    num_isosceles = 0
    m = len(included_indices)
    # For efficiency, store distances between included points
    included_distances = {}
    for i in range(m):
        idx_a = included_indices[i]
        for j in range(i + 1, m):
            idx_b = included_indices[j]
            d_ab = distance_matrix[idx_a][idx_b]
            included_distances[(idx_a, idx_b)] = d_ab

    # Check for isosceles triangles
    for i in range(m):
        idx_a = included_indices[i]
        for j in range(i + 1, m):
            idx_b = included_indices[j]
            d_ab = included_distances[(idx_a, idx_b)]
            for k in range(j + 1, m):
                idx_c = included_indices[k]
                d_ac = included_distances.get((idx_a, idx_c), distance_matrix[idx_a][idx_c])
                d_bc = included_distances.get((idx_b, idx_c), distance_matrix[idx_b][idx_c])

                # Check for isosceles triangle
                if d_ab == d_ac or d_ab == d_bc or d_ac == d_bc:
                    num_isosceles += 1
    return num_isosceles


def state_to_points(args,state):
    """
    Extracts the list of included grid points from the state.
    """
    included_indices = [idx for idx, val in enumerate(state[:args.MYN]) if val == 1]
    included_points = [args.positions[idx] for idx in included_indices]
    return included_indices, included_points

def conj_no_isoleces(args,state):
    """
    Calculates the reward for a given state.
    Reward = Number of points included - penalty * number of isosceles triangles.
    """
    included_indices, included_points = state_to_points(args,state)
    num_points = len(included_indices)
    num_isosceles = count_isosceles_triangles(included_indices,args.distance_matrix)
    penalty = args.INF  # Adjust this value as needed 1000

    # If any isosceles triangles are formed, apply penalty
    reward = num_points - penalty * num_isosceles

    if abs(reward - args.terminal_reward) < 1e-6:
        return 1
    else:
        return -1 *(args.terminal_reward - reward)














##########################################################################
##########################################################################
#  1.  No Three Points on a Line: Place as many points as possible on an n by n grid such that no three points lie on the same straight line
#  2.  Avoiding Right Triangles: Place points on an n×n grid such that no three points form a right triangle.
#  3.  No Collinear Points: Place as many points as possible on an n×n grid such that no subset of points is collinear
#  4.  Golomb Ruler-Type Problems: Place marks along a grid such that the distances between marks are all distinct
#                               Score: Maximize the number of marks while ensuring all pairwise distances are distinct.
#  5.  Latin Square Construction: Construct a Latin square on an n×n grid (i.e., assign values such that no value repeats in any row or column)
#                               Binary decision (variant): Decide whether to assign a specific value to each cell or leave it blank (for a partial Latin square).
#                               Reinforcement learning goal: Maximize coverage of the grid while adhering to Latin square constraints.
#  6.  No Congruent Triangles

#  7.  Maximal Independent Sets on Grid Graphs

#  8.  Maximum Distance Problems

#  9.  Non-attacking Queens Problem

#  10.  Avoiding Parallelograms: Place points on the grid such that no four points form the vertices of a parallelogram

#  11.  Dominating Set Problems: Find the smallest set of grid cells such that every cell is either occupied or adjacent to an occupied cell.








########################################################################################################
########################################################################################################
# ______________________________________________________________________________________________________

#####                   Grid Game #1    -->     No Three Points on a Line


def count_colinear_triplets(included_indices, positions):
    """
    Counts the number of colinear triplets formed by the included points.
    """
    num_colinear = 0
    m = len(included_indices)
    # Check all combinations of three points
    for i in range(m):
        idx_a = included_indices[i]
        x1, y1 = positions[idx_a]
        for j in range(i + 1, m):
            idx_b = included_indices[j]
            x2, y2 = positions[idx_b]
            for k in range(j + 1, m):
                idx_c = included_indices[k]
                x3, y3 = positions[idx_c]
                # Check if the area of the triangle is zero (colinear)
                area = x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2)
                if area == 0:
                    num_colinear += 1
    return num_colinear


def conj_no_three_in_line(args, state):
    '''
    Premise: The objective is to place as many points as possible on an n×n grid such that no three points lie on the same straight line (are collinear). 
    The conjecture revolves around finding the maximum number of points that can be placed under this constraint.
    Scoring function: The reward function penalizes configurations where any three selected points are collinear. The agent aims to maximize the number of points while minimizing the number of collinear triplets.

    Calculates the reward for the 'No Three Points on a Line' game.
    Reward = Number of points included - penalty * number of colinear triplets.
    '''
    included_indices, included_points = state_to_points(args, state)
    num_points = len(included_indices)
    num_colinear = count_colinear_triplets(included_indices, args.positions)
    penalty = args.INF  # Penalty factor for colinear triplets

    # Calculate reward
    reward = num_points - penalty * num_colinear

    if abs(reward - args.terminal_reward) < 1e-6:
        return 1  # Optimal solution found
    else:
        return -1 * (args.terminal_reward - reward)
    
# ______________________________________________________________________________________________________
########################################################################################################
########################################################################################################





########################################################################################################
########################################################################################################
# ______________________________________________________________________________________________________

#####                   Grid Game #2    -->     Avoiding Right Triangles   


def count_right_triangles(included_indices, positions):
    """
    Counts the number of right triangles formed by the included points.
    """
    num_right_triangles = 0
    m = len(included_indices)
    # Precompute squared distances between points
    distances = {}
    for i in range(m):
        idx_a = included_indices[i]
        x1, y1 = positions[idx_a]
        for j in range(i + 1, m):
            idx_b = included_indices[j]
            x2, y2 = positions[idx_b]
            dx = x2 - x1
            dy = y2 - y1
            dist_sq = dx * dx + dy * dy
            distances[(idx_a, idx_b)] = dist_sq
            distances[(idx_b, idx_a)] = dist_sq  # Symmetric

    # Check all triplets for right triangles
    for i in range(m):
        idx_a = included_indices[i]
        for j in range(i + 1, m):
            idx_b = included_indices[j]
            for k in range(j + 1, m):
                idx_c = included_indices[k]
                # Get squared distances
                d_ab = distances[(idx_a, idx_b)]
                d_ac = distances[(idx_a, idx_c)]
                d_bc = distances[(idx_b, idx_c)]
                sides = sorted([d_ab, d_ac, d_bc])
                # Check Pythagorean theorem
                if abs(sides[0] + sides[1] - sides[2]) < 1e-6:
                    num_right_triangles += 1
    return num_right_triangles

def conj_no_right_triangles(args, state):
    """
    Premise: Place as many points as possible on an n×n grid such that no three points form a right triangle. 
    The conjecture seeks the maximal set of points under this constraint.

    Scoring Function: The reward function penalizes configurations where any three selected points form a right triangle. The agent aims to maximize the number of points while avoiding right triangle formations.

    Calculates the reward for the 'Avoiding Right Triangles' game.
    Reward = Number of points included - penalty * number of right triangles.
    """
    included_indices, included_points = state_to_points(args, state)
    num_points = len(included_indices)
    num_right_triangles = count_right_triangles(included_indices, args.positions)
    penalty = args.INF  # Penalty factor for right triangles

    # Calculate reward
    reward = num_points - penalty * num_right_triangles

    if abs(reward - args.terminal_reward) < 1e-6:
        return 1  # Optimal solution found
    else:
        return -1 * (args.terminal_reward - reward)

# ______________________________________________________________________________________________________
########################################################################################################
########################################################################################################






########################################################################################################
########################################################################################################
# ______________________________________________________________________________________________________

#####                   Grid Game #3    --> Golomb Ruler-Type Problems    
def count_duplicate_distances(included_indices, positions):
    """
    Counts the number of duplicate distances among the included points.
    """
    distance_counts = {}
    m = len(included_indices)
    num_duplicates = 0
    # Compute distances between all pairs
    for i in range(m):
        idx_a = included_indices[i]
        x1, y1 = positions[idx_a]
        for j in range(i + 1, m):
            idx_b = included_indices[j]
            x2, y2 = positions[idx_b]
            dx = x2 - x1
            dy = y2 - y1
            dist_sq = dx * dx + dy * dy
            if dist_sq in distance_counts:
                distance_counts[dist_sq] += 1
            else:
                distance_counts[dist_sq] = 1
    # Count duplicate distances
    for count in distance_counts.values():
        if count > 1:
            num_duplicates += count - 1
    return num_duplicates

def conj_golomb_ruler(args, state):
    """
    Premise:
    Place marks along an  n×n grid such that the distances between any two marks are all distinct. 
    The goal is to maximize the number of marks under this constraint.

    Scoring Function: The reward function penalizes configurations where duplicate distances occur between pairs of marks. The agent aims to maximize the number of marks while ensuring all pairwise distances are unique.

    Calculates the reward for the Golomb Ruler problem.
    Reward = Number of marks included - penalty * number of duplicate distances.
    """
    included_indices, included_points = state_to_points(args, state)
    num_marks = len(included_indices)
    num_duplicates = count_duplicate_distances(included_indices, args.positions)
    penalty = args.INF  # Penalty factor for duplicate distances

    # Calculate reward
    reward = num_marks - penalty * num_duplicates

    if abs(reward - args.terminal_reward) < 1e-6:
        return 1  # Optimal solution found
    else:
        return -1 * (args.terminal_reward - reward)


    
# ______________________________________________________________________________________________________
########################################################################################################
########################################################################################################




########################################################################################################
########################################################################################################
# ______________________________________________________________________________________________________

#####                   Grid Game #4    -->   Avoiding Parallelograms  

def count_parallelograms(included_indices, positions):
    """
    Counts the number of parallelograms formed by the included points.
    """
    num_parallelograms = 0
    m = len(included_indices)
    # Convert positions to a set for quick lookup
    point_set = set((positions[idx][0], positions[idx][1]) for idx in included_indices)
    # Check pairs of points to find midpoints
    for i in range(m):
        idx_a = included_indices[i]
        x1, y1 = positions[idx_a]
        for j in range(i + 1, m):
            idx_b = included_indices[j]
            x2, y2 = positions[idx_b]
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            # Look for pairs that share the same midpoint
            for k in range(m):
                if k == i or k == j:
                    continue
                idx_c = included_indices[k]
                x3, y3 = positions[idx_c]
                # Compute potential fourth point
                x4 = 2 * mid_x - x3
                y4 = 2 * mid_y - y3
                if (x4, y4) in point_set:
                    num_parallelograms += 1
    # Each parallelogram is counted multiple times; adjust accordingly
    num_parallelograms = num_parallelograms // 4
    return num_parallelograms

def conj_no_parallelograms(args, state):
    """
    Calculates the reward for the 'Avoiding Parallelograms' game.
    Reward = Number of points included - penalty * number of parallelograms.
    Premise: Place points on an  n×n grid such that no four points form the vertices of a parallelogram. The conjecture focuses on maximizing the number of points under this constraint.

    Scoring Function: The reward function penalizes configurations where any set of four points forms a parallelogram. The agent aims to maximize the number of points while avoiding such configurations.
    """
    included_indices, included_points = state_to_points(args, state)
    num_points = len(included_indices)
    num_parallelograms = count_parallelograms(included_indices, args.positions)
    penalty = args.INF  # Penalty factor for parallelograms

    # Calculate reward
    reward = num_points - penalty * num_parallelograms

    if abs(reward - args.terminal_reward) < 1e-6:
        return 1  # Optimal solution found
    else:
        return -1 * (args.terminal_reward - reward)

    
# ______________________________________________________________________________________________________
########################################################################################################
########################################################################################################




########################################################################################################
########################################################################################################
# ______________________________________________________________________________________________________

#####                   Grid Game #2    -->     


    
# ______________________________________________________________________________________________________
########################################################################################################
########################################################################################################




########################################################################################################
########################################################################################################
# ______________________________________________________________________________________________________

#####                   Grid Game #2    -->     


    
# ______________________________________________________________________________________________________
########################################################################################################
########################################################################################################







