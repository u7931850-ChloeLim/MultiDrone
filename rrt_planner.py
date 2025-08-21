import numpy as np
from multi_drone import MultiDrone

def flatten(q):
    """
    flatten (N,3) array into 1D vector

    Input:
        q: (N,3) array such that N is the number of drones and 3 is the configuration of each drone
    Output:
        A 1D vector of length 3N that contains all drone's configuration
    """
    return q.reshape(-1)

def distance(q1, q2):
    """Calculate the Euclidean distance between two points q1 and q2."""
    return np.linalg.norm(flatten(q1) - flatten(q2))

def steer(q_from, q_to, max_step):
    """Steer from one node to another, step-by-step."""
    d = distance(q_from, q_to)
    # If q_to is within max_step, return q_to directly
    if d <= max_step:
        return q_to
    # Otherwise, move from q_from toward q_to by a step of size max_step
    alpha = max_step / d
    return (1 - alpha) * q_from + alpha * q_to

def get_random_node(sim):
    """Get a random node in the bound and check collision-free."""
    # Get lower and upper bounds
    lower = sim._bounds[:,0]
    upper = sim._bounds[:,1]
    while True:
        # Sample a random node
        q = np.random.uniform(lower, upper, size=(sim.N,3))
        # Check collision-free
        if sim.is_valid(q):
            return q

def nearest_index(V,q):
    """
    Find an index of the nearest node to q.

    Input:
        V: (N,3) array of N in the RRT Tree
        q: Random node to be compared
    Output:
    Index of the nearest node to q.
    """
    # Initialise the best index
    best_idx = 0
    best_dist = distance(V[0], q)
    # Find an index
    for idx in range(1,len(V)):
        dist = distance(V[idx], q)
        if dist < best_dist:
            best_dist = dist
            best_idx = idx
    return best_idx

def build_path(V, parents, goal_i):
    """Make a path from start to goal by following parent links"""
    path = []
    # Start from the goal node
    i = goal_i
    # Loop until there is no parent
    while i is not None:
        path.append(V[i])
        i = parents[i]
    # Reverse the list to make it right direction
    path.reverse()
    return path

def rrt(q_start, K, D, sim, q_goal):
    """
    Input:
        q_start: Initial configuration
        K: Number of iterations
        D: Max step size
        sim: MultiDrone simulation
        q_goal: Goal configuration
    Output:
        Path between q_start and q_goal
    """

    # 1) Initialise G
    V = [q_start]
    parents = [None]

    # 2) For k = 1 to K do
    for _ in range(K):

        # a. get random sample from C
        if np.random.rand() < 0.2:
            q_rand = q_goal
        else:
            q_rand = get_random_node(sim)

        # b. get the nearest node in V to q
        nearest_i = nearest_index(V, q_rand)
        q_near = V[nearest_i]

        # c. If ||q - q'|| = 0 continue
        if distance(q_rand, q_near) == 0.0:
            continue

        # d-f. create q_new by moving from q_near toward q_rand
        q_new = steer(q_near, q_rand, D)

        # g. check if the path from q_near to q_new is collision-free
        if sim.motion_valid(q_near, q_new):
            # i. Add q_new to V
            V.append(q_new)
            parents.append(nearest_i)

            # Check if a q_new already inside the goal area
            if sim.is_goal(q_new):
                return build_path(V, parents, len(V) - 1)

            # Try to connect q_new to the goal
            if distance(q_new, q_goal) <= D and sim.motion_valid(q_new, q_goal):
                # Add q_goal
                V.append(q_goal)
                # Parent is q_new
                parents.append(len(V) - 2)
                # Return the completed path
                return build_path(V, parents, len(V) - 1)

    return None



