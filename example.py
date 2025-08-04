import numpy as np
from multi_drone import MultiDrone

# Initialize the MultiDrone environment
sim = MultiDrone(num_drones=2, environment_file="environment.yaml")

# Obtain the initial configuration and the goal positions
initial_configuration = sim.initial_configuration
goal_positions = sim.goal_positions

# Once the MultiDrone environment is initialized,
# you can use it within a sampling-based motion planner, e.g.
'''
solution_path = my_planner(sim)
'''

# In the planner, you can use the following functions of the MultiDrone environment:

# 1.) Check if a configuration is valid
configuration = np.array([
    [5.0, 4.5, 3.0],
    [3.5, 10.0, 8.0]
], dtype=np.float32)
is_valid = sim.is_valid(configuration)
print(f"is valid: {is_valid}")

# 2.) Check if a straight-line motion between 'start' and 'end' is valid
start = np.array([
    [5.0, 4.5, 3.0],  # The start point of the first drone
    [3.5, 10.0, 8.0]  # The start point of the second drone 
], dtype=np.float32) 
end = np.array([
    [10.0, 20.0, 3.0],  # The end point of the first drone
    [3.5, 20.0, 15.0]  # The end point of the second drone 
], dtype=np.float32)
motion_valid = sim.motion_valid(start, end) 
print(f"motion valid: {motion_valid}")

# 3.) Check if a configuration reached the goal
configuration = np.array([
    [5.0, 4.5, 3.0],
    [3.5, 10.0, 8.0]
], dtype=np.float32)
goal_reached = sim.is_goal(configuration)
print(f"goal reached: {goal_reached}")

# 4.) Visualize a path
paths = [
    np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32), # First waypoints
    np.array([[1, 1, 1], [2, 1, 1]], dtype=np.float32), # Second waypoints
    np.array([[2, 2, 2], [3, 2, 2]], dtype=np.float32), # Third waypoints
]
sim.visualize_paths(paths)