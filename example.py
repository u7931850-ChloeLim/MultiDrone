import numpy as np
from multi_drone import MultiDrone

# Initialize the MultiDrone environment
sim = MultiDrone(num_drones=2, environment_file="environment.yaml")

# Define an initial configuration
initial_configuration = np.array([
    [1.0, 1.0, 1.0], # The initial xyz-position of the first drone
    [3.0, 1.0, 1.0]  # The initial xyz-position of the second drone 
], dtype=np.float32)

# Reset the simulator to the initial drone positions
sim.reset(initial_configuration)

# Check if a configuration is valid
configuration = np.array([
    [5.0, 4.5, 3.0],
    [3.5, 10.0, 8.0]
], dtype=np.float32)
is_valid = sim.is_valid(configuration)
print(f"is valid: {is_valid}")

# Check if a straight-line motion between 'start' and 'end' is valid
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

# Check if a configuration reached the goal
configuration = np.array([
    [5.0, 4.5, 3.0],
    [3.5, 10.0, 8.0]
], dtype=np.float32)
goal_reached = sim.is_goal(configuration)
print(f"goal reached: {goal_reached}")

# Visualize a path
paths = [
    np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32), # First waypoints
    np.array([[1, 1, 1], [2, 1, 1]], dtype=np.float32), # Second waypoint
    np.array([[2, 2, 2], [3, 2, 2]], dtype=np.float32), # Third waypoint
]
sim.visualize_paths(paths)