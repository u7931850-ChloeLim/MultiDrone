import numpy as np
from multi_drone import MultiDrone
from rrt_planner import rrt

def main():
    # 1) Create the simulation with 2 drones and load the environment from YAML
    sim = MultiDrone(num_drones=2, environment_file="environment.yaml")

    # 2) Get start and goal states (shape: (N, 3))
    #    start = drones' initial positions; goal = target positions for each drone
    q_start = sim.initial_configuration
    q_goal  = sim.goal_positions

    # (Optional) Fix randomness for reproducible runs
    np.random.seed(0)

    # 3) Set RRT hyperparameters
    K = 30000   # number of iterations (larger = higher success chance, slower)
    D = 2.0     # max step length per expansion (tune 1.5~2.5)

    # 4) Run the planner
    print("Running RRT...")
    path = rrt(q_start, K, D, sim, q_goal)

    # 5) Check the result and visualize if a path was found
    if path is None:
        print("No path found. Try increasing K, adjusting D, or adding goal bias in rrt().")
    else:
        print(f"Path found! length = {len(path)}")
        sim.visualize_paths(path)  # draws colored trajectories for each drone

if __name__ == "__main__":
    main()
