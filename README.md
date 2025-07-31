
# MultiDrone Simulator

The `MultiDrone` class provides a simple, self-contained simulation environment for multiple drones operating in a 3D space with static obstacles and goal regions.

Each drone is modelled as a sphere and can be individually positioned, checked for collisions, and visualized along a planned trajectory.

## Usage
### Defining environments
Before starting the MultiDrone simulator, we have to define an environment the drones operate in. The environment is defined in a YAML file, for example ```environment.yaml``` . It consists of **obstacles** and **goals**. Obstacles are either box, sphere or cylinder shaped, whereas goals are sphere shaped. The following example shows the definition of an environment with four obstacles and two goals:

```
obstacles:
  - type: box
    position: [10, 10, 1] # The 3D position of the obstacle in the environment
    size: [4, 4, 2] # The side lengths of the box-shaped obstacle
    rotation: [0, 0, 0]  # Euler angles in degrees
    color: red
    
  - type: cylinder
    endpoints: [[15, 10, 0], [15, 10, 15]] # The 3D endpoints of the cylinder
    radius: 2.5 # The radius of the cylinder in meters
    rotation: [0, 0, 0]
    color: red

  - type: sphere
    position: [5, 10, 2]
    radius: 2.0 # The radius of the sphere in meters
    color: red

  - type: box
    position: [18, 10, 1] # The 3D position of the obstacle in the environment
    size: [4, 4, 2] # The side lengths of the box-shaped obstacle
    rotation: [0, 0, 0]  # Euler angles in degrees
    color: red
    
goals:
  - position: [30, 30, 2] # The 3D position of the first goal
    radius: 1.0 # The radius of the first goal
  - position: [20, 30, 2] # The 3D position of the second goal
    radius: 1.0 # The radius of the second goal
```
Note that the number of goals must be consistent with the number of drones in the environment. We assume that the goal of the first drone is the first goal in the environment YAML file, the goal of the second drone is the second goal, and so on.

### Instantiate the simulator
After defining the environment the drones operate in, we can instantiate the simulator via

```
import numpy as np
from multi_drone import MultiDrone

sim = MultiDrone(num_drones=2, environment_file="environment.yaml")
```
The constructor of the ```MultiDrone``` class provides two keyword arguments: ```num_drones``` specifies the number of drones, while ```environment_file``` specifies the path to the YAML environment file. 

### Using the MultiDrone simulator
The MultiDrone class provides several functions for collision checking, goal detection and visualization as listed below. The **state** of the simulation is represented by a NumPy array of shape ```(num_drones, 3)```, which specifies the xyz-position of each drone in the environment.

#### Resetting the simulation
To reset the simulator to an initial state, we can use the ```MultiDrone.reset``` function. For instance, for an environment with two drones, we can use
```
# Define an initial state
initial_state = np.array([
	[1.0, 1.0, 1.0], # The initial xyz-position of the first drone
	[3.0, 1.0, 1.0]  # The initial xyz-position of the second drone 
], dtype=np.float32)
```
```
# Reset the simulator to the initial drone positions
sim.reset(initial_state)
```
#### Collision checking
The MultiDrone class provides two functions for collision checking: ```MultiDrone.collides``` and ```MultiDrone.motion_collides```. The first function checks, for a given state, if any of the drones collide with the obstacles in the environment or with another drone:

```
state = np.array([
	[5.0, 4.5, 3.0],
	[3.5, 10.0, 8.0]
], dtype=np.float32)

collides = sim.collides(state)
```
The second function checks if any of the drones collide with the obstacles or with another drone along a straight-line motion from a start point to an end point:

```
start = np.array([
	[5.0, 4.5, 3.0],  # The start point of the first drone
	[3.5, 10.0, 8.0]  # The start point of the second drone 
], dtype=np.float32)
 
end = np.array([
	[10.0, 20.0, 3.0],  # The end point of the first drone
	[3.5, 20.0, 15.0]  # The end point of the second drone 
], dtype=np.float32)

motion_collides = sim.motion_collides(start, end) 
```
#### Goal detection
The MultiDrone class provides a ```MultiDrone.is_goal``` function which, for a given state, checks if all drones have reached their respective goal areas:

```
state = np.array([
	[5.0, 4.5, 3.0],
	[3.5, 10.0, 8.0]
], dtype=np.float32)

goal_reached = sim.is_goal(state)
```

#### Path visualization
To visualize paths, the MultiDrone class provides the ```MultiDrone.visualize_paths``` function. A path is defined as a sequence of waypoints for all drones:
```
paths = [
    np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32), # First waypoints
    np.array([[1, 1, 1], [2, 1, 1]], dtype=np.float32), # Second waypoint
    np.array([[2, 2, 2], [3, 2, 2]], dtype=np.float32), # Third waypoint
]

sim.visualize_paths(paths)
```
