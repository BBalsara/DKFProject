# code for simulation and running tests
import utilities as util
import numpy as np

# Simulation class



# Cameras (Position, FOV)
fov = np.radians(120)
bearing1 = np.radians(45) 
bearing2 = np.radians(225) 
position1 = np.array([0,0])
position2 = np.array([10,10])
dist_noise = 0.1
angle_noise = 0.1
sensor1 = util.Sensor(position1, fov, bearing1, dist_noise, angle_noise)
sensor2 = util.Sensor(position2, fov, bearing2, dist_noise, angle_noise)
world = util.World(np.array([sensor1, sensor2]))
world.FOVplot()

#